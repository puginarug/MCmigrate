import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_autocorrelation(df, max_lag=None, directional=True):
    """
    Calculate velocity autocorrelation function.
    
    Parameters:
    -----------
    df : DataFrame
        Trajectory data with columns: track_id, step, v_x, v_y
    directional : bool
        If True, compute directional autocorrelation (DACF)
        If False, compute velocity autocorrelation (VACF)
    
    Returns:
    --------
    DataFrame with autocorrelation values
    """
    # Pivot to wide format
    df_vx = df.pivot(index='track_id', columns='step', values='v_x')
    df_vy = df.pivot(index='track_id', columns='step', values='v_y')
    Vx = df_vx.values
    Vy = df_vy.values
    
    n_particles, n_steps = Vx.shape
    acorr_vals = np.empty(n_steps, dtype=float)
    
    for dt in range(n_steps):
        v1x = Vx[:, :n_steps-dt]
        v2x = Vx[:, dt:]
        v1y = Vy[:, :n_steps-dt]
        v2y = Vy[:, dt:]
        
        dot = v1x*v2x + v1y*v2y
        if max_lag is not None and dt >= max_lag:
            break
        if directional:
            # Normalize by magnitudes for DACF
            mag1 = np.sqrt(v1x**2 + v1y**2)
            mag2 = np.sqrt(v2x**2 + v2y**2)
            valid_mask = (mag1 > 0) & (mag2 > 0)
            dot[valid_mask] /= (mag1[valid_mask] * mag2[valid_mask])
            dot[~valid_mask] = np.nan
            acorr_vals[dt] = np.nanmean(dot)
            column_name = 'dacf'
        else:
            # VACF without normalization
            acorr_vals[dt] = np.nanmean(dot)
            column_name = 'vacf'
    
    acorr_df = pd.DataFrame(acorr_vals, columns=[column_name])
    acorr_df['lag'] = np.arange(n_steps)
    acorr_df['dt'] = acorr_df['lag'] * 10  # Convert to minutes
    
    return acorr_df


def calculate_msd(df, max_lag=None):
    """
    Calculate Mean Squared Displacement.
    
    Parameters:
    -----------
    df : DataFrame
        Trajectory data with columns: track_id, step, x, y
    max_lag : int, optional
        Maximum lag to compute MSD for
    
    Returns:
    --------
    DataFrame with MSD values
    """
    tracks = df.groupby('track_id')
    n_steps = df.groupby('track_id').size().max()
    
    if max_lag is None:
        max_lag = n_steps - 1
    else:
        max_lag = min(max_lag, n_steps - 1)
    
    msd_values = np.zeros(max_lag)
    counts = np.zeros(max_lag)
    
    for track_id, track in tracks:
        x = track['x'].values
        y = track['y'].values
        n = len(x)
        
        for lag in range(1, min(n, max_lag + 1)):
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            squared_disp = dx**2 + dy**2
            msd_values[lag-1] += np.sum(squared_disp)
            counts[lag-1] += len(squared_disp)
    
    # Average over all windows and tracks
    msd_values = msd_values / (counts + 1e-10)
    
    msd_df = pd.DataFrame({
        'msd': msd_values,
        'lag': np.arange(1, max_lag + 1),
        'dt': np.arange(1, max_lag + 1) * 10  # Convert to minutes
    })
    
    return msd_df


def plot_dacf(dacf_df, title='Directional Autocorrelation Function'):
    """Plot DACF."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(dacf_df['lag'], dacf_df['dacf'], 'o-', linewidth=2, markersize=6)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time Lag (steps)', fontsize=12)
    ax.set_ylabel('DACF', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.0)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_msd(msd_df, title='Mean Squared Displacement'):
    """Plot MSD with reference lines."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot MSD
    ax.plot(msd_df['lag'], msd_df['msd'], 'o-', linewidth=2, markersize=6, label='MSD')
    
    # Add reference lines
    lags = msd_df['lag'].values[1:]  # Skip lag=0
    if len(lags) > 0:
        # Ballistic motion (∝ t²)
        ballistic = lags[0]**2 * (msd_df['msd'].iloc[1] / lags[0]**2)
        ax.plot(lags, ballistic * (lags / lags[0])**2, 'r--', alpha=0.5, label='Ballistic (∝t²)')
        
        # Diffusive motion (∝ t)
        diffusive = lags[0] * (msd_df['msd'].iloc[1] / lags[0])
        ax.plot(lags, diffusive * (lags / lags[0]), 'g--', alpha=0.5, label='Diffusive (∝t)')
    
    ax.set_xlabel('Time Lag (steps)', fontsize=12)
    ax.set_ylabel('MSD (μm²)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale can be helpful
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def compare_simulations(sim_df, exp_df=None, max_lag=30):
    """
    Compare simulation with experimental data (if provided).
    
    Parameters:
    -----------
    sim_df : DataFrame
        Simulation trajectory data
    exp_df : DataFrame
        Experimental trajectory data (optional)
    max_lag : int
        Maximum lag for analysis
    """
    # Calculate metrics for simulation
    sim_dacf = calculate_autocorrelation(sim_df, max_lag, directional=True)
    sim_msd = calculate_msd(sim_df, max_lag)
    
    if exp_df is not None:
        # Calculate metrics for experimental data
        exp_dacf = calculate_autocorrelation(exp_df, max_lag, directional=True)
        exp_msd = calculate_msd(exp_df, max_lag)
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # DACF comparison
        ax1.plot(exp_dacf['lag'], exp_dacf['dacf'], 'o-', label='Experimental', alpha=0.7)
        ax1.plot(sim_dacf['lag'], sim_dacf['dacf'], 's-', label='Simulation', alpha=0.7)
        ax1.set_xlabel('Time Lag (steps)')
        ax1.set_ylabel('DACF')
        ax1.set_title('Directional Autocorrelation Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MSD comparison
        ax2.loglog(exp_msd['lag'][1:], exp_msd['msd'][1:], 'o-', label='Experimental', alpha=0.7)
        ax2.loglog(sim_msd['lag'][1:], sim_msd['msd'][1:], 's-', label='Simulation', alpha=0.7)
        ax2.set_xlabel('Time Lag (steps)')
        ax2.set_ylabel('MSD (μm²)')
        ax2.set_title('Mean Squared Displacement Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'sim_dacf': sim_dacf,
            'sim_msd': sim_msd,
            'exp_dacf': exp_dacf,
            'exp_msd': exp_msd
        }
    else:
        # Just plot simulation results
        plot_dacf(sim_dacf, 'Simulation DACF')
        plot_msd(sim_msd, 'Simulation MSD')
        
        return {
            'sim_dacf': sim_dacf,
            'sim_msd': sim_msd
        }   