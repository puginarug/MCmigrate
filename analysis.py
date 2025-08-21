"""
analysis.py
Simple analysis functions for DACF and MSD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_dacf(df, max_lag=None):
    """
    Calculate Directional Autocorrelation Function.
    
    Parameters:
    -----------
    df : DataFrame
        Trajectory data with columns: track_id, step, v_x, v_y
    max_lag : int
        Maximum lag to calculate (None = all available)
        
    Returns:
    --------
    DataFrame with columns: lag, msd
    """
    tracks = df.groupby('track_id')
    
    # Find maximum possible lag
    min_track_length = df.groupby('track_id').size().min()
    
    if max_lag is None:
        max_lag = min_track_length - 1
    else:
        max_lag = min(max_lag, min_track_length - 1)
    
    msd_values = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            msd_values.append(0)
        else:
            displacements = []
            
            for track_id, track in tracks:
                x = track['x'].values
                y = track['y'].values
                
                # Calculate displacements for this lag
                for i in range(len(x) - lag):
                    dx = x[i + lag] - x[i]
                    dy = y[i + lag] - y[i]
                    displacements.append(dx**2 + dy**2)
            
            if displacements:
                msd_values.append(np.mean(displacements))
            else:
                msd_values.append(0)
    
    return pd.DataFrame({
        'lag': range(max_lag + 1),
        'msd': msd_values
    })


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
    sim_dacf = calculate_dacf(sim_df, max_lag)
    sim_msd = calculate_msd(sim_df, max_lag)
    
    if exp_df is not None:
        # Calculate metrics for experimental data
        exp_dacf = calculate_dacf(exp_df, max_lag)
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
    Returns:
    --------
    DataFrame with columns: lag, dacf
    """
    # Pivot data to wide format
    vx_pivot = df.pivot(index='track_id', columns='step', values='v_x')
    vy_pivot = df.pivot(index='track_id', columns='step', values='v_y')
    
    vx = vx_pivot.values
    vy = vy_pivot.values
    
    n_cells, n_steps = vx.shape
    
    if max_lag is None:
        max_lag = n_steps - 1
    else:
        max_lag = min(max_lag, n_steps - 1)
    
    dacf_values = []
    
    for lag in range(max_lag + 1):
        # Calculate dot products for this lag
        if lag == 0:
            # Autocorrelation at lag 0 is always 1
            dacf_values.append(1.0)
        else:
            dot_products = []
            
            for i in range(n_steps - lag):
                # Velocity at time t
                v1_x = vx[:, i]
                v1_y = vy[:, i]
                
                # Velocity at time t + lag
                v2_x = vx[:, i + lag]
                v2_y = vy[:, i + lag]
                
                # Calculate normalized dot product for each cell
                for j in range(n_cells):
                    mag1 = np.sqrt(v1_x[j]**2 + v1_y[j]**2)
                    mag2 = np.sqrt(v2_x[j]**2 + v2_y[j]**2)
                    
                    if mag1 > 0 and mag2 > 0:
                        dot = (v1_x[j] * v2_x[j] + v1_y[j] * v2_y[j]) / (mag1 * mag2)
                        dot_products.append(dot)
            
            if dot_products:
                dacf_values.append(np.mean(dot_products))
            else:
                dacf_values.append(0)
    
    return pd.DataFrame({
        'lag': range(max_lag + 1),
        'dacf': dacf_values
    })


def calculate_msd(df, max_lag=None):
    """
    Calculate Mean Squared Displacement.
    
    Parameters:
    -----------
    df : DataFrame
        Trajectory data with columns: track_id, step, x, y
    max_lag : int
        Maximum lag to calculate (None = all available)
        