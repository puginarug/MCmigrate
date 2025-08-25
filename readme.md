# Markov Chain-Based Cell Migration Simulation

A streamlined agent-based simulation for cell migration in a vertical stadium domain with Markov chain-based motion.

## Features

- **Vertical stadium domain** with proper geometry (straight walls + semicircles)
- **Vertical line source gradient** with adjustable length
- **Simple boundary conditions** (cells stop at boundaries)
- **Markov chain angle changes** based on experimental data
- **Log-normal velocity distribution**
- **Chemotaxis and cell-cell repulsion**
- **DACF and MSD analysis**

## Stadium Geometry

The stadium consists of:
- **L**: Length of straight vertical walls
- **R**: Radius of semicircles at top and bottom
- **Total height** = L + 2R
- **Gradient source**: Vertical line at center with adjustable length

## Files

```
├── cell.py              # Simple cell agent
├── stadium.py           # Vertical stadium domain with line source gradient
├── markov.py           # Markov chain for angle changes
├── simulation.py        # Main simulation engine
├── analysis.py          # DACF and MSD calculations
├── visualization.py     # Plotting and animation
└── run_simulation.py    # Main script
```

## Quick Start

### Basic Usage


```python
from simulation import Simulation
from visualization import plot_trajectories

# Create simulation with proper stadium parameters
sim = Simulation(
    n_cells=30,
    stadium_L=60,          # Length of straight walls
    stadium_R=20,          # Radius of semicircles
    source_length=40,      # Length of gradient line source
    chemotaxis_strength=0.3,
    repulsion_strength=0.2,
    initial_distribution='uniform'  # 'uniform' (inside), 'perimeter' (boundary), or 'bottom'
)

# Or, to start cells along the boundary:
# sim = Simulation(..., initial_distribution='perimeter')

# Run simulation
sim.run(n_steps=100)

# Visualize
plot_trajectories(sim)
```

### Command Line

```bash
# Run with defaults
python run_simulation.py

# With experimental data
python run_simulation.py --data your_data.csv

# Custom stadium parameters
python run_simulation.py --L 80 --R 25 --source_length 50

# All parameters
python run_simulation.py --n_cells 50 --n_steps 200 --L 60 --R 20 --chemotaxis 0.4
```

### With Experimental Data

```python
from markov import MarkovChain
from simulation import Simulation
import pandas as pd

# Load data
data = pd.read_csv('experimental_tracks.csv')
tracks = [group[['x', 'y']].values 
          for _, group in data.groupby('track_id')]

# Fit Markov chain
mc = MarkovChain()
mc.fit(tracks, B=12, n=3)

# Run simulation with fitted model
sim = Simulation(
    stadium_L=60,
    stadium_R=20,
    source_length=40,
    markov_chain=mc
)
sim.run(n_steps=100)
```

## Parameters

### Stadium Parameters
- `stadium_L`: Length of straight walls (default: 60 μm)
- `stadium_R`: Radius of semicircles (default: 20 μm)
- `source_length`: Length of vertical gradient source (default: 40 μm)

### Simulation Parameters
- `n_cells`: Number of cells (default: 30)
- `chemotaxis_strength`: Response to gradient, 0-1 (default: 0.3)
- `repulsion_strength`: Cell-cell repulsion, 0-1 (default: 0.2)
- `interaction_radius`: Range for repulsion (default: 10 μm)
- `initial_distribution`: Where cells start: `'bottom'` (default, bottom semicircle), `'uniform'` (random inside stadium), or `'perimeter'` (random along boundary)

### Markov Chain Parameters
- `B`: Number of angle bins (default: 12)
- `n`: n-gram length (default: 3)

## Gradient Field

The gradient emanates from a vertical line source at the center:
- Line extends from `-source_length/2` to `+source_length/2`
- Field strength decreases exponentially with distance from line
- Gradient points radially away from the line source
- Cells move based on gradient direction and strength

## Analysis

```python
from analysis import calculate_dacf, calculate_msd

# Get trajectory data
df = sim.get_dataframe()

# Calculate DACF
dacf = calculate_dacf(df, max_lag=30)

# Calculate MSD
msd = calculate_msd(df, max_lag=30)

# Compare with experimental data
from analysis import compare_simulations
results = compare_simulations(sim_df, exp_df)
```

## Visualization

```python
from visualization import plot_trajectories, create_animation, plot_cell_statistics

# Plot trajectories with gradient field
plot_trajectories(sim, show_gradient=True)

# Create animation
anim = create_animation(sim, save_path='migration.gif')

# Plot statistics
plot_cell_statistics(sim)
```

## Output Files

Running `run_simulation.py` generates:
- `simulation_trajectories.csv` - Cell positions over time
- `dacf_results.csv` - Directional autocorrelation function
- `msd_results.csv` - Mean squared displacement
- `migration.gif` - Animation of cell migration

## Data Format

Expected CSV format for experimental data:
```
track_id,step,x,y
0,0,10.5,20.3
0,1,10.8,21.1
...
```

Or with alternative column names:
```
track_id,normalized_time,x_microns,y_microns
```

## Example Workflow

```python
# 1. Load experimental data
data, tracks = load_experimental_data('data.csv')

# 2. Fit velocity distribution
velocity_params = fit_velocity_distribution(data)

# 3. Fit Markov chain
mc = MarkovChain()
mc.fit(tracks)

# 4. Run simulation with proper stadium
sim = Simulation(
    stadium_L=60,
    stadium_R=20,
    source_length=40,
    velocity_params=velocity_params,
    markov_chain=mc
)
sim.run(n_steps=100)

# 5. Analyze
df = sim.get_dataframe()
dacf = calculate_dacf(df)
msd = calculate_msd(df)

# 6. Visualize
plot_trajectories(sim)
```

## Requirements

```
numpy
pandas
scipy
matplotlib
```

## Key Simplifications

This version is simplified from the full implementation:
- **Single shape**: Only vertical stadium with proper geometry
- **Line source gradient**: Vertical line at center (not point source)
- **Simple boundary**: Cells stop at boundary (no reflection/periodic)
- **Single cell type**: All cells have same properties
- **Basic analysis**: Only DACF and MSD (no collective metrics)

## Tips

1. **Stadium dimensions**: Choose L and R based on your experimental setup
2. **Source length**: Should be ≤ L for gradient to be inside stadium
3. **Chemotaxis strength**: 0.2-0.4 typically gives good migration
4. **Repulsion strength**: Keep < 0.3 to avoid clustering artifacts
5. **Number of steps**: 100-200 steps usually sufficient to see migration

## Troubleshooting

**Cells not migrating away from source:**
- Increase `chemotaxis_strength`
- Check gradient field: `sim.stadium.get_gradient(x, y)`
- Verify source_length is appropriate

**Cells clustering:**
- Reduce `repulsion_strength`
- Increase `interaction_radius`

**Cells stuck at boundaries:**
- This is expected behavior (cells stop at boundaries)
- Reduce chemotaxis if too many cells reach boundary

**Simulation too slow:**
- Reduce `n_cells`
- Reduce `n_steps`

## License

MITdf, max_lag=30)

# Compare with experimental data
from analysis import compare_simulations
results = compare_simulations(sim_df, exp_df)
```

## Visualization

```python
from visualization import plot_trajectories, create_animation, plot_cell_statistics

# Plot trajectories
plot_trajectories(sim, show_gradient=True)

# Create animation
anim = create_animation(sim, save_path='migration.gif')

# Plot statistics
plot_cell_statistics(sim)
```

## Output Files

Running `run_simulation.py` generates:
- `simulation_trajectories.csv` - Cell positions over time
- `dacf_results.csv` - Directional autocorrelation function
- `msd_results.csv` - Mean squared displacement
- `migration.gif` - Animation of cell migration

## Data Format

Expected CSV format for experimental data:
```
track_id,step,x,y
0,0,10.5,20.3
0,1,10.8,21.1
...
```

Or with alternative column names:
```
track_id,normalized_time,x_microns,y_microns
```

## Example Workflow

```python
# 1. Load experimental data
data, tracks = load_experimental_data('data.csv')

# 2. Fit velocity distribution
velocity_params = fit_velocity_distribution(data)

# 3. Fit Markov chain
mc = MarkovChain()
mc.fit(tracks)

# 4. Run simulation
sim = Simulation(
    velocity_params=velocity_params,
    markov_chain=mc
)
sim.run(n_steps=100)

# 5. Analyze
df = sim.get_dataframe()
dacf = calculate_dacf(df)
msd = calculate_msd(df)

# 6. Visualize
plot_trajectories(sim)
```

## Requirements

```
numpy
pandas
scipy
matplotlib
```

## Key Simplifications

This version is simplified from the full implementation:
- **Single shape**: Only vertical stadium (no circles, horizontal stadium)
- **Simple gradient**: Linear upward gradient (no complex fields)
- **Simple boundary**: Cells stop at boundary (no reflection/periodic)
- **Single cell type**: All cells have same properties
- **Basic analysis**: Only DACF and MSD (no collective metrics)

## Tips

1. **Stadium dimensions**: Height should be > width for proper vertical stadium
2. **Chemotaxis strength**: 0.2-0.4 typically gives good migration
3. **Repulsion strength**: Keep < 0.3 to avoid clustering artifacts
4. **Number of steps**: 100-200 steps usually sufficient to see migration

## Troubleshooting

**Cells not migrating upward:**
- Increase `chemotaxis_strength`
- Check gradient is working: `sim.stadium.get_gradient(x, y)`

**Cells clustering:**
- Reduce `repulsion_strength`
- Increase `interaction_radius`

**Simulation too slow:**
- Reduce `n_cells`
- Reduce `n_steps`

## License

MIT
