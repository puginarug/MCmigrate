# Simplified Cell Migration Simulation

A streamlined agent-based simulation for cell migration in a vertical stadium domain with Markov chain-based motion.

## Features

- **Single vertical stadium domain** with linear gradient
- **Simple boundary conditions** (cells stop at boundaries)
- **Markov chain angle changes** based on experimental data
- **Log-normal velocity distribution**
- **Chemotaxis and cell-cell repulsion**
- **DACF and MSD analysis**

## Files

```
├── cell.py              # Simple cell agent
├── stadium.py           # Vertical stadium domain
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

# Create simulation
sim = Simulation(
    n_cells=30,
    stadium_width=40,
    stadium_height=100,
    chemotaxis_strength=0.3,
    repulsion_strength=0.2
)

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

# Custom parameters
python run_simulation.py --n_cells 50 --n_steps 200 --chemotaxis 0.4
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
sim = Simulation(markov_chain=mc)
sim.run(n_steps=100)
```

## Parameters

### Simulation Parameters
- `n_cells`: Number of cells (default: 30)
- `stadium_width`: Width of vertical stadium (default: 40 μm)
- `stadium_height`: Height of vertical stadium (default: 100 μm)
- `chemotaxis_strength`: Response to gradient, 0-1 (default: 0.3)
- `repulsion_strength`: Cell-cell repulsion, 0-1 (default: 0.2)
- `interaction_radius`: Range for repulsion (default: 10 μm)

### Markov Chain Parameters
- `B`: Number of angle bins (default: 12)
- `n`: n-gram length (default: 3)

## Analysis

```python
from analysis import calculate_dacf, calculate_msd

# Get trajectory data
df = sim.get_dataframe()

# Calculate DACF
dacf = calculate_dacf(df, max_lag=30)

# Calculate MSD
msd = calculate_msd(