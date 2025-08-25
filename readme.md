
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

# Create simulation with cells randomly inside the stadium
sim = Simulation(
    n_cells=30,
    stadium_L=60,           # Length of straight walls
    stadium_R=20,           # Radius of semicircles
    source_length=40,       # Length of vertical line source for gradient
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
- `stadium_L`: Length of straight walls in stadium (default: 60 μm)
- `stadium_R`: Radius of semicircles (default: 20 μm)
- `source_length`: Length of vertical line source for gradient (default: 40 μm)
- `chemotaxis_strength`: Response to gradient, 0-1 (default: 0.3)
- `repulsion_strength`: Cell-cell repulsion, 0-1 (default: 0.2)
- `interaction_radius`: Range for repulsion (default: 10 μm)
- `initial_distribution`: Where cells start: `'bottom'` (default, bottom semicircle), `'uniform'` (random inside stadium), or `'perimeter'` (random along boundary)

### Markov Chain Parameters
- `B`: Number of angle bins (default: 12)
- `n`: n-gram length (default: 3)