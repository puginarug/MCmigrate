"""
simulation.py
Simplified simulation engine for cell migration in vertical stadium
"""

import numpy as np
import pandas as pd
from cell import Cell
from stadium import VerticalStadium
from markov import MarkovChain


class Simulation:
    """Simple cell migration simulation."""
    
    def __init__(self, 
                 n_cells=30,
                 stadium_width=40,
                 stadium_height=100,
                 chemotaxis_strength=0.3,
                 repulsion_strength=0.2,
                 interaction_radius=10.0,
                 velocity_params=None,
                 markov_chain=None):
        """
        Initialize simulation.
        
        Parameters:
        -----------
        n_cells : int
            Number of cells
        stadium_width : float
            Width of vertical stadium
        stadium_height : float
            Height of vertical stadium
        chemotaxis_strength : float
            Strength of chemotactic response (0-1)
        repulsion_strength : float
            Strength of cell-cell repulsion (0-1)
        interaction_radius : float
            Range for cell-cell interactions
        velocity_params : dict
            Parameters for velocity distribution
        markov_chain : MarkovChain
            Fitted Markov chain for angle changes
        """
        self.n_cells = n_cells
        self.chemotaxis_strength = chemotaxis_strength
        self.repulsion_strength = repulsion_strength
        self.interaction_radius = interaction_radius
        
        # Default velocity parameters
        if velocity_params is None:
            velocity_params = {'shape': 0.5, 'loc': 0, 'scale': 1.0}
        self.velocity_params = velocity_params
        
        # Initialize stadium
        self.stadium = VerticalStadium(width=stadium_width, height=stadium_height)
        
        # Initialize cells
        positions = self.stadium.sample_initial_positions(n_cells, distribution='bottom')
        self.cells = []
        for i, (x, y) in enumerate(positions):
            self.cells.append(Cell(i, x, y, velocity_params))
        
        # Markov chain for angle changes
        self.markov_chain = markov_chain
        self.markov_states = {}
        
        if markov_chain is not None:
            # Initialize each cell with a random Markov state
            for cell in self.cells:
                self.markov_states[cell.id] = markov_chain.get_random_state()
        
        # Time tracking
        self.time = 0
        self.time_step = 1.0
        
    def step(self):
        """Perform one simulation step."""
        
        for cell in self.cells:
            # 1. Get angle change from Markov chain
            if self.markov_chain is not None and cell.id in self.markov_states:
                angle_change = self.markov_chain.sample_angle_change(self.markov_states[cell.id])
                # Update state (simplified - just use random state)
                self.markov_states[cell.id] = self.markov_chain.get_random_state()
            else:
                # Random walk if no Markov chain
                angle_change = np.random.normal(0, 0.3)
            
            # 2. Calculate chemotaxis bias
            grad_strength, grad_direction = self.stadium.get_gradient(cell.x, cell.y)
            chemotaxis_bias = cell.sense_gradient(grad_strength, grad_direction)
            
            # 3. Calculate repulsion bias
            repulsion_bias = cell.calculate_repulsion(self.cells, self.interaction_radius)
            
            # 4. Update cell position
            cell.update_position(
                angle_change, 
                chemotaxis_bias, 
                repulsion_bias,
                self.chemotaxis_strength,
                self.repulsion_strength
            )
            
            # 5. Apply boundary conditions
            cell.x, cell.y = self.stadium.apply_boundary(cell.x, cell.y)
            
            # Update last position in history if boundary was hit
            cell.x_history[-1] = cell.x
            cell.y_history[-1] = cell.y
        
        self.time += self.time_step
    
    def run(self, n_steps, verbose=True):
        """
        Run simulation for n steps.
        
        Parameters:
        -----------
        n_steps : int
            Number of steps to simulate
        verbose : bool
            Print progress
        """
        for i in range(n_steps):
            self.step()
            
            if verbose and (i + 1) % 25 == 0:
                print(f"Step {i+1}/{n_steps} completed")
        
        if verbose:
            print(f"Simulation completed: {n_steps} steps")
    
    def get_dataframe(self):
        """
        Convert simulation results to DataFrame.
        
        Returns:
        --------
        DataFrame with columns: track_id, step, x, y, v_x, v_y
        """
        data_list = []
        
        for cell in self.cells:
            n_points = len(cell.x_history)
            
            # Calculate velocities
            x_array = np.array(cell.x_history)
            y_array = np.array(cell.y_history)
            
            v_x = np.zeros(n_points)
            v_y = np.zeros(n_points)
            
            v_x[1:] = np.diff(x_array)
            v_y[1:] = np.diff(y_array)
            
            # Create DataFrame for this cell
            cell_data = pd.DataFrame({
                'track_id': cell.id,
                'step': range(n_points),
                'x': x_array,
                'y': y_array,
                'v_x': v_x,
                'v_y': v_y
            })
            
            data_list.append(cell_data)
        
        return pd.concat(data_list, ignore_index=True)
    
    def save_trajectories(self, filename='trajectories.csv'):
        """Save trajectories to CSV file."""
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        print(f"Trajectories saved to {filename}")
        return df