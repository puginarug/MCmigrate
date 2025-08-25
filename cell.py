"""
cell.py
Simplified cell agent for vertical stadium migration
"""

import numpy as np
from scipy.stats import lognorm


class Cell:
    """Simple cell agent."""
    
    def __init__(self, cell_id, x, y, velocity_params):
        """
        Initialize a cell.
        
        Parameters:
        -----------
        cell_id : int
            Unique identifier
        x, y : float
            Initial position
        velocity_params : dict
            Log-normal distribution parameters (shape, loc, scale)
        """
        self.id = cell_id
        self.x = x
        self.y = y
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.velocity_params = velocity_params
        
        # History tracking
        self.x_history = [x]
        self.y_history = [y]
        self.theta_history = [self.theta]
        
    def sense_gradient(self, gradient_strength, gradient_direction):
        """
        Calculate response to gradient.
        
        Parameters:
        -----------
        gradient_strength : float
            Strength of gradient at current position
        gradient_direction : float
            Direction of gradient (angle in radians)
            
        Returns:
        --------
        float : Angle bias toward gradient
        """
        if gradient_strength <= 0:
            return 0
            
        # Calculate angle difference to gradient
        angle_diff = gradient_direction - self.theta
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # Saturation function (simple exponential)
        saturation = 1 - np.exp(-gradient_strength)
        
        return angle_diff * saturation
    
    def calculate_repulsion(self, cells, interaction_radius=10.0):
        """
        Calculate repulsion from nearby cells.
        
        Parameters:
        -----------
        cells : list of Cell
            All cells in simulation
        interaction_radius : float
            Maximum distance for repulsion
            
        Returns:
        --------
        float : Angle bias away from neighbors
        """
        repulse_x = 0
        repulse_y = 0
        
        for other in cells:
            if other.id == self.id:
                continue
                
            dx = self.x - other.x
            dy = self.y - other.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if 0 < dist < interaction_radius:
                # Repulsion strength decreases with distance
                force = (interaction_radius - dist) / interaction_radius
                repulse_x += (dx / dist) * force
                repulse_y += (dy / dist) * force
        
        if repulse_x == 0 and repulse_y == 0:
            return 0
            
        # Convert to angle
        repulsion_angle = np.arctan2(repulse_y, repulse_x)
        angle_diff = repulsion_angle - self.theta
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        return angle_diff * min(np.sqrt(repulse_x**2 + repulse_y**2), 1.0)
    
    def update_position(self, angle_change, chemotaxis_bias, repulsion_bias, 
                       chemotaxis_strength=0.3, repulsion_strength=0.2):
        """
        Update cell position based on various inputs.
        This is the main method that performs one step of cell movement.
        
        Parameters:
        -----------
        angle_change : float
            Base angle change from Markov chain
        chemotaxis_bias : float
            Bias toward chemoattractant
        repulsion_bias : float
            Bias away from other cells
        chemotaxis_strength : float
            Weight for chemotaxis (0-1)
        repulsion_strength : float
            Weight for repulsion (0-1)
            
        Returns:
        --------
        tuple : (new_x, new_y) - the new position before boundary checking
        """
        # Update orientation by combining all angle changes
        self.theta += angle_change
        self.theta += chemotaxis_strength * chemotaxis_bias
        self.theta += repulsion_strength * repulsion_bias
        
        # Normalize angle to [-π, π]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # Sample velocity from log-normal distribution
        velocity = lognorm.rvs(
            self.velocity_params['shape'],
            self.velocity_params['loc'],
            self.velocity_params['scale']
        )
        
        # Calculate new position
        dx = velocity * np.cos(self.theta)
        dy = velocity * np.sin(self.theta)
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Return the new position (will be checked for boundaries in simulation)
        return new_x, new_y
    
    def set_position(self, x, y):
        """
        Set the cell's position after boundary checking and record history.
        
        Parameters:
        -----------
        x, y : float
            Final position after boundary checking
        """
        self.x = x
        self.y = y
        
        # Store history
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.theta_history.append(self.theta)