"""
stadium.py
Vertical stadium domain with gradient field
"""

import numpy as np
import matplotlib.pyplot as plt


class VerticalStadium:
    """Vertical stadium-shaped domain with linear gradient."""
    
    def __init__(self, width=40, height=100, center=(0, 0)):
        """
        Initialize vertical stadium.
        
        Parameters:
        -----------
        width : float
            Width of the stadium (diameter of semicircles)
        height : float
            Total height of the stadium
        center : tuple
            (x, y) center position
        """
        self.width = width
        self.height = height
        self.center = center
        self.radius = width / 2  # Radius of semicircles
        self.rect_height = height - width  # Height of rectangular section
        
        # Gradient parameters (simple linear gradient along y-axis)
        self.gradient_strength = 1.0
        
    def is_inside(self, x, y):
        """Check if point (x, y) is inside the stadium."""
        cx, cy = self.center
        
        # Check rectangular region
        if (cx - self.radius <= x <= cx + self.radius) and \
           (cy - self.rect_height/2 <= y <= cy + self.rect_height/2):
            return True
        
        # Check bottom semicircle
        if y < cy - self.rect_height/2:
            dx = x - cx
            dy = y - (cy - self.rect_height/2)
            if dx**2 + dy**2 <= self.radius**2:
                return True
        
        # Check top semicircle
        if y > cy + self.rect_height/2:
            dx = x - cx
            dy = y - (cy + self.rect_height/2)
            if dx**2 + dy**2 <= self.radius**2:
                return True
        
        return False
    
    def apply_boundary(self, x, y):
        """
        Apply boundary condition - cells stop at boundary.
        
        Parameters:
        -----------
        x, y : float
            Proposed position
            
        Returns:
        --------
        tuple : (x, y) constrained to be inside stadium
        """
        cx, cy = self.center
        
        # If inside, return as is
        if self.is_inside(x, y):
            return x, y
        
        # Find closest point on boundary
        # Check which section we're closest to
        
        # Rectangular region bounds
        if cy - self.rect_height/2 <= y <= cy + self.rect_height/2:
            # Closest to left or right edge
            if x < cx:
                return cx - self.radius, y
            else:
                return cx + self.radius, y
        
        # Bottom semicircle
        elif y < cy - self.rect_height/2:
            center_y = cy - self.rect_height/2
            dx = x - cx
            dy = y - center_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                # Project to circle boundary
                return cx + dx/dist * self.radius, center_y + dy/dist * self.radius
            else:
                return cx, center_y - self.radius
        
        # Top semicircle
        else:
            center_y = cy + self.rect_height/2
            dx = x - cx
            dy = y - center_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                # Project to circle boundary
                return cx + dx/dist * self.radius, center_y + dy/dist * self.radius
            else:
                return cx, center_y + self.radius
    
    def get_gradient(self, x, y):
        """
        Get gradient at position (simple linear gradient along y-axis).
        
        Returns:
        --------
        tuple : (gradient_strength, gradient_direction)
        """
        if not self.is_inside(x, y):
            return 0, 0
        
        # Simple gradient pointing upward (along positive y)
        # Strength decreases near the top
        cy = self.center[1]
        top_y = cy + self.rect_height/2 + self.radius
        
        # Normalize position
        relative_y = (y - (cy - self.rect_height/2 - self.radius)) / self.height
        
        # Gradient strength (weaker near top)
        strength = self.gradient_strength * (1 - relative_y)
        
        # Direction is always upward (π/2 radians)
        direction = np.pi / 2
        
        return strength, direction
    
    def sample_initial_positions(self, n_cells, distribution='bottom'):
        """
        Sample initial cell positions.
        
        Parameters:
        -----------
        n_cells : int
            Number of cells
        distribution : str
            'bottom' - start in bottom semicircle
            'uniform' - uniform throughout stadium
            'perimeter' - along the perimeter
            
        Returns:
        --------
        list : [(x, y)] positions
        """
        positions = []
        cx, cy = self.center
        
        if distribution == 'bottom':
            # Start cells in bottom semicircle
            center_y = cy - self.rect_height/2
            for _ in range(n_cells):
                # Random point in circle
                angle = np.random.uniform(0, 2*np.pi)
                r = np.sqrt(np.random.uniform(0, 1)) * self.radius * 0.8  # 80% of radius
                x = cx + r * np.cos(angle)
                y = center_y + r * np.sin(angle) - self.radius * 0.5  # Bias toward bottom
                positions.append((x, y))
                
        elif distribution == 'uniform':
            # Uniform distribution throughout stadium
            attempts = 0
            while len(positions) < n_cells and attempts < n_cells * 100:
                x = np.random.uniform(cx - self.radius, cx + self.radius)
                y = np.random.uniform(cy - self.height/2, cy + self.height/2)
                if self.is_inside(x, y):
                    positions.append((x, y))
                attempts += 1
                
        elif distribution == 'perimeter':
            # Distribute along perimeter
            # Calculate perimeter
            perimeter = 2 * self.rect_height + 2 * np.pi * self.radius
            
            for i in range(n_cells):
                s = (i / n_cells) * perimeter
                
                if s < self.rect_height:
                    # Left edge
                    x = cx - self.radius
                    y = cy - self.rect_height/2 + s
                elif s < self.rect_height + np.pi * self.radius:
                    # Top semicircle
                    angle = np.pi + (s - self.rect_height) / self.radius
                    x = cx + self.radius * np.cos(angle)
                    y = cy + self.rect_height/2 + self.radius * np.sin(angle)
                elif s < 2 * self.rect_height + np.pi * self.radius:
                    # Right edge
                    x = cx + self.radius
                    y = cy + self.rect_height/2 - (s - self.rect_height - np.pi * self.radius)
                else:
                    # Bottom semicircle
                    angle = (s - 2 * self.rect_height - np.pi * self.radius) / self.radius
                    x = cx + self.radius * np.cos(angle)
                    y = cy - self.rect_height/2 + self.radius * np.sin(angle)
                
                positions.append((x, y))
        
        return positions
    
    def visualize(self, cells=None):
        """Visualize the stadium and optional cell positions."""
        fig, ax = plt.subplots(figsize=(6, 10))
        
        cx, cy = self.center
        
        # Draw stadium outline
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Bottom semicircle
        bottom_y = cy - self.rect_height/2
        x_bottom = cx + self.radius * np.cos(theta)
        y_bottom = bottom_y + self.radius * np.sin(theta)
        mask_bottom = y_bottom <= bottom_y
        ax.plot(x_bottom[mask_bottom], y_bottom[mask_bottom], 'k-', linewidth=2)
        
        # Top semicircle
        top_y = cy + self.rect_height/2
        x_top = cx + self.radius * np.cos(theta)
        y_top = top_y + self.radius * np.sin(theta)
        mask_top = y_top >= top_y
        ax.plot(x_top[mask_top], y_top[mask_top], 'k-', linewidth=2)
        
        # Side edges
        ax.plot([cx - self.radius, cx - self.radius], 
                [bottom_y, top_y], 'k-', linewidth=2)
        ax.plot([cx + self.radius, cx + self.radius], 
                [bottom_y, top_y], 'k-', linewidth=2)
        
        # Show gradient as arrow
        arrow_y = np.linspace(cy - self.height/3, cy + self.height/3, 5)
        for y in arrow_y:
            ax.arrow(cx, y, 0, self.height/20, 
                    head_width=self.radius/10, head_length=self.height/40,
                    fc='red', ec='red', alpha=0.3)
        
        # Plot cells if provided
        if cells is not None:
            for cell in cells:
                ax.scatter(cell.x, cell.y, s=50, c='blue', alpha=0.7)
        
        ax.set_xlim(cx - self.radius * 1.2, cx + self.radius * 1.2)
        ax.set_ylim(cy - self.height/2 * 1.1, cy + self.height/2 * 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Vertical Stadium Domain')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()