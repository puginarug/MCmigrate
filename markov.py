"""
markov.py
Markov chain model for cell angle changes
"""

import numpy as np
import random
from collections import defaultdict, Counter, deque


class MarkovChain:
    """Simple Markov chain for angle changes."""
    
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.states = set()
        
    def fit(self, tracks, B=12, n=3):
        """
        Fit Markov chain to experimental tracks.
        
        Parameters:
        -----------
        tracks : list of arrays
            List of (x, y) coordinate arrays
        B : int
            Number of bins for angle discretization
        n : int
            n-gram length
        """
        self.B = B
        self.n = n
        
        # Extract angle change sequences
        angle_sequences = self._extract_angle_sequences(tracks, B)
        
        # Convert to n-grams
        ngram_sequences = self._make_ngrams(angle_sequences, n)
        
        # Build transition matrix
        for seq in ngram_sequences:
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                current_state = seq[i]
                next_state = seq[i + 1]
                self.transitions[current_state][next_state] += 1
                self.states.add(current_state)
                self.states.add(next_state)
        
        print(f"Markov chain fitted: {len(self.states)} states from {len(tracks)} tracks")
        
    def _extract_angle_sequences(self, tracks, B):
        """Extract discretized angle change sequences from tracks."""
        sequences = []
        
        for xy in tracks:
            if len(xy) < 3:
                continue
                
            # Calculate angles between consecutive steps
            d = np.diff(xy, axis=0)
            theta = np.arctan2(d[:,1], d[:,0])
            
            # Calculate angle changes
            dtheta = np.diff(theta)
            # Normalize to [-π, π]
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            
            # Discretize into B bins
            edges = np.linspace(-np.pi, np.pi, B+1)
            bins = np.digitize(dtheta, edges[:-1]) - 1
            bins[bins == B] = 0  # Wrap around
            
            sequences.append(bins.tolist())
        
        return sequences
    
    def _make_ngrams(self, sequences, n):
        """Convert sequences to n-grams."""
        ngram_sequences = []
        
        for seq in sequences:
            if len(seq) < n + 1:
                continue
                
            # Create n-gram tokens
            window = deque(maxlen=n)
            for i in range(n):
                window.append(seq[i])
            
            tokens = []
            for j in range(n, len(seq)):
                tokens.append("_".join(map(str, window)))
                window.append(seq[j])
            
            ngram_sequences.append(tokens)
        
        return ngram_sequences
    
    def sample_angle_change(self, current_state=None):
        """
        Sample next angle change from Markov chain.
        
        Parameters:
        -----------
        current_state : str
            Current n-gram state
            
        Returns:
        --------
        float : Angle change in radians
        """
        # If no state or state not in transitions, pick random
        if current_state is None or current_state not in self.transitions:
            if self.states:
                current_state = random.choice(list(self.states))
            else:
                # Random angle change if no Markov chain fitted
                return np.random.normal(0, 0.3)
        
        # Get next state based on transitions
        if self.transitions[current_state]:
            next_states = list(self.transitions[current_state].keys())
            weights = list(self.transitions[current_state].values())
            next_state = random.choices(next_states, weights=weights)[0]
        else:
            next_state = random.choice(list(self.states))
        
        # Extract bin index from state
        bin_idx = int(next_state.split('_')[-1])
        
        # Convert bin to angle
        centers = (np.arange(self.B) + 0.5) * 2 * np.pi / self.B - np.pi
        
        return centers[bin_idx]
    
    def get_random_state(self):
        """Get a random state to initialize cells."""
        if self.states:
            return random.choice(list(self.states))
        return None