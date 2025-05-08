"""
Core Adaptability Model - Mathematical model of necessary oscillations 
under conservation constraints in structured systems.

This module implements the core mathematical model described in the paper
"Necessary Oscillations: Adaptability Dynamics Under Fundamental Conservation 
Constraints in Structured Systems"
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional


class AdaptabilityModel:
    """
    A class representing the mathematical model of coherence and adaptability 
    with conservation constraints.
    """
    
    def __init__(self, n_ord: List[int], x0: float = 0):
        """
        Initialize the Adaptability Model with orbital orders and reference point.
        
        Parameters
        ----------
        n_ord : List[int]
            Set of "orbital orders" characterizing the system's internal structural modes.
        x0 : float, optional
            Reference point for the system's configuration space, default is 0.
        """
        self.n_ord = n_ord
        self.x0 = x0
    
    def primary_angle(self, x: float) -> float:
        """
        Calculate the primary angle θ(x) = 2π(x - x0).
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
            
        Returns
        -------
        float
            The primary angle θ(x).
        """
        return 2 * np.pi * (x - self.x0)
    
    def secondary_angle(self, x: float, d: float) -> float:
        """
        Calculate the secondary angle φ(x,d) = dπ(x - x0).
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter representing evolutionary pressure or ordering influence.
            
        Returns
        -------
        float
            The secondary angle φ(x,d).
        """
        return d * np.pi * (x - self.x0)
    
    def coupling_function(self, x: float, d: float, n: int) -> float:
        """
        Calculate the coupling function h_n(x,d) for mode n.
        
        h_n(x,d) = |sin(nθ(x))|^(d/n) · |cos(nφ(x,d))|^(1/n)
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
        n : int
            Orbital order.
            
        Returns
        -------
        float
            The coupling function value h_n(x,d).
        """
        theta = self.primary_angle(x)
        phi = self.secondary_angle(x, d)
        
        sin_term = np.abs(np.sin(n * theta)) ** (d / n)
        cos_term = np.abs(np.cos(n * phi)) ** (1 / n)
        
        return sin_term * cos_term
    
    def temporal_coupling_function(self, x: float, d: float, t: float, n: int) -> float:
        """
        Calculate the time-dependent coupling function h_n(x,d,t) for mode n.
        
        h_n(x,d,t) = |sin(nθ(x))|^(d/n) · |cos(nφ(x,d) + ω_n(d)t)|^(1/n)
        
        where ω_n(d) = √d/n
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
        t : float
            Time.
        n : int
            Orbital order.
            
        Returns
        -------
        float
            The time-dependent coupling function value h_n(x,d,t).
        """
        theta = self.primary_angle(x)
        phi = self.secondary_angle(x, d)
        omega = np.sqrt(d) / n  # Angular frequency for mode n
        
        sin_term = np.abs(np.sin(n * theta)) ** (d / n)
        cos_term = np.abs(np.cos(n * phi + omega * t)) ** (1 / n)
        
        return sin_term * cos_term
    
    def adaptability(self, x: float, d: float) -> float:
        """
        Calculate the adaptability A(x,d) of the system.
        
        A(x,d) = (1/|N_ord|) ∑_{n ∈ N_ord} h_n(x,d)
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
            
        Returns
        -------
        float
            The adaptability A(x,d).
        """
        return np.mean([self.coupling_function(x, d, n) for n in self.n_ord])
    
    def coherence(self, x: float, d: float) -> float:
        """
        Calculate the coherence C(x,d) of the system.
        
        C(x,d) = 1 - A(x,d)
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
            
        Returns
        -------
        float
            The coherence C(x,d).
        """
        return 1 - self.adaptability(x, d)
    
    def adaptability_time(self, x: float, d: float, t: float) -> float:
        """
        Calculate the time-dependent adaptability A(x,d,t) of the system.
        
        A(x,d,t) = (1/|N_ord|) ∑_{n ∈ N_ord} h_n(x,d,t)
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
        t : float
            Time.
            
        Returns
        -------
        float
            The time-dependent adaptability A(x,d,t).
        """
        return np.mean([self.temporal_coupling_function(x, d, t, n) for n in self.n_ord])
    
    def coherence_time(self, x: float, d: float, t: float) -> float:
        """
        Calculate the time-dependent coherence C(x,d,t) of the system.
        
        C(x,d,t) = 1 - A(x,d,t)
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
        t : float
            Time.
            
        Returns
        -------
        float
            The time-dependent coherence C(x,d,t).
        """
        return 1 - self.adaptability_time(x, d, t)
    
    def adaptability_envelope(self, x: float, d: float) -> float:
        """
        Calculate the envelope of time oscillations of A(x,d,t).
        
        A_env(x,d) = (1/|N_ord|) ∑_{n ∈ N_ord} |sin(nθ(x))|^(d/n)
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
            
        Returns
        -------
        float
            The adaptability envelope A_env(x,d).
        """
        theta = self.primary_angle(x)
        return np.mean([np.abs(np.sin(n * theta)) ** (d / n) for n in self.n_ord])
    
    def M_n(self, x: float, n: int) -> float:
        """
        Calculate the exponent factor M_n(x) = -ln|sin(nθ(x))|/n.
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        n : int
            Orbital order.
            
        Returns
        -------
        float
            The exponent factor M_n(x).
        """
        theta = self.primary_angle(x)
        sin_val = np.abs(np.sin(n * theta))
        
        # Avoid taking the log of zero
        if sin_val < 1e-10:
            return np.inf
        
        return -np.log(sin_val) / n
    
    def M_star(self, x: float) -> Tuple[float, List[int]]:
        """
        Calculate the minimum M_n(x) across all n ∈ N_ord and the set N_ord*(x).
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
            
        Returns
        -------
        Tuple[float, List[int]]
            The minimum M_n(x) value and the set of n achieving this minimum.
        """
        m_values = {n: self.M_n(x, n) for n in self.n_ord}
        m_star = min(m_values.values())
        n_star = [n for n, m in m_values.items() if np.isclose(m, m_star)]
        
        return m_star, n_star

    def component_frequencies(self, d: float) -> Dict[int, float]:
        """
        Calculate the component angular frequencies ω_n(d) = √d/n for all n ∈ N_ord.
        
        Parameters
        ----------
        d : float
            Depth parameter.
            
        Returns
        -------
        Dict[int, float]
            Dictionary mapping each n to its angular frequency ω_n(d).
        """
        return {n: np.sqrt(d) / n for n in self.n_ord}