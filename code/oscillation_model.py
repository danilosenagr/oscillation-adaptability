"""
Oscillation Model - Core mathematical model of necessary oscillations 
under conservation constraints in structured systems.

This module implements the mathematical model described in the paper
"Necessary Oscillations: Adaptability Dynamics Under Fundamental Conservation 
Constraints in Structured Systems"
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from matplotlib.colors import LinearSegmentedColormap


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
    
    def compute_adaptability_landscape(self, x_range: Tuple[float, float], 
                                       d_range: Tuple[float, float], 
                                       resolution: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the adaptability landscape A(x,d) over ranges of x and d.
        
        Parameters
        ----------
        x_range : Tuple[float, float]
            Range of x values (min, max).
        d_range : Tuple[float, float]
            Range of d values (min, max).
        resolution : Tuple[int, int]
            Resolution (number of points) for x and d.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x values, d values, and adaptability values A(x,d).
        """
        x_min, x_max = x_range
        d_min, d_max = d_range
        nx, nd = resolution
        
        x_values = np.linspace(x_min, x_max, nx)
        d_values = np.linspace(d_min, d_max, nd)
        
        adaptability_values = np.zeros((nx, nd))
        
        for i, x in enumerate(x_values):
            for j, d in enumerate(d_values):
                adaptability_values[i, j] = self.adaptability(x, d)
        
        return x_values, d_values, adaptability_values
    
    def compute_time_series(self, x: float, d: float, t_range: Tuple[float, float], 
                            nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the time series of adaptability A(x,d,t) and coherence C(x,d,t).
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
        t_range : Tuple[float, float]
            Range of time values (min, max).
        nt : int
            Number of time points.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Time values, adaptability values A(x,d,t), and coherence values C(x,d,t).
        """
        t_min, t_max = t_range
        t_values = np.linspace(t_min, t_max, nt)
        
        adaptability_values = np.array([self.adaptability_time(x, d, t) for t in t_values])
        coherence_values = 1 - adaptability_values
        
        return t_values, adaptability_values, coherence_values
    
    def compute_spectral_density(self, x: float, d: float, t_range: Tuple[float, float], 
                                nt: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the spectral density of adaptability A(x,d,t).
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
        t_range : Tuple[float, float]
            Range of time values (min, max).
        nt : int
            Number of time points.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequency values and power spectral density.
        """
        t_min, t_max = t_range
        t_values = np.linspace(t_min, t_max, nt)
        
        adaptability_values = np.array([self.adaptability_time(x, d, t) for t in t_values])
        
        # Compute the FFT
        fft_values = fft(adaptability_values)
        
        # Compute the frequency values
        sample_spacing = (t_max - t_min) / (nt - 1)
        freq_values = fftfreq(nt, d=sample_spacing)
        
        # Compute the power spectral density
        psd = np.abs(fft_values) ** 2
        
        # Return only the positive frequencies
        positive_freq_mask = freq_values > 0
        
        return freq_values[positive_freq_mask], psd[positive_freq_mask]
    
    def plot_adaptability_landscape(self, x_range: Tuple[float, float], 
                                   d_range: Tuple[float, float], 
                                   resolution: Tuple[int, int] = (100, 100),
                                   cmap: str = 'viridis',
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the adaptability landscape A(x,d).
        
        Parameters
        ----------
        x_range : Tuple[float, float]
            Range of x values (min, max).
        d_range : Tuple[float, float]
            Range of d values (min, max).
        resolution : Tuple[int, int], optional
            Resolution (number of points) for x and d, default is (100, 100).
        cmap : str, optional
            Colormap to use, default is 'viridis'.
        title : str, optional
            Title for the plot, default is None.
        save_path : str, optional
            Path to save the figure, default is None (not saved).
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        x_values, d_values, adaptability_values = self.compute_adaptability_landscape(x_range, d_range, resolution)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the heatmap
        im = ax.imshow(
            adaptability_values.T,  # Transpose to match x(rows) and d(columns) orientation
            extent=[x_range[0], x_range[1], d_range[0], d_range[1]],
            origin='lower',
            aspect='auto',
            cmap=cmap
        )
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Adaptability A(x,d)')
        
        # Set labels
        ax.set_xlabel('Configuration (x)')
        ax.set_ylabel('Depth (d)')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Adaptability Landscape for N_ord = {self.n_ord}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_time_series(self, x: float, d: float, t_range: Tuple[float, float], 
                        nt: int = 1000,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the time series of adaptability A(x,d,t) and coherence C(x,d,t).
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
        t_range : Tuple[float, float]
            Range of time values (min, max).
        nt : int, optional
            Number of time points, default is 1000.
        title : str, optional
            Title for the plot, default is None.
        save_path : str, optional
            Path to save the figure, default is None (not saved).
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        t_values, adaptability_values, coherence_values = self.compute_time_series(x, d, t_range, nt)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(t_values, adaptability_values, label='Adaptability A(x,d,t)', color='blue')
        ax.plot(t_values, coherence_values, label='Coherence C(x,d,t)', color='red')
        
        # Plot envelope if desired
        envelope = self.adaptability_envelope(x, d)
        ax.axhline(y=envelope, color='blue', linestyle='--', alpha=0.7, 
                  label=f'Envelope = {envelope:.4f}')
        
        # Set labels and title
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Value')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Time Series at x = {x}, d = {d}')
        
        # Add a text box with system parameters
        m_star, n_star = self.M_star(x)
        freqs = self.component_frequencies(d)
        
        text = (f'System: N_ord = {self.n_ord}\n'
                f'Parameters: x = {x}, d = {d}\n'
                f'M* = {m_star:.4f}, N_ord* = {n_star}\n'
                f'Frequencies: ' + ', '.join([f'ω_{n} = {freqs[n]:.4f}' for n in self.n_ord]))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        ax.grid(True)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_power_spectrum(self, x: float, d: float, t_range: Tuple[float, float], 
                           nt: int = 1000,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the power spectrum of adaptability A(x,d,t).
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
        t_range : Tuple[float, float]
            Range of time values (min, max).
        nt : int, optional
            Number of time points, default is 1000.
        title : str, optional
            Title for the plot, default is None.
        save_path : str, optional
            Path to save the figure, default is None (not saved).
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        freq_values, psd = self.compute_spectral_density(x, d, t_range, nt)
        
        # Calculate theoretical frequencies
        theoretical_freqs = {n: np.sqrt(d) / (2 * np.pi * n) for n in self.n_ord}
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.semilogy(freq_values, psd, label='Power Spectrum', color='blue')
        
        # Mark theoretical frequencies
        for n, freq in theoretical_freqs.items():
            ax.axvline(x=freq, color='red', linestyle='--', alpha=0.5, 
                      label=f'f_{n} = {freq:.4f} Hz')
        
        # Set labels and title
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Power Spectrum at x = {x}, d = {d}')
        
        # Add a text box with system parameters
        m_star, n_star = self.M_star(x)
        text = (f'System: N_ord = {self.n_ord}\n'
                f'Parameters: x = {x}, d = {d}\n'
                f'M* = {m_star:.4f}, N_ord* = {n_star}\n'
                f'Theoretical Frequencies: ' + 
                ', '.join([f'f_{n} = {freq:.4f} Hz' for n, freq in theoretical_freqs.items()]))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        ax.grid(True)
        # Set a reasonable legend that doesn't duplicate entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def verify_conservation_law(self, x_samples: int = 100, d_samples: int = 10,
                               x_range: Tuple[float, float] = (-1, 1),
                               d_range: Tuple[float, float] = (1, 30)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Verify the conservation law C(x,d) + A(x,d) = 1 numerically.
        
        Parameters
        ----------
        x_samples : int, optional
            Number of x samples, default is 100.
        d_samples : int, optional
            Number of d samples, default is 10.
        x_range : Tuple[float, float], optional
            Range of x values (min, max), default is (-1, 1).
        d_range : Tuple[float, float], optional
            Range of d values (min, max), default is (1, 30).
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The sum of C+A for each sample, and the maximum absolute deviation from 1.
        """
        x_values = np.linspace(x_range[0], x_range[1], x_samples)
        d_values = np.linspace(d_range[0], d_range[1], d_samples)
        
        conservation_sums = np.zeros((x_samples, d_samples))
        
        for i, x in enumerate(x_values):
            for j, d in enumerate(d_values):
                c = self.coherence(x, d)
                a = self.adaptability(x, d)
                conservation_sums[i, j] = c + a
        
        # Calculate max absolute deviation from 1
        max_deviation = np.max(np.abs(conservation_sums - 1))
        
        return conservation_sums, max_deviation
    
    def verify_exponential_decay(self, x: float, d_range: Tuple[float, float], 
                                nd: int = 100) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Verify that adaptability A(x,d) decays exponentially with d.
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d_range : Tuple[float, float]
            Range of d values (min, max).
        nd : int, optional
            Number of d samples, default is 100.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, float]
            d values, adaptability values, fitted exponent, and R² value.
        """
        d_min, d_max = d_range
        d_values = np.linspace(d_min, d_max, nd)
        
        adaptability_values = np.array([self.adaptability(x, d) for d in d_values])
        
        # Take log of adaptability for non-zero values
        mask = adaptability_values > 0
        log_adapt = np.log(adaptability_values[mask])
        d_masked = d_values[mask]
        
        if len(log_adapt) < 2:
            return d_values, adaptability_values, np.nan, np.nan
        
        # Linear fit to log(adaptability) vs d
        coeffs = np.polyfit(d_masked, log_adapt, 1)
        poly = np.poly1d(coeffs)
        
        # Calculate R² value
        y_fit = poly(d_masked)
        ss_tot = np.sum((log_adapt - np.mean(log_adapt))**2)
        ss_res = np.sum((log_adapt - y_fit)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # The exponent is the slope of the fit
        exponent = coeffs[0]
        
        return d_values, adaptability_values, exponent, r_squared

    def plot_exponential_decay(self, x: float, d_range: Tuple[float, float], 
                              nd: int = 100,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the exponential decay of adaptability A(x,d) with d.
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d_range : Tuple[float, float]
            Range of d values (min, max).
        nd : int, optional
            Number of d samples, default is 100.
        title : str, optional
            Title for the plot, default is None.
        save_path : str, optional
            Path to save the figure, default is None (not saved).
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        d_values, adaptability_values, exponent, r_squared = self.verify_exponential_decay(x, d_range, nd)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Linear plot
        ax1.plot(d_values, adaptability_values, 'o-', label='A(x,d)')
        ax1.set_xlabel('Depth (d)')
        ax1.set_ylabel('Adaptability A(x,d)')
        ax1.set_title('Linear Scale')
        ax1.grid(True)
        
        # Logarithmic plot
        mask = adaptability_values > 0
        ax2.semilogy(d_values[mask], adaptability_values[mask], 'o-', label='A(x,d)')
        
        # Add the fitted exponential decay
        if not np.isnan(exponent):
            # Plot the fit on the log scale
            fit_y = np.exp(exponent * d_values[mask] + np.polyfit(d_values[mask], np.log(adaptability_values[mask]), 1)[1])
            ax2.semilogy(d_values[mask], fit_y, 'r--', 
                       label=f'Fit: A ∝ e^({exponent:.4f}d), R² = {r_squared:.4f}')
        
        ax2.set_xlabel('Depth (d)')
        ax2.set_ylabel('Adaptability A(x,d) (log scale)')
        ax2.set_title('Logarithmic Scale')
        ax2.grid(True)
        
        # Add information about the point and theoretical prediction
        m_star, n_star = self.M_star(x)
        theoretical_exponent = -m_star
        
        text = (f'Configuration: x = {x}\n'
                f'System: N_ord = {self.n_ord}\n'
                f'M* = {m_star:.4f}, N_ord* = {n_star}\n'
                f'Theoretical exponent = {theoretical_exponent:.4f}\n'
                f'Fitted exponent = {exponent:.4f}\n'
                f'R² = {r_squared:.4f}')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.05, 0.05, text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)
        
        ax2.legend()
        
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f'Exponential Decay of Adaptability for x = {x}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def verify_temporal_oscillations(self, x: float, d: float, t_range: Tuple[float, float], 
                                    nt: int = 1000) -> Tuple[float, float, float]:
        """
        Verify that adaptability A(x,d,t) oscillates in time and conserves A+C=1.
        
        Parameters
        ----------
        x : float
            Current configuration of the system.
        d : float
            Depth parameter.
        t_range : Tuple[float, float]
            Range of time values (min, max).
        nt : int, optional
            Number of time points, default is 1000.
            
        Returns
        -------
        Tuple[float, float, float]
            Maximum absolute deviation from 1 in A+C, peak-to-peak amplitude of A, and mean A.
        """
        t_values = np.linspace(t_range[0], t_range[1], nt)
        
        # Compute A(x,d,t) and C(x,d,t)
        adaptability_values = np.array([self.adaptability_time(x, d, t) for t in t_values])
        coherence_values = np.array([self.coherence_time(x, d, t) for t in t_values])
        
        # Check conservation law
        conservation_sum = adaptability_values + coherence_values
        max_deviation = np.max(np.abs(conservation_sum - 1))
        
        # Characterize oscillations
        peak_to_peak = np.max(adaptability_values) - np.min(adaptability_values)
        mean_adaptability = np.mean(adaptability_values)
        
        return max_deviation, peak_to_peak, mean_adaptability


def demonstrate():
    """Simple demonstration of the AdaptabilityModel class."""
    # Create a model with harmonic orbital orders
    model = AdaptabilityModel([1, 2, 3])
    
    # Plot the adaptability landscape
    fig1 = model.plot_adaptability_landscape((-1, 1), (1, 30))
    fig1.savefig('/Users/bobbarclay/osolationadaptability/figures/demo_adaptability_landscape.png', dpi=300, bbox_inches='tight')
    
    # Plot time series and power spectrum for a specific point
    fig2 = model.plot_time_series(0.25, 15.0, (0, 50))
    fig2.savefig('/Users/bobbarclay/osolationadaptability/figures/demo_time_series.png', dpi=300, bbox_inches='tight')
    
    fig3 = model.plot_power_spectrum(0.25, 15.0, (0, 200))
    fig3.savefig('/Users/bobbarclay/osolationadaptability/figures/demo_power_spectrum.png', dpi=300, bbox_inches='tight')
    
    # Test exponential decay
    fig4 = model.plot_exponential_decay(0.25, (1, 30))
    fig4.savefig('/Users/bobbarclay/osolationadaptability/figures/demo_exponential_decay.png', dpi=300, bbox_inches='tight')
    
    # Verify conservation law
    conservation_sums, max_deviation = model.verify_conservation_law()
    print(f"Conservation Law Test: Max deviation from C+A=1: {max_deviation:.2e}")
    
    # Test time-dependent conservation
    max_dev, peak_to_peak, mean_adapt = model.verify_temporal_oscillations(0.25, 15.0, (0, 100), 1000)
    print(f"Time-dependent Conservation Test: Max deviation: {max_dev:.2e}")
    print(f"Oscillation amplitude: {peak_to_peak:.4f}")
    print(f"Mean adaptability: {mean_adapt:.4f}")
    
    print("All figures saved to the 'figures' directory.")


if __name__ == "__main__":
    demonstrate()