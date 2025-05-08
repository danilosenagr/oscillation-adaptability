"""
Visualization functions for adaptability model results.

This module provides plotting functions to visualize adaptability model
results and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Dict, Union, Optional
import sys
import os
import pandas as pd

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.adaptability_model import AdaptabilityModel
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from computational_utils import (
    compute_adaptability_landscape,
    compute_time_series,
    compute_spectral_density,
    verify_exponential_decay
)


def plot_adaptability_landscape(model: AdaptabilityModel,
                               x_range: Tuple[float, float],
                               d_range: Tuple[float, float],
                               resolution: Tuple[int, int] = (100, 100),
                               cmap: str = 'viridis',
                               title: Optional[str] = None,
                               save_path: Optional[str] = None,
                               ax=None) -> plt.Figure:
    """
    Plot the adaptability landscape A(x,d).

    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
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
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, default is None (create new figure).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    x_values, d_values, adaptability_values = compute_adaptability_landscape(model, x_range, d_range, resolution)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

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
        ax.set_title(f'Adaptability Landscape for N_ord = {model.n_ord}')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_time_series(model: AdaptabilityModel,
                    x: float,
                    d: float,
                    t_range: Tuple[float, float],
                    nt: int = 1000,
                    title: Optional[str] = None,
                    save_path: Optional[str] = None,
                    ax=None) -> plt.Figure:
    """
    Plot the time series of adaptability A(x,d,t) and coherence C(x,d,t).

    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
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
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, default is None (create new figure).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    t_values, adaptability_values, coherence_values = compute_time_series(model, x, d, t_range, nt)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    ax.plot(t_values, adaptability_values, label='Adaptability A(x,d,t)', color='blue')
    ax.plot(t_values, coherence_values, label='Coherence C(x,d,t)', color='red')

    # Plot envelope if desired
    envelope = model.adaptability_envelope(x, d)
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
    m_star, n_star = model.M_star(x)
    freqs = model.component_frequencies(d)

    text = (f'System: N_ord = {model.n_ord}\n'
            f'Parameters: x = {x}, d = {d}\n'
            f'M* = {m_star:.4f}, N_ord* = {n_star}\n'
            f'Frequencies: ' + ', '.join([f'ω_{n} = {freqs[n]:.4f}' for n in model.n_ord]))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_power_spectrum(model: AdaptabilityModel,
                       x: float,
                       d: float,
                       t_range: Tuple[float, float],
                       nt: int = 1000,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None,
                       ax=None) -> plt.Figure:
    """
    Plot the power spectrum of adaptability A(x,d,t).

    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
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
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, default is None (create new figure).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    freq_values, psd = compute_spectral_density(model, x, d, t_range, nt)

    # Calculate theoretical frequencies
    theoretical_freqs = {n: np.sqrt(d) / (2 * np.pi * n) for n in model.n_ord}

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

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
    m_star, n_star = model.M_star(x)
    text = (f'System: N_ord = {model.n_ord}\n'
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


def plot_exponential_decay(model: AdaptabilityModel,
                          x: float,
                          d_range: Tuple[float, float],
                          nd: int = 100,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the exponential decay of adaptability A(x,d) with d.

    Parameters
    ----------
    model : AdaptabilityModel
        The adaptability model instance to use.
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
    d_values, adaptability_values, exponent, theoretical_exponent, r_squared = verify_exponential_decay(model, x, d_range, nd)

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
    m_star, n_star = model.M_star(x)

    text = (f'Configuration: x = {x}\n'
            f'System: N_ord = {model.n_ord}\n'
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


def plot_modal_contributions(df_modal: pd.DataFrame,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the relative contributions of different modes to the total adaptability.

    Parameters
    ----------
    df_modal : pd.DataFrame
        DataFrame with relative mode contributions at each depth.
    title : str, optional
        Title for the plot, default is None.
    save_path : str, optional
        Path to save the figure, default is None (not saved).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Get mode columns
    mode_cols = [col for col in df_modal.columns if col.startswith('Mode')]

    # Plot relative contributions
    df_pivot = df_modal.set_index('Depth')[mode_cols]
    df_pivot.plot(kind='area', stacked=True, ax=ax1)

    ax1.set_xlabel('Depth (d)')
    ax1.set_ylabel('Relative Contribution')
    ax1.set_title('Modal Contributions vs Depth')
    ax1.grid(True)
    ax1.legend(title='Modes')

    # Plot entropy vs depth
    ax2.plot(df_modal['Depth'], df_modal['Normalized Entropy'], 'o-', color='purple')
    ax2.set_xlabel('Depth (d)')
    ax2.set_ylabel('Normalized Entropy')
    ax2.set_title('Mode Distribution Complexity vs Depth')
    ax2.grid(True)

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle('Modal Structure Analysis')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_multiple_landscapes(models: List[AdaptabilityModel],
                           model_names: List[str],
                           x_range: Tuple[float, float],
                           d_range: Tuple[float, float],
                           resolution: Tuple[int, int] = (100, 100),
                           cmap: str = 'viridis',
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot multiple adaptability landscapes for comparison.

    Parameters
    ----------
    models : List[AdaptabilityModel]
        List of adaptability model instances to use.
    model_names : List[str]
        Names for each model.
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
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for i, (model, name, ax) in enumerate(zip(models, model_names, axes)):
        plot_adaptability_landscape(model, x_range, d_range, resolution, cmap, name, ax=ax)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_spectral_fingerprints(models: List[AdaptabilityModel],
                              model_names: List[str],
                              x: float,
                              d: float,
                              t_range: Tuple[float, float],
                              nt: int = 1000,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot spectral fingerprints for multiple models at the same point.

    Parameters
    ----------
    models : List[AdaptabilityModel]
        List of adaptability model instances to use.
    model_names : List[str]
        Names for each model.
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
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 4*n_models))

    if n_models == 1:
        axes = [axes]

    max_psd = 0
    for i, (model, name, ax) in enumerate(zip(models, model_names, axes)):
        freq_values, psd = compute_spectral_density(model, x, d, t_range, nt)
        max_psd = max(max_psd, psd.max())

    for i, (model, name, ax) in enumerate(zip(models, model_names, axes)):
        freq_values, psd = compute_spectral_density(model, x, d, t_range, nt)

        # Calculate theoretical frequencies
        theoretical_freqs = {n: np.sqrt(d) / (2 * np.pi * n) for n in model.n_ord}

        ax.semilogy(freq_values, psd, label='Power Spectrum', color='blue')

        # Mark theoretical frequencies
        for n, freq in theoretical_freqs.items():
            ax.axvline(x=freq, color='red', linestyle='--', alpha=0.5,
                      label=f'f_{n} = {freq:.4f} Hz')

        # Set consistent y limit
        ax.set_ylim(0.1, max_psd * 1.1)

        # Set labels and title
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title(f'{name}: N_ord = {model.n_ord}')

        ax.grid(True)
        # Set a reasonable legend that doesn't duplicate entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'Spectral Fingerprints at x = {x}, d = {d}', fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig