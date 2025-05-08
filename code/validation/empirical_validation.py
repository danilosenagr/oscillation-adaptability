"""
Simple validation script to test the main conservation law.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Get the base directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "code"))

from model.adaptability_model import AdaptabilityModel

def test_conservation_law():
    """Test the conservation law C(x,d) + A(x,d) = 1"""
    print("Testing Conservation Law: C(x,d) + A(x,d) = 1")
    
    # Create models with different orbital order sets
    models = [
        ("Harmonic", AdaptabilityModel([1, 2, 3])),
        ("Odd Harmonic", AdaptabilityModel([1, 3, 5])),
        ("Mixed", AdaptabilityModel([2, 3, 5]))
    ]
    
    x_samples = 200
    d_samples = 20
    x_range = (-1, 1)
    d_range = (1, 30)
    
    x_values = np.linspace(x_range[0], x_range[1], x_samples)
    d_values = np.linspace(d_range[0], d_range[1], d_samples)
    
    for name, model in models:
        print(f"\nTesting model: {name}")
        max_deviation = 0
        sum_values = []
        
        # Test at a few specific points for detailed output
        test_points = [
            (-0.5, 5.0), (0.0, 10.0), (0.25, 15.0), (0.5, 20.0)
        ]
        
        print("\nDetailed test points:")
        for x, d in test_points:
            c = model.coherence(x, d)
            a = model.adaptability(x, d)
            sum_ca = c + a
            print(f"  x={x:.2f}, d={d:.2f}: C={c:.10f}, A={a:.10f}, C+A={sum_ca:.16f}")
        
        # Calculate overall statistics
        conservation_sums = np.zeros((x_samples, d_samples))
        
        for i, x in enumerate(x_values):
            for j, d in enumerate(d_values):
                c = model.coherence(x, d)
                a = model.adaptability(x, d)
                conservation_sums[i, j] = c + a
                max_deviation = max(max_deviation, abs(conservation_sums[i, j] - 1))
        
        mean_sum = np.mean(conservation_sums)
        std_sum = np.std(conservation_sums)
        
        print(f"\nOverall statistics:")
        print(f"  Mean C+A: {mean_sum:.16f}")
        print(f"  Standard deviation: {std_sum:.16e}")
        print(f"  Maximum absolute deviation from 1: {max_deviation:.16e}")
        
        # Make a histogram of the deviations
        deviations = conservation_sums.flatten() - 1
        
        plt.figure(figsize=(10, 6))
        plt.hist(deviations, bins=50)
        plt.title(f'Deviations from C+A=1 for {name} Model')
        plt.xlabel('Deviation')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        plt.axvline(x=0, color='red', linestyle='--', label='Ideal (No Deviation)')
        stats_text = (f'Mean: {mean_sum:.10f}\n'
                     f'Std Dev: {std_sum:.10e}\n'
                     f'Max |Dev|: {max_deviation:.10e}')
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.legend()
        save_path = os.path.join(BASE_DIR, "figures", f"conservation_test_{name.lower().replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    
    print("\nConservation law testing complete.")

def test_exponential_decay():
    """Test the exponential decay of adaptability with depth."""
    print("\nTesting Exponential Decay of Adaptability")
    
    # Create model
    model = AdaptabilityModel([1, 2, 3])  # Harmonic model
    
    # Test configurations
    x_values = [0.125, 0.25, 0.375]
    d_range = (1, 30)
    nd = 100
    d_values = np.linspace(d_range[0], d_range[1], nd)
    
    # Create subplots for each x value
    fig, axes = plt.subplots(len(x_values), 1, figsize=(10, 4 * len(x_values)))
    
    for i, x in enumerate(x_values):
        # Calculate theoretical exponent
        m_star, n_star = model.M_star(x)
        theoretical_exponent = -m_star
        
        # Calculate adaptability values
        adaptability_values = np.array([model.adaptability(x, d) for d in d_values])
        
        # Fit exponential decay (log-linear fit)
        mask = adaptability_values > 0
        if sum(mask) >= 2:  # Need at least 2 points for linear fit
            log_adapt = np.log(adaptability_values[mask])
            d_masked = d_values[mask]
            
            coeffs = np.polyfit(d_masked, log_adapt, 1)
            fitted_exponent = coeffs[0]
            
            # Calculate fit quality
            y_fit = np.polyval(coeffs, d_masked)
            r_squared = 1 - np.sum((log_adapt - y_fit)**2) / np.sum((log_adapt - np.mean(log_adapt))**2)
            
            # Compute relative error
            rel_error = 100 * abs(fitted_exponent - theoretical_exponent) / abs(theoretical_exponent)
            
            # Plot data and fit
            ax = axes[i] if len(x_values) > 1 else axes
            
            # Plot adaptability values
            ax.semilogy(d_values, adaptability_values, 'o', markersize=3, label='Data')
            
            # Plot fitted curve
            fit_curve = np.exp(coeffs[0] * d_values + coeffs[1])
            ax.semilogy(d_values, fit_curve, 'r-', label=f'Fit: e^({fitted_exponent:.6f}·d)')
            
            # Add reference line with theoretical exponent
            ref_line = np.exp(theoretical_exponent * d_values + coeffs[1])
            ax.semilogy(d_values, ref_line, 'g--', label=f'Theory: e^({theoretical_exponent:.6f}·d)')
            
            ax.set_xlabel('Depth (d)')
            ax.set_ylabel('Adaptability A(x,d)')
            ax.set_title(f'Exponential Decay at x = {x}')
            
            # Add info text
            info_text = (f'Configuration x = {x}\n'
                        f'M* = {m_star:.6f}, N_ord* = {n_star}\n'
                        f'Theoretical exp = {theoretical_exponent:.6f}\n'
                        f'Fitted exp = {fitted_exponent:.6f}\n'
                        f'Relative error = {rel_error:.2f}%\n'
                        f'R² = {r_squared:.6f}')
            
            ax.text(0.95, 0.95, info_text, transform=ax.transAxes,
                   horizontalalignment='right', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.grid(True)
            ax.legend()
            
            print(f"x = {x}:")
            print(f"  M* = {m_star:.6f}, N_ord* = {n_star}")
            print(f"  Theoretical exponent: {theoretical_exponent:.6f}")
            print(f"  Fitted exponent: {fitted_exponent:.6f}")
            print(f"  Relative error: {rel_error:.2f}%")
            print(f"  R² = {r_squared:.6f}")
    
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "figures", "exponential_decay_test.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    print("Exponential decay testing complete.")

def test_temporal_oscillations():
    """Test the temporal oscillations and the conservation law in time."""
    print("\nTesting Temporal Oscillations and Conservation in Time")
    
    # Create model
    model = AdaptabilityModel([1, 2, 3])  # Harmonic model
    
    # Test parameters
    x = 0.25
    d = 15.0
    t_range = (0, 50)
    nt = 1000
    t_values = np.linspace(t_range[0], t_range[1], nt)
    
    # Calculate time series
    adaptability_values = np.array([model.adaptability_time(x, d, t) for t in t_values])
    coherence_values = np.array([model.coherence_time(x, d, t) for t in t_values])
    conservation_sums = adaptability_values + coherence_values
    
    # Calculate statistics
    max_deviation = np.max(np.abs(conservation_sums - 1))
    peak_to_peak = np.max(adaptability_values) - np.min(adaptability_values)
    mean_adaptability = np.mean(adaptability_values)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Time series plot
    ax1.plot(t_values, adaptability_values, label='Adaptability A(x,d,t)', color='blue')
    ax1.plot(t_values, coherence_values, label='Coherence C(x,d,t)', color='red')
    ax1.axhline(y=model.adaptability_envelope(x, d), color='blue', linestyle='--', 
               label=f'Envelope = {model.adaptability_envelope(x, d):.4f}')
    
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Time Series at x = {x}, d = {d}')
    ax1.grid(True)
    ax1.legend()
    
    # Conservation plot
    ax2.plot(t_values, conservation_sums, label='C(x,d,t) + A(x,d,t)', color='green')
    ax2.axhline(y=1, color='black', linestyle='--', label='Conservation Law: C+A=1')
    
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Sum Value')
    ax2.set_title(f'Conservation Law in Time: C(x,d,t) + A(x,d,t) = 1')
    
    # Set a very narrow y-axis range to see any deviations
    mean_sum = np.mean(conservation_sums)
    ax2.set_ylim(mean_sum - 5*max_deviation, mean_sum + 5*max_deviation)
    
    ax2.grid(True)
    ax2.legend()
    
    # Add info text
    m_star, n_star = model.M_star(x)
    freqs = model.component_frequencies(d)
    
    info_text = (f'Parameters: x = {x}, d = {d}\n'
                f'M* = {m_star:.4f}, N_ord* = {n_star}\n'
                f'Mean A = {mean_adaptability:.6f}\n'
                f'Oscillation amplitude = {peak_to_peak:.6f}\n'
                f'Max |deviation| from C+A=1: {max_deviation:.2e}')
    
    ax1.text(0.05, 0.05, info_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, "figures", "temporal_oscillations_test.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Calculate frequency spectrum
    from scipy.fft import fft, fftfreq
    
    fft_values = fft(adaptability_values)
    sample_spacing = (t_range[1] - t_range[0]) / (nt - 1)
    freq_values = fftfreq(nt, d=sample_spacing)
    psd = np.abs(fft_values) ** 2
    
    # Theoretical frequencies
    theoretical_freqs = {n: np.sqrt(d) / (2 * np.pi * n) for n in model.n_ord}
    
    # Plot spectrum
    plt.figure(figsize=(12, 6))
    
    # Only plot positive frequencies
    positive_freq_mask = freq_values > 0
    plt.semilogy(freq_values[positive_freq_mask], psd[positive_freq_mask], label='Power Spectrum')
    
    # Mark theoretical frequencies
    for n, freq in theoretical_freqs.items():
        plt.axvline(x=freq, color='red', linestyle='--', alpha=0.7, 
                   label=f'f_{n} = {freq:.4f} Hz')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Power Spectrum at x = {x}, d = {d}')
    plt.grid(True)
    
    # Create reasonable legend without duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    spectrum_path = os.path.join(BASE_DIR, "figures", "power_spectrum_test.png")
    plt.savefig(spectrum_path, dpi=300, bbox_inches='tight')
    print(f"Spectrum plot saved to: {spectrum_path}")
    
    print("Temporal oscillations testing complete.")
    print(f"Maximum deviation from C+A=1: {max_deviation:.2e}")
    print(f"Peak-to-peak oscillation amplitude: {peak_to_peak:.6f}")
    print(f"Mean adaptability: {mean_adaptability:.6f}")

if __name__ == "__main__":
    print("=== Running Empirical Validation Tests ===")
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(BASE_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Run tests
    test_conservation_law()
    test_exponential_decay()
    test_temporal_oscillations()
    
    print("\nAll empirical validation tests completed successfully!")