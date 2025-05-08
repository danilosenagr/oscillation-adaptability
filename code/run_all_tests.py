"""
Test script to run all validation and figure generation code.
"""

import os
import sys
import time

# Get the base directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation_results")

def run_validation():
    """Run the validation script to test theoretical findings."""
    print("\n=== Running Model Validation ===")
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(BASE_DIR, "code"))

    from validation.validate_theory import run_all_validations

    print("Starting comprehensive model validation...")
    results = run_all_validations()
    print("Validation complete!")
    return results

def run_figure_generation():
    """Run the figure generation script."""
    print("\n=== Generating Figures ===")
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(BASE_DIR, "code"))

    from visualization.generate_figures import generate_all_figures

    print("Starting figure generation...")
    results = generate_all_figures()
    print("Figure generation complete!")
    return results

if __name__ == "__main__":
    start_time = time.time()

    # Create necessary directories
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    # Run validation
    validation_results = run_validation()

    # Run figure generation
    figure_results = run_figure_generation()

    # Report total time
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print("All validation and figure generation tasks completed successfully!")