"""
Calculates the mean roof difference and mean other difference and mean and standard deviation over these stats.
Gets occlusion results in a numerical form.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def read_values_from_file(path):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            return [float(line.strip()) for line in lines if line.strip()]
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return None


if __name__ == '__main__':
    print("Occlusion Statistics Calculation\n")

    examples = []

    while True:
        print(f"\n=== Example {len(examples) + 1} ===")
        mode = input("Enter 'f' to load from file, or 'm' to enter manually: ").strip().lower()

        roof_cells = []
        non_roof_cells = []

        if mode == "f":
            roof_file = input("Path to roof values file: ").strip()
            non_roof_file = input("Path to non-roof values file: ").strip()

            roof_cells = read_values_from_file(roof_file)
            non_roof_cells = read_values_from_file(non_roof_file)

            if roof_cells is None or non_roof_cells is None:
                print("Skipping due to file error.")
                continue

        else:
            print("Now, enter each roof cell:")
            while True:
                roof_cell = input('Enter a roof value (or "c" to continue): ')
                if roof_cell.lower() == "c":
                    break
                try:
                    roof_cells.append(float(roof_cell))
                except ValueError:
                    print("Invalid number.")

            print("Now, enter each non-roof cell:")
            while True:
                non_roof_cell = input('Enter a non-roof value (or "c" to continue): ')
                if non_roof_cell.lower() == "c":
                    break
                try:
                    non_roof_cells.append(float(non_roof_cell))
                except ValueError:
                    print("Invalid number.")

        if not roof_cells or not non_roof_cells:
            print("Skipping due to missing data.")
            continue

        roof_mean_diff = np.mean(roof_cells)
        non_roof_mean_diff = np.mean(non_roof_cells)

        print(f"Roof Mean Difference: {roof_mean_diff:.4f}")
        print(f"Non-Roof Mean Difference: {non_roof_mean_diff:.4f}")

        fix = input("Fix values before saving? (y/n): ").lower()
        if fix == "y":
            roof_fix = input("Enter corrected roof mean (or blank to keep): ").strip()
            non_roof_fix = input("Enter corrected non-roof mean (or blank to keep): ").strip()
            if roof_fix: roof_mean_diff = float(roof_fix)
            if non_roof_fix: non_roof_mean_diff = float(non_roof_fix)

        examples.append([roof_mean_diff, non_roof_mean_diff])

        cont = input("Add another example? (y/n): ").strip().lower()
        if cont != "y":
            break

    # --- Final Stats and Save (optional)
    examples = np.array(examples)
    mean_roof = np.mean(examples[:, 0])
    std_roof = np.std(examples[:, 0])
    mean_non_roof = np.mean(examples[:, 1])
    std_non_roof = np.std(examples[:, 1])

    print("\nFinal Statistics:")
    print(f"Roof:     Mean = {mean_roof:.4f}, SD = {std_roof:.4f}")
    print(f"Non-Roof: Mean = {mean_non_roof:.4f}, SD = {std_non_roof:.4f}")

    # Convert examples list to numpy array
    examples = np.array(examples)

    # --------- PLOT 1: SCATTER PLOT ----------
    plt.figure(figsize=(5, 5))
    plt.scatter(examples[:, 1], examples[:, 0], alpha=0.8)  # swapped columns

    # Calculate bounds from data
    x_min, x_max = examples[:, 1].min(), examples[:, 1].max()
    y_min, y_max = examples[:, 0].min(), examples[:, 0].max()

    # Add small padding (10% of the range)
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1

    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    # Plot the reference line within the zoomed range
    low = min(x_min - x_pad, y_min - y_pad)
    high = max(x_max + x_pad, y_max + y_pad)
    plt.plot([low, high], [low, high], 'r--', label='y = x')

    plt.xlabel("Non-Roof Mean Difference")
    plt.ylabel("Roof Mean Difference")
    plt.title("Per-Example: Non-Roof vs Roof")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_nonroof_vs_roof.png", dpi=300)
    plt.close()

    # --------- PLOT 2: BOX PLOT ----------
    plt.figure(figsize=(5, 5))
    plt.boxplot([examples[:, 0], examples[:, 1]], labels=["Roof", "Non-Roof"])
    plt.ylabel("Mean Occlusion Difference")
    plt.title("Distribution Across Examples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("boxplot_occlusion_distribution.png", dpi=300)
    plt.close()

    # --------- PLOT 3: BAR PLOT (mean and SD) ----------
    means = [np.mean(examples[:, 0]), np.mean(examples[:, 1])]
    stds = [np.std(examples[:, 0]), np.std(examples[:, 1])]

    plt.figure(figsize=(5, 5))
    plt.bar(["Roof", "Non-Roof"], means, yerr=stds, capsize=10)
    plt.ylabel("Mean Occlusion Difference")
    plt.title("Overall Mean ± SD")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("barplot_mean_sd.png", dpi=300)
    plt.close()