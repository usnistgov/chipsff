import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

# Reset to default matplotlib style
plt.style.use('default')

def read_band_yaml(filename):
    """
    Reads a band.yaml file and extracts distances, frequencies, labels, and ticks.
    """
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)

    distances = []
    frequencies = []
    labels = []
    ticks = []
    label_dict = {}

    # Process phonon data
    for phonon_point in data['phonon']:
        distance = phonon_point['distance']
        distances.append(distance)

        # Extract frequencies
        bands = phonon_point['band']
        freqs = [band['frequency'] for band in bands]
        frequencies.append(freqs)

        # Check if 'label' is present
        if 'label' in phonon_point:
            label = phonon_point['label']
            labels.append(label)
            ticks.append(distance)
            label_dict[distance] = label

    distances = np.array(distances)
    frequencies = np.array(frequencies)

    return distances, frequencies, labels, ticks, label_dict

def calculate_error_metrics(distances1, frequencies1, distances2, frequencies2):
    """
    Calculates error metrics between two phonon band structures.
    Interpolates frequencies if necessary.
    """
    # Interpolate frequencies2 onto distances1 if they are different
    if not np.array_equal(distances1, distances2):
        print("Distances are different. Interpolating frequencies.")
        frequencies2_interp = []
        for i in range(frequencies2.shape[1]):
            interp_func = interp1d(distances2, frequencies2[:, i], kind='linear', fill_value="extrapolate")
            frequencies2_interp.append(interp_func(distances1))
        frequencies2 = np.array(frequencies2_interp).T

    # Ensure frequencies have the same shape
    min_modes = min(frequencies1.shape[1], frequencies2.shape[1])
    frequencies1 = frequencies1[:, :min_modes]
    frequencies2 = frequencies2[:, :min_modes]

    # Calculate MAE and RMSE
    abs_diff = np.abs(frequencies1 - frequencies2)
    squared_diff = (frequencies1 - frequencies2) ** 2

    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(squared_diff))

    # Calculate Pearson correlation coefficient for each mode
    correlations = []
    for i in range(min_modes):
        corr_coef, _ = pearsonr(frequencies1[:, i], frequencies2[:, i])
        correlations.append(corr_coef)

    mean_correlation = np.mean(correlations)

    error_metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'Mean_Correlation': mean_correlation,
        'Correlations_Per_Mode': correlations
    }

    return error_metrics, frequencies1, frequencies2

def plot_band_structures(distances_list, frequencies_list, labels_list, ticks_list, filenames):
    """
    Plots multiple phonon band structures on the same graph.
    """
    plt.figure(figsize=(10, 6))

    colors = ['blue', 'red']
    linestyles = ['-', '--']

    for i, (distances, frequencies) in enumerate(zip(distances_list, frequencies_list)):
        num_branches = frequencies.shape[1]
        for j in range(num_branches):
            plt.plot(distances, frequencies[:, j], color=colors[i], linestyle=linestyles[i], linewidth=1.0)

    # Set labels and ticks
    plt.xlabel('Wave Vector', fontsize=14)
    plt.ylabel('Frequency (cm$^{-1}$)', fontsize=14)
    plt.title('Phonon Band Structure Comparison', fontsize=16)

    # Use only high-symmetry points for ticks
    all_ticks = sorted(set(ticks_list[0] + ticks_list[1]))
    all_labels = []
    for tick in all_ticks:
        label = ''
        for label_dict in labels_list:
            if tick in label_dict:
                label = label_dict[tick]
                break
        all_labels.append(label)

    plt.xticks(all_ticks, all_labels, fontsize=12)
    plt.xlim(distances_list[0][0], distances_list[0][-1])

    # Improve aesthetics
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Create a legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=2, linestyle='-'),
        Line2D([0], [0], color='red', lw=2, linestyle='--')
    ]
    plt.legend(custom_lines, ['DFT', 'MLFF'], loc='upper right', fontsize=12)

    # Save and show the plot
    plt.savefig("phonon_band_comparison.png", dpi=300)
    plt.show()

def plot_parity(frequencies1, frequencies2):
    """
    Plots a parity plot of eigenvalues between DFT and MLFF as a scatter plot.
    """
    # Flatten the frequency arrays
    freqs_dft = frequencies1.flatten()
    freqs_mlff = frequencies2.flatten()

    # Plot parity plot using plt.plot with linestyle='None'
    plt.figure(figsize=(6, 6))
    plt.plot(freqs_dft, freqs_mlff, linestyle='None', marker='o', markersize=5,
             markerfacecolor='green', markeredgecolor='none', alpha=0.6)

    # Plot the y = x line
    min_freq = min(np.min(freqs_dft), np.min(freqs_mlff))
    max_freq = max(np.max(freqs_dft), np.max(freqs_mlff))
    plt.plot([min_freq, max_freq], [min_freq, max_freq], 'k--')  # y = x line

    plt.xlabel('DFT Frequencies (cm$^{-1}$)', fontsize=14)
    plt.ylabel('MLFF Frequencies (cm$^{-1}$)', fontsize=14)
    plt.title('Parity Plot of Phonon Frequencies', fontsize=16)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xlim([min_freq, max_freq])
    plt.ylim([min_freq, max_freq])
    plt.tight_layout()
    plt.savefig("phonon_parity_plot.png", dpi=300)
    plt.show()

# Main script
if __name__ == "__main__":
    # File paths (adjust these to your band.yaml file paths)
    band_yaml_file1 = 'band-dft.yaml'    # DFT band.yaml file
    band_yaml_file2 = 'band-mlff.yaml'   # MLFF band.yaml file

    # Read data from the first band.yaml file (DFT)
    distances1, frequencies1, labels1, ticks1, label_dict1 = read_band_yaml(band_yaml_file1)

    # Read data from the second band.yaml file (MLFF)
    distances2, frequencies2, labels2, ticks2, label_dict2 = read_band_yaml(band_yaml_file2)

    # Calculate error metrics and get aligned frequencies
    error_metrics, frequencies1_aligned, frequencies2_aligned = calculate_error_metrics(
        distances1, frequencies1, distances2, frequencies2
    )
    print("Error Metrics between the two phonon band structures:")
    print(f"Mean Absolute Error (MAE): {error_metrics['MAE']:.4f} cm^-1")
    print(f"Root Mean Square Error (RMSE): {error_metrics['RMSE']:.4f} cm^-1")
    print(f"Mean Pearson Correlation Coefficient: {error_metrics['Mean_Correlation']:.4f}")

    # Plot the phonon band structures
    plot_band_structures(
        [distances1, distances2],
        [frequencies1, frequencies2],
        [label_dict1, label_dict2],
        [ticks1, ticks2],
        [band_yaml_file1, band_yaml_file2]
    )

    # Plot the parity plot as a scatter plot
    plot_parity(frequencies1_aligned, frequencies2_aligned)
