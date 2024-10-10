import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from jarvis.db.webpages import Webpage
from jarvis.db.figshare import data
import plotly.express as px

# Function to extract phonon band structure data from JARVIS
def get_phonon_band_structure(jid):
    """Get phonon band structure data from JARVIS webpage data for a given jid."""
    # Create a Webpage object for the given jid
    w = Webpage(jid=jid)
    # Access the data from w.data
    data_dict = w.data["basic_info"]["main_elastic"]["main_elastic_info"]

    # Extract the relevant elements directly from the dictionary
    distances_str = data_dict['phonon_bandstructure_distances'].strip("'")
    frequencies_str = data_dict['phonon_bandstructure_frequencies'].strip("'")

    # Parse distances (Wave vector path)
    distances = np.array([float(x) for x in distances_str.split(',') if x])

    # Parse frequencies (phonon bands at each q-point)
    frequency_blocks = frequencies_str.split(';')
    frequencies = []
    for block in frequency_blocks:
        freq_values = [float(x) for x in block.strip().split(',') if x]
        if len(freq_values) > 0:  # Filter out empty bands
            frequencies.append(freq_values)

    # Ensure frequencies can be converted to a 2D array
    if any(len(freq) != len(distances) for freq in frequencies):
        raise ValueError(f"Mismatch between the length of distances ({len(distances)}) and frequencies.")

    # Convert to a 2D array
    frequencies = np.array(frequencies).T  # Transpose to match (num_qpoints, num_bands)

    return distances, frequencies

# Function to read phonon data from band.yaml files
def read_band_yaml(band_yaml_path):
    """
    Read band.yaml file and extract distances and frequencies.
    """
    with open(band_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)

    # Extract distances and frequencies
    distances = []
    frequencies = []

    for point in data_yaml['phonon']:
        distances.append(point['distance'])
        freq_at_point = [band['frequency'] for band in point['band']]
        frequencies.append(freq_at_point)

    distances = np.array(distances)
    frequencies = np.array(frequencies)

    return distances, frequencies

# Function to compare phonon data and compute error metrics
def compare_phonon_data(distances_ref, frequencies_ref, distances_calc, frequencies_calc):
    """
    Compare the reference and calculated phonon frequencies.
    Returns MAE, MAD, and Pearson correlation coefficient.
    """
    # Define a common distance grid
    common_distances = np.linspace(
        max(distances_ref.min(), distances_calc.min()),
        min(distances_ref.max(), distances_calc.max()),
        num=1000
    )

    # Interpolate frequencies onto the common grid
    num_bands = min(frequencies_ref.shape[1], frequencies_calc.shape[1])
    frequencies_ref_interp = np.zeros((len(common_distances), num_bands))
    frequencies_calc_interp = np.zeros((len(common_distances), num_bands))

    for i in range(num_bands):
        interp_ref = interp1d(distances_ref, frequencies_ref[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_calc = interp1d(distances_calc, frequencies_calc[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")
        frequencies_ref_interp[:, i] = interp_ref(common_distances)
        frequencies_calc_interp[:, i] = interp_calc(common_distances)

    # Flatten the arrays for error calculation
    ref_flat = frequencies_ref_interp.flatten()
    calc_flat = frequencies_calc_interp.flatten()

    # Calculate error metrics
    mae = mean_absolute_error(ref_flat, calc_flat)
    mad = np.mean(np.abs(calc_flat - np.mean(calc_flat)))
    pearson_corr, _ = pearsonr(ref_flat, calc_flat)

    return mae, mad, pearson_corr

# Function to plot both phonon band structures together
def plot_phonon_band_structures(distances_ref, frequencies_ref, distances_calc, frequencies_calc, jid, calculator_type, output_dir):
    """
    Plot the phonon band structures from JARVIS and MLFF calculations together.
    """
    num_bands = min(frequencies_ref.shape[1], frequencies_calc.shape[1])

    plt.figure(figsize=(10, 6))

    # Plot JARVIS phonon bands
    for i in range(num_bands):
        plt.plot(distances_ref, frequencies_ref[:, i], 'r--', linewidth=1.0, label='JARVIS' if i == 0 else '')

    # Plot MLFF phonon bands
    for i in range(num_bands):
        plt.plot(distances_calc, frequencies_calc[:, i], 'b', linewidth=1.0, label='MLFF' if i == 0 else '')

    plt.xlabel('Wave Vector Distance', fontsize=14)
    plt.ylabel('Frequency (cm^-1)', fontsize=14)
    plt.title(f'Phonon Band Structure Comparison for {jid} ({calculator_type})', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plot_filename = os.path.join(output_dir, f'{jid}_{calculator_type}_phonon_comparison.png')
    plt.savefig(plot_filename)
    plt.close()

# Function to collect error data, including phonon data
def collect_error_data(base_dir):
    error_data_list = []
    phonon_error_data_list = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_error_dat.csv'):
                error_dat_path = os.path.join(root, file)
                # Read the CSV file into a DataFrame
                df = pd.read_csv(error_dat_path)
                # Extract jid and calculator_type from the filename or directory name
                filename = os.path.basename(error_dat_path)
                base_filename = filename.replace('_error_dat.csv', '')
                parts = base_filename.split('_')
                jid = parts[0]
                calculator_type = '_'.join(parts[1:])

                # Add jid and calculator_type to the DataFrame
                df['jid'] = jid
                df['calculator_type'] = calculator_type

                # Append the DataFrame to the list
                error_data_list.append(df)

                # Now process phonon data
                try:
                    # Get JARVIS phonon data
                    distances_ref, frequencies_ref = get_phonon_band_structure(jid)

                    # Get MLFF phonon data
                    band_yaml_path = os.path.join(root, 'band.yaml')
                    if not os.path.isfile(band_yaml_path):
                        raise FileNotFoundError(f"band.yaml not found in {root}")

                    distances_calc, frequencies_calc = read_band_yaml(band_yaml_path)

                    # Compare phonon data
                    mae, mad, pearson_corr = compare_phonon_data(
                        distances_ref, frequencies_ref, distances_calc, frequencies_calc
                    )

                    # Plot phonon band structures
                    plot_phonon_band_structures(
                        distances_ref, frequencies_ref,
                        distances_calc, frequencies_calc,
                        jid, calculator_type, root
                    )

                    # Append phonon error data
                    phonon_error_data_list.append({
                        'jid': jid,
                        'calculator_type': calculator_type,
                        'phonon_mae': mae,
                        'phonon_mad': mad,
                        'phonon_pearson_corr': pearson_corr
                    })
                except Exception as e:
                    print(f"Error processing phonon data for {jid} in {root}: {e}")

    # Concatenate all error DataFrames
    if error_data_list:
        all_errors_df = pd.concat(error_data_list, ignore_index=True)
    else:
        all_errors_df = pd.DataFrame()

    # Concatenate all phonon error DataFrames
    if phonon_error_data_list:
        phonon_errors_df = pd.DataFrame(phonon_error_data_list)
        # Merge with all_errors_df
        all_errors_df = all_errors_df.merge(phonon_errors_df, on=['jid', 'calculator_type'], how='left')
    else:
        print('No phonon error data collected.')

    return all_errors_df

# Function to aggregate errors, including phonon errors
def aggregate_errors(all_errors_df):
    # Group by calculator_type
    grouped = all_errors_df.groupby('calculator_type')

    # Error columns
    error_columns = ['err_a', 'err_b', 'err_c', 'err_form', 'err_vol',
                     'err_kv', 'err_c11', 'err_c44', 'err_surf_en', 'err_vac_en',
                     'phonon_mae', 'phonon_mad', 'phonon_pearson_corr']

    # Compute mean errors
    mean_errors = grouped[error_columns].mean()

    # Compute counts of missing entries per error column
    missing_counts = grouped[error_columns].apply(lambda x: x.isna().sum())

    # Rename columns in missing_counts
    missing_counts.columns = [col + '_missing_count' for col in missing_counts.columns]

    # Compute total number of entries per calculator_type
    total_counts = grouped.size().reset_index(name='total_entries')

    # Reset index to 'calculator_type' for mean_errors and missing_counts
    mean_errors = mean_errors.reset_index()
    missing_counts = missing_counts.reset_index()

    # Merge dataframes
    composite_df = mean_errors.merge(missing_counts, on='calculator_type')
    composite_df = composite_df.merge(total_counts, on='calculator_type')

    # Compute missing percentages
    for col in error_columns:
        missing_count_col = col + '_missing_count'
        percentage_col = col + '_missing_percentage'
        composite_df[percentage_col] = (composite_df[missing_count_col] / composite_df['total_entries']) * 100

    return composite_df

# Function to count missing entries per calculator_type
def count_missing_entries(all_errors_df):
    error_columns = ['err_a', 'err_b', 'err_c', 'err_form', 'err_vol',
                     'err_kv', 'err_c11', 'err_c44', 'err_surf_en', 'err_vac_en',
                     'phonon_mae', 'phonon_mad', 'phonon_pearson_corr']
    # Mark rows with any NaN in error columns
    all_errors_df['has_missing'] = all_errors_df[error_columns].isnull().any(axis=1)

    # Group by calculator_type
    grouped = all_errors_df.groupby('calculator_type')

    # Total number of entries per calculator_type
    total_counts = grouped.size().rename('total_entries')

    # Number of entries with missing data per calculator_type
    missing_counts = grouped['has_missing'].sum().rename('missing_entries')

    # Combine into a DataFrame
    missing_summary = pd.concat([total_counts, missing_counts], axis=1)

    # Compute percentage of missing entries
    missing_summary['missing_percentage'] = (missing_summary['missing_entries'] / missing_summary['total_entries']) * 100

    return missing_summary.reset_index()

# Function to plot the composite scorecard, including phonon errors
def plot_composite_scorecard(df):
    """Plot the composite scorecard for all calculators"""
    df = df.set_index('calculator_type')

    # Select only the error columns for plotting
    error_columns = ['err_a', 'err_b', 'err_c', 'err_form', 'err_vol',
                     'err_kv', 'err_c11', 'err_c44', 'err_surf_en', 'err_vac_en',
                     'phonon_mae', 'phonon_mad', 'phonon_pearson_corr']

    # Ensure the columns exist in df
    error_columns = [col for col in error_columns if col in df.columns]

    # Create a DataFrame with only error columns
    error_df = df[error_columns]

    fig = px.imshow(error_df, text_auto=True, aspect="auto", labels=dict(color="Mean Error"))
    fig.update_layout(title="Composite Error Metrics by Calculator Type")
    # Save the plot
    fig.write_image("composite_error_scorecard.png")
    fig.show()

# Function to plot missing data percentages
def plot_missing_percentages(df):
    """Plot the missing data percentages for all calculators"""
    df = df.set_index('calculator_type')

    # Select missing percentage columns
    percentage_columns = [col for col in df.columns if col.endswith('_missing_percentage')]

    # Create a DataFrame with missing percentage columns
    percentage_df = df[percentage_columns]

    fig = px.imshow(percentage_df, text_auto=True, aspect="auto", labels=dict(color="Missing %"))
    fig.update_layout(title="Missing Data Percentages by Calculator Type")
    # Save the plot
    fig.write_image("missing_data_percentages.png")
    fig.show()

if __name__ == '__main__':
    # Set the base directory where your output directories are located
    base_dir = '.'  # Change this to your actual base directory if needed

    # Collect all error data, including phonon errors
    all_errors_df = collect_error_data(base_dir)

    if not all_errors_df.empty:
        # Aggregate the errors
        composite_df = aggregate_errors(all_errors_df)
        # Save the aggregated errors to a CSV file
        composite_df.to_csv('composite_error_data.csv', index=False)
        print('Aggregated error data saved to composite_error_data.csv')

        # Count missing entries per calculator_type
        missing_summary = count_missing_entries(all_errors_df)
        # Save missing entries summary to CSV
        missing_summary.to_csv('missing_entries_summary.csv', index=False)
        print('Missing entries summary saved to missing_entries_summary.csv')

        # Optionally, merge missing_summary into composite_df
        composite_df = composite_df.merge(missing_summary, on='calculator_type')
        # Save updated composite_df
        composite_df.to_csv('composite_error_data_with_missing.csv', index=False)

        # Plot the composite scorecard
        plot_composite_scorecard(composite_df)

        # Plot missing data percentages
        plot_missing_percentages(composite_df)
    else:
        print('No error data files found.')

