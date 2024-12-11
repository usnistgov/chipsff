import os
import pandas as pd
import numpy as np
import json
import matplotlib

matplotlib.use("Agg")  # Use 'Agg' backend for headless environments
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from jarvis.db.webpages import Webpage
import plotly.express as px
import yaml
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import zipfile
from pathlib import Path

# =========================
# Leaderboard Generation with Total Time
# =========================


def write_leaderboard_files(base_dir, output_dir):
    # Define the properties to track with their respective sub-keys
    property_keys = {
        "a": ("energy", "initial_a", "final_a"),
        "b": ("energy", "initial_b", "final_b"),
        "c": ("energy", "initial_c", "final_c"),
        "vol": ("energy", "initial_vol", "final_vol"),
        "form_en": ("form_en", "form_energy_entry", "form_energy"),
        "c11": ("elastic_tensor", "c11_entry", "c11"),
        "c44": ("elastic_tensor", "c44_entry", "c44"),
        "kv": ("modulus", "kv_entry", "kv"),
        "surf_en": ("surface_energy", None, None),
        "vac_en": ("vacancy_energy", None, None),
    }

    # Initialize dictionaries to store contributions and ground truth
    contributions = {}
    ground_truth = {
        key: {"train": {}, "test": {}} for key in property_keys.keys()
    }

    # Iterate over immediate subdirectories in base_dir
    for entry in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, entry)
        if os.path.isdir(subdir_path) and entry.startswith("JVASP-"):
            # Split the directory name to extract JVASP-ID and calculator_type
            parts = entry.split("_", 1)
            if len(parts) != 2:
                print(
                    f"Skipping directory '{entry}' as it does not conform to naming convention."
                )
                continue
            jasp_id, calculator_type = parts
            jasp_id = jasp_id.strip()
            calculator_type = calculator_type.strip()

            # Path to results.json
            results_filename = f"{entry}_results.json"
            results_path = os.path.join(subdir_path, results_filename)
            if not os.path.isfile(results_path):
                print(f"results.json not found in '{subdir_path}'. Skipping.")
                continue

            # Load the results.json
            with open(results_path, "r") as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in '{results_path}': {e}")
                    continue

            # Extract total time from the results (assuming 'total_time' key exists in the JSON)
            total_time = results.get("total_time", None)

            # Process each property
            for prop, keys in property_keys.items():
                main_key, entry_key, pred_key = keys
                if main_key not in results:
                    continue  # Skip if property is missing

                if prop in ["a", "b", "c", "vol"]:
                    energy = results["energy"]
                    target = energy.get(entry_key)
                    prediction = energy.get(pred_key)
                    if target is not None and prediction is not None:
                        contributions.setdefault(
                            (prop, calculator_type), []
                        ).append(
                            {
                                "id": jasp_id,
                                "target": target,
                                "prediction": prediction,
                                "total_time": total_time,  # Include total time in the record
                            }
                        )
                        ground_truth[prop]["test"][jasp_id] = target

                elif prop == "form_en":
                    form_en = results["form_en"]
                    target = form_en.get(entry_key)
                    prediction = form_en.get(pred_key)
                    if target is not None and prediction is not None:
                        contributions.setdefault(
                            (prop, calculator_type), []
                        ).append(
                            {
                                "id": jasp_id,
                                "target": target,
                                "prediction": prediction,
                                "total_time": total_time,  # Include total time in the record
                            }
                        )
                        ground_truth[prop]["test"][jasp_id] = target

                elif prop in ["c11", "c44"]:
                    elastic = results["elastic_tensor"]
                    target = elastic.get(entry_key)
                    prediction = elastic.get(pred_key)
                    if target is not None and prediction is not None:
                        contributions.setdefault(
                            (prop, calculator_type), []
                        ).append(
                            {
                                "id": jasp_id,
                                "target": target,
                                "prediction": prediction,
                                "total_time": total_time,  # Include total time in the record
                            }
                        )
                        ground_truth[prop]["test"][jasp_id] = target

                elif prop == "kv":
                    modulus = results["modulus"]
                    target = modulus.get(entry_key)
                    prediction = modulus.get(pred_key)
                    if target is not None and prediction is not None:
                        contributions.setdefault(
                            (prop, calculator_type), []
                        ).append(
                            {
                                "id": jasp_id,
                                "target": target,
                                "prediction": prediction,
                                "total_time": total_time,  # Include total time in the record
                            }
                        )
                        ground_truth[prop]["test"][jasp_id] = target

                elif prop == "surf_en":
                    for surf in results["surface_energy"]:
                        surf_name = surf.get("name")
                        if (
                            surf_name
                            and "surf_en_entry" in surf
                            and "surf_en" in surf
                        ):
                            contributions.setdefault(
                                (prop, calculator_type), []
                            ).append(
                                {
                                    "id": surf_name,
                                    "target": surf["surf_en_entry"],
                                    "prediction": surf["surf_en"],
                                    "total_time": total_time,  # Include total time in the record
                                }
                            )
                            ground_truth[prop]["test"][surf_name] = surf[
                                "surf_en_entry"
                            ]

                elif prop == "vac_en":
                    for vac in results["vacancy_energy"]:
                        vac_name = vac.get("name")
                        if (
                            vac_name
                            and "vac_en_entry" in vac
                            and "vac_en" in vac
                        ):
                            contributions.setdefault(
                                (prop, calculator_type), []
                            ).append(
                                {
                                    "id": vac_name,
                                    "target": vac["vac_en_entry"],
                                    "prediction": vac["vac_en"],
                                    "total_time": total_time,  # Include total time in the record
                                }
                            )
                            ground_truth[prop]["test"][vac_name] = vac[
                                "vac_en_entry"
                            ]

    # Write contribution and ground truth files for each property and calculator_type
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate CSV.zip files per property per calculator_type
    for (prop, calculator_type), records in contributions.items():
        # Define file names without '-energy-' segment
        contribution_file = f"AI-SinglePropertyPrediction-{prop}-{calculator_type}_pretrained-test-mae.csv"
        contribution_path = os.path.join(output_dir, contribution_file)
        # Create DataFrame and save to CSV
        df = pd.DataFrame(records)
        df.to_csv(contribution_path, index=False)

        # Zip the CSV file
        zip_filename = contribution_path + ".zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(contribution_path, os.path.basename(contribution_path))
        os.remove(contribution_path)  # Remove the original CSV file
        print(f"Generated and zipped '{zip_filename}'.")

    # Generate single JSON.zip files per property
    for prop, gt in ground_truth.items():
        # Define file name without '-energy-' segment
        ground_truth_file = f"dft_3d_{prop}.json"
        ground_truth_path = os.path.join(output_dir, ground_truth_file)

        # Ensure "train" is an empty dictionary and "test" contains the ground truth
        ground_truth_content = {"train": {}, "test": gt["test"]}

        # Save ground truth JSON
        with open(ground_truth_path, "w") as json_file:
            json.dump(ground_truth_content, json_file, indent=2)

        # Zip the JSON file
        zip_filename = ground_truth_path + ".zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(ground_truth_path, os.path.basename(ground_truth_path))
        os.remove(ground_truth_path)  # Remove the original JSON file
        print(f"Generated and zipped '{zip_filename}'.")

    print(f"All leaderboard files have been written to '{output_dir}'.")


# =========================
# Phonon Data Processing
# =========================


def get_phonon_band_structure(jid):
    """Get phonon band structure data from JARVIS webpage data for a given jid."""
    # Create a Webpage object for the given jid
    w = Webpage(jid=jid)
    # Access the data from w.data
    data_dict = (
        w.data.get("basic_info", {})
        .get("main_elastic", {})
        .get("main_elastic_info", {})
    )

    if not data_dict:
        raise ValueError(f"No phonon data found for {jid}.")

    # Extract the relevant elements directly from the dictionary
    distances_str = data_dict.get("phonon_bandstructure_distances", "").strip(
        "'"
    )
    frequencies_str = data_dict.get(
        "phonon_bandstructure_frequencies", ""
    ).strip("'")

    if not distances_str or not frequencies_str:
        raise ValueError(f"Incomplete phonon data for {jid}.")

    # Parse distances (Wave vector path)
    distances = np.array([float(x) for x in distances_str.split(",") if x])

    # Parse frequencies (phonon bands at each q-point)
    frequency_blocks = frequencies_str.strip().split(";")
    frequencies = []
    for block in frequency_blocks:
        freq_values = [float(x) for x in block.strip().split(",") if x]
        if len(freq_values) > 0:  # Filter out empty bands
            frequencies.append(freq_values)

    # Ensure frequencies can be converted to a 2D array
    if any(len(freq) != len(distances) for freq in frequencies):
        raise ValueError(
            f"Mismatch between the length of distances ({len(distances)}) and frequencies."
        )

    # Convert to a 2D array
    frequencies = np.array(
        frequencies
    ).T  # Transpose to match (num_qpoints, num_bands)

    return distances, frequencies


def read_band_yaml(band_yaml_path):
    """
    Read band.yaml file and extract distances and frequencies.
    """
    with open(band_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)

    # Extract distances and frequencies
    distances = []
    frequencies = []

    for point in data_yaml["phonon"]:
        distances.append(point["distance"])
        freq_at_point = [band["frequency"] for band in point["band"]]
        frequencies.append(freq_at_point)

    distances = np.array(distances)
    frequencies = np.array(frequencies)

    return distances, frequencies


def compare_phonon_data(
    distances_ref, frequencies_ref, distances_calc, frequencies_calc
):
    """
    Compare the reference and calculated phonon frequencies.
    Returns MAE, MAD, and Pearson correlation coefficient.
    """
    # Define a common distance grid
    common_distances = np.linspace(
        max(distances_ref.min(), distances_calc.min()),
        min(distances_ref.max(), distances_calc.max()),
        num=1000,
    )

    # Interpolate frequencies onto the common grid
    num_bands = min(frequencies_ref.shape[1], frequencies_calc.shape[1])
    frequencies_ref_interp = np.zeros((len(common_distances), num_bands))
    frequencies_calc_interp = np.zeros((len(common_distances), num_bands))

    for i in range(num_bands):
        interp_ref = interp1d(
            distances_ref,
            frequencies_ref[:, i],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_calc = interp1d(
            distances_calc,
            frequencies_calc[:, i],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        frequencies_ref_interp[:, i] = interp_ref(common_distances)
        frequencies_calc_interp[:, i] = interp_calc(common_distances)

    # Calculate MAE per band and average
    mae_per_band = np.mean(
        np.abs(frequencies_ref_interp - frequencies_calc_interp), axis=0
    )
    mae = np.mean(mae_per_band)

    # Calculate MAD per band and average
    mad_per_band = np.mean(
        np.abs(
            frequencies_calc_interp - np.mean(frequencies_calc_interp, axis=0)
        ),
        axis=0,
    )
    mad = np.mean(mad_per_band)

    # Flatten the arrays for Pearson correlation
    ref_flat = frequencies_ref_interp.flatten()
    calc_flat = frequencies_calc_interp.flatten()
    pearson_corr, _ = pearsonr(ref_flat, calc_flat)

    return mae, mad, pearson_corr


def plot_phonon_band_structures(
    distances_ref,
    frequencies_ref,
    distances_calc,
    frequencies_calc,
    jid,
    calculator_type,
    output_dir,
    mae,
    mad,
    pearson_corr,
):
    """
    Plot the phonon band structures from JARVIS and MLFF calculations together.
    """
    num_bands = min(frequencies_ref.shape[1], frequencies_calc.shape[1])

    plt.figure(figsize=(10, 6), dpi=300)  # Increased DPI for better resolution

    # Plot JARVIS phonon bands
    for i in range(num_bands):
        plt.plot(
            distances_ref,
            frequencies_ref[:, i],
            "r--",
            linewidth=1.0,
            label="JARVIS" if i == 0 else "",
        )

    # Plot MLFF phonon bands
    for i in range(num_bands):
        plt.plot(
            distances_calc,
            frequencies_calc[:, i],
            "b",
            linewidth=1.0,
            label="MLFF" if i == 0 else "",
        )

    plt.xlabel("Wave Vector Distance", fontsize=14)
    plt.ylabel("Frequency (cm$^{-1}$)", fontsize=14)
    plt.title(
        f"Phonon Band Structure Comparison for {jid} ({calculator_type})",
        fontsize=16,
    )
    plt.legend()
    plt.grid(True)

    # Add error metrics as text annotations on the plot
    textstr = "\n".join(
        (
            f"MAE: {mae:.4f}",
            f"MAD: {mad:.4f}",
            f"Pearson Corr: {pearson_corr:.4f}",
        )
    )
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.tight_layout()

    # Save the figure
    plot_filename = os.path.join(
        output_dir, f"{jid}_{calculator_type}_phonon_comparison.png"
    )
    plt.savefig(plot_filename, dpi=300)  # Ensure high-resolution save
    plt.close()
    print(f"Saved phonon comparison plot for {jid} as '{plot_filename}'")


# =========================
# Error Data Collection and Aggregation
# =========================


def collect_error_data(base_dir):
    error_data_list = []
    phonon_error_data_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_error_dat.csv"):
                error_dat_path = os.path.join(root, file)
                # Read the CSV file into a DataFrame
                df = pd.read_csv(error_dat_path)
                # Extract jid and calculator_type from the filename or directory name
                filename = os.path.basename(error_dat_path)
                base_filename = filename.replace("_error_dat.csv", "")
                parts = base_filename.split("_", 1)
                if len(parts) != 2:
                    print(
                        f"Skipping file '{filename}' as it does not conform to naming convention."
                    )
                    continue
                jid, calculator_type = parts
                jid = jid.strip()
                calculator_type = calculator_type.strip()

                # Add jid and calculator_type to the DataFrame
                df["jid"] = jid
                df["calculator_type"] = calculator_type

                # Sum total_time from the error data
                total_time = df["time"].sum() if "time" in df.columns else 0
                df["total_time"] = total_time

                # Append the DataFrame to the list
                error_data_list.append(df)

                # Now process phonon data
                try:
                    # Get JARVIS phonon data
                    distances_ref, frequencies_ref = get_phonon_band_structure(
                        jid
                    )

                    # Get MLFF phonon data
                    band_yaml_path = os.path.join(root, "band.yaml")
                    if not os.path.isfile(band_yaml_path):
                        raise FileNotFoundError(
                            f"band.yaml not found in {root}"
                        )

                    distances_calc, frequencies_calc = read_band_yaml(
                        band_yaml_path
                    )

                    # Compare phonon data
                    mae, mad, pearson_corr = compare_phonon_data(
                        distances_ref,
                        frequencies_ref,
                        distances_calc,
                        frequencies_calc,
                    )

                    # Plot phonon band structures without zooming and x=y line
                    plot_phonon_band_structures(
                        distances_ref,
                        frequencies_ref,
                        distances_calc,
                        frequencies_calc,
                        jid,
                        calculator_type,
                        root,
                        mae,
                        mad,
                        pearson_corr,
                    )

                    # Append phonon error data
                    phonon_error_data_list.append(
                        {
                            "jid": jid,
                            "calculator_type": calculator_type,
                            "phonon_mae": mae,
                            "phonon_mad": mad,
                            "phonon_pearson_corr": pearson_corr,
                        }
                    )
                except Exception as e:
                    print(
                        f"Error processing phonon data for {jid} in {root}: {e}"
                    )

    # Concatenate all error DataFrames
    if error_data_list:
        all_errors_df = pd.concat(error_data_list, ignore_index=True)
    else:
        all_errors_df = pd.DataFrame()

    # Concatenate all phonon error DataFrames
    if phonon_error_data_list:
        phonon_errors_df = pd.DataFrame(phonon_error_data_list)
        # Merge with all_errors_df
        all_errors_df = all_errors_df.merge(
            phonon_errors_df, on=["jid", "calculator_type"], how="left"
        )
    else:
        print("No phonon error data collected.")

    return all_errors_df


def aggregate_errors(all_errors_df):
    # Group by calculator_type
    grouped = all_errors_df.groupby("calculator_type")

    # Error columns
    error_columns = [
        "err_a",
        "err_b",
        "err_c",
        "err_form",
        "err_vol",
        "err_kv",
        "err_c11",
        "err_c44",
        "err_surf_en",
        "err_vac_en",
        "phonon_mae",
        "phonon_mad",
        "phonon_pearson_corr",
    ]

    # Compute mean errors
    mean_errors = grouped[error_columns].mean()

    # Compute counts of missing entries per error column
    missing_counts = grouped[error_columns].apply(lambda x: x.isna().sum())

    # Rename columns in missing_counts
    missing_counts.columns = [
        col + "_missing_count" for col in missing_counts.columns
    ]

    # Compute total number of entries per calculator_type
    total_counts = grouped.size().rename("total_entries")

    # Sum total_time per calculator_type
    total_time_df = grouped["total_time"].sum().rename("total_time")

    # Reset index to 'calculator_type' for mean_errors and missing_counts
    mean_errors = mean_errors.reset_index()
    missing_counts = missing_counts.reset_index()
    total_time_df = total_time_df.reset_index()

    # Merge dataframes
    composite_df = mean_errors.merge(missing_counts, on="calculator_type")
    composite_df = composite_df.merge(total_counts, on="calculator_type")
    composite_df = composite_df.merge(total_time_df, on="calculator_type")

    # Compute missing percentages
    for col in error_columns:
        missing_count_col = col + "_missing_count"
        percentage_col = col + "_missing_percentage"
        composite_df[percentage_col] = (
            composite_df[missing_count_col] / composite_df["total_entries"]
        ) * 100

    return composite_df


def plot_composite_scorecard(df):
    """Plot the composite scorecard for all calculators"""
    df = df.set_index("calculator_type")

    # Select only the error columns for plotting
    error_columns = [
        "err_a",
        "err_b",
        "err_c",
        "err_form",
        "err_vol",
        "err_kv",
        "err_c11",
        "err_c44",
        "err_surf_en",
        "err_vac_en",
        "phonon_mae",
        "phonon_mad",
        "phonon_pearson_corr",
    ]

    # Ensure the columns exist in df
    error_columns = [col for col in error_columns if col in df.columns]

    # Create a DataFrame with only error columns
    error_df = df[error_columns]

    # Round specific columns to desired decimal places
    error_df[["err_kv", "err_c11", "err_c44", "phonon_mae", "phonon_mad"]] = (
        error_df[
            ["err_kv", "err_c11", "err_c44", "phonon_mae", "phonon_mad"]
        ].round(1)
    )
    # Round other columns to 2 decimal places
    other_cols = [
        col
        for col in error_df.columns
        if col
        not in ["err_kv", "err_c11", "err_c44", "phonon_mae", "phonon_mad"]
    ]
    error_df[other_cols] = error_df[other_cols].round(2)

    # Update column names for better readability
    error_df = error_df.rename(
        columns={
            "err_a": "a",
            "err_b": "b",
            "err_c": "c",
            "err_form": "Eform",
            "err_vol": "Volume",
            "err_kv": "Kv",
            "err_c11": "C11",
            "err_c44": "C44",
            "err_surf_en": "Surface Energy",
            "err_vac_en": "Vacancy Energy",
            "phonon_mae": "Phonon MAE",
            "phonon_mad": "Phonon MAD",
            "phonon_pearson_corr": "Phonon Pearson Corr",
        }
    )

    fig = px.imshow(
        error_df,
        text_auto=True,
        aspect="auto",
        labels=dict(color="Mean Error"),
    )
    fig.update_layout(
        title="Composite Error Metrics by Calculator Type",
        font=dict(size=16),
    )
    fig.update_traces(textfont_size=14)  # Increase font size for annotations
    # Save the plot with higher resolution
    fig.write_image(
        "composite_error_scorecard.png", scale=3
    )  # Increased scale for better resolution
    print("Saved composite error scorecard as 'composite_error_scorecard.png'")


def plot_missing_percentages(df):
    """Plot the missing data percentages for all calculators"""
    df = df.set_index("calculator_type")

    # Select missing percentage columns
    percentage_columns = [
        col for col in df.columns if col.endswith("_missing_percentage")
    ]

    # Create a DataFrame with missing percentage columns
    percentage_df = df[percentage_columns].round(1)  # Round to 1 decimal place

    # Update column names for better readability
    percentage_df.columns = [
        col.replace("err_", "").replace("_missing_percentage", "")
        for col in percentage_df.columns
    ]
    percentage_df.columns = [
        col.replace("form", "Eform")
        .replace("vol", "Volume")
        .replace("kv", "Kv")
        .replace("c11", "C11")
        .replace("c44", "C44")
        .replace("surf_en", "Surface Energy")
        .replace("vac_en", "Vacancy Energy")
        .replace("phonon_mae", "Phonon MAE")
        .replace("phonon_mad", "Phonon MAD")
        .replace("phonon_pearson_corr", "Phonon Pearson Corr")
        for col in percentage_df.columns
    ]

    fig = px.imshow(
        percentage_df,
        text_auto=".1f",
        aspect="auto",
        labels=dict(color="Missing %"),
    )
    fig.update_layout(
        title="Missing Data Percentages by Calculator Type",
        font=dict(size=16),
    )
    fig.update_traces(textfont_size=14)
    # Save the plot with higher resolution
    fig.write_image(
        "missing_data_percentages.png", scale=3
    )  # Increased scale for better resolution
    print(
        "Saved missing data percentages plot as 'missing_data_percentages.png'"
    )


# =========================
# Scalar Properties Processing
# =========================


def collect_scalar_properties_data(base_dir):
    data_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_results.json"):
                results_path = os.path.join(root, file)
                with open(results_path, "r") as f:
                    try:
                        results = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in '{results_path}': {e}")
                        continue

                dirname = os.path.basename(root)
                parts = dirname.split("_", 1)
                if len(parts) != 2:
                    print(
                        f"Skipping directory '{dirname}' as it does not conform to naming convention."
                    )
                    continue
                jid, calculator_type = parts
                jid = jid.strip()
                calculator_type = calculator_type.strip()

                # Collect calculated data
                scalar_data_calc = {
                    "jid": jid,
                    "calculator_type": calculator_type,
                    "a": results.get("energy", {}).get("final_a", np.nan),
                    "b": results.get("energy", {}).get("final_b", np.nan),
                    "c": results.get("energy", {}).get("final_c", np.nan),
                    "volume": results.get("energy", {}).get(
                        "final_vol", np.nan
                    ),
                    "formation_energy": results.get("form_en", {}).get(
                        "form_energy", np.nan
                    ),
                    "bulk_modulus": results.get("modulus", {}).get(
                        "kv", np.nan
                    ),
                    "c11": results.get("elastic_tensor", {}).get(
                        "c11", np.nan
                    ),
                    "c44": results.get("elastic_tensor", {}).get(
                        "c44", np.nan
                    ),
                }
                data_list.append(scalar_data_calc)

                # Collect reference data
                scalar_data_ref = {
                    "jid": jid,
                    "calculator_type": "JARVIS",
                    "a": results.get("energy", {}).get("initial_a", np.nan),
                    "b": results.get("energy", {}).get("initial_b", np.nan),
                    "c": results.get("energy", {}).get("initial_c", np.nan),
                    "volume": results.get("energy", {}).get(
                        "initial_vol", np.nan
                    ),
                    "formation_energy": results.get("form_en", {}).get(
                        "form_energy_entry", np.nan
                    ),
                    "bulk_modulus": results.get("modulus", {}).get(
                        "kv_entry", np.nan
                    ),
                    "c11": results.get("elastic_tensor", {}).get(
                        "c11_entry", np.nan
                    ),
                    "c44": results.get("elastic_tensor", {}).get(
                        "c44_entry", np.nan
                    ),
                }
                data_list.append(scalar_data_ref)
    df = pd.DataFrame(data_list)
    print(f"Total scalar property records collected: {len(df)}")
    print(
        f"Calculators found in scalar properties: {df['calculator_type'].unique()}"
    )
    return df


def create_scalar_parity_plots_consistent(scalar_df, properties):
    calculator_types = scalar_df["calculator_type"].unique().tolist()
    calculator_types = [ct for ct in calculator_types if ct != "JARVIS"]
    print("Calculator types found:", calculator_types)

    # Define distinct marker shapes for different calculator types
    marker_shapes = [
        "o",
        "s",
        "D",
        "^",
        "v",
        ">",
        "<",
        "p",
        "*",
        "h",
    ]  # Extend as needed
    shape_map = {
        ct: marker_shapes[i % len(marker_shapes)]
        for i, ct in enumerate(calculator_types)
    }

    # Define colors for calculator types, including new models
    color_map = {
        "alignn_ff": "#1f77b4",
        "chgnet": "#ff7f0e",
        "mace": "#2ca02c",
        "sevennet": "#d62728",
        "matgl": "#9467bd",
        "matgl-direct": "#8c564b",
        "orb": "#e377c2",
        "orb-v2": "#17becf",
        # Add more calculator types and colors as needed
    }

    # Units for properties
    units = {
        "a": " (Å)",
        "b": " (Å)",
        "c": " (Å)",
        "volume": " (Å³)",
        "formation_energy": " (eV/atom)",
        "bulk_modulus": " (GPa)",
        "c11": " (GPa)",
        "c44": " (GPa)",
    }

    # Axis labels mapping
    axis_label_map = {
        "formation_energy": "E$_{form}$",
        "bulk_modulus": "K$_v$",
        "c11": "C$_{11}$",
        "c44": "C$_{44}$",
        # Other properties can use their names directly
    }

    # Collect MAE from the aggregated error data to ensure consistency
    all_errors_df = collect_error_data(".")
    composite_df = aggregate_errors(all_errors_df)

    # Map properties to their corresponding error columns
    property_error_col_map = {
        "a": "err_a",
        "b": "err_b",
        "c": "err_c",
        "volume": "err_vol",
        "formation_energy": "err_form",
        "bulk_modulus": "err_kv",
        "c11": "err_c11",
        "c44": "err_c44",
    }

    mae_dict = {}
    for index, row in composite_df.iterrows():
        calculator_type = row["calculator_type"]
        for prop in properties:
            error_col = property_error_col_map.get(prop)
            if error_col and error_col in row:
                mae_dict[(calculator_type, prop)] = row[error_col]

    for prop in properties:
        print(f"\nProcessing property: {prop}")
        plt.figure(
            figsize=(8, 8), dpi=300
        )  # Increased DPI for better resolution
        jarvis_data = scalar_df[scalar_df["calculator_type"] == "JARVIS"][
            ["jid", prop]
        ].set_index("jid")
        print(f"Jarvis data size for {prop}: {jarvis_data.shape}")
        all_min = []
        all_max = []
        data_found = False  # Flag to check if any data is found
        for calculator in calculator_types:
            print(f"Processing calculator: {calculator}")
            calc_data = scalar_df[scalar_df["calculator_type"] == calculator][
                ["jid", prop]
            ].set_index("jid")
            print(f"Calculator data size for {calculator}: {calc_data.shape}")
            merged_data = jarvis_data.join(
                calc_data, lsuffix="_JARVIS", rsuffix=f"_{calculator}"
            ).dropna()
            print(f"Merged data size for {calculator}: {merged_data.shape}")
            if merged_data.empty:
                print(
                    f"No data to plot for calculator {calculator} and property {prop}"
                )
                continue
            data_found = True

            # Use MAE from the aggregated error data
            mae = mae_dict.get((calculator, prop), np.nan)

            # Plot the scatter with consistent MAE
            plt.scatter(
                merged_data[f"{prop}_JARVIS"],
                merged_data[f"{prop}_{calculator}"],
                label=(
                    f"{calculator} (MAE: {mae:.2f})"
                    if not np.isnan(mae)
                    else f"{calculator}"
                ),
                alpha=0.7,
                edgecolor="k",
                color=color_map.get(calculator, "#000000"),
                marker=shape_map.get(calculator, "o"),
                s=100,
            )
            # Collect min and max for percentile-based limits
            all_min.extend(merged_data[f"{prop}_JARVIS"].values)
            all_min.extend(merged_data[f"{prop}_{calculator}"].values)
            all_max.extend(merged_data[f"{prop}_JARVIS"].values)
            all_max.extend(merged_data[f"{prop}_{calculator}"].values)

        if data_found:
            if all_min and all_max:
                # Determine plot limits based on percentiles
                if prop == "formation_energy":
                    min_val = np.min(all_min + all_max)
                    max_val = np.max(all_min + all_max)
                else:
                    min_val = np.percentile(all_min, 5)
                    max_val = np.percentile(all_max, 95)
                plt.plot(
                    [min_val, max_val], [min_val, max_val], "k--", lw=2
                )  # 1:1 line

                plt.xlim(min_val, max_val)
                plt.ylim(min_val, max_val)

                # Use custom axis labels if available
                x_label = f"{axis_label_map.get(prop, prop)} (JARVIS){units.get(prop, '')}"
                y_label = f"{axis_label_map.get(prop, prop)} (Calculator){units.get(prop, '')}"

                plt.xlabel(x_label, fontsize=16)
                plt.ylabel(y_label, fontsize=16)
                plt.title(
                    f"Parity Plot for {axis_label_map.get(prop, prop)}",
                    fontsize=18,
                )
                plt.legend(fontsize=14)
                plt.grid(True)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                # Save the plot with higher resolution
                plt.savefig(f"{prop}_parity_plot.png", dpi=300)
                plt.close()
                print(
                    f"Saved parity plot for {prop} as '{prop}_parity_plot.png'"
                )
            else:
                print(f"No data available for property '{prop}' to plot.")
        else:
            print(f"No data found to plot for property {prop}.")


# =========================
# Surface Energies Processing
# =========================


def collect_surface_energies_data(base_dir):
    data_list = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_results.json"):
                results_path = os.path.join(root, file)
                with open(results_path, "r") as f:
                    try:
                        results = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in '{results_path}': {e}")
                        continue

                dirname = os.path.basename(root)
                parts = dirname.split("_", 1)
                if len(parts) != 2:
                    print(
                        f"Skipping directory '{dirname}' as it does not conform to naming convention."
                    )
                    continue
                jid, calculator_type = parts
                jid = jid.strip()
                calculator_type = calculator_type.strip()

                surface_energies = results.get("surface_energy", [])
                for surf in surface_energies:
                    # Calculated data
                    data_calc = {
                        "surface_name": surf.get("name"),
                        "calculator_type": calculator_type,
                        "surf_en": surf.get("surf_en", np.nan),
                    }
                    data_list.append(data_calc)
                    # Reference data
                    data_ref = {
                        "surface_name": surf.get("name"),
                        "calculator_type": "JARVIS",
                        "surf_en": surf.get("surf_en_entry", np.nan),
                    }
                    data_list.append(data_ref)

    df = pd.DataFrame(data_list)
    print(f"Total surface energy records collected: {len(df)}")
    print(
        f"Calculators found in surface energies: {df['calculator_type'].unique()}"
    )
    return df


def create_surface_energy_parity_plot(surface_df):
    calculator_types = surface_df["calculator_type"].unique().tolist()
    calculator_types = [ct for ct in calculator_types if ct != "JARVIS"]
    print("Calculator types found in surface energies:", calculator_types)

    # Define distinct marker shapes for different calculator types
    marker_shapes = [
        "o",
        "s",
        "D",
        "^",
        "v",
        ">",
        "<",
        "p",
        "*",
        "h",
    ]  # Extend as needed
    shape_map = {
        ct: marker_shapes[i % len(marker_shapes)]
        for i, ct in enumerate(calculator_types)
    }

    # Define colors for calculator types, including new models
    color_map = {
        "alignn_ff": "#1f77b4",
        "chgnet": "#ff7f0e",
        "mace": "#2ca02c",
        "sevennet": "#d62728",
        "matgl": "#9467bd",
        "matgl-direct": "#8c564b",
        "orb": "#e377c2",
        "orb-v2": "#17becf",
        # Add more calculator types and colors as needed
    }

    jarvis_data = surface_df[surface_df["calculator_type"] == "JARVIS"][
        ["surface_name", "surf_en"]
    ].set_index("surface_name")
    all_min = []
    all_max = []
    data_found = False
    plt.figure(figsize=(8, 8), dpi=300)  # Increased DPI for better resolution
    for calculator in calculator_types:
        print(f"Processing calculator: {calculator}")
        calc_data = surface_df[surface_df["calculator_type"] == calculator][
            ["surface_name", "surf_en"]
        ].set_index("surface_name")
        print(f"Calculator data size for {calculator}: {calc_data.shape}")
        merged_data = jarvis_data.join(
            calc_data, lsuffix="_JARVIS", rsuffix=f"_{calculator}"
        ).dropna()
        print(f"Merged data size for {calculator}: {merged_data.shape}")
        if merged_data.empty:
            print(
                f"No data to plot for calculator {calculator} in surface energies."
            )
            continue
        data_found = True

        # Use MAE from the aggregated error data
        mae = mean_absolute_error(
            merged_data["surf_en_JARVIS"], merged_data[f"surf_en_{calculator}"]
        )

        # Plot the scatter with consistent MAE
        plt.scatter(
            merged_data["surf_en_JARVIS"],
            merged_data[f"surf_en_{calculator}"],
            label=f"{calculator} (MAE: {mae:.2f})",
            alpha=0.7,
            edgecolor="k",
            color=color_map.get(calculator, "#000000"),
            marker=shape_map.get(calculator, "o"),
            s=100,
        )
        # Collect min and max for percentile-based limits
        all_min.extend(merged_data["surf_en_JARVIS"].values)
        all_min.extend(merged_data[f"surf_en_{calculator}"].values)
        all_max.extend(merged_data["surf_en_JARVIS"].values)
        all_max.extend(merged_data[f"surf_en_{calculator}"].values)

    if data_found:
        if all_min and all_max:
            # Determine plot limits based on percentiles
            min_val = np.percentile(all_min, 5)
            max_val = np.percentile(all_max, 95)
            plt.plot(
                [min_val, max_val], [min_val, max_val], "k--", lw=2
            )  # 1:1 line

            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)

            plt.xlabel("Surface Energy (JARVIS-DFT) (J/m$^2$)", fontsize=16)
            plt.ylabel("Surface Energy (Calculator) (J/m$^2$)", fontsize=16)
            plt.title(f"Parity Plot for Surface Energies", fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            # Save the plot with higher resolution
            plt.savefig(f"surface_energy_parity_plot.png", dpi=300)
            plt.close()
            print(
                "Saved surface energy parity plot as 'surface_energy_parity_plot.png'"
            )
        else:
            print("No data available to plot surface energy parity plot.")
    else:
        print("No data found to plot surface energy parity plot.")


# =========================
# Vacancy Energies Processing
# =========================


def collect_vacancy_energies_data(base_dir):
    data_list = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_results.json"):
                results_path = os.path.join(root, file)
                with open(results_path, "r") as f:
                    try:
                        results = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in '{results_path}': {e}")
                        continue

                dirname = os.path.basename(root)
                parts = dirname.split("_", 1)
                if len(parts) != 2:
                    print(
                        f"Skipping directory '{dirname}' as it does not conform to naming convention."
                    )
                    continue
                jid, calculator_type = parts
                jid = jid.strip()
                calculator_type = calculator_type.strip()

                vacancy_energies = results.get("vacancy_energy", [])
                for vac in vacancy_energies:
                    # Calculated data
                    data_calc = {
                        "vacancy_name": vac.get("name"),
                        "calculator_type": calculator_type,
                        "vac_en": vac.get("vac_en", np.nan),
                    }
                    data_list.append(data_calc)
                    # Reference data
                    data_ref = {
                        "vacancy_name": vac.get("name"),
                        "calculator_type": "JARVIS",
                        "vac_en": vac.get("vac_en_entry", np.nan),
                    }
                    data_list.append(data_ref)

    df = pd.DataFrame(data_list)
    print(f"Total vacancy energy records collected: {len(df)}")
    print(
        f"Calculators found in vacancy energies: {df['calculator_type'].unique()}"
    )
    return df


def create_vacancy_energy_parity_plot(vacancy_df):
    calculator_types = vacancy_df["calculator_type"].unique().tolist()
    calculator_types = [ct for ct in calculator_types if ct != "JARVIS"]
    print("Calculator types found in vacancy energies:", calculator_types)

    # Define distinct marker shapes for different calculator types
    marker_shapes = [
        "o",
        "s",
        "D",
        "^",
        "v",
        ">",
        "<",
        "p",
        "*",
        "h",
    ]  # Extend as needed
    shape_map = {
        ct: marker_shapes[i % len(marker_shapes)]
        for i, ct in enumerate(calculator_types)
    }

    # Define colors for calculator types, including new models
    color_map = {
        "alignn_ff": "#1f77b4",
        "chgnet": "#ff7f0e",
        "mace": "#2ca02c",
        "sevennet": "#d62728",
        "matgl": "#9467bd",
        "matgl-direct": "#8c564b",
        "orb": "#e377c2",
        "orb-v2": "#17becf",
        # Add more calculator types and colors as needed
    }

    jarvis_data = vacancy_df[vacancy_df["calculator_type"] == "JARVIS"][
        ["vacancy_name", "vac_en"]
    ].set_index("vacancy_name")
    all_min = []
    all_max = []
    data_found = False
    plt.figure(figsize=(8, 8), dpi=300)  # Increased DPI for better resolution
    for calculator in calculator_types:
        print(f"Processing calculator: {calculator}")
        calc_data = vacancy_df[vacancy_df["calculator_type"] == calculator][
            ["vacancy_name", "vac_en"]
        ].set_index("vacancy_name")
        print(f"Calculator data size for {calculator}: {calc_data.shape}")
        merged_data = jarvis_data.join(
            calc_data, lsuffix="_JARVIS", rsuffix=f"_{calculator}"
        ).dropna()
        print(f"Merged data size for {calculator}: {merged_data.shape}")
        if merged_data.empty:
            print(
                f"No data to plot for calculator {calculator} in vacancy energies."
            )
            continue
        data_found = True

        # Use MAE from the aggregated error data
        mae = mean_absolute_error(
            merged_data["vac_en_JARVIS"], merged_data[f"vac_en_{calculator}"]
        )

        # Plot the scatter with consistent MAE
        plt.scatter(
            merged_data["vac_en_JARVIS"],
            merged_data[f"vac_en_{calculator}"],
            label=f"{calculator} (MAE: {mae:.2f})",
            alpha=0.7,
            edgecolor="k",
            color=color_map.get(calculator, "#000000"),
            marker=shape_map.get(calculator, "o"),
            s=100,
        )
        # Collect min and max for percentile-based limits
        all_min.extend(merged_data["vac_en_JARVIS"].values)
        all_min.extend(merged_data[f"vac_en_{calculator}"].values)
        all_max.extend(merged_data["vac_en_JARVIS"].values)
        all_max.extend(merged_data[f"vac_en_{calculator}"].values)

    if data_found:
        if all_min and all_max:
            # Determine plot limits based on percentiles
            min_val = np.percentile(all_min, 5)
            max_val = np.percentile(all_max, 95)
            plt.plot(
                [min_val, max_val], [min_val, max_val], "k--", lw=2
            )  # 1:1 line

            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)

            plt.xlabel("Vacancy Energy (JARVIS-DFT) (eV)", fontsize=16)
            plt.ylabel("Vacancy Energy (Calculator) (eV)", fontsize=16)
            plt.title(f"Parity Plot for Vacancy Energies", fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            # Save the plot with higher resolution
            plt.savefig(f"vacancy_energy_parity_plot.png", dpi=300)
            plt.close()
            print(
                "Saved vacancy energy parity plot as 'vacancy_energy_parity_plot.png'"
            )
        else:
            print("No data available to plot vacancy energy parity plot.")
    else:
        print("No data found to plot vacancy energy parity plot.")


# =========================
# Main Execution
# =========================


def main():
    base_dir = "."  # Set your base directory path
    output_dir = (
        "./leaderboard_files"  # Set your desired output directory path
    )

    # 1. Leaderboard Generation
    print("Starting Leaderboard Generation...")
    write_leaderboard_files(base_dir, output_dir)

    # 2. Scalar Properties and Parity Plots
    print("\nCollecting Scalar Properties Data...")
    scalar_df = collect_scalar_properties_data(base_dir)
    if not scalar_df.empty:
        properties = [
            "a",
            "b",
            "c",
            "volume",
            "formation_energy",
            "bulk_modulus",
            "c11",
            "c44",
        ]
        print("Creating Scalar Parity Plots...")
        create_scalar_parity_plots_consistent(scalar_df, properties)
    else:
        print("No scalar property data found.")

    # 3. Surface Energies and Parity Plot
    print("\nCollecting Surface Energies Data...")
    surface_df = collect_surface_energies_data(base_dir)
    if not surface_df.empty:
        print("Creating Surface Energy Parity Plot...")
        create_surface_energy_parity_plot(surface_df)
    else:
        print("No surface energy data found.")

    # 4. Vacancy Energies and Parity Plot
    print("\nCollecting Vacancy Energies Data...")
    vacancy_df = collect_vacancy_energies_data(base_dir)
    if not vacancy_df.empty:
        print("Creating Vacancy Energy Parity Plot...")
        create_vacancy_energy_parity_plot(vacancy_df)
    else:
        print("No vacancy energy data found.")

    # 5. Error Data Collection and Aggregation
    print("\nCollecting Error Data...")
    all_errors_df = collect_error_data(base_dir)
    if not all_errors_df.empty:
        print("Aggregating Error Data...")
        composite_df = aggregate_errors(all_errors_df)
        composite_df.to_csv("composite_error_data.csv", index=False)
        print("Aggregated error data saved to composite_error_data.csv")

        print("Plotting Composite Scorecard...")
        plot_composite_scorecard(composite_df)

        print("Plotting Missing Data Percentages...")
        plot_missing_percentages(composite_df)
    else:
        print("No error data files found.")


if __name__ == "__main__":
    main()
