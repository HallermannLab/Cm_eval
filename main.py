try:
    import config
except ImportError:
    print(
        "\nERROR: 'config.py' not found.\n"
        "Please create a local 'config.py' by copying 'config_template.py' and "
        "adjusting the paths for your system.\n"
    )
    raise SystemExit(1)
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
import heka_reader
import git_save as myGit
from collections import defaultdict
import json
from statistics import median


# --- parameters ---
A_to_pA = 1e12
V_to_mV = 1e3
F_to_pF = 1e12
window_size_for_median_rolling_filter = 11  # must be odd


def exp_func(t, A, tau):
    return A * np.exp(-t / tau)

def exp_funcY(t, A, tau, y0):
    return A * np.exp(-t / tau) + y0

def exp_func2(t, A, tau1, aRel, tau2):
    return A * (1 - aRel) * np.exp(-t / tau1) + A * aRel * np.exp(-t / tau2)

# Check if series values are provided and valid (not NaN, not empty, and numeric)
def is_valid_series(value):
    if pd.isna(value):
        return False
    try:
        # Try to convert to int - if it fails, it's not a valid series number
        int_val = int(float(value))  # float() first in case it's a string like "1.0"
        return int_val > 0  # Series numbers should be positive
    except (ValueError, TypeError):
        return False


def bootstrap_median_trace(Y, n_boot=1000, random_state=None):
    """
    Compute the median and bootstrap SEM (std of medians) for each time point.

    Parameters
    ----------
    Y : array-like, shape (n_traces, n_timepoints)
        Input traces (NaNs allowed)
    n_boot : int
        Number of bootstrap resamples
    random_state : int, optional
        Seed for reproducibility

    Returns
    -------
    median_trace : array, shape (n_timepoints,)
        Median across traces at each time point
    sem_trace : array, shape (n_timepoints,)
        Bootstrap-based SEM for the median
    """
    Y = np.asarray(Y, dtype=float)
    if random_state is not None:
        np.random.seed(random_state)

    n_traces, n_timepoints = Y.shape
    median_trace = np.nanmedian(Y, axis=0)

    # Initialize bootstrap storage
    boot_medians = np.zeros((n_boot, n_timepoints))

    # Perform bootstrapping
    for i in range(n_boot):
        sample_idx = np.random.choice(n_traces, n_traces, replace=True)
        boot_sample = Y[sample_idx, :]
        boot_medians[i] = np.nanmedian(boot_sample, axis=0)

    # Standard deviation of bootstrapped medians = nonparametric SEM
    sem_trace = np.nanstd(boot_medians, axis=0, ddof=1)

    return median_trace, sem_trace


def analyze_trace(bundle, group_id, series_id, trace_name, axs_start_idx, axs, file_name):
    """
    Analyze a single trace series and return results.

    Args:
        bundle: HEKA data bundle
        group_id: Group ID (0)
        series_id: Series ID from metadata
        trace_name: Name prefix for variables (e.g., "snapshot_3ms_1")
        axs_start_idx: Starting index in axs array for plotting
        axs: Matplotlib axes array for plotting
        file_name: File name for error messages

    Returns:
        dict: Analysis results for this trace
    """
    results = {}

    try:
        # --------------- load traces -----------------
        trace_id = 0
        i_trace = A_to_pA * bundle.data[group_id, series_id, 0, trace_id]
        n_points = len(i_trace)
        sampling_interval = bundle.pul[group_id][series_id][0][trace_id].XInterval
        i_trace_time = np.arange(n_points) * sampling_interval

        trace_id = 1
        v_trace = V_to_mV * bundle.data[group_id, series_id, 0, trace_id]
        n_points = len(v_trace)
        sampling_interval = bundle.pul[group_id][series_id][0][trace_id].XInterval
        v_trace_time = np.arange(n_points) * sampling_interval

        trace_id = 2
        cm_trace = F_to_pF * bundle.data[group_id, series_id, 0, trace_id]
        n_points = len(cm_trace)
        sampling_interval = bundle.pul[group_id][series_id][0][trace_id].XInterval
        time = np.arange(n_points) * sampling_interval

        # remove NaNs from the cm_trace
        valid_mask = ~np.isnan(cm_trace)
        cm_trace = cm_trace[valid_mask]
        time = time[valid_mask]

        # Find t0 i.e. time of stimulation. i.e. where voltage crosses -20 mV in the voltage command trace
        crossing_indices = np.where(np.diff(v_trace > -20))[0]
        if len(crossing_indices) > 0:
            # Get the first crossing point
            t0_index = crossing_indices[0]
            t0 = v_trace_time[t0_index]
            # Find t1 - the next crossing after t0
            later_crossings = crossing_indices[crossing_indices > t0_index]
            if len(later_crossings) > 0:
                t1_index = later_crossings[0]
                t1 = v_trace_time[t1_index]
            else:
                t1 = None
        else:
            t0 = None
            t1 = None

        trace_base_st = 0.1 * t0
        trace_base_end = 0.9 * t0

        # ------------------ Baseline Subtraction -----------------------
        baseline_mask = (time >= trace_base_st) & (time <= trace_base_end)
        baseline_time = time[baseline_mask]
        baseline_values = cm_trace[baseline_mask]

        # -------- Baseline Cm value --------
        baseline = baseline_values.mean()

        # Fit linear function: y = m*x + b and subtract from median filtered trace
        coeffs = np.polyfit(baseline_time, baseline_values, deg=1)
        baseline_fit_line = np.polyval(coeffs, time)
        cm_trace_baseline_subtracted = cm_trace - baseline_fit_line
        # apply median filter using scipy
        cm_trace_baseline_subtracted = median_filter(cm_trace_baseline_subtracted,
                                                     size=window_size_for_median_rolling_filter)

        # Plot original trace with baseline fit
        axs[axs_start_idx].plot(time, cm_trace, label="Original")
        axs[axs_start_idx].plot(time, baseline_fit_line, label="Baseline fit", linestyle="--")
        axs[axs_start_idx].set_title(trace_name)
        axs[axs_start_idx].legend()
        axs[axs_start_idx].set_ylabel("pF")

        # ------------------ Ca trace analysis -----------------------
        baseline_ca_mask = (i_trace_time >= trace_base_st) & (i_trace_time <= trace_base_end)
        baseline_ca_values = i_trace[baseline_ca_mask]
        baseline_ca = baseline_ca_values.mean()

        ca_ss_mask = (i_trace_time >= t0 + 0.5 * (t1 - t0)) & (
                    i_trace_time < t1 - 0.0001)  # 100 us for making sure we don't include the tail current'
        ca_ss_values = i_trace[ca_ss_mask]
        ca_ss = ca_ss_values.mean() - baseline_ca

        ca_tail_mask = (i_trace_time >= t1) & (
                    i_trace_time < t1 + 0.002)  # 2 ms window for finding the peak of the tail current
        ca_tail_values = i_trace[ca_tail_mask]
        ca_tail = ca_tail_values.min() - baseline_ca

        # ----------------- shift time that t0 = 0s ---------------
        time_relative = time - t0
        fit_st = t1 - t0    # we need the relative time here
        fit_end = time_relative[-1]

        # ------------------------  1exp  -----------------------------------
        fit_mask = (time_relative >= fit_st) & (time_relative <= fit_end)
        try:
            popt, _ = curve_fit(exp_func, time_relative[fit_mask], cm_trace_baseline_subtracted[fit_mask],
                                p0=(np.max(cm_trace_baseline_subtracted), 5), bounds=([0, 0], [np.inf, np.inf]))
            A_fit, tau_fit = popt
        except Exception as e:
            print(f"        1-exp fit failed for trace {trace_name} for {file_name}: {e}")
            A_fit, tau_fit = np.nan, np.nan

        # Baseline-subtracted with exponential fit
        fit_plot_x = time_relative[time_relative >= 0]
        fit_plot_y = A_fit * np.exp(-fit_plot_x / tau_fit) if not np.isnan(A_fit) else np.zeros_like(fit_plot_x)
        axs[axs_start_idx + 1].plot(time_relative, cm_trace_baseline_subtracted, label="Baseline-subtracted")
        if not np.isnan(A_fit):
            axs[axs_start_idx + 1].plot(fit_plot_x, fit_plot_y, 'r--', label="Exponential fit")
        axs[axs_start_idx + 1].set_title("Baseline-subtracted + 1-exp fit")
        axs[axs_start_idx + 1].legend()
        axs[axs_start_idx + 1].set_xlabel("Time (s)")
        axs[axs_start_idx + 1].set_ylabel("pF")

        # ------------------------  1expY (exp with y offset)  -----------------------------------
        try:
            # Initial guess: A from max, tau from previous fit or 5, y0 as minimum value
            initial_y0 = np.min(cm_trace_baseline_subtracted[fit_mask])
            initial_A = np.max(cm_trace_baseline_subtracted[fit_mask]) - initial_y0
            initial_tau = tau_fit if not np.isnan(tau_fit) else 5

            popt_y, _ = curve_fit(exp_funcY, time_relative[fit_mask], cm_trace_baseline_subtracted[fit_mask],
                                  p0=(initial_A, initial_tau, initial_y0))
            A_fit_y, tau_fit_y, y0_fit_y = popt_y
        except Exception as e:
            print(f"        1-expY fit failed for trace {trace_name} for {file_name}: {e}")
            A_fit_y, tau_fit_y, y0_fit_y = np.nan, np.nan, np.nan

        # Baseline-subtracted with 1expY fit
        fit_plot_y_expY = A_fit_y * np.exp(-fit_plot_x / tau_fit_y) + y0_fit_y if not np.isnan(
            A_fit_y) else np.zeros_like(fit_plot_x)
        axs[axs_start_idx + 2].plot(time_relative, cm_trace_baseline_subtracted, label="Baseline-subtracted")
        if not np.isnan(A_fit_y):
            axs[axs_start_idx + 2].plot(fit_plot_x, fit_plot_y_expY, 'g--', label="1-ExpY fit")
        axs[axs_start_idx + 2].set_title("Baseline-subtracted + 1-expY fit")
        axs[axs_start_idx + 2].legend()
        axs[axs_start_idx + 2].set_xlabel("Time (s)")
        axs[axs_start_idx + 2].set_ylabel("pF")

        # ------------------------  2exp (double exponential)  -----------------------------------
        try:
            # Initial guess for double exponential
            initial_A = np.max(cm_trace_baseline_subtracted[fit_mask])
            initial_tau1 = tau_fit if not np.isnan(tau_fit) else 5  # fast component
            initial_aRel = 0.5  # equal contribution initially
            initial_tau2 = initial_tau1 * 10  # slow component, 10x slower

            popt_2exp, _ = curve_fit(exp_func2, time_relative[fit_mask], cm_trace_baseline_subtracted[fit_mask],
                                     p0=(initial_A, initial_tau1, initial_aRel, initial_tau2),
                                     bounds=([0, 0, 0, 0], [np.inf, np.inf, 1, np.inf]))
            A_fit_2exp, tau1_fit_2exp, aRel_fit_2exp, tau2_fit_2exp = popt_2exp
        except Exception as e:
            print(f"        2-exp fit failed for trace {trace_name} for {file_name}: {e}")
            A_fit_2exp, tau1_fit_2exp, aRel_fit_2exp, tau2_fit_2exp = np.nan, np.nan, np.nan, np.nan

        # Baseline-subtracted with 2exp fit
        if not np.isnan(A_fit_2exp):
            fit_plot_y_2exp = A_fit_2exp * (1 - aRel_fit_2exp) * np.exp(
                -fit_plot_x / tau1_fit_2exp) + A_fit_2exp * aRel_fit_2exp * np.exp(-fit_plot_x / tau2_fit_2exp)
        else:
            fit_plot_y_2exp = np.zeros_like(fit_plot_x)

        axs[axs_start_idx + 3].plot(time_relative, cm_trace_baseline_subtracted, label="Baseline-subtracted")
        if not np.isnan(A_fit_2exp):
            axs[axs_start_idx + 3].plot(fit_plot_x, fit_plot_y_2exp, 'm--', label="2-Exp fit")
        axs[axs_start_idx + 3].set_title("Baseline-subtracted + 2-exp fit")
        axs[axs_start_idx + 3].legend()
        axs[axs_start_idx + 3].set_xlabel("Time (s)")
        axs[axs_start_idx + 3].set_ylabel("pF")

        # ------------------------  Current trace (5th plot)  -----------------------------------
        # Plot current trace from t0-0.002 to t1+0.002
        if t0 is not None and t1 is not None:
            current_plot_start = t0 - 0.002
            current_plot_end = t1 + 0.002
            current_mask = (i_trace_time >= current_plot_start) & (i_trace_time <= current_plot_end)

            if np.any(current_mask):
                axs[axs_start_idx + 4].plot(i_trace_time[current_mask], i_trace[current_mask], 'c-', linewidth=1,
                                            label="Current")
                axs[axs_start_idx + 4].axvline(x=t0, color='r', linestyle='--', alpha=0.7, label="t0")
                axs[axs_start_idx + 4].axvline(x=t1, color='g', linestyle='--', alpha=0.7, label="t1")
                axs[axs_start_idx + 4].set_title("Current Trace")
                axs[axs_start_idx + 4].legend()
                axs[axs_start_idx + 4].set_xlabel("Time (s)")
                axs[axs_start_idx + 4].set_ylabel("Current (pA)")
                axs[axs_start_idx + 4].grid(True, alpha=0.3)
            else:
                axs[axs_start_idx + 4].text(0.5, 0.5, "No current data\nin time window",
                                            ha='center', va='center', transform=axs[axs_start_idx + 4].transAxes)
                axs[axs_start_idx + 4].set_title("Current Trace - No Data")
        else:
            axs[axs_start_idx + 4].text(0.5, 0.5, "t0 or t1 not found",
                                        ha='center', va='center', transform=axs[axs_start_idx + 4].transAxes)
            axs[axs_start_idx + 4].set_title("Current Trace - No Timing")

        # Store results
        results = {
            f'{trace_name}_baseline': baseline,
            f'{trace_name}_ca_ss': ca_ss,
            f'{trace_name}_ca_tail': ca_tail,
            f'{trace_name}_1exp_A': A_fit,
            f'{trace_name}_1exp_tau': tau_fit,
            f'{trace_name}_1expY_A': A_fit_y,
            f'{trace_name}_1expY_tau': tau_fit_y,
            f'{trace_name}_1expY_y0': y0_fit_y,
            f'{trace_name}_2exp_A': A_fit_2exp,
            f'{trace_name}_2exp_tau1': tau1_fit_2exp,
            f'{trace_name}_2exp_aRel': aRel_fit_2exp,
            f'{trace_name}_2exp_tau2': tau2_fit_2exp,
            'cm_trace_baseline_subtracted': cm_trace_baseline_subtracted,
            'time_relative': time_relative
        }

    except Exception as e:
        print(f"        Error analyzing {trace_name} for {file_name}: {e}")
        # Return NaN values for failed analysis
        results = {
            f'{trace_name}_baseline': np.nan,
            f'{trace_name}_ca_ss': np.nan,
            f'{trace_name}_ca_tail': np.nan,
            f'{trace_name}_1exp_A': np.nan,
            f'{trace_name}_1exp_tau': np.nan,
            f'{trace_name}_1expY_A': np.nan,
            f'{trace_name}_1expY_tau': np.nan,
            f'{trace_name}_1expY_y0': np.nan,
            f'{trace_name}_2exp_A': np.nan,
            f'{trace_name}_2exp_tau1': np.nan,
            f'{trace_name}_2exp_aRel': np.nan,
            f'{trace_name}_2exp_tau2': np.nan,
            'cm_trace_baseline_subtracted': None,
            'time_relative': None
        }

    return results


def plot_combined_group_analysis(all_traces, group_traces, all_time_arrays, group_time_arrays,
                                 trace_types, unique_groups, output_folder_results):
    """
    Generate combined group analysis plots with 3 plots per trace type:
    - Superposition (individual traces)
    - Average ± parametric SEM
    - Median ± bootstrap SEM

    Creates one PDF per group (including "all").
    """

    # Define groups to process (all + individual groups)
    groups_to_process = ["all"] + unique_groups

    for group_name in groups_to_process:
        print(f"Creating combined analysis for group: {group_name}")

        # Create figure with 5 rows (trace types) × 3 columns (plot types)
        fig, axes = plt.subplots(5, 3, figsize=(18, 30))
        fig.suptitle(f"Combined Analysis - {group_name.title()}", fontsize=16)

        for trace_idx, trace_type in enumerate(trace_types):
            # Get traces for this group and trace type
            if group_name == "all":
                traces = all_traces[trace_type]
                time_arrays = all_time_arrays[trace_type]
            else:
                traces = group_traces[trace_type].get(group_name, [])
                time_arrays = group_time_arrays[trace_type].get(group_name, [])

            # Skip if no traces available
            if not traces or len(traces) == 0:
                for col in range(3):
                    axes[trace_idx, col].text(0.5, 0.5, f"No data for {trace_type}",
                                              ha='center', va='center', transform=axes[trace_idx, col].transAxes)
                    axes[trace_idx, col].set_title(f"{trace_type} - No Data")
                continue

            # Find common time base (use the first trace's time array as reference)
            reference_time = time_arrays[0] if time_arrays else np.linspace(0, 1, 1000)

            # Interpolate all traces to common time base
            interpolated_traces = []
            for trace, time_array in zip(traces, time_arrays):
                if len(trace) > 0 and len(time_array) > 0:
                    interpolated_trace = np.interp(reference_time, time_array, trace)
                    interpolated_traces.append(interpolated_trace)

            if not interpolated_traces:
                for col in range(3):
                    axes[trace_idx, col].text(0.5, 0.5, f"No valid data for {trace_type}",
                                              ha='center', va='center', transform=axes[trace_idx, col].transAxes)
                continue

            interpolated_traces = np.array(interpolated_traces)
            n_traces = len(interpolated_traces)

            # Plot 1: Superposition (Individual traces)
            ax1 = axes[trace_idx, 0]
            for i, (trace, time_array) in enumerate(zip(traces, time_arrays)):
                if len(trace) > 0 and len(time_array) > 0:
                    ax1.plot(time_array, trace, alpha=0.3, color='gray', linewidth=0.5)
            ax1.set_title(f"{trace_type} - Superposition (n={n_traces})")
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Capacitance (pF)')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Average ± parametric SEM
            ax2 = axes[trace_idx, 1]
            mean_trace = np.mean(interpolated_traces, axis=0)
            sem_trace = np.std(interpolated_traces, axis=0) / np.sqrt(n_traces) if n_traces > 1 else np.zeros_like(
                mean_trace)

            ax2.plot(reference_time, mean_trace, 'b-', linewidth=2, label=f'Mean (n={n_traces})')
            ax2.fill_between(reference_time, mean_trace - sem_trace, mean_trace + sem_trace,
                             alpha=0.3, color='blue', label='±SEM')
            ax2.set_title(f"{trace_type} - Average ± SEM")
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Capacitance (pF)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Median ± bootstrap SEM
            ax3 = axes[trace_idx, 2]
            try:
                median_trace, bootstrap_sem = bootstrap_median_trace(interpolated_traces, n_boot=1000, random_state=42)

                ax3.plot(reference_time, median_trace, 'r-', linewidth=2, label=f'Median (n={n_traces})')
                ax3.fill_between(reference_time, median_trace - bootstrap_sem, median_trace + bootstrap_sem,
                                 alpha=0.3, color='red', label='±Bootstrap SEM')
                ax3.set_title(f"{trace_type} - Median ± Bootstrap SEM")
                ax3.legend()
            except Exception as e:
                ax3.text(0.5, 0.5, f"Bootstrap failed: {str(e)}",
                         ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title(f"{trace_type} - Bootstrap Failed")

            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Capacitance (pF)')
            ax3.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(output_folder_results, f"combined_analysis_{group_name}.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Combined analysis saved to: {output_path}")

def plot_group_analysis(traces, time_arrays, title, output_path):
    """
    Generate group analysis plots for multiple traces.

    Parameters:
    - traces: List of trace arrays
    - time_arrays: List of corresponding time arrays
    - title: Title for the plots
    - output_path: Path to save the PDF output
    """
    if not traces or len(traces) == 0:
        print(f"Warning: No traces provided for {title}")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot 1: Individual traces
    ax1 = axes[0]
    for i, (trace, time_array) in enumerate(zip(traces, time_arrays)):
        if len(trace) > 0 and len(time_array) > 0:
            ax1.plot(time_array, trace, alpha=0.3, color='gray', linewidth=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current (pA)')
    ax1.set_title('Individual Traces')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average trace with SEM
    ax2 = axes[1]
    if len(traces) > 0:
        # Find common time base (use the first trace's time array as reference)
        reference_time = time_arrays[0] if time_arrays else np.linspace(0, 1, 1000)

        # Interpolate all traces to common time base
        interpolated_traces = []
        for trace, time_array in zip(traces, time_arrays):
            if len(trace) > 0 and len(time_array) > 0:
                # Interpolate trace to reference time
                interpolated_trace = np.interp(reference_time, time_array, trace)
                interpolated_traces.append(interpolated_trace)

        if interpolated_traces:
            interpolated_traces = np.array(interpolated_traces)
            mean_trace = np.mean(interpolated_traces, axis=0)
            # Calculate SEM using numpy instead of scipy
            sem_trace = np.std(interpolated_traces, axis=0) / np.sqrt(len(interpolated_traces)) if len(interpolated_traces) > 1 else np.zeros_like(
                mean_trace)

            ax2.plot(reference_time, mean_trace, 'b-', linewidth=2, label=f'Mean (n={len(interpolated_traces)})')
            ax2.fill_between(reference_time, mean_trace - sem_trace, mean_trace + sem_trace,
                             alpha=0.3, color='blue', label='±SEM')
            ax2.legend()

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Current (pA)')
    ax2.set_title('Average ± SEM')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Statistics summary
    ax3 = axes[2]
    if traces:
        # Calculate some basic statistics
        peak_values = []
        baseline_values = []

        for trace in traces:
            if len(trace) > 0:
                peak_values.append(np.max(np.abs(trace)))
                baseline_values.append(np.mean(trace[:min(100, len(trace))]))  # First 100 points as baseline

        if peak_values:
            ax3.hist(peak_values, bins=min(10, len(peak_values)), alpha=0.7, color='orange', label='Peak Amplitudes')
            ax3.set_xlabel('Peak Amplitude (pA)')
            ax3.set_ylabel('Count')
            ax3.set_title('Peak Amplitude Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

    # Plot 4: Time course analysis
    ax4 = axes[3]
    if traces:
        # Plot peak values over trace index (time course)
        peak_values = []
        for trace in traces:
            if len(trace) > 0:
                peak_values.append(np.max(np.abs(trace)))

        if peak_values:
            ax4.plot(range(len(peak_values)), peak_values, 'ro-', markersize=4, linewidth=1)
            ax4.set_xlabel('Trace Number')
            ax4.set_ylabel('Peak Amplitude (pA)')
            ax4.set_title('Peak Amplitude vs Trace Number')
            ax4.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Group analysis plot saved to: {output_path}")



def Cm_eval():
    # --- Create Output Folders ---
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    output_folder = os.path.join(config.ROOT_FOLDER, f"output_{config.MY_INITIAL}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    output_folder_used_data_and_code = os.path.join(output_folder, "used_data_and_code")
    os.makedirs(output_folder_used_data_and_code, exist_ok=True)

    output_folder_individual_experiments = os.path.join(output_folder, "individual_experiments")
    os.makedirs(output_folder_individual_experiments, exist_ok=True)

    output_folder_results = os.path.join(output_folder, "results")
    os.makedirs(output_folder_results, exist_ok=True)

    # --- Load Metadata ---
    metadata_df = pd.read_excel(config.METADATA_FILE)
    # save used data
    metadata_df.to_excel(os.path.join(output_folder_used_data_and_code, "my_data.xlsx"), index=False)

    # --- Extract experimental groups ---
    if 'groups' in metadata_df.columns:
        group_column = metadata_df['groups']
        # Get unique group names, excluding NaN values
        unique_groups = [str(g) for g in group_column.dropna().unique() if pd.notna(g)]
        print(f"Found {len(unique_groups)} experimental groups: {unique_groups}")
    else:
        print("No 'groups' column found in metadata. Proceeding without group analysis.")
        unique_groups = []

    # Initialize storage for traces by groups for each trace type
    trace_types = ['snapshot_3ms_1', 'snapshot_30ms_1', 'sine_3ms_1', 'snapshot_3ms_2', 'snapshot_30ms_2']

    all_traces = {trace_type: [] for trace_type in trace_types}
    group_traces = {trace_type: {group: [] for group in unique_groups} for trace_type in trace_types}
    all_time_relative = {trace_type: [] for trace_type in trace_types}
    group_time_relative = {trace_type: {group: [] for group in unique_groups} for trace_type in trace_types}

    # === GIT SAVE ===
    # Provide the current script path (only works in .py, not notebooks)
    script_path = __file__ if '__file__' in globals() else None
    myGit.save_git_info(output_folder_used_data_and_code, script_path)

    # --- Results Storage ---
    results = []

    # Create figure for average traces (now 5 trace types)
    fig_avg, axs_avg = plt.subplots(5, 3, figsize=(15, 25))
    axs_avg = axs_avg.flatten()

    # --- Process Each Cell ---
    for cell_count, row in metadata_df.iterrows():
        file_name = row['file_name']
        print(f"Processing cell {cell_count + 1}: {file_name}")

        dat_path = os.path.join(config.EXTERNAL_DATA_FOLDER, file_name)
        try:
            bundle = heka_reader.Bundle(dat_path)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue

        group_id = 0
        fig, axs = plt.subplots(5, 5, figsize=(25, 25))  # Now 5 rows for 5 trace types
        axs = axs.flatten()

        # Initialize result dictionary for this cell
        cell_results = {
            "cell_count": cell_count + 1,
            "file_name": file_name,
        }

        # ==========================================================================================
        # --- Analyze all trace types ---
        # ==========================================================================================

        for trace_idx, trace_type in enumerate(trace_types):
            series_column = f'{trace_type}_series'

            # Check if this trace type is available for this cell
            trace_available = is_valid_series(row.get(series_column, np.nan))

            if trace_available:
                series_id = int(float(row[series_column])) - 1  # Convert to 0-based index
                print(f"        Analyzing {trace_type}")

                # Calculate subplot indices (4 plots per row)
                axs_start_idx = trace_idx * 5

                # Analyze this trace
                trace_results = analyze_trace(bundle, group_id, series_id, trace_type, axs_start_idx, axs, file_name)

                # Add trace results to cell results
                cell_results.update({k: v for k, v in trace_results.items()
                                     if k not in ['cm_trace_baseline_subtracted', 'time_relative']})

                # --- Collect traces for group analysis ---
                if trace_results['cm_trace_baseline_subtracted'] is not None:
                    # Ensure traces are proper numpy arrays with consistent dtype
                    cm_trace_clean = np.asarray(trace_results['cm_trace_baseline_subtracted'], dtype=np.float64)
                    time_relative_clean = np.asarray(trace_results['time_relative'], dtype=np.float64)

                    # Store trace for all cells
                    all_traces[trace_type].append(cm_trace_clean)
                    all_time_relative[trace_type].append(time_relative_clean)

                    # Store trace by group if group information is available
                    if 'groups' in metadata_df.columns and pd.notna(row['groups']):
                        cell_group = str(row['groups'])
                        if cell_group in group_traces[trace_type]:
                            group_traces[trace_type][cell_group].append(cm_trace_clean)
                            group_time_relative[trace_type][cell_group].append(time_relative_clean)

            else:
                print(f"        Skipping {trace_type}")
                # Mark skipped plots
                axs_start_idx = trace_idx * 5
                for i in range(5):
                    axs[axs_start_idx + i].set_title(f"SKIPPED - {trace_type}")
                    axs[axs_start_idx + i].grid(True)

                # Initialize NaN values for skipped analysis
                nan_keys = [
                    f'{trace_type}_baseline', f'{trace_type}_ca_ss', f'{trace_type}_ca_tail',
                    f'{trace_type}_1exp_A', f'{trace_type}_1exp_tau',
                    f'{trace_type}_1expY_A', f'{trace_type}_1expY_tau', f'{trace_type}_1expY_y0',
                    f'{trace_type}_2exp_A', f'{trace_type}_2exp_tau1', f'{trace_type}_2exp_aRel',
                    f'{trace_type}_2exp_tau2'
                ]
                for key in nan_keys:
                    cell_results[key] = np.nan

        # Store results for this cell
        results.append(cell_results)

        # --- Save PDFs for each individual cell ---
        plt.figure(fig.number)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_individual_experiments, f"{1 + cell_count:03d}_{file_name}.pdf"))
        plt.close(fig)

    # ==========================================================================================
    # --- Group Analysis After the Loop ---
    # ==========================================================================================
    # Generate combined plots with 3 subplots per trace type
    plot_combined_group_analysis(all_traces, group_traces, all_time_relative, group_time_relative,
                                trace_types, unique_groups, output_folder_results)

    # ==========================================================================================
    # --- Save all results to Excel ---
    # ==========================================================================================
    # results EXCEL file
    results_df = pd.DataFrame(results)
    excel_output_path = os.path.join(output_folder_results, "results.xlsx")
    results_df.to_excel(excel_output_path, index=False)


def start_browser():
    # Import and start the browser
    from browser import app, win
    win.show()
    if sys.flags.interactive == 0:
        app.exec_()


if __name__ == '__main__':
    Cm_eval()
    start_browser()  # This will start the browser after CC_eval completes