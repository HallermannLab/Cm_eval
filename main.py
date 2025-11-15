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


def Cm_eval():

    # --- Create Output Folders ---
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    output_folder = os.path.join(config.ROOT_FOLDER, f"output_{config.MY_INITIAL}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    output_folder_used_data_and_code = os.path.join(output_folder, "used_data_and_code")
    os.makedirs(output_folder_used_data_and_code, exist_ok=True)

    output_folder_traces = os.path.join(output_folder, "traces")
    os.makedirs(output_folder_traces, exist_ok=True)

    output_folder_results = os.path.join(output_folder, "results")
    os.makedirs(output_folder_results, exist_ok=True)

    # --- Load Metadata ---
    metadata_df = pd.read_excel(config.METADATA_FILE)
    # save used data
    metadata_df.to_excel(os.path.join(output_folder_used_data_and_code, "my_data.xlsx"), index=False)

    # initialize analysis points structure
    analysis_points = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(list)
                )
            )
        )
    )

    # === GIT SAVE ===
    # Provide the current script path (only works in .py, not notebooks)
    script_path = __file__ if '__file__' in globals() else None
    myGit.save_git_info(output_folder_used_data_and_code, script_path)

    # --- Results Storage ---
    results = []


    # --- Process Each Cell ---
    for cell_count, row in metadata_df.iterrows():
        file_name = row['file_name']
        print(f"Processing cell {cell_count + 1}: {file_name}")

        snapshot_3ms_1_series_available = is_valid_series(row['snapshot_3ms_1_series'])

        # Only convert to int and subtract 1 if value is available
        snapshot_3ms_1_series = int(float(row['snapshot_3ms_1_series'])) - 1 if snapshot_3ms_1_series_available else None


        dat_path = os.path.join(config.EXTERNAL_DATA_FOLDER, file_name)
        try:
            bundle = heka_reader.Bundle(dat_path)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue

        group_id = 0
        fig, axs = plt.subplots(5, 4, figsize=(20, 20))
        axs = axs.flatten()

        # ==========================================================================================
        # --- snapshot_3ms_1_series ---
        # ==========================================================================================

        if snapshot_3ms_1_series_available:
            series_id = snapshot_3ms_1_series

            #--------------- load traces -----------------
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
            # n_sweeps = bundle.pul[group_id][series_id].NumberSweeps
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
            else:
                t0 = None

            trace_base_st = 0.1 * t0
            trace_base_end = 0.9 * t0

            # ------------------ Baseline Subtraction -----------------------
            baseline_mask = (time >= trace_base_st) & (time <= trace_base_end)
            baseline_time = time[baseline_mask]
            baseline_values = cm_trace[baseline_mask]

            snapshot_3ms_1_baseline = baseline_values.mean()

            # Fit linear function: y = m*x + b and subtract from median filtered trace
            coeffs = np.polyfit(baseline_time, baseline_values, deg=1)
            baseline_fit_line = np.polyval(coeffs, time)
            cm_trace_baseline_subtracted = cm_trace - baseline_fit_line
            # apply median filter using scipy
            cm_trace_baseline_subtracted = median_filter(cm_trace_baseline_subtracted, size=window_size_for_median_rolling_filter)

            # Plot original trace with baseline fit
            axs[0].plot(time, cm_trace, label="Original")
            axs[0].plot(time, baseline_fit_line, label="Baseline fit", linestyle="--")
            axs[0].set_title("snapshot_3ms_1")
            axs[0].legend()
            axs[0].set_ylabel("pF")

            # ----------------- shift time that t0 = 0s ---------------
            time_relative = time - t0
            fit_st = 0.004
            fit_end = time_relative[-1]

            # ------------------------  1exp  -----------------------------------
            fit_mask = (time_relative >= fit_st) & (time_relative <= fit_end)
            try:
                popt, _ = curve_fit(exp_func, time_relative[fit_mask], cm_trace_baseline_subtracted[fit_mask],
                                    p0=(np.max(cm_trace_baseline_subtracted), 5), bounds=([0, 0], [np.inf, np.inf]))
                A_fit, tau_fit = popt
            except Exception as e:
                print(f"        1-exp fit failed for trace snapshot_3ms_1_series for {file_name}: {e}")
                A_fit, tau_fit = np.nan, np.nan

            snapshot_3ms_1_1exp_A = A_fit
            snapshot_3ms_1_1exp_tau = tau_fit

            # Baseline-subtracted with exponential fit
            fit_plot_x = time_relative[time_relative >= 0]
            fit_plot_y = A_fit * np.exp(-fit_plot_x / tau_fit)
            axs[1].plot(time_relative, cm_trace_baseline_subtracted, label="Baseline-subtracted")
            if not np.isnan(A_fit):
                axs[1].plot(fit_plot_x, fit_plot_y, 'r--', label="Exponential fit")
            axs[1].set_title("Baseline-subtracted + 1-exp fit")
            axs[1].legend()
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("pF")

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
                print(f"        1-expY fit failed for trace snapshot_3ms_1_series for {file_name}: {e}")
                A_fit_y, tau_fit_y, y0_fit_y = np.nan, np.nan, np.nan

            snapshot_3ms_1_1expY_A = A_fit_y
            snapshot_3ms_1_1expY_tau = tau_fit_y
            snapshot_3ms_1_1expY_y0 = y0_fit_y

            # Baseline-subtracted with 1expY fit
            fit_plot_y_expY = A_fit_y * np.exp(-fit_plot_x / tau_fit_y) + y0_fit_y
            axs[2].plot(time_relative, cm_trace_baseline_subtracted, label="Baseline-subtracted")
            if not np.isnan(A_fit_y):
                axs[2].plot(fit_plot_x, fit_plot_y_expY, 'g--', label="1-ExpY fit")
            axs[2].set_title("Baseline-subtracted + 1-expY fit")
            axs[2].legend()
            axs[2].set_xlabel("Time (s)")
            axs[2].set_ylabel("pF")

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
                print(f"        2-exp fit failed for trace snapshot_3ms_1_series for {file_name}: {e}")
                A_fit_2exp, tau1_fit_2exp, aRel_fit_2exp, tau2_fit_2exp = np.nan, np.nan, np.nan, np.nan

            snapshot_3ms_1_2exp_A = A_fit_2exp
            snapshot_3ms_1_2exp_tau1 = tau1_fit_2exp
            snapshot_3ms_1_2exp_aRel = aRel_fit_2exp
            snapshot_3ms_1_2exp_tau2 = tau2_fit_2exp

            # Baseline-subtracted with 2exp fit
            fit_plot_y_2exp = A_fit_2exp * (1 - aRel_fit_2exp) * np.exp(
                -fit_plot_x / tau1_fit_2exp) + A_fit_2exp * aRel_fit_2exp * np.exp(-fit_plot_x / tau2_fit_2exp)
            axs[3].plot(time_relative, cm_trace_baseline_subtracted, label="Baseline-subtracted")
            if not np.isnan(A_fit_2exp):
                axs[3].plot(fit_plot_x, fit_plot_y_2exp, 'm--', label="2-Exp fit")
            axs[3].set_title("Baseline-subtracted + 2-exp fit")
            axs[3].legend()
            axs[3].set_xlabel("Time (s)")
            axs[3].set_ylabel("pF")

        else:
            print(f"        Skipping snapshot_3ms_1_series")
            axs[0].set_title("SKIPPED")
            axs[0].grid(True)
            axs[1].set_title("SKIPPED")
            axs[1].grid(True)
            axs[2].set_title("SKIPPED")
            axs[2].grid(True)
            axs[3].set_title("SKIPPED")
            axs[3].grid(True)

            # Initialize variables for skipped analysis
            snapshot_3ms_1_baseline = np.nan
            snapshot_3ms_1_1exp_A = np.nan
            snapshot_3ms_1_1exp_tau = np.nan
            snapshot_3ms_1_1expY_A = np.nan
            snapshot_3ms_1_1expY_tau = np.nan
            snapshot_3ms_1_1expY_y0 = np.nan
            snapshot_3ms_1_2exp_A = np.nan
            snapshot_3ms_1_2exp_tau1 = np.nan
            snapshot_3ms_1_2exp_aRel = np.nan
            snapshot_3ms_1_2exp_tau2 = np.nan

        # ==========================================================================================
        # --- Store results within the loop ---
        # ==========================================================================================
        results.append({
            "cell_count": cell_count + 1,
            "file_name": file_name,
            "snapshot_3ms_1_baseline": snapshot_3ms_1_baseline,
            "snapshot_3ms_1_1exp_A": snapshot_3ms_1_1exp_A,
            "snapshot_3ms_1_1exp_tau": snapshot_3ms_1_1exp_tau,
            "snapshot_3ms_1_1expY_A": snapshot_3ms_1_1expY_A,
            "snapshot_3ms_1_1expY_tau": snapshot_3ms_1_1expY_tau,
            "snapshot_3ms_1_1expY_y0": snapshot_3ms_1_1expY_y0,
            "snapshot_3ms_1_2exp_A": snapshot_3ms_1_2exp_A,
            "snapshot_3ms_1_2exp_tau1": snapshot_3ms_1_2exp_tau1,
            "snapshot_3ms_1_2exp_aRel": snapshot_3ms_1_2exp_aRel,
            "snapshot_3ms_1_2exp_tau2": snapshot_3ms_1_2exp_tau2
        })


        # --- Save PDF ---
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_traces, f"{1+cell_count:03d}_{file_name}.pdf"))
        plt.close()

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