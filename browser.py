try:
    import config
except ImportError:
    print(
        "\nERROR: 'config.py' not found.\n"
        "Please create a local 'config.py' by copying 'config.py' and "
        "adjusting the paths for your system.\n"
    )
    raise SystemExit(1)
import os, sys
import pyqtgraph as pg
import numpy as np
import heka_reader
from pyqtgraph.Qt import QtWidgets, QtCore
import pandas as pd
import json
from scipy.signal import savgol_filter

pg.setConfigOption('background', 'w')  # white background
pg.setConfigOption('foreground', 'k')  # black labels

V_to_mV = 1e3
F_to_pF = 1e12

sg_polyorder = 3

analysis_points = {}

app = pg.mkQApp()

# Configure Qt GUI:

# Main window + splitters to let user resize panes
win = QtWidgets.QWidget()
layout = QtWidgets.QGridLayout()
win.setLayout(layout)
hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
layout.addWidget(hsplit, 0, 0)
vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
hsplit.addWidget(vsplit)
w1 = QtWidgets.QWidget()
w1l = QtWidgets.QGridLayout()
w1.setLayout(w1l)
vsplit.addWidget(w1)

# Button for loading .dat file
load_btn = QtWidgets.QPushButton("Load...")
w1l.addWidget(load_btn, 0, 0)

# Tree for displaying .pul structure
tree = QtWidgets.QTreeWidget()
tree.setHeaderLabels(['Node', 'Label'])
tree.setColumnWidth(0, 200)
w1l.addWidget(tree, 1, 0)

# Tree for displaying metadata for selected node
data_tree = pg.DataTreeWidget()
vsplit.addWidget(data_tree)

# Create plot widget
plot_widget = pg.GraphicsLayoutWidget()

# Create voltage plot (always present)
voltage_plot = plot_widget.addPlot(row=0, col=0, title="Trace")
voltage_plot.addLegend()
voltage_plot.showGrid(x=True, y=True)

# Initialize derivative plots as None (will be created dynamically)
first_deriv_plot = None #change name later bc its not the deriv anymore! (but keep now for simplicity)
second_deriv_plot = None #keep the structure, maybe I will need it in some time

hsplit.addWidget(plot_widget)

# Resize and show window
hsplit.setStretchFactor(0, 400)
hsplit.setStretchFactor(1, 600)
win.resize(1200, 800)
win.show()


def setup_plots_with_derivatives():
    """
    CM_eval layout (changed from CC_eval:
    Top  = raw capacitance
    Bottom = baseline-subtracted + fits
    """
    global voltage_plot, first_deriv_plot, second_deriv_plot, plot_widget

    # Clear existing layout
    plot_widget.clear()

    # Raw trace (top)
    voltage_plot = plot_widget.addPlot(
        row=0, col=0, title="Raw Capacitance"
    )
    voltage_plot.addLegend()
    voltage_plot.showGrid(x=True, y=True)

    # Processed trace (bottom)
    first_deriv_plot = plot_widget.addPlot(
        row=1, col=0, title="Processed Capacitance + Fits"
    )
    first_deriv_plot.addLegend()
    first_deriv_plot.showGrid(x=True, y=True)

    second_deriv_plot = plot_widget.addPlot(row=2, col=0, title="2nd Derivative (d²V/dt²)")
    second_deriv_plot.showGrid(x=True, y=True)

    # Set row stretch factors for 60/20/20 split
    plot_widget.ci.layout.setRowStretchFactor(0, 60)
    plot_widget.ci.layout.setRowStretchFactor(1, 20)
    plot_widget.ci.layout.setRowStretchFactor(2, 20)

    # Link x-axes for synchronized scrolling
    first_deriv_plot.setXLink(voltage_plot)
    second_deriv_plot.setXLink(voltage_plot)


def setup_plots_voltage_only():
    """Set up the plot layout with only the voltage plot (100% of space)."""
    global voltage_plot, first_deriv_plot, second_deriv_plot, plot_widget

    # Clear existing layout
    plot_widget.clear()

    # Create only voltage plot using full space
    voltage_plot = plot_widget.addPlot(row=0, col=0, title="Trace")
    voltage_plot.addLegend()
    voltage_plot.showGrid(x=True, y=True)

    # Reset derivative plot references
    first_deriv_plot = None
    second_deriv_plot = None


def load_clicked():
    """Display a popup menu with available .dat files from metadata."""
    global analysis_points  # Declare as global to ensure updates are accessible everywhere
    try:
        # Read metadata file
        metadata_df = pd.read_excel(config.METADATA_FILE)
        file_names = metadata_df['file_name'].tolist()

        # Read analysis points
        try:
            analysis_points_path = os.path.join(config.IMPORT_FOLDER, "analysis_points.json")
            if os.path.exists(analysis_points_path):
                with open(analysis_points_path, 'r') as f:
                    analysis_points = json.load(f)
                    # print(f"Loaded analysis points: {analysis_points}")  # Debug check
            else:
                print("analysis_points.json not found")  # Debug output
                analysis_points = {}
        except Exception as e:
            print(f"Error loading analysis points: {e}")
            analysis_points = {}  # Ensure default initialization

        # Create popup menu
        menu = QtWidgets.QMenu()
        for fname in file_names:
            action = menu.addAction(fname)

        # Show menu at button position
        selected_action = menu.exec_(load_btn.mapToGlobal(QtCore.QPoint(0, load_btn.height())))

        if selected_action:
            selected_file = selected_action.text()
            dat_path = os.path.join(config.EXTERNAL_DATA_FOLDER, selected_file)
            if os.path.exists(dat_path):
                load(dat_path)
            else:
                QtWidgets.QMessageBox.warning(
                    win,
                    "File Not Found",
                    f"The file {selected_file} was not found in the import folder."
                )
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            win,
            "Error",
            f"Error loading metadata: {str(e)}"
        )


load_btn.clicked.connect(load_clicked)


def load(file_name):
    """Load a new .dat file into the browser."""
    global bundle, tree_items
    bundle = heka_reader.Bundle(file_name)

    # Clear and update tree
    tree.clear()
    update_tree(tree.invisibleRootItem(), [])
    replot()


def update_tree(root_item, index):
    """Recursively build tree structure."""
    global bundle
    root = bundle.pul
    node = root
    for i in index:
        node = node[i]
    node_type = node.__class__.__name__
    if node_type.endswith('Record'):
        node_type = node_type[:-6]
    try:
        node_type += str(getattr(node, node_type + 'Count'))
    except AttributeError:
        pass
    try:
        node_label = node.Label
    except AttributeError:
        node_label = ''
    item = QtWidgets.QTreeWidgetItem([node_type, node_label])
    root_item.addChild(item)
    item.node = node
    item.index = index
    if len(index) < 2:
        item.setExpanded(True)
    for i in range(len(node.children)):
        update_tree(item, index + [i])


def replot():
    """Update plot and data tree when user selects a trace."""
    global voltage_plot, first_deriv_plot, second_deriv_plot

    # Clear data tree
    data_tree.clear()

    selected = tree.selectedItems()
    if len(selected) < 1:
        # If no selection, set up voltage-only layout
        setup_plots_voltage_only()
        return

    sel = selected[0]
    fields = sel.node.get_fields()
    data_tree.setData(fields)

    for sel in selected:
        index = sel.index
        if len(index) < 4:
            # If not a trace level, set up voltage-only layout
            setup_plots_voltage_only()
            return

        # These are integers from the tree selection
        group_id = index[0]  # e.g., 0 (integer)
        series_id = index[1]  # e.g., 1 (integer)
        sweep_id = index[2]  # e.g., 2 (integer)
        trace_id = index[3]  # e.g., 0 (integer)

        # Check if we have analysis points for this file and indices
        # Convert numeric indices to strings to match JSON structure
        group_key = str(group_id)
        series_key = str(series_id)
        sweep_key = str(sweep_id)
        trace_key = str(trace_id)

        trace = sel.node
        data = bundle.data[index]
        time = np.linspace(trace.XStart, trace.XStart + trace.XInterval * (len(data) - 1), len(data))

        # Get the file name from the bundle
        file_name = os.path.basename(bundle.file_name)

        # Check if we have analysis points for this file and indices
        has_analysis_points = (file_name in analysis_points and
                               group_key in analysis_points[file_name] and
                               series_key in analysis_points[file_name][group_key] and
                               sweep_key in analysis_points[file_name][group_key][series_key] and
                               trace_key in analysis_points[file_name][group_key][series_key][sweep_key])

        if has_analysis_points:
            # Set up layout with derivatives
            setup_plots_with_derivatives()

            # Set labels for all plots
            voltage_plot.setLabel('bottom', trace.XUnit)
            voltage_plot.setLabel('left', trace.Label, units=trace.YUnit)
            first_deriv_plot.setLabel('bottom', trace.XUnit)
            first_deriv_plot.setLabel('left', 'dV/dt (V/s)')
            second_deriv_plot.setLabel('bottom', trace.XUnit)
            second_deriv_plot.setLabel('left', 'd²V/dt² (V/s²)')

            # Plot the main voltage trace
            voltage_plot.plot(time, data, pen='m', name='Voltage')

            trace_data = analysis_points[file_name][group_key][series_key][sweep_key][trace_key]

            # Get the filter_cut_off from analysis_points, with fallback
            smooth_window = trace_data.get('smooth_window', 0.01)

            # use Savitzky-Golay filter for smoothing
            dt = time[1] - time[0]
            sg_window_s = smooth_window / 1000.0
            win_samples = int(round(sg_window_s / dt))
            # make odd and at least polyorder+2
            if win_samples <= sg_polyorder + 1:
                win_samples = sg_polyorder + 3
            if win_samples % 2 == 0:
                win_samples += 1

            # smooth voltages and use numerical derivatives (central differences via np.gradient)
            #NOTE difference with main.py. Here voltage is in V (to show unmodified traces) and dV/dt is in V/s (also used as thershold value) but d2V/dt2 is in V/ms^2 (to remove 1e6 in the number on the axes)
            voltage_filt = savgol_filter(data, window_length=win_samples, polyorder=sg_polyorder)
            d1 = np.gradient(voltage_filt, dt)
            d1_in_V_per_s = savgol_filter(d1, window_length=win_samples, polyorder=sg_polyorder)
            d2 = np.gradient(d1_in_V_per_s, dt)
            d2 = d2 / V_to_mV
            d2 = d2 / V_to_mV
            #the 1st is V/s and 2nd is V/ms^2
            d2_in_V_per_s_s = savgol_filter(d2, window_length=win_samples,
                                    polyorder=sg_polyorder)

            # Plot derivatives
            first_deriv_plot.plot(time, d1, pen='blue', name='1st Derivative (raw)')
            first_deriv_plot.plot(time, d1_in_V_per_s, pen='k', name='1st Derivative (filtered)')
            second_deriv_plot.plot(time, d2, pen='blue', name='2nd Derivative (raw)')
            second_deriv_plot.plot(time, d2_in_V_per_s_s, pen='k', name='2nd Derivative (filtered)')

            points = trace_data
            # Plot each type of point with different symbols and colors
            voltage_symbols = {
                'threshold': ('o', 'red', 'Threshold'),
                'threshold_2nd': ('o', 'brown', 'Threshold_2nd'),
                'half_duration_start': ('s', 'blue', 'Half Duration Start'),
                'half_duration_end': ('s', 'green', 'Half Duration End'),
                'peak': ('t', 'yellow', 'Peak'),
                'ahp': ('d', 'purple', 'AHP'),
                'dvdt_max': ('p', 'cyan', 'Max dV/dt')
            }

            d1_symbols = {
                'threshold_v1': ('o', 'red', 'Threshold')
            }

            d2_symbols = {
                'd2_threshold': ('o', 'brown', 'Threshold_2nd'),
                'd2_peak1': ('t', 'purple', 'Peak1 (Left)'),
                'd2_peak2': ('d', 'yellow', 'Peak2 (Right)')
            }

            # Plot points on voltage plot
            for point_type, (symbol, color, label) in voltage_symbols.items():
                if points.get(point_type):  # If we have points of this type
                    t_points, v_points = zip(*points[point_type])
                    scatter = pg.ScatterPlotItem(
                        x=t_points,
                        y=v_points,
                        symbol=symbol,
                        size=10,
                        pen=pg.mkPen(None),
                        brush=pg.mkBrush(color),
                        name=label
                    )
                    voltage_plot.addItem(scatter)

            # Plot points on first derivative plot
            for point_type, (symbol, color, label) in d1_symbols.items():
                if points.get(point_type):
                    t_points, v_points = zip(*points[point_type])
                    scatter_deriv = pg.ScatterPlotItem(
                        x=t_points,
                        y=v_points,
                        symbol=symbol,
                        size=10,
                        pen=pg.mkPen(None),
                        brush=pg.mkBrush(color),
                        name=label
                    )
                    first_deriv_plot.addItem(scatter_deriv)

            # Plot points on 2nd derivative plot
            for point_type, (symbol, color, label) in d2_symbols.items():
                if points.get(point_type):
                    t_points, v_points = zip(*points[point_type])
                    scatter_deriv = pg.ScatterPlotItem(
                        x=t_points,
                        y=v_points,
                        symbol=symbol,
                        size=10,
                        pen=pg.mkPen(None),
                        brush=pg.mkBrush(color),
                        name=label
                    )
                    second_deriv_plot.addItem(scatter_deriv)

        else:
            # Set up layout with voltage only (no analysis points available)
            setup_plots_voltage_only()

            # Set labels for voltage plot only
            voltage_plot.setLabel('bottom', trace.XUnit)
            voltage_plot.setLabel('left', trace.Label, units=trace.YUnit)

            # Plot the main voltage trace
            voltage_plot.plot(time, data, pen='k', name='Trace')


tree.itemSelectionChanged.connect(replot)

if __name__ == '__main__':
    if sys.flags.interactive == 0:
        app.exec_()