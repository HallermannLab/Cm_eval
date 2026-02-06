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

pg.setConfigOption('background', 'w')  # white background
pg.setConfigOption('foreground', 'k')  # black labels

V_to_mV = 1e3
F_to_pF = 1e12
A_to_pA = 1e12

sg_polyorder = 3

analysis_points = {}

cursor_a = None
cursor_b = None
cursor_a_line = None
cursor_b_line = None
cursor_text = None
last_x = None
last_y = None

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

    # No third plot
    second_deriv_plot = None

    # Layout ratio
    plot_widget.ci.layout.setRowStretchFactor(0, 50)
    plot_widget.ci.layout.setRowStretchFactor(1, 50)

    # Link x-axes for synchronized scrolling
    first_deriv_plot.setXLink(voltage_plot)

def setup_plots_with_calcium():
    """
    Current trace layout with leak subtraction (for Imon-1, trace_id=0):
    Top    = raw current trace
    Bottom = calcium leak subtraction (APS raw, Leak P/4, Ca subtracted)
    """
    global voltage_plot, first_deriv_plot, second_deriv_plot, plot_widget
    plot_widget.clear()

    # Raw current trace (top)
    voltage_plot = plot_widget.addPlot(
        row=0, col=0, title="Current Trace (Imon-1)"
    )
    voltage_plot.addLegend()
    voltage_plot.showGrid(x=True, y=True)

    # Calcium leak subtraction (bottom)
    first_deriv_plot = plot_widget.addPlot(
        row=1, col=0, title="Calcium Current (Leak Subtraction)"
    )
    first_deriv_plot.addLegend()
    first_deriv_plot.showGrid(x=True, y=True)

    # No third plot
    second_deriv_plot = None

    # Layout ratio
    plot_widget.ci.layout.setRowStretchFactor(0, 50)
    plot_widget.ci.layout.setRowStretchFactor(1, 50)

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

def update_cursor_text():
    global cursor_text
    global cursor_a, cursor_b

    if cursor_a is None or cursor_b is None:
        return

    xA, yA = cursor_a.getData()
    xB, yB = cursor_b.getData()

    if len(xA) == 0 or len(xB) == 0:
        return

    xA = xA[0]
    yA = yA[0]
    xB = xB[0]
    yB = yB[0]

    txt = (
        f"A: {xA:.3f} ms, {yA:.2f} pA\n"
        f"B: {xB:.3f} ms, {yB:.2f} pA\n"
        f"Δt: {abs(xB - xA):.3f} ms\n"
        f"ΔI: {abs(yB - yA):.2f} pA"
    )

    cursor_text.setText(txt)

def replot():
    """Update plot and data tree when user selects a trace."""
    global voltage_plot, first_deriv_plot, second_deriv_plot
    global cursor_a, cursor_b, cursor_text

    cursor_a = None
    cursor_b = None
    cursor_text = None

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
        trace_id = index[3]  # e.g., 0 for Imon-1, 2 for Cm

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

        # ========================================================================
        # CASE 1: Imon-1 trace (trace_id = 0) with calcium leak subtraction data
        # ========================================================================
        if trace_id == 0:
            # Check if we have calcium leak subtraction data for this trace
            has_calcium_data = (file_name in analysis_points and
                                group_key in analysis_points[file_name] and
                                series_key in analysis_points[file_name][group_key] and
                                sweep_key in analysis_points[file_name][group_key][series_key] and
                                trace_key in analysis_points[file_name][group_key][series_key][sweep_key] and
                                "calcium_leak" in analysis_points[file_name][group_key][series_key][sweep_key][
                                    trace_key])

            if has_calcium_data:
                # Set up layout with calcium leak subtraction (2 panels)
                setup_plots_with_calcium()

                # Set labels
                voltage_plot.setLabel('bottom', trace.XUnit)
                voltage_plot.setLabel('left', trace.Label, units=trace.YUnit)
                first_deriv_plot.setLabel('bottom', 'Time', units='ms')
                first_deriv_plot.setLabel('left', 'Current', units='pA')

                # Top plot: Raw current trace (Imon-1)
                voltage_plot.plot(
                    time,
                    A_to_pA * data,  # Convert to pA
                    pen='k',
                    name='Imon-1'
                )

                # Bottom plot: Calcium leak subtraction
                first_deriv_plot.clear()

                calcium_data = analysis_points[file_name][group_key][series_key][sweep_key][trace_key]["calcium_leak"]

                # Convert time to ms for better readability
                t_ms = np.array(calcium_data["time"]) * 1e3

                # Plot raw aps current
                if "raw" in calcium_data and len(calcium_data["raw"]) > 0:
                    first_deriv_plot.plot(
                        t_ms,
                        calcium_data["raw"],
                        pen=pg.mkPen('gray', width=2, style=QtCore.Qt.DashLine),
                        name="aps raw"
                    )

                # Plot leak current
                if "leak" in calcium_data and len(calcium_data["leak"]) > 0:
                    first_deriv_plot.plot(
                        t_ms,
                        calcium_data["leak"],
                        pen=pg.mkPen('b', width=1.5),
                        name="Leak (P/4)"
                    )

                # Plot calcium current (subtracted)
                if "ca" in calcium_data and len(calcium_data["ca"]) > 0:
                    first_deriv_plot.plot(
                        t_ms,
                        calcium_data["ca"],
                        pen=pg.mkPen('m', width=2.5),
                        name="Ca (subtracted)"
                    )

                # Add RMS info if available
                if "rms" in calcium_data:
                    rms_text = f"RMS: {calcium_data['rms']:.2f} pA"
                    text_item = pg.TextItem(rms_text, anchor=(0, 1), color='k')
                    # Position at top left of plot
                    y_max = max(calcium_data["ca"]) if len(calcium_data["ca"]) > 0 else 0
                    text_item.setPos(t_ms[0], y_max * 0.95)
                    first_deriv_plot.addItem(text_item)

            else:
                # No calcium data - just show the raw current trace
                setup_plots_voltage_only()
                voltage_plot.setLabel('bottom', trace.XUnit)
                voltage_plot.setLabel('left', trace.Label, units=trace.YUnit)
                voltage_plot.plot(time, A_to_pA * data, pen='k', name='Imon-1')

        # ========================================================================
        # CASE 2: Cm trace (trace_id = 2) with capacitance analysis
        # ========================================================================
        elif trace_id == 2:
            # Check if we have analysis points for Cm trace
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
            first_deriv_plot.setLabel('left', 'Capacitance', units='pF')

            # Top plot: Raw Cm data (pF)
            voltage_plot.plot(
                time,
                data,
                pen='m',
                name='Raw Cm'
            )

            # Lower plot: Processed Cm data (pF)
            proc = analysis_points[file_name][group_key][series_key][sweep_key][trace_key]

            # Load arrays
            time_rel = np.array(proc["time_rel"])
            cm_bs = np.array(proc["cm_bs"])

            # Clear bottom plot
            first_deriv_plot.clear()

            # Baseline-subtracted trace
            first_deriv_plot.plot(
                time_rel,
                cm_bs,
                pen=pg.mkPen('k', width=2),
                name="Capacitance (processed)"
            )

            # 1-exp fit
            if "cm_1exp" in proc and len(proc["cm_1exp"]) > 2:
                x, y = zip(*proc["cm_1exp"])
                first_deriv_plot.plot(
                    x, y,
                    pen=pg.mkPen('r', width=2),
                    name="1-exp"
                )

            # 1-expY fit
            if "cm_1expY" in proc and len(proc["cm_1expY"]) > 2:
                x, y = zip(*proc["cm_1expY"])
                first_deriv_plot.plot(
                    x, y,
                    pen=pg.mkPen('g', width=2),
                    name="1-expY"
                )

            # 2-exp fit
            if "cm_2exp" in proc and len(proc["cm_2exp"]) > 2:
                x, y = zip(*proc["cm_2exp"])
                first_deriv_plot.plot(
                    x, y,
                    pen=pg.mkPen('m', width=2),
                    name="2-exp"
                )


        else:
            # Set up layout with voltage only (no analysis points available)
            setup_plots_voltage_only()

            # Set labels for voltage plot only
            voltage_plot.setLabel('bottom', trace.XUnit)
            voltage_plot.setLabel('left', 'Capacitance', units='pF')

        # ========================================================================
        # CASE 3: Other traces (e.g., trace_id = 1 for voltage)
        # ========================================================================
        else:
            # Just show the raw trace
            setup_plots_voltage_only()
            voltage_plot.setLabel('bottom', trace.XUnit)
            voltage_plot.setLabel('left', trace.Label, units=trace.YUnit)
            voltage_plot.plot(time, data, pen='k', name='Trace')


tree.itemSelectionChanged.connect(replot)

if __name__ == '__main__':
    if sys.flags.interactive == 0:
        app.exec_()