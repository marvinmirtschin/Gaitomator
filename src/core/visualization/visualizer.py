import os
import time
import warnings

# import ipywidgets as ipyw
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# matplotlib.use("agg")
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
# needed for 3d plots although it is marked as unused: from mpl_toolkits.mplot3d import Axes3D
# from IPython.display import display
from matplotlib import cm
from plotly import express as px, subplots
from scipy import signal

from src.core import constants
from src.core.constants import DATA_FRAMES_KEY
from src.core.utility import file_handling


def plot_data_dimensions(data_list, labels=None, title="", predefined_colors=None, save_figure=False):
    """creates a figure with a large 3D chart and 3 1D charts"""
    # create figure
    fig, (fig_xs, fig_ys, fig_zs) = plt.subplots(3, 1, sharex=True, figsize=(15, 5))

    # set figure properties
    fig.suptitle(title, fontsize=12)

    # create 3D chart
    fig_xyz = fig.add_subplot(121, projection='3d')

    if labels is None:
        labels = []

    # setup figure grid
    gs = gridspec.GridSpec(3, 2)

    # plot all data
    for index in range(len(data_list)):
        data = data_list[index]

        if len(labels) > index:
            label = labels[index]
        else:
            label = str(index)

        # color = plt.cm.RdYlBu(index)
        if predefined_colors is not None:
            color = predefined_colors[label]
        else:
            color = get_rgba_color(number_of_distinct_colors=len(data_list), index=index)

        # plot classified data into 3D chart
        fig_xyz.scatter(data.dataList_values_0,
                        data.dataList_values_1,
                        data.dataList_values_2,
                        alpha=0.2, marker="o", color=color, label=label)

        setup_sub_plot(gs=gs, fig=fig, sub_plot=fig_xs, data=data, index=index, dimension=0, size=len(data_list),
                       predefined_colors=predefined_colors, user_label=label)
        setup_sub_plot(gs=gs, fig=fig, sub_plot=fig_ys, data=data, index=index, dimension=1, size=len(data_list),
                       predefined_colors=predefined_colors, user_label=label)
        setup_sub_plot(gs=gs, fig=fig, sub_plot=fig_zs, data=data, index=index, dimension=2, size=len(data_list),
                       predefined_colors=predefined_colors, user_label=label)

    fig_xyz.legend(loc=2, prop={'size': 10})

    if save_figure:
        if not os.path.exists(file_handling.get_current_working_directory() + "/" + "figures"):
            os.makedirs(file_handling.get_current_working_directory() + "/" + "figures")
        fig.savefig('figures/full_figure_{}.png'.format(round(time.time() * 1000)))

    return fig


# helper for the 1D charts
def setup_sub_plot(gs, fig, sub_plot, data, index, dimension, size, predefined_colors=None, user_label=None):
    x = data["dataList_values_" + str(dimension)]
    y = np.zeros_like(x) + index

    if predefined_colors is not None:
        color = predefined_colors[user_label]
    else:
        color = get_rgba_color(number_of_distinct_colors=size, index=index)

    scatter = sub_plot.scatter(x, y, alpha=0.2, marker="o", color=color)
    sub_plot.set_ylim([-10, 10])
    sub_plot.set_position(gs[dimension, 1].get_position(fig))
    sub_plot.yaxis.set_visible(False)
    return scatter


def reduce_features(importances, feature_names, threshold):
    deletion_indices = []
    for index in range(len(importances)):
        if importances[index] < threshold:
            deletion_indices.append(index)
    importances = np.delete(importances, deletion_indices)
    feature_names = np.delete(feature_names, deletion_indices)
    return importances, feature_names


def plot_feature_importances_for_model(model, feature_names, threshold=None, return_fig=False):
    # create data copies
    importances = model.feature_importances_
    plot_feature_importances(importances=importances, feature_names=feature_names.copy(), threshold=threshold,
                             return_fig=return_fig)


def plot_feature_importances(importances, feature_names, threshold=None, return_fig=False):
    # if threshold is specified remove features with importance under threshold
    if threshold is not None:
        importances, feature_names = reduce_features(importances=importances, feature_names=feature_names,
                                                     threshold=threshold)

    importances, feature_names = __sort_lists_equally(sort_list=importances,
                                                      reference_list=feature_names)

    # create plot
    n_features = len(feature_names)
    plt.barh(range(n_features), importances, height=0.8, align='center')

    # set plot size
    fig = plt.gcf()
    height_minimum = max(0.25 * n_features, 4)
    fig.set_size_inches(9, height_minimum, forward=True)

    max_importance = max(importances)

    # adjust axis
    ax = plt.gca()
    ax.tick_params(axis='y', which='major', pad=12)
    ax.autoscale(tight=True)
    ax.set_xlim([0, max_importance])

    # label plot
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    if return_fig:
        return fig


def show_progress(value=0, progress_widget=None, description=""):
    """
    if progress_widget is None:
        if description == "":
            progress_widget = ipyw.widgets.FloatProgress(value=value, min=0, max=1.0, step=0.01,
                                                         orientation='horizontal', bar_style='info')
        else:
            progress_widget = ipyw.widgets.FloatProgress(value=value, min=0, max=1.0, step=0.01,
                                                         orientation='horizontal', bar_style='info',
                                                         description=description)
        display(progress_widget)
    """
    progress_widget.value = value
    return progress_widget


def show_label(value="", label_widget=None):
    """Called from notebooks"""
    """
    if label_widget is None:
        label_widget = ipyw.widgets.Label(value=value)
        display(label_widget)
    """
    label_widget.value = value
    return label_widget


def get_directory_for_html_plots():
    return os.path.join(file_handling.get_python_directory(), "reports", "html/")


def get_visualization_data(visualization_types, device_id, values, value_counts, number_of_items, index):
    data_list = []
    if constants.VIS_TYPE_BAR in visualization_types:
        trace = get_bar_visualization_data(device_id=device_id, values=values, value_counts=value_counts,
                                           number_of_items=number_of_items, index=index)
        data_list.append(trace)

    if constants.VIS_TYPE_LINE in visualization_types:
        trace = get_line_visualization_data(device_id=device_id, values=values, value_counts=value_counts,
                                            number_of_items=number_of_items, index=index)
        data_list.append(trace)
    return data_list


def get_bar_visualization_data(device_id, values, value_counts, number_of_items, index):
    name = 'Device {id}'.format(id=device_id) + ' with a Value Count of {values}'.format(values=sum(value_counts))
    legend_group = 'group{index}'.format(index=index)
    color = get_hex_color(number_of_distinct_colors=number_of_items, index=index)
    marker = dict(color=color)
    trace = go.Bar(y=values, name=name, legendgroup=legend_group, opacity=.2, marker=marker)
    return trace


def get_line_visualization_data(device_id, values, value_counts, number_of_items, index):
    # window length has to be uneven
    if len(values) % 2 == 0:
        if len(values) > 1:
            window_length = len(values) - 1
        else:
            window_length = 1
    else:
        window_length = len(values)

    # Savgol filter for window length
    filter_ = signal.savgol_filter(values, window_length=window_length, polyorder=min(10, window_length - 1))
    name = 'Device {id}'.format(id=device_id) + ' with a Value Count of {values}'.format(values=sum(value_counts))
    legend_group = 'group{index}'.format(index=index)

    color = get_hex_color(number_of_distinct_colors=number_of_items, index=index)
    line = dict(color=color)

    trace = go.Scatter(y=filter_, name=name, mode='lines', legendgroup=legend_group, line=line)
    return trace


def get_layout_for_feature_plot(feature_type, dimension, depth, use_absolute_values=False):
    title = "Feature {feature_type} for Dimension {dimension} and Depth {depth}" \
        .format(feature_type=feature_type, dimension=dimension, depth=depth)
    x_axis = dict(title='Bin')
    if use_absolute_values:
        y_axis = dict(title='Absolute Counts')
    else:
        y_axis = dict(title='Count (%)')

    layout = go.Layout(title=title, barmode='group', xaxis=x_axis, yaxis=y_axis)
    return layout


def create_file_name_for_feature_plot(feature_type, dimension, depth):
    """Create file name for plotly visualizations."""
    directory = get_directory_for_html_plots()
    file_name = 'feature-{feature_type}-dimension-{dimension}-depth-{depth}.html' \
        .format(feature_type=feature_type, dimension=dimension, depth=depth)
    return directory + file_name


def create_plotly(data_list, layout, file_name):
    """Create visualization of data with plotly."""
    fig = go.Figure(data=data_list, layout=layout)
    plotly.offline.plot(fig, filename=file_name)


# check for available color maps here: <a href="https://matplotlib.org/stable/tutorials/colors/colormaps.html">Color Maps</a>
def get_rgba_color(number_of_distinct_colors, index):
    color_map = cm.get_cmap("Spectral")
    return color_map(index / (number_of_distinct_colors - 1))


def get_hex_color(number_of_distinct_colors, index):
    r, g, b, a = get_rgba_color(number_of_distinct_colors=number_of_distinct_colors, index=index)
    r = int(255 * r)
    g = int(255 * g)
    b = int(255 * b)
    return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)


def visualize_3d_clusters(data, y, feature_type):
    if data.shape[1] < 3:  # data should have three dimensions
        warnings.warn("Data has less than three dimension. No 3D visualization is plotted.")
    else:
        labels = np.unique(y)
        data_list = list()

        for i, label in enumerate(labels):
            selected_data = data[y == label]
            # create visualization data
            trace = get_trace(selected_data, label)
            data_list.append(trace)

        # get layout
        title = "Visualization_Feature_Vector_{feature_type}".format(feature_type=feature_type)
        layout = dict(title=title)

        fig = go.Figure(data=data_list, layout=layout)
        plotly.offline.plot(fig, filename=os.path.join(file_handling.get_or_create_html_visualizations_directory(),
                                                       title + "_{time}.html".format(time=round(time.time()))))


def get_trace(data, label):
    if data.shape[1] > 3:
        warnings.warn("Data is just visualized for first three feature dimensions")

    return go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        name=label,
        marker=dict(
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8
        )
    )


"""Visualizations for Classification matrices"""


def visualize_classification_matrix(pd_dataframe, title):
    # Set table styles - CSS properties
    styles = [
        dict(selector="th", props=[  # table header
            ('font-size', '11px'),
            ('text-align', 'center'),
            ('font-weight', 'bold')]),
        dict(selector="caption", props=[  # caption
            ('font-size', '12px'),
            ('color', 'white'),
            ('font-weight', 'bold')]),
    ]

    s = pd_dataframe.style.applymap(color_classification) \
        .set_caption(title) \
        .set_precision(2) \
        .set_table_styles(styles)

    # display(s)


def color_classification(value):
    """
    Colors elements in a data frame
    green if positive and red if
    negative. Does not color NaN
    values.
    """

    if value > 0.66:
        color = 'red'
    elif value < 0.33:
        color = 'green'
    else:
        color = 'yellow'

    return 'color: %s' % color


"""Visualization for walking cycles"""


def visualize_concatenated_cycles_from_list(cycle_list, title, sensor, subtitle="", label_name="step_cycle", label_list=None,
                                            reset_index=True, trace_mode='lines+markers'):
    try:
        number_of_columns = cycle_list[0].shape[1]
    except KeyError:
        # instability data frame
        cycle_list = cycle_list[DATA_FRAMES_KEY].values
        number_of_columns = cycle_list[0].shape[1]

    subplot_titles = ["{sensor} {column}".format(sensor=sensor, column=str(column)) for column in
                      range(0, cycle_list[0].shape[1])]
    fig = subplots.make_subplots(rows=number_of_columns, cols=1,
                                 subplot_titles=subplot_titles)

    for dimension in range(0, number_of_columns):
        number_of_cycles = len(cycle_list)

        offset = 0
        color_picker = dimension
        for i in range(0, number_of_cycles):
            cycle_df = cycle_list[i]
            if not isinstance(cycle_df, pd.DataFrame):
                continue
            if reset_index:
                cycle_df = cycle_df.reset_index()
                cycle_df.drop("index", axis=1, inplace=True)

            if label_list is not None:
                if "points" in label_list[i]:
                    trace_mode = 'markers'
                else:
                    trace_mode = 'lines+markers'

            trace = go.Scatter(
                x=np.array(cycle_df.index.tolist()) + offset,
                y=cycle_df.iloc[:, dimension],
                mode=trace_mode,
                name=cycle_df.columns[dimension] if i is 0 else None,
                legendgroup=dimension,
                line={'color': get_hex_color(number_of_distinct_colors=2 * number_of_columns, index=color_picker)},
                showlegend=i == 0
            )
            offset += len(cycle_df)
            color_picker = (color_picker + number_of_columns) % (2 * number_of_columns)

            # last two parameters are for grid visualization (dimensions x 1)
            fig.append_trace(trace, dimension + 1, 1)
    fig['layout'].update(title=str(title + "<br>{subtitle}".format(subtitle=subtitle)))
    file_name = os.path.join(file_handling.get_or_create_html_visualizations_directory(),
                             title.replace(" ", "_") + "_{time}.html".format(time=round(time.time())))
    plotly.offline.plot(fig, filename=file_name)


def visualize_single_cycles_from_list(cycle_list, title, sensor, subtitle="", label_name="step_cycle", label_list=None,
                                      reset_index=True, trace_mode='lines+markers'):
    try:
        number_of_columns = cycle_list[0].shape[1]
    except KeyError:
        # instability data frame
        cycle_list = cycle_list[DATA_FRAMES_KEY].values
        number_of_columns = cycle_list[0].shape[1]

    subplot_titles = ["{sensor} {column}".format(sensor=sensor, column=str(column)) for column in
                      range(0, cycle_list[0].shape[1])]
    fig = subplots.make_subplots(rows=number_of_columns, cols=1,
                                 subplot_titles=subplot_titles)

    for dimension in range(0, number_of_columns):
        number_of_cycles = len(cycle_list)

        for i in range(0, number_of_cycles):
            cycle_df = cycle_list[i]
            if not isinstance(cycle_df, pd.DataFrame):
                continue
            if reset_index:
                cycle_df = cycle_df.reset_index()
                cycle_df.drop("index", axis=1, inplace=True)

            if label_list is not None:
                if "points" in label_list[i]:
                    trace_mode = 'markers'
                else:
                    trace_mode = 'lines+markers'

            trace = go.Scatter(
                x=cycle_df.index.tolist(),
                y=cycle_df.iloc[:, dimension],
                mode=trace_mode,
                name='{label_name}_{no}'.format(label_name=label_name, no=i if label_list is None else label_list[i]),
                legendgroup=i,
                line={'color': get_hex_color(number_of_distinct_colors=number_of_cycles, index=i)},
                showlegend=False if dimension > 0 else True
            )

            # last two parameters are for grid visualization (dimensions x 1)
            fig.append_trace(trace, dimension + 1, 1)
    fig['layout'].update(
        title=str(title + "<br>{subtitle}".format(subtitle=subtitle)))
    file_name = os.path.join(file_handling.get_or_create_html_visualizations_directory(),
                             title.replace(" ", "_") + "_{time}.html".format(time=round(time.time())))
    plotly.offline.plot(fig, filename=file_name)


def visualize_mean_and_bounded_cycles(cycle_dict, title, sensor):
    number_of_columns = cycle_dict["mean"].shape[1]
    fig = subplots.make_subplots(rows=number_of_columns, cols=1,
                                 subplot_titles=("{} x".format(sensor), "{} y".format(sensor),
                                                 "{} z".format(sensor), "{} magnitude".format(sensor)))

    for dimension in range(0, number_of_columns):
        upper_bound = go.Scatter(
            name='Upper Bound',
            legendgroup=1,
            x=cycle_dict["upper_bound"].index.tolist(),
            y=cycle_dict["upper_bound"].iloc[:, dimension],
            mode='lines',
            line=dict(color='lightblue'),
            fill=None)

        trace = go.Scatter(
            name='Measurement',
            legendgroup=2,
            x=cycle_dict["mean"].index.tolist(),
            y=cycle_dict["mean"].iloc[:, dimension],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            fill='tonexty')

        lower_bound = go.Scatter(
            name='Lower Bound',
            legendgroup=3,
            x=cycle_dict["lower_bound"].index.tolist(),
            y=cycle_dict["lower_bound"].iloc[:, dimension],
            fill='tonexty',
            line=dict(color='lightblue'),
            mode='lines')

        fig.append_trace(upper_bound, dimension + 1, 1)
        fig.append_trace(trace, dimension + 1, 1)
        fig.append_trace(lower_bound, dimension + 1, 1)

    fig['layout'].update(
        title=title)

    file_name = os.path.join(file_handling.get_or_create_html_visualizations_directory(),
                             title.replace(" ", "_") + "_{time}.html".format(time=round(time.time())))
    plotly.offline.plot(fig, filename=file_name)


def visualize_cycles_as_boxplots(cycle_list, title, sensor):
    fig = subplots.make_subplots(rows=5, cols=1,
                                 subplot_titles=("{} x".format(sensor), "{} y".format(sensor),
                                                 "{} z".format(sensor), "{} magnitude".format(sensor),
                                                 "number of cycles per user"))

    fig = get_trace_list_of_cycles_as_boxplots(cycle_list, fig)[0]
    fig['layout'].update(
        title=title)

    file_name = os.path.join(file_handling.get_or_create_html_visualizations_directory(),
                             title.replace(" ", "_") + "_{time}.html".format(time=round(time.time())))
    plotly.offline.plot(fig, filename=file_name)


def get_trace_list_of_cycles_as_boxplots(cycle_list, fig, user=None, color=None):
    concatenated_cycles = pd.concat(cycle_list, axis=0)
    grouped = concatenated_cycles.groupby(level=0)

    for dimension in range(0, concatenated_cycles.shape[1]):

        for i in range(0, grouped.ngroups):
            trace = go.Box(
                y=grouped.get_group(i).iloc[:, dimension],
                legendgroup=i if user is None else str(user),
                name='{i}'.format(i=i),  # cycle index
                marker=dict(
                    color='rgb(214, 12, 140)' if color is None else color,
                ),
                showlegend=False if any([i > 0, dimension > 0]) else True
            )
            fig.append_trace(trace, dimension + 1, 1)

    # add a bar chart for number of cycles
    if user is not None:
        fig.append_trace(
            go.Bar(
                x=["user_{}".format(user), ],
                y=[len(cycle_list), ],
                text=[len(cycle_list), ],
                textposition='auto',
                name="user_{user}".format(user=user),
                marker=dict(
                    color='rgb(214, 12, 140)' if color is None else color,
                ),
            ), 5, 1)

    return fig, concatenated_cycles


def visualize_histograms(histogram_list, feature_type="", dimension="", depth=""):
    visualization_data_list = []
    for index, histogram in enumerate(histogram_list):
        visualization_data = get_visualization_data(visualization_types=("bar", "list"),
                                                    device_id=index, values=histogram,
                                                    value_counts=histogram,
                                                    number_of_items=len(histogram_list),
                                                    index=index)
        visualization_data_list += visualization_data
    layout = get_layout_for_feature_plot(feature_type=feature_type, dimension=dimension,
                                         depth=depth, use_absolute_values=False)
    create_plotly(data_list=visualization_data_list, layout=layout,
                  file_name="histogram_of_{feature_type}_{dimension}_{depth}".format(feature_type=feature_type,
                                                                                     dimension=dimension, depth=depth))


def get_trace_for_characteristic_curves(x_values_list, y_values_list, characteristic_metric_list,
                                        label_list, legend_name, align_same_labels_to_colors=False):
    trace_list = []
    max_colors = len(label_list) + 1
    if align_same_labels_to_colors:
        unique_labels = np.unique(label_list)
        max_colors = len(unique_labels) + 1
        color_labels = [np.where(unique_labels == label)[0] for label in label_list]

    for i, characteristic_curve in enumerate(label_list):
        color_index = i + 1
        if align_same_labels_to_colors:
            # noinspection PyUnboundLocalVariable
            color_index = int(color_labels[i]) + 1

        trace = go.Scatter(x=x_values_list[i], y=y_values_list[i],
                           mode='lines',
                           line={'color': get_hex_color(number_of_distinct_colors=max_colors, index=color_index)},
                           name=r'$%s \text{ }(%s = %0.2f)$' % (
                               label_list[i], legend_name, characteristic_metric_list[i]))
        trace_list.append(trace)
    return trace_list


def get_trace_for_average_characteristic_curve(thresholds, average_curve, legend_label, number_of_colors=10.):
    return (go.Scatter(x=thresholds, y=average_curve,
                       mode='lines',
                       line={'color': get_hex_color(number_of_distinct_colors=number_of_colors, index=0), 'width': 4},
                       name=legend_label))


def get_line_by_chance_trace():
    # the "guessing" curve
    return (go.Scatter(x=[0, 1], y=[0, 1],
                       mode='lines',
                       line=dict(color='navy', dash='dash'),
                       showlegend=False))


def get_filled_between_traces(thresholds, lower_bound, upper_bound):
    trace_upper = (go.Scatter(x=thresholds, y=upper_bound,
                              fill=None,
                              mode='lines',
                              line={'color': 'rgba(220,160,140,0.1)', 'width': 0.5},
                              showlegend=False,
                              ))
    trace_lower = (go.Scatter(x=thresholds, y=lower_bound,
                              fill='tonexty', fillcolor='rgba(220,160,140,0.1)',
                              mode='lines',
                              line={'color': 'rgba(220,160,140,0.1)', 'width': 0.5},
                              showlegend=False))

    return trace_upper, trace_lower


def visualize_trace_list(trace_list, title, x_axis, y_axis):
    layout = go.Layout(title=title,
                       xaxis=dict(title=x_axis),
                       yaxis=dict(title=y_axis))
    fig = go.Figure(data=trace_list, layout=layout)
    directory = file_handling.get_or_create_html_visualizations_directory()
    os.makedirs(directory, exist_ok=True)
    file_name = title.replace(" ", "_") + "_{time}.html".format(time=round(time.time()))
    file_path = os.path.join(directory, file_name)
    plotly.offline.plot(fig, filename=file_path, include_mathjax="cdn")


def visualize_smoothed_z_score_algorithm_for_cycle_detection(pandas_series, translated_signal, moving_average, moving_standard_deviation):
    label_list = ["original", "peak_signal", "moving_average", "moving_standard_deviation"]
    data_for_visualization_list = [
        pd.DataFrame(pandas_series, index=pandas_series.index),
        pd.DataFrame(translated_signal, index=pandas_series.index),
        pd.DataFrame(moving_average, index=pandas_series.index),
        pd.DataFrame(moving_standard_deviation, index=pandas_series.index),
    ]
    visualize_single_cycles_from_list(data_for_visualization_list,
                                      title="Smoothed-z algorithm", sensor="", label_name="",
                                      label_list=label_list, reset_index=False)


def visualize_eigenstep_projections(eigenstep_projections, title="", label_column="labels", in_3D=False, density=False):
    # to check dimensions
    if eigenstep_projections.shape[1] <= 2:
        warnings.warn("Eigensteps cannot be plotted due to less dimensions of components.")

    if in_3D:
        fig = px.scatter_3d(eigenstep_projections, x=0, y=1, z=2, color=label_column)
    else:
        if density:
            fig = px.density_contour(eigenstep_projections, x=0, y=1, color=label_column, marginal_x="rug", marginal_y="histogram")
        else:
            fig = px.scatter(eigenstep_projections, x=0, y=1, color=label_column, marginal_x="histogram", marginal_y="histogram")

    fig.update_layout(title_text=title)

    directory = file_handling.get_or_create_html_visualizations_directory()
    os.makedirs(directory, exist_ok=True)
    file_name = "exploring_eigenstep_visualization_{}.html".format(round(time.time()))
    file_path = os.path.join(directory, file_name)
    plotly.offline.plot(fig, filename=file_path)


def __sort_lists_equally(sort_list, reference_list, ascending=False):
    """Sort first list in the specified order. The second list will have its values ordered according to the changes in
    the first list.
    """
    if len(sort_list) != len(reference_list):
        return

    result = {}
    list_x = list(sort_list)
    list_y = list(reference_list)
    for i in range(len(sort_list)):
        if ascending:
            minimum_item = min(list_x)
            index = list_x.index(minimum_item)
        else:
            maximum_item = max(list_x)
            index = list_x.index(maximum_item)
        key = list_y[index]
        result[key] = list_x[index]

        del list_x[index]
        del list_y[index]
    return list(result.values()), list(result.keys())
