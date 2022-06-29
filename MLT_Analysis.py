import numpy as np
import astropy.units as u
import matplotlib.ticker as tkr
import matplotlib.colors as cols
import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
import matplotlib.dates as matdates
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import scipy.signal._savitzky_golay as sav_gol
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from matplotlib.path import Path as mplPath
import scipy.ndimage
import datetime
import json
import copy
import enum

import Contours
import MLT
import SpotTools
from Logger import PrintProgress
import Logger

# Fixes a matplotlib/pandas warning that shows up when plotting dates.
# I have no idea what it does but it gets rid of the warning.
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

class Outcome(enum.Enum):
    NewSpot = 0
    Inherited = 1
    Tracked = 2
    Displaced_by_size = 3
    Displaced_by_distance = 4


def get_euclidean_sqr_dist(coord1, coord2):
    return (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2


class MLT_Analyser:
    def __init__(self, _path_man, _config_parser, _comm, _spot_group=None):
        self.path_man = _path_man
        self.spot_group = _spot_group
        self.config_parser = _config_parser
        self.config_name = self.config_parser.get('Options', 'config_name')

        # MPI
        self.comm = _comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.output_path = self.path_man.getDir('output')
        self.graph_path = self.path_man.getDir('output_analysis', posix_path=(self.output_path / 'analysis'))
        self.combo_graph_path = None
        self.one_graph_path = None
        self.scanline_graph_path = None
        self.og_sizes_graph_path = None
        self.polar_contour_graph_path = None
        self.area_sizes_graph_path = None
        self.velocity_area_graph_path = None
        self.multi_parameter_graph_path = None
        self.roi_plot_path = None
        self.parameters_path = self.path_man.getDir('parameters')
        self.parameters_plot_path = self.path_man.getDir('parameters_plot', posix_path=(self.parameters_path / 'layer_plot'))

        # Parameters
        # Defines viewport in one-graphs
        self.viewport_ranges = json.loads(self.config_parser.get('MLT_Analysis', 'viewport'))
        self.viewport_aspect_ratio = json.loads(self.config_parser.get('MLT_Analysis', 'viewport_aspect_ratio'))
        # Defines the distance which centre of clusters will be measured to in get_cluster_params()
        self.region_centre = json.loads(self.config_parser.get('MLT_Analysis', 'centre'))
        # Set to 'dynamic' if no limit, otherwise an array specifying the upper and lower y axis limits for the angles
        # plot.
        self.graph_y_limits = json.loads(self.config_parser.get('MLT_Analysis', 'graph_y_limits'))
        self.param_tolerances = self.config_parser.parse_section_as_dict('MLT_Tolerances', 'float')
        self.cluster_track_direction = self.config_parser.get('MLT_Analysis', 'cluster_track_direction')
        self.do_velocity_filter = self.config_parser.getboolean('MLT_Analysis', 'do_velocity_filter')
        self.median_filter_window_size = self.config_parser.getint('MLT_Tolerances', 'median_filter_window_size')
        self.flare_times = self.config_parser.get_list_as_datetimes('Options', 'flare_times')
        self.cme_times = self.config_parser.get_list_as_datetimes('Options', 'cme_times')

        self.velocity_stride = self.config_parser.getint('MLT_Analysis', 'velocity_stride')
        self.angles_normalise = self.config_parser.get('MLT_Analysis', 'angles_normalise')
        self.plot_stack_separation = self.config_parser.getint('MLT_Analysis', 'plot_stack_separation')
        self.velocity_size_normalise = self.config_parser.get('MLT_Analysis', 'velocity_size_normalise')
        self.minimum_cluster_size = self.config_parser.getfloat('MLT_Analysis', 'minimum_cluster_size')
        self.number_of_clusters_to_track = self.config_parser.getint('MLT_Analysis', 'number_of_clusters_to_track')
        self.stack_plot_memory = self.config_parser.getint('MLT_Analysis', 'stack_plot_memory')
        self.max_distance_from_centre = self.config_parser.getint('MLT_Analysis', 'max_distance_from_centre')
        self.figsize = self.config_parser.get_list_as_float('MLT_Analysis', 'figsize')
        self.dpi = self.config_parser.getint('MLT_Analysis', 'fig_dpi')
        self.fig_font_size = self.config_parser.getint('MLT_Analysis', 'fig_font_size')
        self.label_clusters_on_roi = self.config_parser.getboolean('MLT_Analysis', 'label_clusters_on_roi')
        self.highlight_bad_data = self.config_parser.get('MLT_Analysis', 'highlight_bad_data').strip().lower()
        self.bad_data_ranges = self.config_parser.get_list_as_datetimes('MLT_Analysis', 'bad_data_ranges')
        self.plot_cme_times = self.config_parser.getboolean('MLT_Analysis', 'plot_cme_times')
        self.plot_flare_times = self.config_parser.getboolean('MLT_Analysis', 'plot_flare_times')
        self.displacement_size_differential_threshold = self.config_parser.getfloat('MLT_Analysis', 'displacement_size_differential_threshold')

        self.singles_graph_styles = self.config_parser.get_list_as_string('Plot_Single_Graph', 'graph_styles')
        self.singles_graph_colours = self.config_parser.get_list_as_int('Plot_Single_Graph', 'graph_colours')

        self.layers_show_centres_tracked  = self.config_parser.getboolean('MLT_Layers_Plot','show_centres_tracked')
        self.layers_tracer_line_memory = self.config_parser.getint('MLT_Layers_Plot', 'layers_tracer_line_memory')

        self.graph_type = self.config_parser.get('MLT_Analysis', 'graph_type')
        self.plots_type = self.config_parser.get('MLT_Analysis', 'plots').strip('[]').replace(' ', '').split(',')
        self.output_format = self.config_parser.get('MLT_Analysis', 'output_format')
        self.colour_map_plots = self.config_parser.get('MLT_Analysis', 'colour_map_plots')
        self.colour_map_spot = self.config_parser.get('MLT_Analysis', 'colour_map_spot')
        self.tracking_method = self.config_parser.get('MLT_Analysis', 'tracking_method')
        self.calculate_missing_params = self.config_parser.getboolean('MLT_Analysis', 'calculate_missing_params')
        self.time_label = self.config_parser.get('MLT_Analysis', 'time_label')
        self.plotting_stride = self.config_parser.getint('MLT_Analysis','plotting_stride')
        self.do_debug_tracking_plots = self.config_parser.getboolean('MLT_Analysis','do_debug_tracking_plots')
        self.y_axis_labels = {'angle': r'$\theta$ ($^\circ$)',
                              'velocity': r'$\omega$ ($^\circ h^{-1}$)',
                              'size': r'$r_{eq}$ (pix)',
                              'threshold_ratio': 'Threshold Ratio',
                              'global_intensity': 'Intensity (arb. units)',
                              'intensity': 'Intensity (arb. units)',
                              'ellipticity': 'Ellipticity (unitless)',
                              'roi_darkest_intensity': 'Intensity (arb. units)',
                              'roi_screen': 'Distance (pix)',
                              'roi_print': 'Distance (arcsec)',
                              'delta_1st_angle': 'Angle 1st order diff',
                              'delta_1st_size': 'Size 1st order diff',
                              'delta_1st_ellipticity': 'Ellipticity 1st order diff',
                              'delta_2nd_angle': 'Angle 2nd order diff',
                              'delta_2nd_size': 'Size 2nd order diff',
                              'delta_2nd_ellipticity': 'Ellipticity 2nd order diff',
                              'centre': 'Centre position (pix)',
                              'interpolated_velocity': r'(intrp) $\omega$ ($^\circ h^{-1}$)',
                              'smoothed_angle': r'(smoothed) $\theta$ ($^\circ$)',
                              'perimeter_length': 'Perim. length (pix)',
                              'delta_perimeter': r'$\Delta$ Perim. Length',
                              'delta_size': r'$\Delta \sqrt{Area}$ (pix)',
                              'delta_ellipticity': r'$\Delta$Ellipticity',
                              'cluster_instability': 'Cluster Instability (arb.)',
                              'intercluster_distance': 'Distance (pix)'
                              }
        self.time_axis_labels = {'days': '%d/%m',
                                 'hours': '%H:%M',
                                 'days_small': '%d',
                                 'days_long': '%H:%M %d/%m'}
        self.NOAA_number = self.config_parser.get('Options', 'NOAA_number')

        self.colour_map_limits = self.config_parser.get('MLT_Analysis', 'colour_map_limits')
        self.graphs_layout = self.config_parser.get('MLT_Analysis', 'graphs_layout')
        if self.colour_map_limits != 'dynamic':
            self.colour_map_limits = json.loads(self.colour_map_limits)

        self.export_clusters_to_txt = self.config_parser.getboolean('MLT_Analysis','export_clusters_to_txt')

        # Old function vars
        self.layer_index = 3
        self.max_separation_distance = self.config_parser.getint('MLT_Analysis','max_separation_distance')
        self.max_area_disparity = 0.2
        # maximum gap in time between two clusters being considered the same (in seconds)
        self.max_time_delta = 4 * 3600
        self.start_date = datetime.datetime(2014, 9, 10, 17, 0, 0)
        self.end_date = datetime.datetime(2014, 9, 10, 19, 0, 0)
        self.cluster_list = None

    def run(self, sunspot_group, thresholds, start_date=None, end_date=None):
        """Handles setup of input and output for all the graph types and launches them."""
        self.spot_group = sunspot_group
        if start_date is not None:
            self.start_date = start_date
        if end_date is not None:
            self.end_date = end_date
        if self.colour_map_limits == 'dynamic':
            self.colour_map_limits = [min(thresholds) - 0.01, max(thresholds) + 0.03]

        # Do Combo
        start, stop = SpotTools.get_date_range_indices(sunspot_group, self.start_date, self.end_date)
        if start - self.velocity_stride < 0:
            start = 0
        else:
            start -= self.velocity_stride
        if self.graph_type == 'velocity_area_distribution':
            self.velocity_against_area_distribution(sunspot_group.history[start:stop], thresholds)
        elif self.graph_type == 'velocity_area':
            self.velocity_area_graph_path = self.path_man.getDir('velocity_area' + self.config_name,
                                                                 posix_path=(self.graph_path / (
                                                                             'velocity_area' + self.config_name)))
            self.velocity_against_area(sunspot_group.history[start:stop], thresholds)
        elif self.graph_type == 'multi_parameter':
            self.multi_parameter_graph_path = self.path_man.getDir('output_multi_parameter' + self.config_name,
                                                                   posix_path=(self.graph_path / (
                                                                           'multi_parameter' + self.config_name)))
            self.multi_parameter_plot(sunspot_group.history[start:stop], thresholds, plots=self.plots_type)
        elif self.graph_type == 'roi':
            self.roi_plot_path = self.path_man.getDir('output_roi_plot' + self.config_name,
                                                      posix_path=(self.graph_path / (
                                                                           'roi_plot' + self.config_name)))
            self.roi_plot(sunspot_group.history[start:stop], thresholds)
        elif self.graph_type == 'cluster_count':
            self.plot_cluster_histograms(sunspot_group.history[start:stop], thresholds)
        elif self.graph_type == 'rotation_activity':
            self.global_parameters_plot(sunspot_group.history[start:stop], thresholds)
        elif self.graph_type == 'mlt_layers':
            self.mlt_plot_path = self.path_man.getDir('output_mlt_plot' + self.config_name,
                                                      posix_path=(self.graph_path / (
                                                                           'mlt_layers' + self.config_name)))
            self.plot_mlt_layers(sunspot_group.history[start:stop], thresholds)
        elif self.graph_type == 'single_graph':
            self.single_graph_path = self.path_man.getDir('output_single_graph' + self.config_name,
                                                       posix_path=(self.graph_path))
            self.single_graph_plot(sunspot_group.history[start:stop], thresholds, plots=self.plots_type)
        elif self.graph_type == 'mlt_count':
            self.cluster_statistics_plot(sunspot_group.history[start:stop], thresholds)
        else:
            raise ValueError("Value for graph type '{0}' not valid.".format(self.graph_type))

    def get_cluster_parent_area(self, cluster, all_cluster_list, max_separation, max_time_delta):
        """Given a cluster, and a list of cluster lists, find the cluster list that ends in the cluster closest in
        space to the input cluster that is also within max_separation and has a similar area. This function returns a
        number equal to the index of the cluster list that satisfies this condition."""
        cluster_list_ends = []
        cluster_list_starts = []
        closest_cluster_index = None
        closest_dist = max_separation ** 2
        closest_area = 1000  # cluster.size * self.max_area_disparity

        if len(all_cluster_list) == 0:
            return None, None

        for i in range(0, len(all_cluster_list)):
            # Add last value in each list to a new list for easier iteration.
            cluster_list_ends.append(all_cluster_list[i][-1])
            cluster_list_starts.append(all_cluster_list[i][0])

        # If the list length is 0, means the cluster list is empty so return None
        if len(cluster_list_ends) == 0:
            return None, None

        # Find all potential end points that are close enough to be the same cluster
        potential_indices = []
        for i in range(0, len(cluster_list_ends)):
            potential_match = cluster_list_ends[i]
            sqr_dist = get_euclidean_sqr_dist(cluster.centre, potential_match.centre)
            time_delta = (cluster.datetime - potential_match.datetime).seconds
            if sqr_dist < closest_dist and 0 < time_delta < max_time_delta:
                potential_indices.append(i)
                closest_dist = sqr_dist

        # Go over all potential matches and find the cluster most similar area
        for i in potential_indices:
            potential_match = cluster_list_starts[i]
            area_disparity = abs(cluster.size - potential_match.size)
            if area_disparity < closest_area:
                closest_cluster_index = i
                closest_area = area_disparity

        # Debug wtf is going on
        msg = "Current cluster size: {0} [{2}] ({1}) Potential matches:".format(cluster.size, cluster.threshold_ratio,
                                                                                cluster.centre)
        for i in potential_indices:
            potential_match = cluster_list_starts[i]
            msg += " {0} [{2}] ({1})".format(potential_match.size, potential_match.threshold_ratio,
                                             potential_match.centre)
        msg += " | went with {0}".format(closest_area)
        Logger.log(msg, Logger.LogLevel.debug)

        return closest_cluster_index, closest_area



    # This seems to be the one I'm using for area tracking - 22/04/2020
    def track_constant_area(self, sunspots, threshold, max_separation, cluster_to_track=0):
        """
        Track a single sunspot keeping the area as consistent as possible through time by allowing clusters to
        change threshold layer. This is intended for use when tracking spots over short timescales (a few hours) during
        large flare events, where white-light contamination could affect the cluster appearance. The reference point
        is assumed to be the first image. This tracking method is not stable over long periods or with spots undergoing
        significant changes in shape.

        Checks each cluster in the following image to find the best match by cluster size first. The cluster with the
        smallest difference in size that is also within the max separation distance is chosen as the best match.

        :param sunspots: list. The sunspot group list.
        :param threshold: float. The threshold ratio
        :param max_separation: float. Maximum distance in pixels between two clusters.
        :param cluster_to_track: int. The spot number of the spot to track.
        :return:
        """
        cluster_history = []
        mlt_initial = self.path_man.loadMLT(sunspots[0].mlt_path)
        layer_initial = SpotTools.first(l for l in mlt_initial.layers if l.threshold_ratio == threshold)
        cluster_initial = layer_initial.mlt_clusters[cluster_to_track]
        cluster_history.append(cluster_initial)

        for i in range(1, len(sunspots)):
            last_cluster = cluster_history[-1]
            try:
                mlt = self.path_man.loadMLT(sunspots[i].mlt_path)
            except TypeError:
                Logger.log("[MLT_Analyser] No MLT file found for sunspot {0}. Skipping..".format(
                    str(sunspots[i].timestamp)), Logger.LogLevel.verbose)
                continue

            potential_clusters = []
            for layer in mlt.layers:
                try:
                    potential_clusters.extend(layer.mlt_clusters)
                except AttributeError:
                    # Not all spots will have all layers, so if no layer found, move on.
                    continue

            # Set default next cluster to be in same layer as last one.
            likely_next_cluster = SpotTools.first(
                c for c in potential_clusters if c.threshold_ratio == last_cluster.threshold_ratio)
            try:
                closest_area = abs(likely_next_cluster.size - cluster_initial.size)
            except AttributeError:
                # This is likely due to likely_next_cluster not existing because of a missing layer. So just set the
                # current closes area to a large number and let the code find the best match in the next round.
                closest_area = 1000

            for potential_match in potential_clusters:
                area_disparity = abs(cluster_initial.size - potential_match.size)
                if area_disparity < closest_area:
                    sqr_dist = get_euclidean_sqr_dist(last_cluster.centre, potential_match.centre)
                    if sqr_dist < max_separation ** 2:
                        likely_next_cluster = potential_match
                        closest_area = area_disparity

            cluster_history.append(likely_next_cluster)
        return cluster_history

    # This is the main way of doing it. 08/06/2020
    def multi_track_clusters(self, sunspots, thresholds):
        Logger.debug("[MLT_Analysis - multi_track_clusters] Running multi-tracking...")
        # Initialise
        tracked_clusters = [[] for threshold in thresholds]
        cluster_count = 0
        mlt_previous = None
        spot_order = None
        if self.cluster_track_direction == "forwards":
            spot_order = range(0,len(sunspots))
        elif self.cluster_track_direction == "backwards":
            spot_order = range(len(sunspots)-1, -1, -1)
        else:
            raise ValueError("Cluster track direction \"{0}\" not recognised. Please check config file.".format(self.cluster_track_direction))

        Logger.debug("[MLT_Analysis - multi_track_clusters] len(sunspots) = {0}".format(len(sunspots)))
        # Requires thresholds be put in descending order. Can't simply sort them or it might cause errors reading the correct data.
        if thresholds[0] < thresholds[-1]:
            raise Exception("Thresholds must be given in descending order. e.g. [0.55, 0.35, 0.2, 0.1]")
        # make a plotting function for debugging
        def plot_cluster(graph, xpoints, ypoints, outcome, layer_index):
            graph.scatter(xpoints, ypoints, marker='s', s=((72. / fig.dpi) ** 2) * 2,
                          color=SpotTools.cvd_colours[outcome.value])
            graph.set_xlim(self.viewport_ranges[0])
            graph.set_ylim(self.viewport_ranges[1])
            graph.set_title("Threshold Ratio: {0}".format(thresholds[layer_index]))
            graph.set_xlabel("Position (pix)")
            graph.set_ylabel("Position (pix)")


        # Do the things
        tracking_outcomes = {}
        prog_tracking = PrintProgress(0, len(sunspots), label="[MLT_Analysis] Running Multi-Tracking for {0} spots... ".format(len(sunspots)))
        for spot_index in spot_order:
            #Logger.debug("[MLT_Analysis - multi_track_clusters] Spot index: {0}".format(spot_index))
            if sunspots[spot_index].mlt_path is None:
                Logger.debug("[MLT_Analysis - multi_track_clusters] MLT path is none for date {0}!".format(sunspots[spot_index].filename))
                continue
            mlt_current = self.path_man.loadMLT(sunspots[spot_index].mlt_path)
            if mlt_current is None:
                Logger.debug("[MLT_Analysis - multi_track_clusters] MLT **object** is none for date {0}!".format(sunspots[spot_index].filename))
                continue
            Logger.debug("[MLT_Analysis - multi_track_clusters] - Analysing layer: {0}".format(mlt_current.filename))
            mlt_current.print_layers()

            # Record time difference between frames to keep things scaled when dealing with patchy data.
            if mlt_previous is not None:
                time_between_layers = (mlt_current.timestamp - mlt_previous.timestamp).total_seconds() * u.second
                Logger.debug(
                    "[MLT_Analysis - multi_track_clusters] time_between_layers: {0} ".format(time_between_layers)
                    + "-- mlt_current.timestamp: {0} ".format(mlt_current.timestamp)
                    + "-- mlt_previous.timestamp: {0}".format(mlt_previous.timestamp))
            else:
                time_between_layers = 1 * u.hour

            if time_between_layers == 0. * u.second:
                Logger.log("[MLT_Analysis - multi_track_clusters] Time between frames is 0 seconds, skipping frame.")
                continue

            # Diagnostic telemetry
            tracking_outcome = {Outcome.NewSpot:0, Outcome.Inherited:0, Outcome.Tracked:0, Outcome.Displaced_by_size:0,
                                Outcome.Displaced_by_distance:0}  # Tally of which outcomes occur most

            # Debug plots
            if self.do_debug_tracking_plots:
                fig = plt.figure(figsize=(12,12), dpi=120)
                filename = sunspots[spot_index].timestamp.strftime('%Y-%m-%d_%H-%M-%S')
                plt.title(sunspots[spot_index].timestamp)
                grid_dimensions = (4, 4)
                graphs = [plt.subplot2grid(grid_dimensions, ((x // 4), x % 4)) for x in range(0, 16)]
                # Plot first graph
                try:
                    roi = self.path_man.loadROI(sunspots[spot_index].ROI_path)
                except:
                    continue
                graphs[0].imshow(roi.data, cmap=self.colour_map_spot, aspect='auto')
                graphs[0].set_xlim(self.viewport_ranges[0])
                graphs[0].set_ylim(self.viewport_ranges[1])
                graphs[0].set_title(filename)

            # for each layer in this image
            for layer_index in range(len(thresholds)-1, -1, -1):
                # Logger.debug("[MLT_Analysis - multi_track_clusters] "
                #              + "Layer Index: {0} ".format(layer_index)
                #              + "Threshold ratio: {0}".format(thresholds[layer_index]))
                layer_dictionary = {}
                clusters_in_current_layer = SpotTools.first(l.mlt_clusters for l in mlt_current.layers if l.threshold_ratio == thresholds[layer_index])
                if clusters_in_current_layer is None:
                    Logger.debug("[MLT_Analysis - multi_track_clusters] No clusters in this layer! Skipping...")
                    continue

                try:
                    clusters_in_previous_image = SpotTools.first(l.mlt_clusters for l in mlt_previous.layers if l.threshold_ratio == thresholds[layer_index])
                except AttributeError:
                    clusters_in_previous_image = None
                    # Also sort first layer so biggest at top
                    clusters_in_current_layer.sort(key=lambda x: x.size, reverse=True)

                # Nth LAYER ====================================================================================
                # If this is the Nth layer, check the last image to see if any previously defined cluster has a
                # similar center point. If not, this is a new cluster.
                if layer_index == len(thresholds)-1:
                    for cluster in clusters_in_current_layer:
                        # Get data for cluster on debug plot
                        xpoints = []
                        ypoints = []
                        for point in cluster.points:
                            xpoints.append(point[0])
                            ypoints.append(point[1])
                        #Logger.debug("[MLT_Analysis - multi_track_clusters] Tracking cluster number {0} (centre {1})".format(cluster.number, cluster.centre))
                        # If first layer, just add in order found
                        if clusters_in_previous_image is None:
                            #Logger.debug("[MLT_Analysis - multi_track_clusters]     First layer and image. Assigning new ID")
                            layer_dictionary[cluster_count] = cluster.id
                            cluster.number = cluster_count
                            cluster_count += 1
                            tracking_outcome[Outcome.NewSpot] += 1
                            if self.do_debug_tracking_plots:
                                plot_cluster(graphs[layer_index+1],xpoints, ypoints, Outcome.NewSpot, layer_index)
                        else:
                            # Track by Position
                            #Logger.debug("[MLT_Analysis - multi_track_clusters]     First layer, tracking from previous...")
                            layer_dictionary, cluster_count, outcome = self.trk_from_prev_position(layer_dictionary,
                                                                                          cluster_count,
                                                                                          cluster,
                                                                                          clusters_in_previous_image,
                                                                                          time_between_layers)
                            tracking_outcome[outcome] += 1
                            if self.do_debug_tracking_plots:
                                plot_cluster(graphs[layer_index+1],xpoints, ypoints, outcome, layer_index)

                else:
                    # < Nth LAYER ==============================================================================
                    # First, check if any cluster in the layer ABOVE has its centre point inside this cluster's
                    # perimeter. If this is the case, and there is only one, inherit the number. If there is more than
                    # one result then find out which one is closer and inherit from that one. If there are no clusers
                    # in the layer above that are within the boundaries of this cluster, attempt to track from the
                    # previous image. If that doesn't work, then congratulations - you're a new spot.
                    clusters_in_layer_above = SpotTools.first(l.mlt_clusters for l in mlt_current.layers if l.threshold_ratio == thresholds[layer_index+1])
                    for cluster in clusters_in_current_layer:
                        # check layer above you for being inside another cluster
                        #parent = SpotTools.first([c for c in clusters_in_previous_layer if [cluster.centre] in c.points], default=None)
                        # Had to expand out nice statement above because c.points is a list of numpy arrays and so
                        # that simple statement above couldn't quite do what I wanted it to.
                        #Logger.debug("[MLT_Analysis - multi_track_clusters] Tracking next cluster (centre {0} / size {1})".format(cluster.centre, cluster.size))
                        xpoints = []
                        ypoints = []
                        for point in cluster.points:
                            xpoints.append(point[0])
                            ypoints.append(point[1])
                        parent = None
                        matches_in_layer_above = []
                        if clusters_in_layer_above is not None:
                            for c in clusters_in_layer_above:
                                centre = c.centre.tolist()
                                points = [p.tolist() for p in cluster.points]
                                if centre in points:
                                    matches_in_layer_above.append(c)
                                # !! Find by convex hull
                                # try:
                                #     hull = ConvexHull(cluster.points)
                                #     hull_path = mplPath(np.array(cluster.points)[hull.vertices])
                                #     if hull_path.contains_point(centre):
                                #         matches_in_layer_above.append(c)
                                # except:
                                #     # !! Find parent by "if contained in my borders" approach !!
                                #     points = [p.tolist() for p in cluster.points]
                                #     if centre in points:
                                #         matches_in_layer_above.append(c)
                            if len(matches_in_layer_above) == 1:
                                parent = matches_in_layer_above[0]
                            elif len(matches_in_layer_above) > 0:
                                distances = [(get_euclidean_sqr_dist(cluster.centre, m.centre), m) for m in matches_in_layer_above]
                                distances.sort(key=lambda x: x[0])
                                parent = distances[0][1]

                        if parent is not None and parent.number not in layer_dictionary.keys():
                            layer_dictionary[parent.number] = cluster.id
                            cluster.number = parent.number
                            tracking_outcome[Outcome.Inherited] += 1
                            if self.do_debug_tracking_plots:
                                plot_cluster(graphs[layer_index+1], xpoints, ypoints, Outcome.Inherited, layer_index)
                            #Logger.debug("[MLT_Analysis - multi_track_clusters]     Match accepted.")
                            continue
                        if spot_index > 0 and clusters_in_previous_image is not None:
                            #Logger.debug("[MLT_Analysis - multi_track_clusters]     Could not find parent, trying to track...")
                            layer_dictionary, cluster_count, outcome = self.trk_from_prev_position(layer_dictionary,
                                                                                          cluster_count,
                                                                                          cluster,
                                                                                          clusters_in_previous_image,
                                                                                          time_between_layers)
                            tracking_outcome[outcome] += 1
                            if self.do_debug_tracking_plots:
                                plot_cluster(graphs[layer_index+1], xpoints, ypoints, outcome, layer_index)

                        else:
                            #Logger.debug("[MLT_Analysis - multi_track_clusters]     Could not track, assigning new ID ({0}).".format(cluster_count))
                            layer_dictionary[cluster_count] = cluster.id
                            cluster.number = cluster_count
                            cluster_count += 1
                            tracking_outcome[Outcome.NewSpot] += 1
                            if self.do_debug_tracking_plots:
                                plot_cluster(graphs[layer_index+1], xpoints, ypoints, Outcome.NewSpot, layer_index)
                # ------------------------------------------------------
                # Re-organise the layer dictionary to be sorted into the global cluster list, so that it can
                # be read by the rest of the program:
                #
                #   all_cluster_list
                #          |             |--> cluster_1 [ img0, img1, ..]
                #          |---> 0.55 ---|--> cluster_2 [ img0, img1, ..]
                #          |             |--> cluster_3 [ img0, img1, ..]
                #          |
                #          |             |--> cluster_1 [ img0, img1, ..]
                #          |---> 0.35 ---|--> cluster_2 [ img0, img1, ..]
                #          |             |--> cluster_3 [ img0, img1, ..]
                #
                # ------------------------------------------------------
                Logger.debug("[MLT_Analysis - multi_track_clusters] Layer complete! spot_index: {0}".format(spot_index)
                             + ", layer_index: {0}".format(layer_index)
                             + ", len(layer_dictionary): {0}".format(len(layer_dictionary))
                             + ", cluster_count: {0}".format(cluster_count)
                             + "\nTracking outcome - {0}".format(tracking_outcome))

                for key, value in layer_dictionary.items():
                    while len(tracked_clusters[layer_index]) <= key:
                        tracked_clusters[layer_index].append([])
                    tracked_clusters[layer_index][key].append(value)

            # Update diagnostics
            for key in tracking_outcome.keys():
                if key in tracking_outcomes:
                    tracking_outcomes[key].append(tracking_outcome[key])
                else:
                    tracking_outcomes[key] = [tracking_outcome[key]]
            # save debug plot
            if self.do_debug_tracking_plots:
                plt.tight_layout()
                plt.savefig(str(self.parameters_plot_path / (filename + '.png')))
                plt.close()
            mlt_previous = mlt_current
        prog_tracking.update()

        # Plot diagnostics
        self.diagnostics_plot_tracking_outcomes(tracking_outcomes)
        # Sort tracked cluster lists based on timestamp so that data is consistent across forward/backward tracking
        Logger.debug("[MLT_Analysis - multi_track_clusters] Sorting clusters by date...")
        for threshold in tracked_clusters:
            for cluster_list in threshold:
                cluster_list.sort(key=lambda clist: SpotTools.parse_string_to_datetime(clist.split('#')[2]))
        Logger.debug("[MLT_Analysis - multi_track_clusters] Sorted.")
        # Print some information about the shape of the data.
        total_string = ""
        for i in range(0, len(tracked_clusters)):
            total_string += "\nCluster count in Layer {0}: {1}".format(thresholds[i], len(tracked_clusters[i]))
        Logger.log("[MLT_Analysis - multi_track_clusters] Final tracked clusters results:"
                   + total_string)
        Logger.debug("[MLT_Analysis - Cluster cache info: {0}]".format(self.path_man.get_cluster_from_id.cache_info()))
        return tracked_clusters

    def diagnostics_plot_tracking_outcomes(self, tracking_outcomes):
        """Plots a graph showing how many clusters were tracked by position, inheritance, or appeared as a new spot."""
        fig = plt.figure(figsize=(3,3), dpi=300)
        count_plotted = 0
        for key, value in tracking_outcomes.items():
            plt.plot(list(range(len(tracking_outcomes[key]))), tracking_outcomes[key],
                     c=SpotTools.cvd_colours[count_plotted], label=key.name)
            count_plotted += 1
        #plt.legend()
        plt.title("Tracking Outcomes")
        plt.ylabel("Frequency")
        plt.xlabel("Frame")
        plt.rcParams.update({"font.size" : 6})
        plt.tight_layout()
        plt.savefig(self.parameters_path / "diagnostics_tracking_outcomes.png")


    def trk_from_prev_position(self, layer_dict, total_cluster_counter, cluster, clusters_in_previous_image, time_between_layers):
        """
        Check the list of clusters in previous image for any that have a similar centre point to the input cluster.
        Add cluster to the layer_dict by either associating it with a previously identified cluster, or by giving it
        a new id, then return the updated layer_dict and total_cluster_counter.

        :param layer_dict:
        :param total_cluster_counter:
        :param cluster:
        :param clusters_in_previous_image:
        :return: layer_dict, total_cluster_counter
        """
        dists_to_prev_clusters = self.sort_clusters_by_distance(cluster, clusters_in_previous_image)
        outcome = None
        #Logger.debug("[MLT_Analysis - trk_from_prev_pos]        Distances to previous clusters: {0}".format(["{0} : {1}".format(dists_to_prev_clusters[i][1].number, dists_to_prev_clusters[i][0]) for i in range(len(dists_to_prev_clusters))]))
        target_cluster = dists_to_prev_clusters[0][1]
        maximum_separation = abs(self.max_separation_distance * (u.pixel / u.hour) * time_between_layers.to(u.hour))
        if (dists_to_prev_clusters[0][0] * u.pixel) < maximum_separation:
            #Logger.debug("[MLT_Analysis - trk_from_prev_pos]        Found previous cluster! (ID: {0})".format(dists_to_prev_clusters[0][1].number))
            if not target_cluster.number in layer_dict:
                layer_dict[target_cluster.number] = cluster.id
                cluster.number = target_cluster.number
                outcome = Outcome.Tracked
            else:
                Logger.debug("[MLT_Analysis - trk_from_prev_pos] Conflict found, attempting to resolve via displacement...")
                existing_match = self.path_man.get_cluster_from_id(layer_dict[target_cluster.number])
                #Logger.debug("[MLT_Analysis - trk_from_prev_pos]        Multiple matches found, attempting to resolve conflict...")
                # Logger.debug("[MLT_Analysis - trk_from_prev_pos]        "
                #              + "Target cluster center/size: {0} / {1}".format(target_cluster.centre,
                #                                                               target_cluster.size)
                #              + "--- Old match center/size: {0} / {1} ".format(existing_match.centre,
                #                                                               existing_match.size)
                #              + "--- Conflict match center/size: {0} / {1} ".format(cluster.centre, cluster.size))
                size_differential = abs(existing_match.size - cluster.size) / max(existing_match.size, cluster.size)
                Logger.debug("[MLT_Analysis - trk_from_prev_pos] Size differential: {0}".format(size_differential))

                if size_differential > self.displacement_size_differential_threshold:
                    Logger.debug("[MLT_Analysis - trk_from_prev_pos] Size differential greater than {0}, displacing via size.".format(self.displacement_size_differential_threshold))
                    winning_match = existing_match if existing_match.size > cluster.size else cluster
                    losing_match = cluster if winning_match == existing_match else existing_match
                    layer_dict[target_cluster.number] = winning_match.id
                    winning_match.number = target_cluster.number
                    layer_dict[total_cluster_counter] = losing_match.id
                    losing_match.number = total_cluster_counter
                    total_cluster_counter += 1
                    outcome = Outcome.Displaced_by_size
                else:
                    Logger.debug("[MLT_Analysis - trk_from_prev_pos] Differential too small, displacing via distance.")
                    sorted_by_distance = self.sort_clusters_by_distance(target_cluster,
                                                                        [existing_match,
                                                                         cluster])
                    # Logger.debug("[MLT_Analysis - trk_from_prev_pos]        "
                    #              + "Closest cluster: {0} / {1}".format(sorted_by_distance[0][1].centre,
                    #                                                    sorted_by_distance[0][1].size)
                    #              + "Relabelled cluster: {0} / {1}".format(sorted_by_distance[1][1].centre,
                    #                                                       sorted_by_distance[1][1].size)
                    #              )
                    layer_dict[target_cluster.number] = sorted_by_distance[0][1].id
                    sorted_by_distance[0][1].number = target_cluster.number
                    layer_dict[total_cluster_counter] = sorted_by_distance[1][1].id
                    sorted_by_distance[1][1].number = total_cluster_counter
                    total_cluster_counter += 1
                    outcome = Outcome.Displaced_by_distance
        else:
            Logger.debug("[MLT_Analysis - trk_from_prev_pos] No clusters within {0} (nearest: {1}) -- New spot.".format(maximum_separation, dists_to_prev_clusters[0][0] * u.pixel))
            layer_dict[total_cluster_counter] = cluster.id
            cluster.number = total_cluster_counter
            total_cluster_counter += 1
            outcome = Outcome.NewSpot
            #Logger.debug("[MLT_Analysis - trk_from_prev_pos]        No match found. New cluster!")

        return layer_dict, total_cluster_counter, outcome

    def sort_clusters_by_distance(self, reference_cluster, clusters):
        """
        Given a refernce cluster for a centre point, return a list of clusters sorted by their distance to the reference
        point, with the smallest first.
        :param reference_cluster:
        :param clusters:
        :return: [(float, obj), ...] List of tuples [(sqr_distance_to_cluster, cluster), ...]
        """
        dists_to_reference = [(np.sqrt(get_euclidean_sqr_dist(reference_cluster.centre, c.centre)), c) for c in clusters]
        dists_to_reference.sort(key=lambda y: y[0])
        return dists_to_reference

    def track_clusters_by_area(self, sunspots, max_separation, max_time_delta):
        """Tracks clusters through MLT layers based on their area (size) and centre point. This does not rely on
        thresholds and so clusters can be tracked cross-layer."""
        all_clusters_list = []
        for i in range(0, len(sunspots)):
            # Try loading MLT. Not all spots will have MLT files (apparently) so skip if there is nothing there.
            try:
                mlt = self.path_man.loadMLT(sunspots[i].mlt_path)
            except TypeError:
                Logger.log("[MLT_Analyser] No MLT file found for sunspot {0}. Skipping..".format(
                    str(sunspots[i].timestamp)), Logger.LogLevel.verbose)
                continue

            clusters = []
            for layer in mlt.layers:
                try:
                    clusters.extend(layer.mlt_clusters)
                except AttributeError:
                    # Not all spots will have all layers, so if no layer found, move on.
                    continue

            clusters.sort(key=lambda x: x.size, reverse=True)
            # cluster_matches = []
            for cluster in clusters:
                parent_cluster_index = self.get_cluster_parent_area(cluster, all_clusters_list,
                                                                    max_separation, max_time_delta)
                if parent_cluster_index is None:
                    all_clusters_list.append([cluster])
                else:
                    all_clusters_list[parent_cluster_index].append(cluster)

        return all_clusters_list

    def is_too_jumpy(self, data, threshold):
        avg_delta_angle = 0.
        for j in range(1, len(data)):
            avg_delta_angle += abs(data[j] - data[j - 1])
        avg_delta_angle = avg_delta_angle / len(data)
        return avg_delta_angle > threshold

    def get_interpolated_velocity(self, parameters):
        # Angles
        angles = parameters["smoothed_angle"]
        # Interpolation of angles
        data_points_per_hour = 10  # TODO: Should be a config parameter eventually
        interp_count = int(abs((parameters["time"][0] - parameters["time"][-1]).total_seconds() / (60 * 60)) * data_points_per_hour)
        interp_times = np.linspace(parameters["matplot_time"][0], parameters["matplot_time"][-1], interp_count)
        parameters["interpolated_time"] = interp_times
        try:
            a_spline = CubicSpline(parameters["matplot_time"], angles)
        except ValueError as ve:
            Logger.debug("[MLT_Analysis - get_interpolated_velocity] Parameters['matplot_time'] = {0}".format(parameters["matplot_time"]))
            return None
        interp_angles = a_spline(interp_times)
        # Get velocities from interpolated angles
        stride = self.velocity_stride
        interp_velocities = [0 for s in range(stride)]
        for i in range(stride,len(interp_times)):
            angle_delta = interp_angles[i] - interp_angles[i-stride]
            time_delta = (interp_times[i] - interp_times[i-stride]) * 24.0  # matplot times are in fractions of a day! 
            Logger.debug("[MLT_Analysis - get_smooth_velocity] angle_delta: {0} time_delta: {1}".format(angle_delta, time_delta))
            velocity = angle_delta / time_delta
            interp_velocities.append(velocity)
        return interp_velocities

    def get_angular_velocity(self, timestamp_list, angle_list, delta_angles, stride):
        velocities = []
        if len(angle_list) < 11:
            smooth_angles = angle_list
        else:
            smooth_angles = sav_gol.savgol_filter(angle_list, 11, 3)
        for i in range(0, len(smooth_angles)):
            if i < stride:
                velocities.append(0.0)
                continue
            delta_angle = smooth_angles[i] - smooth_angles[i - stride]
            delta_time = (timestamp_list[i] - timestamp_list[i - stride]).seconds / 3600
            velocity = delta_angle / delta_time
            if len(self.param_tolerances) == 0:
                velocities.append(velocity)
            else:
                if abs(delta_angles[i]) > abs(self.param_tolerances['velocity']):
                    # overwrite the current one, and previous one
                    #velocities[i-1] = np.nan
                    velocities.append(np.nan)
                    Logger.debug("[MLT_Analysis - get_angular_velocity]"
                                 + " Angle diff ({0})".format(delta_angles[i])
                                 + " greater than cutoff ({0}), ignoring!".format(self.param_tolerances['velocity']))
                else:
                    velocities.append(velocity)
                # if abs(delta_angles[i - 1] > abs(self.delta_2_cutoffs[0])):
                #     velocities[i] = np.nan
        return velocities

    def get_ellipticity(self, axes):
        return (axes[1] - axes[0]) / axes[1]

    def differentiate_parameter(self, parameter, timestamp_list, stride, do_smooth=False, abs_threshold=None, relative=False):
        """
        Returns the differential of the given parameter, replacing outliers with NaNs if abs_threshold specified.
        :param parameter:
        :param timestamp_list:
        :param stride:
        :param do_smooth: Apply a savgol filter to the data before differentiation?
        :param abs_threshold: remove values above specified threshold?
        :param relative: return values as a change relative to the size of the i-stride-th value.
        :return:
        """
        parameter_diff = []
        # Smooth the data first?
        if do_smooth:
            if len(parameter) > 11:
                parameter = sav_gol.savgol_filter(parameter, 11, 3)
        # Start differentiation
        for i in range(0, len(parameter)):
            if i < stride:
                parameter_diff.append(np.nan)
                continue
            delta_parameter = parameter[i] - parameter[i-stride]
            delta_time = (timestamp_list[i] - timestamp_list[i - stride]).seconds / 3600.
            try:
                if relative:
                    differentiated = (delta_parameter / delta_time) / parameter[i-stride]
                else:
                    differentiated = delta_parameter / delta_time
            except ZeroDivisionError:
                Logger.debug("[MLT_Analysis - differentiate_parameter] Delta time was 0 between two frames, repeating last value...")
                differentiated = parameter_diff[i-stride]
            # Debug rogue nan's 2020-10-08
            if differentiated == np.nan:
                Logger.debug("[MLT_Analysis - differentiate_parameter] NaN found."
                             + " delta_parameter = {0} delta_time = {1}".format(delta_parameter, delta_time))
            if abs_threshold is None:
                parameter_diff.append(differentiated)
            else:
                if abs(differentiated) < abs_threshold:
                    parameter_diff.append(differentiated)
                else:
                    parameter_diff.append(np.nan)
        return parameter_diff

    def perimeter_polar_coords(self, perimeter_points, centre_coords):
        """Returns a list of coordinates representing the perimeter in polar coordinates."""
        formatted_cartesian = Contours.reformat_perimeter(perimeter_points)
        perimeter_polar = Contours.cart2polData(formatted_cartesian[0], formatted_cartesian[1], centre_coords)
        perimeter_polar = perimeter_polar.tolist()
        return Contours.reformat_perimeter(perimeter_polar)


    def get_cluster_params(self, cluster_list, threshold_layer, spot_index, thresholds, tryload=True):
        """Get a number of cluster parameters for each cluster in cluster list and return as a dictionary.
         Used to plot cluster parameters against times"""
        # extract cluster in here so that we have access to threshold_layer and spot_index for saving data
        if tryload:
            loaded_parameters = self.path_man.loadParameters(thresholds[threshold_layer], spot_index)
            if loaded_parameters is not None:
                Logger.log("[MLT_Analysis - get_cluster_params] Loaded paramters for {0}_{1} from file".format(spot_index, thresholds[threshold_layer]))
                return loaded_parameters
        try:
            cluster_list = cluster_list[threshold_layer][spot_index]
        except IndexError as ie:
            Logger.debug("[MLT_Analysis - get_cluster_params] "
                         + "cluster_list has no entry for threshold_layer = {0}".format(threshold_layer)
                         + " and spot_index = {0}. Returning None.".format(spot_index))
            return None
        Logger.debug("[MLT_Analysis - get_cluster_params] Getting parameters for spot {0} at threshold {1}".format(spot_index, thresholds[threshold_layer]))
        parameters = {}
        parameters['time'] = []
        parameters['matplot_time'] = []
        parameters['angle'] = []
        parameters['size'] = []
        parameters['centre'] = []
        parameters['distance'] = []
        parameters['velocity'] = []
        parameters['interpolated_velocity'] = []
        parameters['threshold_ratio'] = []
        parameters['intensity'] = []
        parameters['global_intensity'] = []
        parameters['equivalent_radius'] = []
        parameters['ellipticity'] = []
        parameters['roi_darkest_intensity'] = []
        parameters['perimeter_length'] = []
        parameters['delta_perimeter'] = []
        parameters['polar_perimeter'] = []
        parameters['perimeter_variance'] = []
        parameters['delta_size'] = []
        parameters['delta_ellipticity'] = []
        parameters['cluster_instability'] = []

        prog_extraction = PrintProgress(0, len(cluster_list)+1, label="[MLT_Analysis]     Extracing cluster data...")
        none_clusters = 0
        for i in range(0, len(cluster_list)):
            # updating the progress tracker at the start because of lots of opportunities to skip an iteration
            prog_extraction.update()
            previous_c = self.path_man.get_cluster_from_id(cluster_list[i-1])
            c = self.path_man.get_cluster_from_id(cluster_list[i])
            if c is None:
                none_clusters += 1
                continue
            if c.ellipse_parameters is None:
                # Logger.log("c.ellipse_parameters are None! Recalculating...", Logger.LogLevel.debug)
                if len(c.points) < 10:
                    continue
                elli = MLT.MultiLevelThresholding.do_ellipse_fit(c.points)
                if elli is None:
                    continue
                c.ellipse_parameters = elli
            # Check for duplicate images. Skip if that is the case.
            if i > 0 and c.datetime == previous_c.datetime:
                continue
            c.number = spot_index  # A horrible workaround for the fact that cluster numbers are apparently not saved
            parameters['time'].append(c.datetime)
            parameters['matplot_time'].append(matdates.date2num(c.datetime))
            parameters['size'].append(np.sqrt(c.size) / np.pi)
            parameters['centre'].append(c.centre)
            parameters['equivalent_radius'].append(np.sqrt(c.ellipse_parameters[1][0] * c.ellipse_parameters[1][1])*0.5)
            parameters['threshold_ratio'].append(c.threshold_ratio)
            parameters['distance'].append(np.sqrt(get_euclidean_sqr_dist(self.region_centre, c.centre)))
            parameters['intensity'].append(c.threshold)
            parameters['global_intensity'].append(c.threshold * (1.0 / c.threshold_ratio))
            parameters['ellipticity'].append(self.get_ellipticity(c.ellipse_parameters[1]))
            parameters['perimeter_length'].append(len(Contours.getPerimeter(c.points)))
            parameters['polar_perimeter'].append(self.perimeter_polar_coords(Contours.getPerimeter(c.points), c.centre))
            #parameters['perimeter_variance'].append(0 if previous_c is None else self.polar_perimeter_variation(parameters['polar_perimeter'][i],parameters['polar_perimeter'][i-1]))
            angle = (c.ellipse_parameters[2] / (2 * np.pi)) * 360.
            # Remove wrapping around -90/90
            try:
                last_angle = parameters['angle'][i - 1]
                angle = self.fix_wrapping(angle, last_angle)
            except IndexError:
                pass
            parameters['angle'].append(angle)
        prog_extraction.update()  # Add update at the end for final tick to 100%

        Logger.log("[MLT_Analysis - get_cluster_parameters] None cluster count: {0}".format(none_clusters))
        # If there are no angles for a spot, because unable to fit an ellipse, then don't try to process further
        if len(parameters['angle']) == 0:
            Logger.debug("[MLT_Analysis - get_cluster_parameters] Could not find angles for cluster!"
                         + "\ntime = {0}".format(parameters['time'])
                         + "\nthreshold_ratio = {0}".format(parameters['threshold_ratio'])
                         + "\nlen(size) = {0}".format(len(parameters['size'])))
            return None

        if self.angles_normalise and len(parameters['angle']) != 0:
            parameters['angle'] = (np.array(parameters['angle'])
                                   - parameters['angle'][0]).tolist()

        parameters["smoothed_angle"] = scipy.ndimage.median_filter(parameters["angle"],
                                                                   size=self.median_filter_window_size,
                                                                   mode='nearest')
        # Calculate First Order derivatives
        #parameters['delta_perimeter'] = np.diff(parameters['perimeter_length'], prepend=parameters['perimeter_length'][0])
        parameters['delta_perimeter'] = self.differentiate_parameter(parameters["perimeter_length"], parameters["time"],
                                                                     1, relative=True)
        parameters['delta_size'] = self.differentiate_parameter(parameters['size'], parameters['time'],
                                                                1, relative=True)
        parameters['delta_ellipticity'] = self.differentiate_parameter(parameters['ellipticity'], parameters['time'],
                                                                       1)
        # Sum up all the changes in parameters as a benchmark for how consistent a cluster is over time.
        parameters['cluster_instability'] = [parameters['delta_perimeter'][i] + parameters['delta_size'][i] + parameters['delta_ellipticity'][i] for i in range(len(parameters['delta_perimeter']))]
        parameters['velocity'] = self.differentiate_parameter(parameters['smoothed_angle'], parameters['time'],
                                                              self.velocity_stride, do_smooth=False,
                                                              abs_threshold=None)
        if self.do_velocity_filter:
            parameters["interpolated_velocity"] = self.get_interpolated_velocity(parameters)

        # Uncomment when debugging parameter values.
        # Logger.debug("[MLT_Analysis - get_parameters] List of Parameter values:"
        #              + "\nParameters[\"angle\"] = {0}".format(parameters['angle'])
        #              + "\nParameters[\"velocity\"] = {0}".format(parameters['velocity'])
        #              + "\nParameters[\"size\"] = {0}".format(parameters['size']))

        # Save data and return parameters
        self.path_man.saveParameters(parameters, c.threshold_ratio, spot_index, self.parameters_path)

        return parameters

    def get_paramters(self, snapshot_list, thresholds, spot_indices, calculate_missing=None):
        """Try to load parameter files from disk, if a file is not available, queue it to be made. This procedure makes
        use of MPI to get things done faster."""
        if calculate_missing is None:
            calculate_missing = self.calculate_missing_params
        parameters_todo = []
        parameters_done = [{} for threshold in thresholds]
        for k in range(len(thresholds)):
            for spot_index in spot_indices:
                loaded_parameters = self.path_man.loadParameters(thresholds[k], spot_index)
                if loaded_parameters is None:
                    # add to "to-do" list
                    parameters_todo.append([k,spot_index])
                else:
                    # add to done list
                    parameters_done[k][spot_index] = loaded_parameters
                    Logger.log("[MLT_Analysis - get_parameters] {0}_{1} loaded!".format(spot_index, thresholds[k]))

        # If not all parameters saved to disk gotta do the long arduous task of getting all the cluster data.
        if len(parameters_todo) > 0 and calculate_missing:
            # No quick way of doing this yet
            Logger.log("[MLT_Analysis - get_paramters] {0} cases could not be loaded and will have to be calculated.".format(len(parameters_todo)))
            all_cluster_lists = self.multi_track_clusters(snapshot_list, thresholds)
            all_cluster_lists = self.sort_all_cluster_list(all_cluster_lists, thresholds)
            if self.export_clusters_to_txt:
                self.path_man.export_clusters_to_text(all_cluster_lists, thresholds,
                                                      list(range(self.number_of_clusters_to_track)))
            # Start splitting up the tasks amoung cpus and getting parameters done.
            start, stop = SpotTools.divide_tasks(self, len(parameters_todo), self.size)
            prog = PrintProgress(start, stop,
                                 label="[MLT_Analysis - get_parameters] "
                                       + "Node {0} applying to spots {1} to {2} Progress: ".format(self.rank, start, stop))
            for i in range(start, stop):
                k,spot_index = parameters_todo[i]
                cluster_params = self.get_cluster_params(all_cluster_lists, k, spot_index, thresholds)
                if cluster_params is not None:
                    parameters_done[k][spot_index] = cluster_params
            self.comm.allgather(parameters_done)
        elif not self.calculate_missing_params:
            Logger.log("[MLT_Analysis - get_parameters] Calculate_missing_parameters is set to False. "
                       + "{0} parameters were not loaded and will not be calculated.".format(len(parameters_todo)))

        return parameters_done

    def get_parameter_at_time(self, parameter_slice, timestamp, parameters_to_get):
        """Given a slice of a parameter variable (i.e. parameters[k][spot_index]), returns the values of the elements in
        paramters_to_get if there is a cluster at the timestamp."""
        # Check if there is a match, return none if there are no clusters at the timestamp.
        search_result = np.where(np.array(parameter_slice['time']) == timestamp)
        Logger.debug("[MLT_Analysis - get_parameter_at_time]"
                     + " search_result: {0}".format(search_result)
                     + " timestamp: {0}".format(timestamp)
                     + " type(timestamp): {0}".format(type(timestamp))
                     + " type(parameter_slice['time']): {0}".format(type(parameter_slice['time']))
                     + " parameter_slice['time'][0]: {0}".format(parameter_slice['time'][0])
                     + " type(parameter_slice['time'][0]): {0}".format(type(parameter_slice['time'][0])))
        if len(search_result[0]) == 0:
            # returned as an array of None's when there is more than one variable requested.
            return None if len(parameters_to_get) ==1 else [None for element in parameters_to_get]
        results = []
        for parameter in parameters_to_get:
            results.append(parameter_slice[parameter][search_result[0][0]])
        return results[0] if len(results) == 1 else results


    def fix_wrapping(self, angle, last_angle):
        """Stop wrapping around -90/90 and instead wrap around -180/180"""
        if angle is None or last_angle is None:
            return angle
        difference_too_large = abs(last_angle - angle) > self.param_tolerances['velocity']
        if last_angle < 0 and angle > 0 and difference_too_large:
            return angle - 180
        if last_angle > 0 and angle < 0 and difference_too_large:
            return angle + 180
        else:
            return angle


    def angles_on_axis(self, plt, cluster_list, layer, plot_index):
        """Given pyplot window, plot the graph of major axis angle against time for all clusters that are large
        enough."""
        count_plotted = 0
        for j in range(0, len(cluster_list)):
            parameters = self.get_cluster_params(cluster_list[j])
            parameters['time'] = matdates.date2num(parameters['time'])

            # Skip if the size is too small
            try:
                if np.mean(parameters['size']) < self.minimum_cluster_size:
                    continue
            except IndexError:
                continue

            # Plot the cluster
            plt.plot_date(parameters['time'], parameters['angle'], color=SpotTools.get_colour(count_plotted),
                          linestyle=SpotTools.get_line_style(count_plotted), marker=None)
            count_plotted += 1

        date_formatter = matdates.DateFormatter('%d/%m')
        # ax = plt.axes()
        # ax.xaxis.set_major_formatter(date_formatter)
        layer_colour = ["Green", "Blue", "Purple", "Pink"]
        plt.title("Clusters at threshold {0}I ({1})".format(layer, layer_colour[plot_index]))
        plt.xlabel("Time")
        plt.ylabel("Major axis angle (deg)")

    def get_roi_intensities(self, sunspot_list):
        intensities = {}
        intensities['average'] = []
        intensities['darkest'] = []
        intensities['brightest'] = []
        for i in range(0, len(sunspot_list)):
            roi = self.path_man.loadROI(sunspot_list[i].ROI_path)
            intensities['average'].append(np.mean(roi.data))
            intensities['darkest'].append(np.min(roi.data))
            intensities['brightest'].append(np.max(roi.data))
        return intensities

    def plot_on_single_graph(self, ax, cluster_list, parameters, layer, plot_type, colour_map, plot_index, spot_index, intensities=None,
                             colour_style='normal', timestamp=None):
        """Plots the angle data for the cluster list on plt. Designed to be plot on the same axes as other graphs."""
        count_plotted = 0
        if parameters is None:
            Logger.log("[MLT_Analysis - plot_on_single_graph] Parameters are None. No data to plot.")
            return parameters
        try:
            y_limits = self.graph_y_limits[plot_index]
            if y_limits == []:
                y_limits = 'dynamic'
        except IndexError:
            y_limits = 'dynamic'

        # Define end. for some unknown reason.
        # UPDATE: Think this is a workaround for some archaic code that likely has significant historical
        # and anthropological value to society. However, now cluster_list is just a list of 1 element so it's just a
        # change that i'm not committed to enough to properly change.
        end = len(cluster_list)
        #end = 1
        if plot_type == 'velocity':
            end = 1
        for j in range(0, end):
            # Define a new dict for the variables to use, so we don't end up modifying the input dict.
            # Update: Using Deepcopy because i forgot its really hard to explicitly copy a dict in python
            modified_parameters = copy.deepcopy(parameters)

            Logger.debug("[MLT_Analysis - plot_on_single_graph] "
                         + "len(parameters['time']): {0}\n".format(len(modified_parameters['time']))
                         + "len(parameters['angle']): {0}\n".format(len(modified_parameters['angle']))
                         + "len(parameters['size']): {0}\n".format(len(modified_parameters['size']))
                         + "len(parameters['velocity']): {0}\n".format(len(modified_parameters['velocity']))
                         + "len(parameters['ellipticity']): {0}\n".format(len(modified_parameters['ellipticity'])))

            # Get the time from datetime to the matplotlib format.
            try:
                modified_parameters['time'] = matdates.date2num(modified_parameters['time'])
            except AttributeError as e:
                Logger.debug("[MLT_Analysis - plot_on_single_graph] Attribute error found. Parameters['time']:\n{0}.".format(parameters['time']))
                raise e

            # Normalise angles? - only do after all other computation involving angles!
            if self.plot_stack_separation and len(modified_parameters['angle']) != 0:
                modified_parameters['angle'] = (np.array(modified_parameters['angle'])
                                       + self.plot_stack_separation * count_plotted).tolist()

            # Little hacky work around to get proper roi intensities without having to complete code something new
            if intensities is not None:
                modified_parameters['roi_darkest_intensity'] = intensities['darkest']

            # Skip if the spot cluster is too far away from the specified region centre
            try:
                if np.mean(modified_parameters['distance']) > self.max_distance_from_centre:
                    continue
            except IndexError:
                continue

            # Skip if spot cluster too small.
            try:
                if np.mean(modified_parameters['size']) < self.minimum_cluster_size:
                    continue
            except IndexError:
                continue

            # Set up colours
            if plot_type == 'roi_darkest_intensity':
                colour = 'black'
            else:
                colour = colour_map.get_rgb(layer)

            if colour_style == 'faded':
                colour = SpotTools.ensure_colour_faded(colour)

            # Do plots
            if len(modified_parameters[plot_type]) == 0:
                Logger.log("[MLT_Analysis - plot_on_single_graph] Parameter {0} has no values".format(plot_type))

            Logger.debug("[MLT_Analysis - plot_on_single_graph] plot_type: {0}".format(plot_type))
            if plot_type == "centre":
                centre_x_coords = []
                centre_y_coords = []
                for centre in modified_parameters["centre"]:
                    centre_x_coords.append(centre[0])
                    centre_y_coords.append(centre[1])
                ax.plot(centre_x_coords, centre_y_coords, color=colour,
                              linestyle=SpotTools.get_line_style(count_plotted, switch_on_iterations=False),
                              marker=None, label="{0}%".format(round(layer*100,1)))
                ax.set_xlabel(self.y_axis_labels[plot_type])
                # ax.set_xlim(self.viewport_ranges[0][0], self.viewport_ranges[0][1])
                # ax.set_ylim(self.viewport_ranges[1][0], self.viewport_ranges[1][1])
                # prevent matplotlib.date2num messing up the axes
                ax.xaxis.set_major_formatter(tkr.ScalarFormatter())
                timestamp = None
            elif plot_type == "interpolated_velocity":
                ax.plot_date(modified_parameters['interpolated_time'], modified_parameters[plot_type], color=colour,
                              linestyle=SpotTools.get_line_style(count_plotted, switch_on_iterations=False),
                              marker=None, label="{0}%".format(round(layer*100,1)))
                ax.set_xlim(matdates.date2num(self.start_date), matdates.date2num(self.end_date))
                try:
                    x_axis_label = self.time_axis_labels[self.time_label]
                except KeyError:
                    Logger.log("[MLT_Analysis - plot_on_single_graph] WARNING! x axis label \'{0}\' not recognised. Using default.".format(self.time_label))
                    x_axis_label = "%d/%m %H:%M"
                ax.xaxis.set_major_formatter(matdates.DateFormatter(x_axis_label))
                # Scrobbler
                if timestamp is not None:
                    Logger.debug("[MLT_Analysis - plot_on_single_graph] axvline"
                                 + " plot_index {0}".format(plot_index)
                                 + " spot_index {0}".format(spot_index)
                                 + " threshold {0}".format(layer)
                                 + " timestamp {0}".format(timestamp))
                    ax.axvline(matdates.date2num(timestamp), color='black')
                # y=0 line
                ax.axhline(c='black', lw=1)
                # Plot flare times
                if self.flare_times is not None:
                    for time in self.flare_times:
                        ax.axvline(matdates.date2num(time), lw=1, color="red")
            elif plot_type == "intercluster_distance":
                ydata = modified_parameters[plot_type]
                for key in ydata:
                    ax.plot_date(modified_parameters['time'], ydata[key] - ydata[key][0], color=SpotTools.colourListViridis[key],
                                  linestyle=SpotTools.get_line_style(count_plotted, switch_on_iterations=False),
                                  marker='o', markevery=[0], label="{0}%".format(key))
                ax.set_xlim(matdates.date2num(self.start_date), matdates.date2num(self.end_date))
                try:
                    x_axis_label = self.time_axis_labels[self.time_label]
                except KeyError:
                    Logger.log("[MLT_Analysis - plot_on_single_graph] WARNING! x axis label \'{0}\' not recognised. Using default.".format(self.time_label))
                    x_axis_label = "%d/%m %H:%M"
                ax.xaxis.set_major_formatter(matdates.DateFormatter(x_axis_label))
                # Scrobbler
                if timestamp is not None:
                    Logger.debug("[MLT_Analysis - plot_on_single_graph] axvline"
                                 + " plot_index {0}".format(plot_index)
                                 + " spot_index {0}".format(spot_index)
                                 + " threshold {0}".format(layer)
                                 + " timestamp {0}".format(timestamp))
                    ax.axvline(matdates.date2num(timestamp), color='black')
                # y=0 line
                ax.axhline(c='black', lw=1)
                # Plot flare times
                if self.flare_times is not None:
                    for time in self.flare_times:
                        ax.axvline(matdates.date2num(time), lw=1, color="red")
                if self.cme_times is not None:
                    for time in self.cme_times:
                        ax.axvline(matdates.date2num(time), lw=1, color="red", linestyle='dotted')
            else:
                ax.plot_date(modified_parameters['time'], modified_parameters[plot_type], color=colour,
                              linestyle=SpotTools.get_line_style(count_plotted, switch_on_iterations=False),
                              marker=None, label="{0}%".format(round(layer*100,1)))
                ax.set_xlim(matdates.date2num(self.start_date), matdates.date2num(self.end_date))
                try:
                    x_axis_label = self.time_axis_labels[self.time_label]
                except KeyError:
                    Logger.log("[MLT_Analysis - plot_on_single_graph] WARNING! x axis label \'{0}\' not recognised. Using default.".format(self.time_label))
                    x_axis_label = "%d/%m %H:%M"
                ax.xaxis.set_major_formatter(matdates.DateFormatter(x_axis_label))
                # Scrobbler
                if timestamp is not None:
                    Logger.debug("[MLT_Analysis - plot_on_single_graph] axvline"
                                 + " plot_index {0}".format(plot_index)
                                 + " spot_index {0}".format(spot_index)
                                 + " threshold {0}".format(layer)
                                 + " timestamp {0}".format(timestamp))
                    ax.axvline(matdates.date2num(timestamp), color='black')
                # y=0 line
                ax.axhline(c='black', lw=1)
                # Plot flare times
                if self.flare_times is not None:
                    for time in self.flare_times:
                        ax.axvline(matdates.date2num(time), lw=1, color="red")
                if self.cme_times is not None:
                    for time in self.cme_times:
                        ax.axvline(matdates.date2num(time), lw=1, color="red", linestyle='dotted')

            # Modifications to finished plot
            ax.set_ylabel("[C{0}] ".format(spot_index) + self.y_axis_labels[plot_type])
            if y_limits != 'dynamic':
                ax.set_ylim(y_limits[0], y_limits[1])
            count_plotted += 1

            # returns the parameters calculated of last cluster so that they can be numbered in the right place on the
            # roi plot.
            return parameters

    def ranges_from_list_indices(self, index_list, data=None, return_values=False):
        """
        Returns a list of index pairs that mark the start/end of ranges within the input list.
        e.g turns: [4,5,6,7,80,90,91,92,96] into [[4,7],[80,81],[90,93],[96,97]].

        Args:
            index_list: list<int>. Indexes in data set "data".
            data: list. Data set. Only required if return_values is True
            return_values: boool. if True then returns the data values instead of just indices. Data must be supplied.

        Returns: list<[start,stop]>. A list of start and stop values pairs (also a list).
        i.e. [[4,7],[80,81],[90,93],[96,97]]

        """
        if len(index_list) == 0:
            return None
        list_of_ranges = []
        sublist = [index_list[0],]
        Logger.debug("[MLT_Analysis - ranges_from_list_indices] initial list of ranges: {0}".format(sublist))
        for i in range(1, len(index_list)):
            if index_list[i] != index_list[i-1] + 1:
                sublist.append(index_list[i-1])
            if len(sublist) == 2:
                list_of_ranges.append(sublist)
                sublist = [index_list[i]]
            Logger.debug("[MLT_Analysis - ranges_from_list_indices] New index found,"
                         + " sublist: {0}, list_of_ranges: {1}".format(sublist,list_of_ranges))
        # if we end on an odd number, make it it's own small range and add to the list.
        if len(sublist) == 1:
            sublist.append(sublist[0])
            list_of_ranges.append(sublist)
        # If two numbers in the sublists are the same, then make the range at least 1 wide.
        for element in list_of_ranges:
            if element[0] == element[1]:
                element[1] = element[0] - 1 if element[0] - 1 >= 0 else 0
        Logger.debug("[MLT_Analysis - ranges_from_list_indices] Final list of indices: {0}".format(list_of_ranges))
        # If the values are wanted instead of indexes, do that.
        if return_values:
            for element in list_of_ranges:
                element[0] = data[element[0]]
                element[1] = data[element[1]]
        return list_of_ranges

    def find_datetimes_in_list(self, list_datetimes, datetimes_to_find, return_indices=True):
        """
        Given a list of datetimes, find all datetimes in the datetimes_to_find list that exist within the listed
        datetimes. Returns a list of indices of list_datetimes that is the nearest datetime match.

        Args:
            return_indices: boo. If True returns the index in the original list, if false returns the value
            list_datetimes: list<datetime>. Input list of datetimes.
            datetimes_to_find: list<datetime>. The datetimes to find.

        Returns:

        """
        list_indices = []
        for i in range(len(datetimes_to_find)):
            for ii in range(len(list_datetimes) - 1):
                date_start = list_datetimes[ii]
                date_end = list_datetimes[ii + 1]
                if date_end > datetimes_to_find[i] > date_start:
                    if return_indices:
                        list_indices.append(list_datetimes.index(date_start))
                    else:
                        list_indices.append(date_start)
        if len(list_indices) == 0:
            return None
        return list_indices

    def plot_average_parameter(self, plt, cluster_list, plot_type, plot_index):
        """A different implementation of plot_on_single_graph that allows for the average of parameters in different
        layers to be plotted."""
        # Init
        count_plotted = 0
        if plot_index == 0:
            y_limits = self.graph_1_y_limits
        else:
            y_limits = self.graph_2_y_limits
        end = len(cluster_list)
        # if plot_type == 'velocity':
        #     end = 1
        if plot_type == 'average_velocity':
            plot_type = 'velocity'

        # Get average Parameter
        layer_params = []
        for j in range(0, end):
            parameters = self.get_cluster_params(cluster_list[j][0], normalise=self.angles_normalise,
                                                 separation=self.plot_stack_separation * count_plotted)

            parameters['time'] = matdates.date2num(parameters['time'])
            layer_params.append(parameters)

        average_parameter = []
        for i in range(0, len(layer_params[0]['time'])):
            params = []
            for j in range(0, len(layer_params)):
                params.append(layer_params[j][plot_type][i])
            average_parameter.append(np.mean(params))

        # Do plots
        colour = 'black'
        plt.plot_date(parameters['time'], average_parameter, color=colour,
                      linestyle=SpotTools.get_line_style(count_plotted, switch_on_iterations=False),
                      marker=None, label="average {0}".format(plot_type))
        plt.xlim(matdates.date2num(self.start_date), matdates.date2num(self.end_date))
        plt.ylabel(self.y_axis_labels[plot_type])
        if y_limits != 'dynamic':
            plt.ylim(y_limits[0], y_limits[1])
        count_plotted += 1

    def plot_polar_perimeter(self, plt, cluster_list, layer, bounds, colour_map):
        for i in range(bounds[0], bounds[1]):
            cluster = cluster_list[0][i]
            perimeter_cart = Contours.getPerimeter(cluster.points)
            perimeter_cart = Contours.reformat_perimeter(perimeter_cart)
            perimeter_pol = Contours.cart2polData(perimeter_cart[0], perimeter_cart[1], cluster.centre)
            perimeter_pol = perimeter_pol.tolist()
            perimeter_pol.sort(key=lambda x: x[1])
            perimeter_pol = Contours.reformat_perimeter(perimeter_pol)

            plt.plot(np.array(perimeter_pol[1]),
                     np.array(perimeter_pol[0]) + ((i-bounds[0]) * self.plot_stack_separation),
                     c=colour_map.get_rgb(layer))

    def plot_roi_on_plt(self, plt, fig, snapshot, thresholds, colourmap):
        """Used to plot an ROI showing a sunspot onto a plt object."""
        # Load ROI
        try:
            roi = self.path_man.loadROI(snapshot.ROI_path)
        except:
            return False

        # Plot background
        plt.imshow(roi.data, cmap='inferno', aspect='auto')

        # Load MLT
        try:
            mlt = self.path_man.loadMLT(snapshot.mlt_path)
        except:
            return False

        # Plot contours for the given thresholds
        layers = [l for l in mlt.layers if l.threshold_ratio in thresholds]
        Logger.log("number of layers: {0}".format(len(layers)), Logger.LogLevel.debug)

        for j in range(0, len(layers)):
            layer = layers[j]
            if layer is None:
                continue
            for cluster in layer.mlt_clusters:
                if len(cluster.points) < 10:
                    # Reject because too small
                    continue

                # Attempt at plotting only the perimeter
                perimeter = Contours.getPerimeter(cluster.points)
                # change perimeter from a list of [[x,y], ...] coords to a list of [[x],[y]] coords.
                perimeter = [list(x) for x in zip(*perimeter)]                
                plt.scatter(perimeter[0], perimeter[1], c=[colourmap.get_rgb(layer.threshold_ratio)],
                            label=str(layer.threshold_ratio) + r"$I_{quiet sun}$",
                            marker='s', s=(72. / fig.dpi) ** 2)

                # Getting Ellipse fit
                elli, xx, yy = MLT.MultiLevelThresholding.fit_ellipse(perimeter[0], perimeter[1])
                if elli is None:
                    continue

                # Record parameters and plot.
                # cluster.ellipse_parameters = elli
                ellipse_colour = colourmap.get_rgb(layer.threshold_ratio)
                ellipse_colour = SpotTools.ensure_colour_bright(ellipse_colour)
                plt.plot(xx, yy, c=ellipse_colour, ms=(72. / fig.dpi))
                

        # Set up Contour plot axes
        plt.xlim(self.viewport_ranges[0])
        plt.ylim(self.viewport_ranges[1])
        plt.xlabel("Distance (pix)")
        plt.ylabel("Distance (pix)")
        return True

    def plot_clusters_on_roi(self, ax, fig, snapshot, clusters, colourmap):
        """Used to plot an ROI showing a sunspot onto a plt object."""
        # Load ROI
        try:
            roi = self.path_man.loadROI(snapshot.ROI_path)
        except:
            return False

        #ax = plt.axes()

        # Plot background
        ax.imshow(roi.data, cmap=self.colour_map_spot, aspect='auto')

        # Plot contours for the given clusters
        for cl in clusters:
            cluster = self.path_man.get_cluster_from_id(cl)
            if cluster is None:
                Logger.log("[MLT_Analysis - plot_clusters_on_roi] WARN: Cluster \'{0}\' returned as None".format(cl))
                continue
            if len(cluster.points) < 10:
                # Reject because too small
                continue

            # Attempt at plotting only the perimeter
            perimeter = Contours.getPerimeter(cluster.points)
            # TODO Make this into arcsec.
            # change perimeter from a list of [[x,y], ...] coords to a list of [[x],[y]] coords.
            perimeter = [list(x) for x in zip(*perimeter)]
            ax.scatter(perimeter[0], perimeter[1], c=[colourmap.get_rgb(cluster.threshold_ratio)],
                        label=str(round(cluster.threshold_ratio*100)) + "%",
                        marker='s', s=(72. / fig.dpi) ** 2)

            # Getting Ellipse fit
            elli, xx, yy = MLT.MultiLevelThresholding.fit_ellipse(perimeter[0], perimeter[1])
            if elli is None:
                continue

            # Record parameters and plot.
            # cluster.ellipse_parameters = elli
            ellipse_colour = colourmap.get_rgb(cluster.threshold_ratio)
            ellipse_colour = SpotTools.ensure_colour_bright(ellipse_colour)
            ax.plot(xx, yy, c=ellipse_colour, ms=(72. / fig.dpi))

        # Set up Contour plot axes
        ax.set_xlim(self.viewport_ranges[0])
        ax.set_ylim(self.viewport_ranges[1])
        if self.output_format == 'print':
            xticks = ax.get_xticklabels()
            Logger.debug("[ROI_Plot] x-axis tick labels: {0}".format(xticks))
            xtick_range = self.viewport_ranges[0][1] * roi.pixel_scale[0] -\
                          self.viewport_ranges[0][0] * roi.pixel_scale[0]
            xtick_sep = round(xtick_range / len(xticks))
            xtick_start = round(roi.centre_arcsec[0] - xtick_range/2)
            new_xticks = []
            for i in range(len(xticks)):
                new_xticks.append(xtick_start + xtick_sep * i)

            yticks = ax.get_yticklabels()
            ytick_range = self.viewport_ranges[1][1] * roi.pixel_scale[1] -\
                          self.viewport_ranges[1][0] * roi.pixel_scale[1]
            ytick_sep = round(ytick_range / len(yticks))
            ytick_start = round(roi.centre_arcsec[1] - ytick_range/2)
            new_yticks = []
            for i in range(len(yticks)):
                new_yticks.append(ytick_start + ytick_sep * i)

            ax.set_xticklabels(new_xticks)
            ax.set_yticklabels(new_yticks)
            ax.set_xlabel(self.y_axis_labels["roi_print"])
            ax.set_ylabel(self.y_axis_labels["roi_print"])
        else:
            ax.set_xlabel(self.y_axis_labels["roi_screen"])
            ax.set_ylabel(self.y_axis_labels["roi_screen"])
        return True

    def velocity_against_area_distribution(self, snapshot_list, thresholds):
        """Make a rainbow graph showing all the velocities for each cluster based on distance from centre"""
        colour_map = SpotTools.MplColorHelper(self.colour_map_plots, self.colour_map_limits[0], self.colour_map_limits[1])
        if self.output_format == "print":
            plt.rcParams.update({'font.size': self.fig_font_size})
        # Get the clusters
        all_cluster_lists = []
        prog_get_clusters = PrintProgress(0, len(thresholds), label="[MLT_Analysis] Tracking clusters...")
        for threshold in thresholds:
            if self.tracking_method == "area":
                tracked_cluster = self.track_constant_area(snapshot_list, threshold,
                                                           self.max_separation_distance, cluster_to_track=0)
                all_cluster_lists.append([tracked_cluster])
            elif self.tracking_method == "threshold":
                all_cluster_lists.append(self.track_clusters(snapshot_list, threshold,
                                                             self.max_separation_distance, self.max_time_delta))
            prog_get_clusters.update()

        # There will be one entry in the parameter list below for each layer in the spot.
        layer_params = []
        for j in range(0, len(all_cluster_lists)):
            parameters = self.get_cluster_params(all_cluster_lists[j][0], normalise=self.angles_normalise,
                                                 separation=0)
            parameters['time'] = matdates.date2num(parameters['time'])
            layer_params.append(parameters)

        Logger.debug("[Velocity Distribution] Number of layers: {0}".format(len(layer_params)))
        # Do plot
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        flare_mean_radius = []
        flare_mean_velocity = []
        preflare_mean_radius = []
        preflare_mean_velocity = []
        postflare_mean_radius = []
        postflare_mean_velocity = []
        #nonflare_mean_radius = []
        #nonflare_mean_velocity = []
        # TODO: Make this into a configuration setting
        flare_start_time = datetime.datetime.strptime("2011-02-15_01-47-32", "%Y-%m-%d_%H-%M-%S")
        flare_end_time = datetime.datetime.strptime("2011-02-15_04-20-32", "%Y-%m-%d_%H-%M-%S")
        # For each threshold level (0 -> len(layer_params)), plot the equivalent radius and velocity, ignoring the
        # first self.velocity_stride number of entries that will be 0.
        for j in range(0, len(layer_params)):
            colour = colour_map.get_rgb(thresholds[j])
            preflare_times = layer_params[j]['time'][np.where(layer_params[j]['time'] < matdates.date2num(flare_start_time))]
            postflare_times = layer_params[j]['time'][np.where(layer_params[j]['time'] > matdates.date2num(flare_end_time))]
            nonflare_radius_mean = 0.
            nonflare_velocity_mean = 0.
            
            # Pre-flare plots
            plt.scatter(layer_params[j]['equivalent_radius'][self.velocity_stride:len(preflare_times)],
                        layer_params[j]['velocity'][self.velocity_stride:len(preflare_times)],
                        color=colour,
                        #edgecolor='black',
                        marker='o')
            preflare_mean_radius.append(np.mean(np.array(layer_params[j]['equivalent_radius'][self.velocity_stride:len(preflare_times)])))
            preflare_mean_velocity.append(np.mean(np.array(layer_params[j]['velocity'][self.velocity_stride:len(preflare_times)])))
            
            # During flare
            plt.scatter(layer_params[j]['equivalent_radius'][len(preflare_times):-len(postflare_times)],
                        layer_params[j]['velocity'][len(preflare_times):-len(postflare_times)],
                        color=colour,
                        edgecolor='black',
                        marker='X')
            flare_mean_radius.append(np.mean(np.array(layer_params[j]['equivalent_radius'][len(preflare_times):-len(postflare_times)])))
            flare_mean_velocity.append(np.mean(np.array(layer_params[j]['velocity'][len(preflare_times):-len(postflare_times)])))
            
            # Post flare
            plt.scatter(layer_params[j]['equivalent_radius'][-len(postflare_times):],
                        layer_params[j]['velocity'][-len(postflare_times):],
                        color=colour,
                        #edgecolor='black',
                        marker='^')
            postflare_mean_radius.append(np.mean(np.array(layer_params[j]['equivalent_radius'][-len(postflare_times):])))
            postflare_mean_velocity.append(np.mean(np.array(layer_params[j]['velocity'][-len(postflare_times):])))
            
            #nonflare_mean_radius.append(nonflare_radius_mean/2.)
            #nonflare_mean_velocity.append(nonflare_velocity_mean/2.)
            
            
        # Plot Averages
        #plt.plot(mean_radius, mean_velocity, color='black', marker='x')
        plt.plot(preflare_mean_radius, preflare_mean_velocity, color='black', linestyle='dashed')
        plt.plot(postflare_mean_radius, postflare_mean_velocity, color='black', linestyle='dotted')
        plt.plot(flare_mean_radius, flare_mean_velocity, color='black')
        plt.axhline(color='black',lw=1)
        # Format plot
        plt.xlabel("Radius of Equivalent Circle (arcsec)")
        plt.ylabel(r"Major Axis Velocity $\omega$ ($\circ h^{-1}$)")
        plt.title("Major Axis Velocity Distribution between {0} UT - {1} UT".format(
                   datetime.datetime.strftime(self.start_date, "%H:%M"),
                   datetime.datetime.strftime(self.end_date, "%H:%M")
                   )
        )
        fig.tight_layout()
        plt.savefig(str(self.output_path / "velocity_area_distribution.png"))
        Logger.log("[MLT_Analysis] Velocity-Area distribution plotted!")

    def velocity_against_area(self, snapshot_list, thresholds):
        """Plot a line graph of velocity against area to create an animation of the rotation rates over time"""
        colour_map = SpotTools.MplColorHelper(self.colour_map_plots, self.colour_map_limits[0], self.colour_map_limits[1])
        if self.output_format == "print":
            plt.rcParams.update({'font.size': self.fig_font_size})
        # Get the clusters
        all_cluster_lists = []
        prog_get_clusters = PrintProgress(0, len(thresholds), label="[MLT_Analysis] Tracking clusters...")
        for threshold in thresholds:
            if self.tracking_method == "area":
                tracked_cluster = self.track_constant_area(snapshot_list, threshold,
                                                           self.max_separation_distance, cluster_to_track=0)
                all_cluster_lists.append([tracked_cluster])
            elif self.tracking_method == "threshold":
                all_cluster_lists.append(self.track_clusters(snapshot_list, threshold,
                                                             self.max_separation_distance, self.max_time_delta))
            prog_get_clusters.update()

        layer_params = []
        for j in range(0, len(all_cluster_lists)):
            parameters = self.get_cluster_params(all_cluster_lists[j][0], normalise=self.angles_normalise,
                                                 separation=0)
            parameters['time'] = matdates.date2num(parameters['time'])
            layer_params.append(parameters)

        # Do plot
        prog_plot = PrintProgress(0, len(snapshot_list), label="[MLT_Analysis] Plotting velocity-area plots")
        plot_history = 5
        for i in range(0, len(snapshot_list)):
            fig = plt.figure(figsize=(16, 9), dpi=90)
            filename = snapshot_list[i].timestamp.strftime('%Y-%m-%d_%H-%M-%S')
            y_limits = 'dynamic' if self.graph_y_limits[0] == [] else self.graph_y_limits[0]

            # Plot the previous plot_history number of plots in a faded colour.
            for k in range(1, plot_history):
                if i - k < 0:
                    continue
                layer_eqiv_circ_rad = []
                layer_velocities = []
                for j in range(0, len(layer_params)):
                    layer_eqiv_circ_rad.append(layer_params[j]['equivalent_radius'][i - k])
                    layer_velocities.append(layer_params[j]['velocity'][i - k])
                # if self.velocity_size_normalise:
                #     layer_velocities = np.array(layer_velocities) - layer_velocities[-1]
                # Gotta convert the hex code int he string to an actual colour object.
                colour = SpotTools.ensure_colour_faded(cols.to_rgba(mcd.XKCD_COLORS['xkcd:light grey']))
                plt.plot(layer_eqiv_circ_rad, layer_velocities, color=colour, marker='o')

            # Plot this layer's line
            layer_eqiv_circ_rad = []
            layer_velocities = []
            for j in range(0, len(layer_params)):
                layer_eqiv_circ_rad.append(layer_params[j]['equivalent_radius'][i])
                layer_velocities.append(layer_params[j]['velocity'][i])
            # if self.velocity_size_normalise:
            #     layer_velocities = np.array(layer_velocities) - layer_velocities[-1]
            colour = 'black'
            plt.plot(layer_eqiv_circ_rad, layer_velocities, color=colour, marker='o')

            # Format plot
            if y_limits != 'dynamic':
                plt.ylim(y_limits[0], y_limits[1])
            plt.xlim(4,13)
            plt.xlabel("Radius of Equivalent Circle (pixels)")
            plt.ylabel("Major Axis Velocity (deg per hour)")
            plt.title(snapshot_list[i].timestamp)
            fig.tight_layout()
            plt.savefig(str(self.velocity_area_graph_path / (filename + ".png")), transparent=True)
            plt.close(fig)
            prog_plot.update()

    def get_clusters_for_plot(self, snapshot_list, thresholds):
        """Returns a list of lists that contain the clusters tracked through time. Used to then plot data for each spot"""
        all_cluster_lists = []
        prog_get_clusters = PrintProgress(0, len(thresholds), label="[MLT_Analysis] Getting clusters for plot...")
        for threshold in thresholds:
            if self.tracking_method == "area":
                # Separte list for each cluster. Because of course. Why the @#$% is it done like this?!
                tracked_clusters = []
                for i in range(0,self.number_of_clusters_to_track):
                # tracked_clusters.append(self.track_constant_area(snapshot_list, threshold,
                                                               # self.max_separation_distance, cluster_to_track=0))
                    try:
                        tracked_clusters.append(self.track_constant_area(snapshot_list, threshold,
                                                                         self.max_separation_distance, cluster_to_track=i))
                    except IndexError:
                        Logger.debug("[MLT_Analysis - Area Tracking] Could not find cluster number {0}, skipping...".format(i))

                all_cluster_lists.append(tracked_clusters)
            elif self.tracking_method == "threshold":
                all_cluster_lists.append(self.track_clusters(snapshot_list, threshold,
                                                             self.max_separation_distance, self.max_time_delta))
            prog_get_clusters.update()
        return all_cluster_lists

    def sort_all_cluster_list(self, all_cluster_lists, thresholds):
        """
        Sort the cluster list by cluster lifetime. This needs to be done carefully to make sure the cluster indices
        still correctly correlate with one another.
        Each cluster is given a number which is mapped to it's size (which in turn corresponds to the number of images
        it is seen in - it's "lifetime"). This list of tuples can then be sorted into decending size order.
        However this ordered list cannot distinguish between threshold layers, and so frequently you'll get 1 spot
        taking up multiple consequtive spots in the list - so you can't just take x number from the top of this list.
        Hence a list of used_indices is required and checked before adding clusters to the final sorted list to avoid
        repititions.
        To speed up the code, an upper limit of self.number_of_clusters_to_track is used to break out the loop early,
        as we often don't care for the many thousands of small single frame spots that exist.
        :param all_cluster_lists:
        :param thresholds:
        :return:
        """
        key_length_list = []
        for threshold_index in range(0, len(all_cluster_lists)):
            for cluster_index in range(0, len(all_cluster_lists[threshold_index])):
                key_length_list.append((cluster_index, len(all_cluster_lists[threshold_index][cluster_index])))
        key_length_list.sort(key=lambda e: e[1], reverse=True)
        Logger.debug("[MLT_Analysis sort_all_cluster_list] key_length_list sorted: {0}".format(key_length_list))

        used_indices = []
        new_cluster_list = [[] for threshold in thresholds]
        for i in range(0, len(key_length_list)):
            key = key_length_list[i][0]
            Logger.debug("[MLT_Analysis sort_all_cluster_list] Key: {0} Size: {1}".format(key, key_length_list[i][1]))
            if key not in used_indices:
                Logger.debug("[MLT_Analysis sort_all_cluster_list] New key accepted.")
                for j in range(0, len(thresholds)):
                    try:
                        new_cluster_list[j].append(all_cluster_lists[j][key])
                    except IndexError:
                        Logger.debug("[MLT_Analysis sort_all_cluster_list] Could not find "
                                     + "entry at threshold {0} ".format(thresholds[j])
                                     + "for key {0}.".format(key))
                used_indices.append(key)
            if(len(used_indices) >= self.number_of_clusters_to_track):
                break
        for i in range(len(thresholds)):
            Logger.debug("[MLT_Analysis sort_all_cluster_list] New cluster List: "
                         + " Threshold {0} has {1} clusters".format(thresholds[i], len(new_cluster_list[i])))
        return new_cluster_list


    def single_graph_plot(self, snapshot_list, thresholds, plots):
        """
        Plots the average of the requested parameter for all clusters on the same graph.
        :param snapshot_list:
        :param thresholds:
        :param plots:
        :return:
        """
        colour_map = SpotTools.MplColorHelper(self.colour_map_plots, self.colour_map_limits[0], self.colour_map_limits[1])
        if self.output_format == "print":
            plt.rcParams.update({'font.size': self.fig_font_size})
        colour_style = 'normal'

        # Get clusters
        #all_cluster_lists = self.get_clusters_for_plot(snapshot_list, thresholds)
        # all_cluster_lists = self.multi_track_clusters(snapshot_list, thresholds)
        # all_cluster_lists = self.sort_all_cluster_list(all_cluster_lists, thresholds)
        # if self.export_clusters_to_txt:
        #     self.path_man.export_clusters_to_text(all_cluster_lists, thresholds, list(range(self.number_of_clusters_to_track)))

        # Extract the spots to track and graph types from the config file and store for reference of spot index
        plots_details = []
        for j in range(0, len(plots)):
            spot_index, plot_type = plots[j].split('_', maxsplit=1)
            spot_index = int(spot_index)
            plots_details.append([spot_index, plot_type])
        Logger.debug("[MLT_Analysis - single_graph_plot] plots_details: {0}".format(plots_details))

        # Get parameters for requested clusters.
        #parameters = [{} for threshold in thresholds]
        # Find out the clusters the users want from the config file
        spot_index_set = set()
        for j in range(0, len(plots_details)):
            spot_index, plot_type = plots_details[j]
            spot_index_set.add(spot_index)
        # Construct the parameter list
        # for k in range(0, len(thresholds)):
        #     for spot_index in spot_index_set:
        #         cluster_parameters = self.get_cluster_params(all_cluster_lists, k, spot_index, thresholds)
        #         parameters[k][spot_index] = cluster_parameters

        parameters = self.get_paramters(snapshot_list, thresholds, spot_index_set)

        # Get data for plotting averages. Means going a few for-loops deep to reformat the data.
        # A dictionary is made of matdates as keys and a list of lists taht contain the values at each threshold for
        # each cluster. These can then be averaged at the end for the final plot.
        plot_type = plots_details[0][1]
        spot = []
        for index in spot_index_set:
            times = {}
            spot.append(times)
            for t in range(0,len(thresholds)):
                try:
                    params = parameters[t][index]
                except KeyError:
                    Logger.debug("[MLT_Analysis - single_graph_plot] "
                                 + "Could not find spot index \'{0}\' in layer \'{1}\'.".format(index, t))
                    continue
                for time_index in range(0,len(parameters[t][index]["matplot_time"])):
                    if params['matplot_time'][time_index] in times:
                        times[params['matplot_time'][time_index]].append(params[plot_type][time_index])
                    else:
                        times[params['matplot_time'][time_index]] = [params[plot_type][time_index]]

        # Set up the plot window
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        title = "{0} against time for NOAA {1}".format(plot_type, self.NOAA_number)
        filename = title + ".png"
        plt.title(title)
        ax = fig.add_subplot(111)

        # Plot the data
        for s in range(len(spot)):
            data_values = spot[s]
            for timestamp in data_values:
                data_values[timestamp] = np.mean(data_values[timestamp])
            data_values = dict(sorted(data_values.items(), key=lambda item: item[0]))
            ax.plot_date(data_values.keys(), list(data_values.values()),
                         color=SpotTools.colourListViridis[self.singles_graph_colours[s]],
                         linestyle=self.singles_graph_styles[s],
                         label="Spot {0}".format(s), marker=None, lw=2)

        # Final formatting of plot
        # Y axis limits and label
        try:
            y_limits = self.graph_y_limits[0]
            if y_limits == []:
                y_limits = 'dynamic'
        except IndexError:
            y_limits = 'dynamic'
        if y_limits != 'dynamic':
            ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_ylabel(self.y_axis_labels[plot_type])

        # X-axis limits and label
        ax.set_xlim(matdates.date2num(self.start_date), matdates.date2num(self.end_date))
        try:
            x_axis_label = self.time_axis_labels[self.time_label]
        except KeyError:
            Logger.log(
                "[MLT_Analysis - plot_on_single_graph] WARNING! x axis label \'{0}\' not recognised. Using default.".format(
                    self.time_label))
            x_axis_label = "%d/%m %H:%M"
        ax.xaxis.set_major_formatter(matdates.DateFormatter(x_axis_label))
        ax.set_xlabel("Time (day/month)")
        ax.axhline(color='black',lw=1)

        # Figure labels
        if self.flare_times is not None and self.plot_flare_times:
            for time in self.flare_times:
                ax.axvline(matdates.date2num(time), lw=1, color="red")
        if self.cme_times is not None and self.plot_cme_times:
            for time in self.cme_times:
                ax.axvline(matdates.date2num(time), lw=1, color="red", linestyle='dotted')

        plt.legend()
        plt.tight_layout()

        plt.savefig(str(self.single_graph_path / filename))

    def get_distances_to_clusters(self, parameters_at_threshold, this_index, spot_indices):
        """Finds the distances between centres of all the clusters in the current threshold, returning a dictionary
        with the key being the spot index and the value being a list of distances at the time."""
        distances = {}
        for i in spot_indices:
            distances[i] = []
            for j in range(0,len(parameters_at_threshold[this_index]["time"])):
                try:
                    sqr_dist = get_euclidean_sqr_dist(parameters_at_threshold[this_index]["centre"][j],
                                                           parameters_at_threshold[i]["centre"][j])
                    distances[i].append(np.sqrt(sqr_dist))
                except IndexError:
                     # catch case where one list ends before the other
                    distances[i].append(np.nan)
        return distances


    def multi_parameter_plot(self, snapshot_list, thresholds, plots):
        """
        ##############################################################################################################
        !!!!!! USE THIS ONE FOR PARAMETER PLOTTING!!!!
        ##############################################################################################################
        """
        colour_map = SpotTools.MplColorHelper(self.colour_map_plots, self.colour_map_limits[0], self.colour_map_limits[1])
        if self.output_format == "print":
            plt.rcParams.update({'font.size': self.fig_font_size})
        colour_style = 'normal'

        # Get data for ROI
        # Dunno why this is here, but not touching that hornets nest yet - I'm touching it! 06/03/2020
        if any([t for t in plots if 'roi_darkest_intensity' in t]):
            roi_intensities = self.get_roi_intensities(snapshot_list)
        else:
            roi_intensities = None

        # Extract the spots to track and graph types from the config file and store for reference of spot index
        plots_details = []
        for j in range(0, len(plots)):
            spot_index, plot_type = plots[j].split('_', maxsplit=1)
            spot_index = int(spot_index)
            plots_details.append([spot_index, plot_type])
        Logger.debug("[MLT_Analysis - multi_parameter_plot] plots_details: {0}".format(plots_details))

        # Find out the clusters the users want from the config file
        spot_index_set = set()
        for j in range(0, len(plots_details)):
            spot_index, plot_type = plots_details[j]
            spot_index_set.add(spot_index)

        # Get or load parameter list
        parameters = self.get_paramters(snapshot_list, thresholds, spot_index_set)

        # Not functional. Disabled 2022-01-27
        # Add meta-parameters to list (parameters that can only be obtained with knowledge of other clusters).
        # for k in range(0, len(thresholds)):
        #     for spot_index in spot_index_set:
        #         parameters[k][spot_index]["intercluster_distance"] = self.get_distances_to_clusters(parameters[k], spot_index, spot_index_set)


        prog_make_image = PrintProgress(0, len(snapshot_list),
                                        label="[MLT_Analysis] Plotting multi-parameter graphs...")
        for i in range(0, len(snapshot_list), self.plotting_stride):
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            filename = snapshot_list[i].timestamp.strftime('%Y-%m-%d_%H-%M-%S')
            plt.title(snapshot_list[i].timestamp)

            # Set image layout
            if self.graphs_layout == "vertical_roi":
                # use length of plots[] array to determine how many graphs to make/how big the gridspec should be.
                Logger.debug("[MLT_Analysis - multi_parameter_plot] Setting up graphs for vertical plot...")
                grid_dimensions = (len(plots), 3)
                Logger.debug("[MLT_Analysis - multi_parameter_plot] Grid dimensions: {0}".format(grid_dimensions))
                gridspec.GridSpec(*grid_dimensions)
                graphs = [plt.subplot2grid(grid_dimensions, (x, 1), colspan=2) for x in range(0, grid_dimensions[0])]

                # Handle plotting the ROI image in the left pane.
                ax_roi = plt.subplot2grid(grid_dimensions, (0, 0),
                                          colspan=self.viewport_aspect_ratio[0],
                                          rowspan=self.viewport_aspect_ratio[1])
            elif self.graphs_layout == "horizontal_roi":
                Logger.debug("[MLT_Analysis - multi_parameter_plot] Setting up graphs for horizontal plot...")
                col_count = int(np.ceil(len(plots)/2))
                grid_dimensions = (4, col_count)
                Logger.debug("[MLT_Analysis - multi_parameter_plot] Grid dimensions: {0}".format(grid_dimensions))
                gridspec.GridSpec(*grid_dimensions)
                graphs = [plt.subplot2grid(grid_dimensions, ((x // col_count)+self.viewport_aspect_ratio[1], x % col_count)) for x in range(0, len(plots))]
                #Logger.debug("[MLT_Analysis - multi_parameter_plot] Made {0} axes".format(len(graphs)))
                ax_roi = plt.subplot2grid(grid_dimensions, (0, 0),
                                          colspan=self.viewport_aspect_ratio[0],
                                          rowspan=self.viewport_aspect_ratio[1])

            clusters = []
            # This is so I can plot the clusters on the ROI window. The IDs have to be reconstructed from the parameter
            # list, so we have to find all the indices in the
            for threshold in parameters:
                for spot in threshold.items():
                    centre, threshold_ratio = self.get_parameter_at_time(spot[1], snapshot_list[i].timestamp, ['centre','threshold_ratio'])  #np.where(np.array(spot[1]['time']) == snapshot_list[i].timestamp)
                    if not all(x is None for x in [centre, threshold_ratio]):
                        id = "{0},{1}#{2}#{3}".format(centre[0],centre[1],threshold_ratio,snapshot_list[i].timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"))
                        clusters.append(id)
            Logger.debug("[MLT_Analysis] Cluster list count for ROI plot: {0}".format(len(clusters)))
            roi_plotted = self.plot_clusters_on_roi(ax_roi, fig, snapshot_list[i], clusters, colour_map)
            # If no ROI or some error, just skip to next image.
            if not roi_plotted:
                continue

            # Now doing all the graphs on the side.
            for j in range(0, len(plots)):
                spot_index, plot_type = plots_details[j]
                if plot_type.startswith('avg_'):
                    colour_style = 'faded'
                    do_average_plot = True
                    plot_type = plot_type[4:]
                    parameters_for_average = {}
                else:
                    colour_style = 'normal'
                    do_average_plot = False
                clusters_with_text_plotted = []
                for k in range(0, len(thresholds)):
                    params = None
                    # Catches cases where the layer only has 1 cluster in it and just moves on to next one.
                    try:
                        #Logger.debug("[MLT_Analysis - multi_parameter_plot] i: {3} Plot: {0} Threshold: {1} Snapshot time: {2}".format(j, k, snapshot_list[i].timestamp, i))
                        #cluster list all_cluster_lists[k][spot_index]
                        params = self.plot_on_single_graph(graphs[j], [0],
                                                           parameters[k][spot_index], thresholds[k],
                                                           plot_type, colour_map, plot_index=j, spot_index=spot_index,
                                                           intensities=roi_intensities, colour_style=colour_style,
                                                           timestamp=snapshot_list[i].timestamp)
                        if params is None:
                            Logger.log("[MLT_Analysis - multi_parameter_plot] Params was None for"
                                       + " threshold {0} and spot_index {1} ".format(k, spot_index)
                                       + "after plot_on_single_graph! Attempting to continue with original data...")
                            params = parameters[k][spot_index]
                        if do_average_plot:
                            # collect data, making sure every value is attributed to the correct timestamp
                            for l in range(0,len(params['time'])):
                                if params['time'][l] in parameters_for_average:
                                    parameters_for_average[params['time'][l]].append(params[plot_type][l])
                                else:
                                    parameters_for_average[params['time'][l]] = [params[plot_type][l]]
                    except KeyError:
                        Logger.debug("[MLT_Analysis - multi_parameter_plot]"
                                     + " No spot {0} at threshold {1}. Skipping...".format(spot_index, thresholds[k]))
                        prog_make_image.update(self.plotting_stride)
                    except IndexError as ie:
                        Logger.debug("[MLT_Analysis - multi_parameter_plot] Index error encountered"
                                     + "\nj = {0}".format(j)
                                     + "\nk = {0}".format(k)
                                     + "\nspot_index = {0}".format(spot_index)
                                     + "\nlen(graphs) = {0}".format(len(graphs))
                                     + "\ngraphs = {0}".format(graphs)
                                     + "\nlen(thresholds) = {0}".format(len(thresholds))
                                     + "\nlen(parameters) = {0}".format(len(parameters))
                                     + "\nlen(parameters[0]) = {0}".format(len(parameters[0]))
                                     + "\nlen(parameters[0][0]) = {0}".format(len(parameters[0][0])))
                        raise ie

                    #cluster_centre = np.mean(params['centre'], axis=0)
                    if params is None:
                        continue
                    #Logger.debug("[MLT_Analysis - multi_parameter_plot] typeof(Params): {0} len(params): {1}".format(type(params), len(params)))
                    # Write the cluster number on the ROI window?
                    if self.label_clusters_on_roi:
                        if j not in clusters_with_text_plotted:
                            centre = self.get_parameter_at_time(params, snapshot_list[i].timestamp, ['centre'])
                            if centre is not None:
                                txt = ax_roi.text(centre[0]+10.0,centre[1]-10.0, '{0}'.format(j))
                                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                                clusters_with_text_plotted.append(j)

                    # If highlighting of bad data is required, do that here.
                    if self.highlight_bad_data == "manual" and not do_average_plot:
                        data = list(params['time'])
                        highlight_x_coords = self.find_datetimes_in_list(data, self.bad_data_ranges)
                        highlight_x_coords = self.ranges_from_list_indices(highlight_x_coords)
                        if highlight_x_coords is not None:
                            for timerange in highlight_x_coords:
                                graphs[j].axvspan(params['time'][timerange[0]],
                                                  params['time'][timerange[1]],
                                                  color=SpotTools.ensure_colour_faded(
                                                      colour_map.get_rgb(thresholds[k])),
                                                  alpha=0.5)
                    elif self.highlight_bad_data == "automatic" and not do_average_plot:
                        if plot_type in self.param_tolerances.keys():
                            data = np.array(params[plot_type])
                            # Logger.debug(
                            #     "[MLT_Analysis - plot_on_single_graph] Highlight bad {0} data...".format(plot_type))
                            highlight_x_coords = np.where(abs(data) > self.param_tolerances[plot_type])[0]
                            # Logger.debug("[MLT_Analysis - plot_on_single_graph] highlight_x_coords = {0}".format(
                            #     highlight_x_coords))
                            highlight_x_coords = self.ranges_from_list_indices(highlight_x_coords)
                            # If there are bad points, shade them on the graph
                            if highlight_x_coords is not None:
                                # Logger.debug("[MLT_Analysis - plot_on_single_graph] highlight_x_coords = {0}".format(
                                #     highlight_x_coords))
                                for timerange in highlight_x_coords:
                                    # Logger.debug(
                                    #     "[MLT_Analysis - plot_on_single_graph] timerange = {0}".format(timerange))
                                    graphs[j].axvspan(params['time'][timerange[0]],
                                               params['time'][timerange[1]],
                                               color=SpotTools.ensure_colour_faded(colour_map.get_rgb(thresholds[k])),
                                               alpha=0.5)

                if do_average_plot:
                    for timestamp in parameters_for_average:
                        parameters_for_average[timestamp] = np.mean(parameters_for_average[timestamp])
                    # Ensure time order
                    parameters_for_average = dict(sorted(parameters_for_average.items(), key=lambda item: item[0]))
                    graphs[j].plot_date(parameters_for_average.keys(), list(parameters_for_average.values()),
                                        color='black', marker=None, lw=2, linestyle='-')
                    if self.highlight_bad_data == "manual":
                        # if parameters_for_average[timestamp] is None:
                        #     continue
                        data = list(parameters_for_average.keys())

                        highlight_x_coords = self.find_datetimes_in_list(data,self.bad_data_ranges)
                        Logger.debug("[MLT_Analysis - highlight_bad_data] find_datetimes_in_list: {0}".format(highlight_x_coords))
                        highlight_x_coords = [[data[highlight_x_coords[x-1]], data[highlight_x_coords[x]]] for x in range(1,len(highlight_x_coords))]
                        #highlight_x_coords = self.ranges_from_list_indices(highlight_x_coords, data=data, return_values=True)
                        Logger.debug("[MLT_Analysis - highlight_bad_data] ranges_from_list_indices: {0}".format(highlight_x_coords))
                        if highlight_x_coords is not None:
                            for timerange in highlight_x_coords:
                                Logger.debug("[MLT_Analysis - highlight_bad_data] timerange: {0}".format(timerange)
                                             + " parameters_for_average[timerange[0]] = {0}".format(parameters_for_average[timerange[0]])
                                             + " parameters_for_average[timerange[1]] = {0}".format(parameters_for_average[timerange[1]]))
                                graphs[j].axvspan(parameters_for_average[timerange[0]],
                                                  parameters_for_average[timerange[1]],
                                                  color='black',
                                                  alpha=0.2)
                    elif self.highlight_bad_data == "automatic":
                        if plot_type in self.param_tolerances.keys():
                            data = np.array(parameters_for_average[plot_type])
                            # Logger.debug(
                            #     "[MLT_Analysis - plot_on_single_graph] Highlight bad {0} data...".format(plot_type))
                            # highlight_x_coords = np.where(abs(data) > self.param_tolerances[plot_type])[0]
                            # Logger.debug("[MLT_Analysis - plot_on_single_graph] highlight_x_coords = {0}".format(
                            #     highlight_x_coords))
                            highlight_x_coords = self.ranges_from_list_indices(highlight_x_coords)
                            # If there are bad points, shade them on the graph
                            if highlight_x_coords is not None:
                                # Logger.debug("[MLT_Analysis - plot_on_single_graph] highlight_x_coords = {0}".format(
                                #     highlight_x_coords))
                                for timerange in highlight_x_coords:
                                    # Logger.debug(
                                    #     "[MLT_Analysis - plot_on_single_graph] timerange = {0}".format(timerange))
                                    graphs[j].axvspan(params['time'][timerange[0]],
                                                      params['time'][timerange[1]],
                                                      color='black',
                                                      alpha=0.2)




            #graphs[-1].set_xlabel("Time (Day / HH:MM)")
            fig.colorbar(colour_map.scalarMap)
            # Tidy up, plot, and save
            plt.tight_layout()
            #leg = graphs[0].legend(bbox_to_anchor=(1.05, 1.0), loc=2)
            plt.savefig(str(self.multi_parameter_graph_path / (filename + '.png')))
            plt.close()
            prog_make_image.update(self.plotting_stride)


    def roi_plot(self, snapshot_list, thresholds):
        """Plots only the ROI and clusters. No graphs. Uses self.velocity_stride as a stride for skipping large
         numbers of files"""
        colour_map = SpotTools.MplColorHelper(self.colour_map_plots, self.colour_map_limits[0], self.colour_map_limits[1])

        # Get clusters
        #all_cluster_lists = self.get_clusters_for_plot(snapshot_list, thresholds)
        # all_cluster_lists = self.multi_track_clusters(snapshot_list, thresholds)
        # all_cluster_lists = self.sort_all_cluster_list(all_cluster_lists, thresholds)
        
        # Get Parameters
        parameters = self.get_paramters(snapshot_list, thresholds, range(0, self.number_of_clusters_to_track))


        prog_make_image = PrintProgress(0, len(snapshot_list),
                                        label="[MLT_Analysis] Plotting ROI graphs...")
        for i in range(0, len(snapshot_list), self.plotting_stride):
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            filename = snapshot_list[i].timestamp.strftime('%Y-%m-%d_%H-%M-%S')
            plt.title(snapshot_list[i].timestamp)
            ax = plt.subplot(111)

            # This is so I can plot the clusters on the ROI window. The IDs have to be reconstructed from the parameter
            # list, so we have to find all the indices in the
            clusters = []
            for threshold in parameters:
                for spot in threshold.items():
                    centre, threshold_ratio = self.get_parameter_at_time(spot[1], snapshot_list[i].timestamp, ['centre','threshold_ratio'])  #np.where(np.array(spot[1]['time']) == snapshot_list[i].timestamp)
                    if not all(x is None for x in [centre, threshold_ratio]):
                        id = "{0},{1}#{2}#{3}".format(centre[0],centre[1],threshold_ratio,snapshot_list[i].timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"))
                        clusters.append(id)
            Logger.debug("[MLT_Analysis] Cluster list count for ROI plot: {0}".format(len(clusters)))
            if len(clusters) == 0:
                Logger.debug("[MLT_Analysis - ROI_plot] No clusters were found! Skipping...")
                continue
            roi_plotted = self.plot_clusters_on_roi(ax, fig, snapshot_list[i], clusters, colour_map)
            
            # Plot number of the cluster next to it on the ROI
            if self.label_clusters_on_roi:
                clusters_with_text_plotted = []
                for threshold in parameters:
                    for spot in threshold.items():
                        if spot[0] not in clusters_with_text_plotted:
                            centre = self.get_parameter_at_time(spot[1], snapshot_list[i].timestamp, ['centre'])
                            if centre is not None:
                                txt = ax.text(centre[0] + 10.0, centre[1] - 10.0, '{0}'.format(spot[0]))
                                txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
                                clusters_with_text_plotted.append(spot[0])
            
            #leg = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            fig.colorbar(colour_map.scalarMap)
            fig.tight_layout()
            plt.savefig(str(self.roi_plot_path / (filename + '.png')))
            plt.close(fig)
            prog_make_image.update(self.plotting_stride)


    def plot_cluster_histograms(self, snapshot_list, thresholds):
        """
        For each image, plot a histogram counting the number of clusters found at each threshold layer. Tool to
        help determine the best threshold levels to choose for further analysis.
        :param snapshot_list:
        :param thresholds:
        :return:
        """
        prog_cluster_hist = PrintProgress(0, len(snapshot_list), label="[MLT_Analysis] Plotting cluster histograms...")
        cluster_counts = [[] for threshold in thresholds]
        for i in range(len(snapshot_list)):
            # load
            if snapshot_list[i].mlt_path is None:
                continue
            mlt = self.path_man.loadMLT(snapshot_list[i].mlt_path)

            # Get cluster counts
            for j in range(0, len(thresholds)):
                layer = mlt.find_layer_by_threshold(thresholds[j])
                if layer is not None:
                    cluster_counts[j].append(len(layer.mlt_clusters))
                else:
                    cluster_counts[j].append(0)

            prog_cluster_hist.update()
        average_cluster_count = [np.mean(c) for c in cluster_counts]

        # Plot
        Logger.debug("[MLT_Analysis - plot_cluster_histograms] Thresholds: {0}".format(thresholds)
                     + " average_cluster_count: {0}".format(average_cluster_count))
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)
        ax.bar(thresholds, average_cluster_count, width=0.0045)
        ax.set_xlabel("Threshold values")
        ax.set_ylabel("Mean cluster count")
        ax.set_title("Cluster count by threshold level")
        plt.savefig(str(self.output_path / ('cluster_count_histogram.png')))

    def plot_mlt_layers(self, snapshot_list, thresholds):
        """
        Plots each layer of the image side-by-side.
        :param snapshot_list:
        :param thresholds:
        :return:
        """
        colour_map = SpotTools.MplColorHelper(self.colour_map_plots, self.colour_map_limits[0],
                                              self.colour_map_limits[1])
        if self.output_format == "print":
            plt.rcParams.update({'font.size': self.fig_font_size})
        colour_style = 'normal'
        # For centre tracking that doesn't rely on parameters list
        # Each sub-list is a bunch of coordinates for centre points in each threshold layer. They are unordered because
        # currently there is no way to link clusters to their spot_indices directly.
        cluster_centre_dict = [[],[],[],[],[],[],[]]

        # Get Parameters
        parameters = self.get_paramters(snapshot_list, thresholds, range(self.number_of_clusters_to_track))

        prog_make_image = PrintProgress(0, len(snapshot_list),
                                        label="[MLT_Analysis] Plotting MLT layer graphs...")

        for i in range(0, len(snapshot_list), self.plotting_stride):
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            filename = snapshot_list[i].timestamp.strftime('%Y-%m-%d_%H-%M-%S')
            plt.title(snapshot_list[i].timestamp)
            grid_dimensions = (2, 4)
            graphs = [plt.subplot2grid(grid_dimensions, ((x // 4), x % 4)) for x in range(0, 8)]

            # Plot first graph - HMI image
            try:
                roi = self.path_man.loadROI(snapshot_list[i].ROI_path)
            except:
                continue
            graphs[0].imshow(roi.data, cmap=self.colour_map_spot, aspect='auto')
            graphs[0].set_xlim(self.viewport_ranges[0])
            graphs[0].set_ylim(self.viewport_ranges[1])
            graphs[0].set_title(filename)

            # Get all the clusters.
            try:
                mlt = self.path_man.loadMLT(snapshot_list[i].mlt_path)
                if mlt is None:
                    raise RuntimeError("mlt object is None. Skipping.")
            except:
                prog_make_image.update()
                continue

            # Plot all other thresholds as different graphs
            if len(thresholds) > 7:
                # This is lazy, but I just need to get this code out. There is no technical reason why more can't
                # be done, but I cba to implement the catch-all situation right now.
                raise ValueError("Number of thresholds must be less than 8 for this plot type.")

            for j in range(len(thresholds)):
                layer = mlt.find_layer_by_threshold(thresholds[j])
                if layer is None:
                    Logger.debug("[MLT_Analysis - plot_mlt_layers] MLT Layer is null for threshold {0}.".format(thresholds[j]))
                    continue
                #threshold_clusters = [c for c in layer.mlt_clusters if c.threshold_ratio == layer.threshold_ratio]
                for k in range(len(layer.mlt_clusters)):
                    xpoints = []
                    ypoints = []
                    for point in layer.mlt_clusters[k].points:
                        xpoints.append(point[0])
                        ypoints.append(point[1])
                    # append data regarding centre point to master list
                    cluster_centre_dict[j].append([np.mean(xpoints), np.mean(ypoints)])

                    # plot on grid with +1 offset from hmi image
                    graphs[j+1].scatter(xpoints, ypoints, marker='s', s=((72. / fig.dpi) ** 2) * 2,
                                      color=SpotTools.get_colour(k))
                    graphs[j+1].set_xlim(self.viewport_ranges[0])
                    graphs[j+1].set_ylim(self.viewport_ranges[1])
                    graphs[j+1].set_title("Threshold Ratio: {0}".format(layer.threshold_ratio))
                    graphs[j+1].set_xlabel("Position (pix)")
                    graphs[j+1].set_ylabel("Position (pix)")
                
                # Plot tracer lines
                # if self.layers_show_centres_tracked:
                #     if self.layers_only_track_numbered:
                #         end_track_centres = self.number_of_clusters_to_track
                #     else:
                #         end_track_centres = len(parameters[j])
                #     for m in range(end_track_centres):
                #         try:
                #             if parameters is None or parameters[j] is None or parameters[j][m] is None:
                #                 continue
                #         except KeyError:
                #             continue
                #         try:
                #             centres = parameters[j][m]["centre"]
                #         except IndexError as ie:
                #             if parameters[j][m] is None:
                #                 continue
                #             else:
                #                 Logger.debug("[MLT_Analysis - plot_mlt_layers] Tracer: "
                #                 + "j = {0}, m = {1}".format(j,m)
                #                 + "\nlen(parameters[j]) = {0}".format(len(parameters[j]))
                #                 + "\nlen(parameters[j][m]) = {0}".format(len(parameters[j][m]))
                #                 + "\ntype(parameters[j]) = {0}".format(len(parameters[j]))
                #                 + "\ntype(parameters[j][m]) = {0}".format(len(parameters[j][m]))
                #                 + "\nparameters[j] = {0}".format(parameters[j])
                #                 + "\nparameters[j][m] = {0}".format(parameters[j][m]))
                #                 raise ie
                #
                #         Logger.debug("[MLT_Analysis - plot_mlt_layers] plotting centre lines..."
                #                      + "\ncluster_to_track (m) = {0}".format(m)
                #                      + "\nthresholds (j) = {0}".format(j)
                #                      + "\nparameters[j][m][\"centre\"] = {0}".format(parameters[j][m]["centre"]))
                #         centre_x_coords = []
                #         centre_y_coords = []
                #         for centre in centres:
                #             centre_x_coords.append(centre[0])
                #             centre_y_coords.append(centre[1])
                #         graphs[j+1].plot(centre_x_coords, centre_y_coords, color="black")

            # Plot the tracer lines
            if self.layers_show_centres_tracked:
                for j in range(len(thresholds)):
                        centre_x_coords = []
                        centre_y_coords = []
                        start_point = len(cluster_centre_dict[j]) - self.layers_tracer_line_memory if len(cluster_centre_dict[j]) - self.layers_tracer_line_memory > 0 else 0
                        for jj in range (start_point, len(cluster_centre_dict[j])):
                            centre = cluster_centre_dict[j][jj]
                            centre_x_coords.append(centre[0])
                            centre_y_coords.append(centre[1])
                        graphs[j + 1].scatter(centre_x_coords, centre_y_coords, marker='.', color="black",
                                              s=((72. / fig.dpi) ** 2) * 2)

            # Tidy up, plot, and save
            plt.tight_layout()
            plt.savefig(str(self.mlt_plot_path / (filename + '.png')))
            plt.close()
            prog_make_image.update()

    #==================================================================================================================
    #                                                  Global Parameters
    #==================================================================================================================

    def calculate_rotation_activity(self, parameters, si):
        """
        Uses the weighted rotation velocities of all clusters in an active region to calculate a value that approximates
        the rotational activity in the region.

            weighted_rotation_activity = sum_over_layers(sum_over_clusters(abs(velocity) * size))

        :param parameters:
        :param si: Snapshot index
        :return:
        """
        # ti = threshold index
        # ci = cluster index
        weighted_rotation_activity = 0.0
        time_index = None
        cluster_count = 0
        for ti in range(len(parameters)):
            for ci in range(len(parameters[ti])):
                if parameters[ti][ci] is None:
                    continue
                try:
                    if time_index is None:
                        time_index = parameters[ti][ci]["time"][si]
                    weighted_rotation_activity += abs(parameters[ti][ci]["velocity"][si]) * parameters[ti][ci]["size"][si]
                    cluster_count += 1
                except IndexError:
                    continue
        if cluster_count == 0:
            weighted_rotation_activity = 0.0
        else:
            weighted_rotation_activity = weighted_rotation_activity/float(cluster_count)
        return time_index, weighted_rotation_activity, cluster_count


    def cluster_statistics_plot(self, snapshot_list, thresholds):
        """Plots the number of clusters over time and other timeseries data"""
        colour_map = SpotTools.MplColorHelper(self.colour_map_plots, self.colour_map_limits[0],
                                              self.colour_map_limits[1])
        if self.output_format == "print":
            plt.rcParams.update({'font.size': self.fig_font_size})
        colour_style = 'normal'

        timestamps = []
        clusters_per_layer = {}

        # Get Data
        for i in range(len(snapshot_list)):
            try:
                mlt = self.path_man.loadMLT(snapshot_list[i].mlt_path)
            except TypeError:
                continue
            if mlt is None:
                continue
            timestamps.append(snapshot_list[i].timestamp)
            for threshold_ratio in np.flip(thresholds):
                layer = mlt.find_layer_by_threshold(threshold_ratio)
                cluster_count = 0 if layer is None else len(layer.mlt_clusters)
                if threshold_ratio in clusters_per_layer:
                    clusters_per_layer[threshold_ratio].append(cluster_count)
                else:
                    clusters_per_layer[threshold_ratio] = [cluster_count]

        # Plot
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)
        colours = colour_map.get_discrete_list(len(clusters_per_layer.keys()))
        ax.stackplot(matdates.date2num(timestamps), clusters_per_layer.values(),
                     labels=clusters_per_layer.keys(), edgecolor='k',
                     colors=colours)
        date_formatter = matdates.DateFormatter('%d/%m')
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_xlabel("Time")
        ax.set_ylabel("Clusters Detected")
        leg = plt.legend(bbox_to_anchor=(1,1), loc="upper left", frameon=False)
        plt.tight_layout()
        plt.savefig(self.output_path / "clusters_distribution.png")


    def global_parameters_plot(self, snapshot_list, thresholds):
        """
        Plots a number of global parameters for the active region.
        :param snapshot_list:
        :param thresholds:
        :return:
        """
        colour_map = SpotTools.MplColorHelper(self.colour_map_plots, self.colour_map_limits[0],
                                              self.colour_map_limits[1])
        if self.output_format == "print":
            plt.rcParams.update({'font.size': self.fig_font_size})
        colour_style = 'normal'

        # Get clusters
        all_cluster_lists = self.multi_track_clusters(snapshot_list, thresholds)
        all_cluster_lists = self.sort_all_cluster_list(all_cluster_lists, thresholds)

        # Get parameters for requested clusters.
        parameters = [{} for threshold in thresholds]

        # Construct the parameter list
        for k in range(0, len(thresholds)):
            for spot_index in range(self.number_of_clusters_to_track):
                cluster_parameters = self.get_cluster_params(all_cluster_lists, k, spot_index, thresholds)
                parameters[k][spot_index] = cluster_parameters

        # Calculate Global Parameters
        Logger.log("[MLT_Analysis - global_parameters_plot] Calculating global parameters...")

        timelist = []
        weighted_rotation_activity = []
        data = []
        for i in range(len(snapshot_list)):
            timestamp, wra, cluster_count = self.calculate_rotation_activity(parameters, i)
            # TODO: track cluster count too
            if timestamp is not None:
                timelist.append(timestamp)
                weighted_rotation_activity.append(wra)
                data.append([timestamp, wra])
        data.sort(key=lambda x: x[0])
        Logger.debug("[MLT_Analysis - global_parameters_plot] Data: \n{0}".format(data))


        # Make plots
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        filename = "global_rotation_activity.png"
        plt.title("NOAA {0} Global Weighted Rotation Activity".format(self.NOAA_number))
        ax = plt.subplot(111)

        x,y = zip(*data)
        ax.plot_date(matdates.date2num(x), y,
                     linestyle='-')

        # Plot flare times
        if self.flare_times is not None:
            for time in self.flare_times:
                ax.axvline(matdates.date2num(time), lw=1, color="red")

        ax.xaxis.set_major_formatter(matdates.DateFormatter(self.time_axis_labels['days_long']))
        ax.set_xlabel("Time")
        ax.set_ylabel("Rotation activity (arb. units)")
        plt.savefig(self.output_path / filename)
        plt.close()


def mlt_exploded_plot(roi, mlt_layers, xrange=[0,600], yrange=[0,600]):
    """Produces a 3d plot showing the sunspot at the 0th level, then stacking MLT layers on the Z axis."""
    colourmap = SpotTools.MplColorHelper('viridis_r',0.10,0.55)
    fig = plt.figure(figsize=(8,6), dpi=300)
    ax = fig.add_subplot(111,projection='3d')
    xx = np.arange(xrange[0], xrange[1])
    yy = np.arange(yrange[0], yrange[1])
    zz = np.array(roi.data)
    ax.contourf(X=xx, Y=yy, Z=zz[xrange[0]:xrange[1], yrange[0]:yrange[1]], cmap='Greys_r', zdir='z', offset=0, levels=20)
    
    # Add each layers points
    for i in range(0, len(mlt_layers)):
        layer = mlt_layers[i]
        if layer is None:
            continue
        for cluster in layer.mlt_clusters:
#            ax.scatter(cluster.points[0][:], cluster.points[1][:], zs=(i+1)*10000, zdir='z', 
#                       c=SpotTools.colours_combo_contour[i], alpha=0.7, marker='s', s=((72. / fig.dpi) ** 2) * 2)
            xpoints = []
            ypoints = []
            for point in cluster.points:
                xpoints.append(point[0])
                ypoints.append(point[1])
            ax.scatter(xpoints, ypoints, zs=(i+2)*0.5, zdir='z', c=[colourmap.get_rgb(layer.threshold_ratio)], alpha=0.7,
                           marker='s', s=((72. / fig.dpi) ** 2) * 2, label='{0}%'.format(int(layer.threshold_ratio*100)))

    # Fix plot axes
    # X axis
    xticks = ax.get_xticklabels()
    xtick_range = xrange[1] * roi.pixel_scale[0] - \
                  xrange[0] * roi.pixel_scale[0]
    xtick_sep = round(xtick_range / len(xticks))
    xtick_start = round(roi.centre_arcsec[0] - xtick_range / 2)
    new_xticks = []
    for i in range(len(xticks)):
        new_xticks.append(xtick_start + xtick_sep * i)

    # Y axis
    yticks = ax.get_yticklabels()
    ytick_range = yrange[1] * roi.pixel_scale[1] - \
                  yrange[0] * roi.pixel_scale[1]
    ytick_sep = round(ytick_range / len(yticks))
    ytick_start = round(roi.centre_arcsec[1] - ytick_range / 2)
    new_yticks = []
    for i in range(len(yticks)):
        new_yticks.append(ytick_start + ytick_sep * i)

    # Z axis
    #new_zticks = ['100']
    #new_zticks.extend([round(z.threshold_ratio * 100) for z in mlt_layers])

    ax.set_xticklabels(new_xticks)
    ax.set_yticklabels(new_yticks)
    #ax.set_zticklabels(new_zticks)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_zlim([0,2.5])
    ax.set_xlabel("Distance (arcsec)")
    ax.set_ylabel("Distance (arcsec)")
    ax.set_zlabel("")

    #ax.legend()
    plt.show()
    plt.close()

