# My Libraries
import SpotTools
import Contours
import Logger
import EllipseFit

# Maths and Phys libs
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from pyclustering.cluster.optics import optics
import time

# Data holders --------------------------------------------------------------
class Cluster:
    def __init__(self, _coordinates, _datetime, _threshold, _threshold_ratio, _ellipse_parameters=None):
        self.points = _coordinates
        self.datetime = _datetime
        self.threshold = _threshold
        self.threshold_ratio = _threshold_ratio
        self.ellipse_parameters = _ellipse_parameters


class Layer():
    def __init__(self, _threshold, _threshold_ratio, _clusters, _cluster_ordering):
        self.threshold = _threshold
        self.threshold_ratio = _threshold_ratio
        self.clusters = _clusters
        self.cluster_ordering = _cluster_ordering

def calculate_pixel_weight(pixel_value, weighting_threshold):
    """Returns the number of times a pixel value must be repeated to provide a weighting such that lower intensity
    pixels have a  higher weight."""
    return int(np.ceil((weighting_threshold - pixel_value)/(weighting_threshold / 10)))

def extract_data(data, threshold_value, use_weighting=False):
    """Extract a list of the points that contain data and their co-ordinates for passing into the OPTICS code"""
    (x, y) = np.shape(data)
    # Square the array
    if x != y:
        if y < x:
            x = y
        else:
            y = x
    extracted_data = []
    for i in range(0, y):
        for j in range(0, x):
            if data[j, i] > 0:
                pixel_weight = 1 if not use_weighting else calculate_pixel_weight(data[j,i], threshold_value)
                for w in range(pixel_weight):
                    extracted_data.append((i + (w/10) * pow(-1,w), j + (w/10) * pow(-1,w)))
    return np.array(extracted_data)

# Ellipse Fitting ---------------------------------------------

def fit_ellipse(x_perimeter, y_perimeter):
    """Fit an ellipse to a group of x and y coordinates. Returns 3 values: the ellipse parameters in a list [centre,
    axes, angle to normal], the x coords of the ellipse, the y coords of the ellipse."""
    try:
        elli = EllipseFit.fitEllipse_b2ac(np.transpose(x_perimeter), np.transpose(y_perimeter))
        centre = elli[0]
        phi = elli[2]
        axes = elli[1]

        R = np.arange(0, 2.0 * np.pi, 0.01)
        a, b = axes
        xx = centre[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
        yy = centre[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
        return elli, xx, yy
    except ArithmeticError:
        Logger.debug("[MLT - fit_ellipse] Could not find ellipse for given parameters!")
        return None, None, None

def do_ellipse_fit(points):
    perimeter = Contours.getPerimeter(points)
    perimeter = [list(x) for x in zip(*perimeter)]
    elli, xx, yy = fit_ellipse(perimeter[0], perimeter[1])
    return elli

# OPTICS stuff ------------------------------------------------------------------

def apply_optics(dataset, eps, min_pts):
    """Apply the OPTICS clustering algorithm to the data and return the resulting clusters."""
    instance = optics(dataset, eps, min_pts)
    instance.process()
    return instance.get_clusters(), instance.get_ordering()

def make_layer(data, threshold, threshold_ratio, timestamp, epsilon=5, min_points=4):
    """Make a layer by applying a thresholding to the data and finding the clusters.
    :param epsilon:
    :param min_points:
    """
    data_thresh = (data < threshold) * data
    # If the result array contains no non-zero values, it is empty so return no clusters.
    if data_thresh.max() == 0.0:
        return None

    # Reformat the data to be put into OPTICS and get clusters
    tic = time.perf_counter()
    data_formatted = extract_data(data_thresh, threshold, use_weighting=False)
    cluster_indices, cluster_ordering = apply_optics(data_formatted, epsilon, min_points)

    # Reformat the output of the OPTICS algorithm to use the data coords and not the array indices. Then put the
    # data into a SpotData.Cluster() class to hold the data nicely.
    clusters = []
    for cluster in cluster_indices:
        cluster_coords = []
        for i in range(0, len(cluster)):
            coord = data_formatted[cluster[i]]
            cluster_coords.append(np.array([np.floor(coord[0]), np.floor(coord[1])]))
        elli = do_ellipse_fit(cluster_coords)
        clusters.append(Cluster(_coordinates=cluster_coords, _datetime=timestamp,
                                         _threshold=threshold, _threshold_ratio=threshold_ratio,
                                         _ellipse_parameters=elli))

    # Make a Layer object to save
    layer = Layer(threshold, threshold_ratio, clusters, cluster_ordering)
    toc = time.perf_counter()

    # Print Diagnostics
    print("[Diagnostics] Run {3}-{4}-{5} -- Number of clusters found: {0} -- Data length: {1} -- Time Elapsed: {2:0.4f}s".format(len(clusters), len(np.where(data_thresh > 0)[0]), toc - tic,threshold_ratio,epsilon,min_points))

    return layer


def load_roi(_path):
    path = Path(_path)
    with path.open('rb') as f:
        roi = pickle.load(f, encoding='bytes')
    return roi

def plot_layers(graph, layers, roi, epsilon, min_points):
    graph.imshow(roi.data, cmap='Greys_r')
    for i in range(len(layers)):
        layer = layers[i]
        if layer is None:
            continue
        for ci in range(len(layer.clusters)):
            cluster = layer.clusters[ci]
            # Attempt at plotting only the perimeter
            perimeter = Contours.getPerimeter(cluster.points)
            # change perimeter from a list of [[x,y], ...] coords to a list of [[x],[y]] coords.
            perimeter = [list(x) for x in zip(*perimeter)]
            graph.scatter(perimeter[0], perimeter[1], c=SpotTools.colourList[ci],
                       label=str(layer.threshold_ratio) + r"$I_{quiet sun}$",
                       marker='s', s=(72. / graph.figure.dpi) ** 2)
            graph.set_xlabel("Distance (px)")
            graph.set_ylabel("Distance (px)")
            # 11944: [250,900] 11166: [250,800] and [100,600]
            graph.set_xlim([600,0])
            graph.set_ylim([600,0])
            graph.set_title("Epsilon = {0} Min Points = {1}".format(epsilon, min_points))


def apply_mlt_to_roi(roi_path, thresholds, epsilon_values=[2,5,10,15], min_points_values=[1,4,10,20], output_path=None, filename=None):
    if output_path is None:
        output_path = '/mnt/alpha/work/PhD/DataArchive/sunspots/'
    if filename is None:
        filename = '11289_2011-09-09_16-38.png'
    # Load ROI file
    roi = load_roi(roi_path)

    # determine gird layout of plot
    grid_width = len(epsilon_values)
    grid_height = len(min_points_values)
    fig = plt.figure(figsize=(7, 7), dpi=300)
    print("grid_width: {0}".format(grid_width))
    print("grid_height: {0}".format(grid_height))

    plots = []
    for x in range(grid_width):
        for y in range(grid_height):
            plots.append(plt.subplot2grid((grid_height, grid_width), (y,x)))
    print("Len(plots): {0}".format(len(plots)))

    # apply optics to ROI for each combination of epsilon/min_pts for each threshold in thresholds.
    for i in range(0, grid_height):
        for ii in range(0, grid_width):
            layers = []
            for threshold in thresholds:
                layers.append(make_layer(roi.data, roi.qsun_intensity * threshold,
                                         threshold, roi.timestamp, epsilon=epsilon_values[ii],
                                         min_points=min_points_values[i]))
            index = i*grid_height + ii
            print("plot iteration: {0}".format(index))
            graph = plots[index]
            plot_layers(graph, layers, roi, epsilon=epsilon_values[ii], min_points=min_points_values[i])

    # save the picture
    plt.tight_layout()
    plt.savefig(output_path + filename, transparent=True)


if __name__ == "__main__":
    # NOAA_11166-hourly/roi/2011-03-08_18-34-14 // NOAA_11166-hourly/roi/2011-03-10_03-14-45
    # 2014-09-05_16-hourly/roi/2014-09-08_10-00-36
    # /mnt/alpha/work/PhD/DataArchive/sunspots/NOAA_11944-hourly/roi/2014-01-08_00-46-09.roi
    roi_path = "/mnt/alpha/work/PhD/DataArchive/sunspots/NOAA_11289-hourly/roi/2011-09-09_16-38-52.roi"
    epsilon_values = [1,2,5,10]
    min_pts_values = [1,4,10,20]
    thresholds = [0.50]
    plt.rcParams.update({'font.size': 6})
    apply_mlt_to_roi(roi_path, thresholds, epsilon_values=epsilon_values, min_points_values=min_pts_values)