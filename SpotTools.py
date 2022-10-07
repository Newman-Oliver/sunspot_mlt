import numpy as np
import matplotlib as mpl
import matplotlib.colors as cols
import matplotlib.pyplot as plt
from matplotlib import cm
import datetime

colourList = ['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:brown', 'xkcd:red', 'xkcd:pink', 'xkcd:light blue',
              'xkcd:teal', 'xkcd:orange', 'xkcd:light green', 'xkcd:magenta', 'xkcd:yellow', 'xkcd:sky blue',
              'xkcd:grey', 'xkcd:lime green', 'xkcd:light purple', 'xkcd:violet', 'xkcd:dark green', 'xkcd:turquoise',
              'xkcd:lavender', 'xkcd:dark blue', 'xkcd:tan', 'xkcd:cyan', 'xkcd:forest green', 'xkcd:mauve',
              'xkcd:dark purple', 'xkcd:bright green', 'xkcd:maroon', 'xkcd:olive', 'xkcd:salmon', 'xkcd:beige',
              'xkcd:royal blue', 'xkcd:hot pink', 'xkcd:rose', 'xkcd:mustard']

colourListViridis = ['#45065A', '#47C06E', '#31648D', '#1E988A', '#B7DD29', '#C2DF22', '#EEE51B']

cvd_colours = ['#7530a0', '#5da853', '#528ad5', '#ebcd28', '#6f6297', '#4e866d', '#2bbacc']
# alternative red_blue scheme: ['#c83a31', '#4938d3', '#b05ff7', '#6f5362', '#4eadbb', '#f19849']

colours_combo_ellipse = ['xkcd:green', 'xkcd:royal blue', 'xkcd:light purple', 'xkcd:light pink', 'xkcd:indigo']

colours_combo_contour = ['xkcd:sea green', 'xkcd:bright blue', 'xkcd:violet', 'xkcd:dark pink', 'xkcd:purple']

colours_combo_light = ['xkcd:sea green', 'xkcd:bright blue', 'xkcd:light purple', 'xkcd:light pink', 'xkcd:indigo']

colour_maps = ['Purples', 'Greens', 'Blues', 'Oranges', 'Reds']

markers = ["circle", "cross", "star", "triangle_up", "diamond", "plus"]

markers_symbol = ["o", "X", "*", "^", "p"]

line_styles = ['-', '--', ':', '-.']


def parse_string_to_datetime(date_string):
    """Attempt to get a datetime object from a string using different formats"""
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d_%H:%M:%S', '%Y-%m-%d_%H:%M:%S.%f',
                '%Y-%m-%d %H-%M-%S', '%Y-%m-%d %H-%M-%S.%f', '%Y-%m-%d_%H-%M-%S', '%Y-%m-%d_%H-%M-%S.%f'):
        try:
            return datetime.datetime.strptime(date_string, fmt)
        except ValueError:
            pass
    raise ValueError('Could not parse string \'{0}\' to datetime.'.format(date_string))


def is_coordinate_in_list(coordinate, list_of_coords):
    """
    Returns true if the coordinate given is in the list of coordinates, otherwise false.
    :param coordinate: a python list containing an x and y value. e.g. [0,0]
    :param list_of_coords: A python list of coordinates to check against. e.g. [[0,0],[1,2],[2,2]]
    :return: Bool
    """
    return any(x in list(list_of_coords) for x in [coordinate])


def pixel_scale_from_centres(centre_in_pixels,centre_in_arcsec, image_size=4096):
    """Uses the difference in values for the centre point in pixels and arcsec to determine image scale"""
    centre_in_pixels = np.array(centre_in_pixels)
    centre_in_arcsec = np.array(centre_in_arcsec)
    origin_pixels = centre_in_pixels - (round(image_size/2))
    scale = centre_in_arcsec / origin_pixels
    return scale


def get_date_range_indices(spot_group, start, end):
    """From a given spot group, find the indices of the snapshots that match the start and end dates"""
    # Get input in datetime.datetime format
    if isinstance(start, datetime.datetime):
        start_date = start
    else:
        start_date = datetime.datetime.strptime(start, '%Y-%m-%d_%H-%M-%S')
    if isinstance(end, datetime.datetime):
        end_date = end
    else:
        end_date = datetime.datetime.strptime(end, '%Y-%m-%d_%H-%M-%S')

    start_index = None
    end_index = None

    for i in range(0, len(spot_group.history)):
        if spot_group.history[i].timestamp > start_date and start_index is None:
            start_index = i
        if spot_group.history[i].timestamp > end_date and end_index is None:
            end_index = i
    return start_index, end_index


def divide_tasks(self, dataLength, cpuCount):
    """Provides the start and stop indices for an MPI loop that uses cpuCount CPUs to iterate over a list of length
    dataLength"""
    N = dataLength
    count = np.floor(N / cpuCount)
    remainder = N % cpuCount

    # Get start and stop points for each MPI process
    if self.rank < remainder:
        start = self.rank * (count + 1)
        stop = start + count
    else:
        start = (self.rank * count) + remainder
        stop = start + (count - 1)

    return np.int64(start), np.int64(stop)


class MplColorHelper:
    """Used to get a colour from a colour map. Instantiates a copy of the map with the given min and max values and
    provides a colour for a given value (val) between the min and max."""

    def __init__(self, cmap_name, start_val, stop_val):
        self.start = start_val
        self.end = stop_val
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

    def get_discrete_list(self, colour_count):
        """Returns a list of colour_count length of evenly spaced colours from the colour map"""
        step = (self.end - self.start) / colour_count
        return [self.get_rgb(self.start + (x * step)) for x in range(colour_count)]


def get_colour_map(index):
    return colour_maps[index % len(colour_maps)]


def get_colour(index, switch_on_iterations=False):
    if switch_on_iterations:
        return colourList[int(index/len(line_styles))]
    else:
        return colourList[index % len(colourList)]

def ensure_colour_bright(colour):
    """If the input colour has a hsv value of less than 0.5, set it to 0.5. Returns the same or brighter colour"""
    hsv_colour = cols.rgb_to_hsv(colour[:3])
    if hsv_colour[2] < 0.5:
        hsv_colour[2] = 0.5
    rgba_colour = cols.to_rgba(cols.hsv_to_rgb(hsv_colour), 1.0)
    return rgba_colour

def ensure_colour_dark(colour):
    """If the input colour has a hsv value of greater than 0.5, set it to 0.5. Returns the same or darker colour"""
    hsv_colour = cols.rgb_to_hsv(colour[:3])
    if hsv_colour[2] > 0.5:
        hsv_colour[2] = 0.5
    rgba_colour = cols.to_rgba(cols.hsv_to_rgb(hsv_colour), 1.0)
    return rgba_colour

def ensure_colour_faded(colour):
    """If the input colour has a hsv saturation of more than 0.5, set it to 0.5. Returns the same or lighter colour"""
    hsv_colour = cols.rgb_to_hsv(colour[:3])
    if hsv_colour[1] > 0.5:
        hsv_colour[1] = 0.5
    rgba_colour = cols.to_rgba(cols.hsv_to_rgb(hsv_colour), 1.0)
    return rgba_colour

def get_line_style(index, switch_on_iterations=True):
    """Returns a line style from a preset list based on current iteration.
    :param index: Current iteration number
    :param switch_on_iterations: If true, a new line style is returned for every time the index loops around the
    available colours. e.g. all values from 0 to len(colourlist) will be '-', then from len(colourlist) to
    2*len(colourlist) will be '--' etc. If false, the linestyle will loop around the linestyle list instead."""
    if switch_on_iterations:
        return line_styles[int(index/len(colourList))]
    else:
        return line_styles[index % len(line_styles)]

def get_marker(index, use_symbol=False):
    # Switches marker every time colour loops back around
    if use_symbol:
        return markers_symbol[int(index/len(colourList))]
    else:
        return markers[int(index/len(colourList))]


def get_marker_cmp(index):
    # Switches the marker every time the colour map colours loop around
    return markers_symbol[int(index/len(colour_maps))]


def bresenhamLow(x0, y0, x1, y1):
    # Applies the Bresenham line approximation for line gradients 0 to -1
    deltaX = float(x1) - float(x0)
    deltaY = float(y1) - float(y0)
    yError = 1.0

    if deltaY < 0:
        yError = -1.0
        deltaY = -deltaY

    error = (2.0 * deltaY) - deltaX
    y = y0
    outputX = []
    outputY = []

    for x in range(int(x0), int(x1)):
        outputX.append(int(x))
        outputY.append(int(y))
        if error > 0:
            y = y + yError
            error = error - 2*deltaX
        error = error + 2*deltaY

    output = []
    output.append(outputX)
    output.append(outputY)
    return output

def bresenhamHigh(x0, y0, x1, y1):
    # Applies the Bresenham line approximation for line gradients 0 to +1
    deltaX = float(x1) - float(x0)
    deltaY = float(y1) - float(y0)
    xError = 1.0

    if deltaX < 0:
        xError = -1.0
        deltaX = -deltaX

    error = (2.0 * deltaX) - deltaY
    x = x0
    outputX = []
    outputY = []

    for y in range (int(y0), int(y1)):
        outputX.append(int(x))
        outputY.append(int(y))
        if error > 0:
            x = x + xError
            error = error - 2*deltaY
        error = error + 2*deltaX

    output = []
    output.append(outputX)
    output.append(outputY)
    return output

def getPointsOnLine(centre, radius, angle, from_centre=True):
    # Get the coordinates of the points on the line going through point 'centre' at a given angle (in rads). If
    # from_centre is true, the line originates at 'centre', if false, the line is a line that passes through 'centre'
    # with length of 2*radius.

    if from_centre:
        x0 = centre[0]
        y0 = centre[1]

        x1 = centre[0] + (radius * np.cos(angle))
        y1 = centre[1] + (radius * np.sin(angle))
    else:
        x0 = centre[0] - (radius * np.cos(angle))
        y0 = centre[1] - (radius * np.sin(angle))

        x1 = centre[0] + (radius * np.cos(angle))
        y1 = centre[1] + (radius * np.sin(angle))

    output = []

    if np.abs(y1-y0) < np.abs(x1-x0):
        if x0 > x1:
            output = bresenhamLow(x1, y1, x0, y0)
        else:
            output = bresenhamLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            output = bresenhamHigh(x1, y1, x0, y0)
        else:
            output = bresenhamHigh(x0, y0, x1, y1)

    return np.array(output)

def first(iterable, default=None, key=None):
    if key is None:
        for el in iterable:
            if el:
                return el
    else:
        for el in iterable:
            if key(el):
                return iterable
    return default
