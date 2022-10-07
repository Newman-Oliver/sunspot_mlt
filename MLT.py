# My Libraries
from numpy.linalg import LinAlgError

import SpotData
import SpotTools
import Contours
import Logger
from Logger import PrintProgress
import EllipseFit

# Maths and Phys libs
import numpy as np
import matplotlib.pyplot as plt
from pyclustering.cluster.optics import optics

# Misc
from threading import Thread


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class MultiLevelThresholding():

    def __init__(self, _path_manager, _comm, _config_parser, _min_pts = 4, _eps = 5):
        self.min_pts = _min_pts
        self.eps = _eps
        self.path_man = _path_manager
        self.config_parser = _config_parser
        self.output_path = self.path_man.getDir('output')
        self.mlt_path = self.path_man.getDir('mlt')
        self.image_path = self.path_man.getDir('output_mlt',posix_path=(self.output_path / 'mlt'))

        self.overwrite_layers = self.config_parser.getboolean('MLT', 'overwrite_layers')
        self.use_weighting = self.config_parser.getboolean('MLT', 'use_weighting')
        self.eps = self.config_parser.getfloat('MLT','epsilon')
        self.min_pts = self.config_parser.getint('MLT','min_points')

        # MPI
        self.comm = _comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def apply_optics(self, dataset, eps, min_pts):
        """Apply the OPTICS clustering algorithm to the data and return the resulting clusters."""
        instance = optics(dataset, eps, min_pts)
        instance.process()
        return instance.get_clusters(), instance.get_ordering()

    def calculate_pixel_weight(self, pixel_value, weighting_threshold):
        """Returns the number of times a pixel value must be repeated to provide a weighting such that lower intensity
        pixels have a  higher weight."""
        return int(np.ceil((weighting_threshold - pixel_value) / (weighting_threshold / 10)))

    def extract_data(self, data, threshold_value, use_weighting=False):
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
                    pixel_weight = 1 if not use_weighting else self.calculate_pixel_weight(data[j, i], threshold_value)
                    for w in range(pixel_weight):
                        extracted_data.append((i + (w / 10) * pow(-1, w), j + (w / 10) * pow(-1, w)))
        return np.array(extracted_data)

    @staticmethod
    def fit_ellipse(x_perimeter, y_perimeter):
        """Fit an ellipse to a group of x and y coordinates. Returns 3 values: the ellipse parameters in a list [centre,
        axes, angle to normal], the x coords of the ellipse, the y coords of the ellipse."""
        if len(x_perimeter) < 7 or len(y_perimeter) < 7:
            Logger.debug("[MLT - fit_ellipse] Not enough points in cluster to fit ellipse. Skipping.")
            return None, None, None
        try:
            elli = EllipseFit.fitEllipse_scipy(np.transpose(x_perimeter), np.transpose(y_perimeter))
            centre = elli[0]
            phi = elli[2]
            axes = elli[1]

            R = np.arange(0, 2.0 * np.pi, 0.01)
            a, b = axes
            xx = centre[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
            yy = centre[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
            return elli, xx, yy
        except LinAlgError:
            Logger.debug("[MLT - fit_ellipse] Could not find ellipse for given parameters!")
            return None, None, None
        except ArithmeticError:
            Logger.debug("[MLT - fit_ellipse] Could not find ellipse for given parameters!")
            return None, None, None

    def make_layer(self, data, threshold, threshold_ratio, timestamp, counter_dict={}):
        """Make a layer by applying a thresholding to the data and finding the clusters."""
        data_thresh = (data < threshold) * data
        # If the result array contains no non-zero values, it is empty so return no clusters.
        if data_thresh.max() == 0.0:
            return None

        # Reformat the data to be put into OPTICS and get clusters
        data_formatted = self.extract_data(data_thresh, threshold, use_weighting=self.use_weighting)

        # Put the apply_optics into a thread with a 5 min timeout so it doesn't hang if apply_optics does.
        # Hacky workaround for issue #6
        Logger.debug("[MLT - make_layer] Applying OPTICS...")
        optics_thread = ThreadWithReturnValue(target=self.apply_optics, args=(data_formatted, self.eps, self.min_pts))
        optics_thread.start()
        cluster_result = optics_thread.join(5*60)
        if cluster_result is None:
            Logger.log("Warning! OPTICS timed out for layer {0} for data at {1}.".format(threshold_ratio, timestamp))
            return None
        cluster_indices, cluster_ordering = cluster_result
        Logger.debug("[MLT - make_layer] OPTICS complete. Iterating over {0} clusters...".format(len(cluster_indices)))

        # Reformat the output of the OPTICS algorithm to use the data coords and not the array indices. Then put the
        # data into a SpotData.Cluster() class to hold the data nicely.
        clusters = []
        for cluster in cluster_indices:
            counter_dict["clusters"] = counter_dict["clusters"] + 1 if "clusters" in counter_dict.keys() else 1
            cluster_coords = []
            for i in range(0, len(cluster)):
                cluster_coords.append(data_formatted[cluster[i]])
            elli = self.do_ellipse_fit(cluster_coords)
            if elli is None:
                counter_dict["none_clusters"] = counter_dict["none_clusters"] + 1 if "none_clusters" in counter_dict.keys() else 1
                Logger.debug("[MLT - make_layer] Ellipse parameters for cluster {0} returned None!".format(counter_dict["clusters"]))
            clusters.append(SpotData.Cluster(_coordinates=cluster_coords, _datetime=timestamp,
                                             _threshold=threshold, _threshold_ratio=threshold_ratio,
                                             _ellipse_parameters=elli))

        # Make a Layer object to save
        layer = SpotData.Layer(threshold, threshold_ratio, clusters, cluster_ordering)

        return layer

    @staticmethod
    def do_ellipse_fit(points):
        perimeter = Contours.getPerimeter(points)
        perimeter = [list(x) for x in zip(*perimeter)]
        elli, xx, yy = MultiLevelThresholding.fit_ellipse(perimeter[0], perimeter[1])
        return elli

    def pretty_plot(self, layers, roi, filename):
        """Debug code for plotting the clusters in each layer. Used to track down an issue in clustering but also good
        for visualisation."""
        fig = plt.figure(figsize=(16, 9), dpi=90)
        plt.rcParams.update({'font.size': 12})
        ax = fig.add_subplot(111, projection='3d')
        xx = np.arange(180, 420)
        yy = np.arange(180, 420)
        zz = np.array(roi.data)
        ax.contourf(X=xx, Y=yy, Z=zz[180:420, 180:420], cmap='inferno', zdir='z', offset=0)

        for i in range(0, len(layers)):
            layer = layers[i]
            if layer is None:
                continue
            for cluster in layer.mlt_clusters:
                for point in cluster.points:
                    ax.scatter(point[0], point[1], zs=i+1, zdir='z', c=SpotTools.colours_combo_contour[i], alpha=0.7,
                               marker='s', s=((72. / fig.dpi) ** 2)*2)

        ax.set_ylabel("Distance (pix)")
        ax.set_ylim([420, 180])
        ax.set_xlabel("Distance (pix)")
        ax.set_xlim([420, 180])
        ax.set_zlabel("Threshold level")
        ax.set_zlim([0, 5])
        plt.savefig(str(self.output_path / ('pretty_' + filename + '.png')))
        plt.close(fig)

    def plot_layers(self, layers, roi, filename=""):
        """Produce a plot of the clusters and save it to a file"""
        fig = plt.figure(figsize=(16,9), dpi=90)
        plt.rcParams.update({'font.size': 16})
        ax = fig.add_subplot(111)
        ax.imshow(roi.data, cmap='inferno')
        for i in range(0, len(layers)):
            layer = layers[i]
            if layer is None:
                continue
            for cluster in layer.mlt_clusters:
                if len(cluster.points) < 10:
                    # Reject because too few to get good perimeter
                    continue

                # Attempt at plotting only the perimeter
                perimeter = Contours.getPerimeter(cluster.points)
                # change perimeter from a list of [[x,y], ...] coords to a list of [[x],[y]] coords.
                perimeter = [list(x) for x in zip(*perimeter)]
                ax.scatter(perimeter[0], perimeter[1], c=SpotTools.colours_combo_contour[i],
                           label=str(layer.threshold_ratio) + r"$I_{quiet sun}$",
                           marker='s', s=(72./fig.dpi)**2)

                # Getting Ellipse fit
                elli, xx, yy = self.fit_ellipse(perimeter[0], perimeter[1])
                if elli is None:
                    continue

                # Record parameters and plot.
                cluster.ellipse_parameters = elli
                ax.plot(xx, yy, c=SpotTools.colours_combo_ellipse[i], ms=(72. / fig.dpi))

                # Annotate the ellipses with their eccentricity
                ecc = np.sqrt(1 - ((min(elli[1]) ** 2) / (max(elli[1]) ** 2)))
                ax.annotate(str(round(ecc,2)),
                            (xx[(i*20) % len(xx)], yy[(i*20) % len(yy)]),
                            textcoords="offset points",
                            color=SpotTools.colours_combo_light[i],
                            xytext=(i*10, 20 - (i*10)),
                            ha='left')

        ax.set_xlim([600,0])
        ax.set_ylim([450,180])
        ax.set_xlabel("Distance (pix)")
        ax.set_ylabel("Distance (pix)")
        ax.set_title(str(roi.timestamp))
        leg = plt.legend(bbox_to_anchor=(1., .7),
                         bbox_transform=plt.gcf().transFigure,
                         markerscale=9)
        plt.savefig(str(self.image_path / (filename + '.png')))
        plt.close(fig)

    def run_thresholding_on_roi(self, roi, thresholds, filename=""):
        """Runs the MLT process for a single ROI image"""
        counter_dict = {}
        thresholds_ratio = 1  # Make a fake ratio
        layers = []
        for threshold in thresholds:
            layers.append(self.make_layer(roi.data, threshold, thresholds_ratio, roi.timestamp,counter_dict))
        self.plot_layers(layers, roi, filename=filename)

    def run_thresholding_on_list(self, spot_list, _threshold_ratios):
        """Runs the MLT process on a list of sunspot groups."""
        threshold_ratios = _threshold_ratios
        start, stop = SpotTools.divide_tasks(self, len(spot_list.history), self.size)
        prog = PrintProgress(start, stop,
                             label="[MLT] Node {0} applying to spots {1} to {2} Progress: ".format(self.rank,
                                                                                                   start,
                                                                                                   stop))
        for i in range(start, stop):
            group = spot_list.history[i]
            counter_dict = {}

            # Load or make MLT
            group_layers = None
            # Try load
            if group.mlt_path is not None:
                group_layers = self.path_man.loadMLT(group.mlt_path)
            # Else make
            if group_layers is None:
                group_layers = SpotData.MLT_Layers(_timestamp=group.timestamp)

            # Load ROI or skip if unavailable
            if group.ROI_path is not None:
                try:
                    roi = self.path_man.loadROI(group.ROI_path)
                except FileNotFoundError:
                    Logger.log("[MLT] ERR: Group {0} does not have an associated ROI! Skipping...".format(
                        str(group.timestamp)))
                    continue
            else:
                Logger.log("[MLT] ERR: Group {0} does not have an associated ROI! Skipping...".format(str(group.timestamp)))
                continue
            filename = str(roi.timestamp).replace(':', '-')
            thresholds = np.round(threshold_ratios * roi.qsun_intensity)
            Logger.debug("[MLT - run_thresholding_on_roi] Threshold pixel values: {0}".format(thresholds))

            layers = []
            for j in range(0, len(thresholds)):
                threshold = thresholds[j]
                if not self.overwrite_layers:
                    # only overwrite same layer if specified
                    if threshold_ratios[j] in group_layers.layer_thresholds:
                        continue
                layers.append(self.make_layer(roi.data, threshold, threshold_ratios[j], roi.timestamp, counter_dict))



            for layer in layers:
                group_layers.add_or_replace_layers(layer)
            group.mlt_path = group_layers.filename
            self.path_man.saveMLTData(group_layers, str(self.mlt_path))
            prog.update()
            if i == stop - 1:
                Logger.log("counter_dict: {0}".format(counter_dict))


if __name__ == "__main__":
    doAll = False
    sd = SpotData.SpotData(_base_dir='/home/rig12/Work/sunspots/2014-09-05_16-hourly')
    thresholds = [30000,20000,15000,8000]

    if doAll:
        mlt = MultiLevelThresholding(sd)
        roiList = sd.loadROIList(sd.getDir('roi'))
        prog = PrintProgress(0, len(roiList), interval=5, label="[MLT] Applying MLT... Progress: ")
        roiList.sort(key=lambda r: r.timestamp)
        for i in range(0,len(roiList)):
            filename = str(roiList[i].timestamp).replace(':','-')
            mlt.run_thresholding_on_roi(roiList[i], thresholds, filename=filename)
            prog.update()
    else:
        mlt = MultiLevelThresholding(sd)
        roi = sd.loadROI('2014-09-10_18-00-35')
        mlt.run_thresholding_on_roi(roi, thresholds, filename=str(roi.timestamp).replace(':', '-'))
