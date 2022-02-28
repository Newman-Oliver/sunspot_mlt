#Maths stuff
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#SunPy and Astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.coordinates.ephemeris import get_earth
from limb_darkening import limbdark

#Other
from pyclustering.cluster.optics import optics
import warnings
import time
import pickle

#My stuff
from SpotData import *
import SpotTools
from Logger import PrintProgress
import Logger


class SpotTracker():
    '''
    Identifies spot groups in a set of fits files. Use the trackSpots() method to produce a list of all the groups in
    a set of images. Is more accurate for higher cadance images (45s-1hr optimal).
    '''
    def __init__(self, path_manager, _comm, _config_parser, external_fits_dir=''):
        # Vars
        self.path_man = path_manager
        self.config_parser = _config_parser
        self.sun_rad_buffer = 10  # padding in pixels to remove edge effects. default: 10
        self.spot_intensity_threshold = self.config_parser.getfloat('Tracking','spot_intensity_threshold')  # % threshold for detection of spots. default: 0.5
        self.intensity_weighting = 3000.0
        self.fits_dir = self.path_man.getDir('fits')
        self.external_fits_dir = self.path_man.getDir('fits_ext',
                                                posix_path=external_fits_dir) if external_fits_dir is not '' else None
        self.figure_save_dir = self.path_man.getDir('tracking')
        self.numpy_save_dir = self.path_man.getDir('tracking_numpy', self.figure_save_dir / 'numpy')
        self.group_data_save_dir = self.path_man.getDir('tracking_group_data', self.figure_save_dir / 'group_data')

        if str(external_fits_dir) is not '':
            self.fits_files = [x for x in self.external_fits_dir.glob('*.fits') if x.is_file()]
        else:
            self.fits_files = [x for x in self.fits_dir.glob('*.fits') if x.is_file()]

        # Sort fits_files as glob takes from the os's order, the logic of which is lost to all but a few druids who
        # have foresaken the rest of us mere mortals and have ascended to the great directory above.
        #self.fits_files = sorted(self.fits_files)

        # Options
        # OPTICS parameters
        self.spot_epsilon = self.config_parser.getint('Tracking', 'optics_spot_epsilon')
        self.spot_min_pts = self.config_parser.getint('Tracking', 'optics_spot_min_pts')
        self.group_epsilon = self.config_parser.getint('Tracking', 'optics_group_epsilon')
        self.group_min_pts = self.config_parser.getint('Tracking', 'optics_group_min_pts')
        self.save_group_ordering = self.config_parser.getboolean('Tracking', 'save_group_ordering')
        # Plot figures of each individual frame and save them?
        self.make_figures_per_fits = self.config_parser.getboolean('Tracking', 'make_figures_per_fits')
        # Use the numpy data from previous runs if available? - mostly used to debug things later in the code that don't need the data to be re-acquired.
        self.useSavedData = self.config_parser.getboolean('Tracking', 'use_numpy_save_Data')
        # Used to make the last picture of all the groups.
        self.make_final_figure = self.config_parser.getboolean('Tracking', 'make_final_figure')
        # Make a figure after on each node after they have done their tracking.
        self.make_figures_per_node = self.config_parser.getboolean('Tracking', 'make_figures_per_node')
        # Make a figure showing the OPTICS dendrogram for each fits file?
        self.make_group_dendrogram = self.config_parser.getboolean('Tracking', 'make_group_dendrogram')
        self.load_group_data = self.config_parser.getboolean('Tracking', 'load_group_data')
        self.load_tracking_data = self.config_parser.getboolean('Tracking', 'load_tracking_data')
        self.do_per_node_pre_stitching = self.config_parser.getboolean('Tracking', 'per_node_pre_stitching')
        self.tracking_tolerance = self.config_parser.getint('Tracking', 'tracking_sensitivity')
        warnings.simplefilter('ignore',category=RuntimeWarning)

        # MPI
        self.comm = _comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def calculateWeight(self, pixelValue, threshold):
        '''Provides the weighting for a pixel value such that smaller values return a higher weight. The scaling is linear.'''
        return int(np.ceil((threshold - pixelValue) / self.intensity_weighting))

    def calculateQuietSun(self, correctedMap):
        return int(np.nanmean(correctedMap.data))

    def identifySpots(self, fitsPath, fitsMap, tryLoad = False):
        '''Finds pixels on a FITS image that are below a threshold considered to be a sunspot and return an array of their co-ordinates weighted
            such that darker co-ordinates are repeated based on how dark they appear. Then save this array to a numpy file for faster retrieval
            in subsequent runs.'''
        # Define a circle within which to search for sunspots
        sunRad = (fitsMap.rsun_obs / fitsMap.scale[0])
        sunCirc = Circle([fitsMap.reference_pixel[0].value,fitsMap.reference_pixel[1].value], sunRad.value - self.sun_rad_buffer)

        # Load
        if(tryLoad):
            try:
                Logger.debug("[Tracking - identifySpots] Attempting to load numpy data...")
                data = np.load(self.numpy_save_dir / (fitsPath.stem + ".npy"))
                return data
            except:
                Logger.debug("[Tracking - identifySpots] Load failed.")
                pass


        # Find sunspots in the Circle.
        weighted_spot_coords = []
        #rows = fitsMap.data.shape[0]
        #cols = fitsMap.data.shape[1]
        intensity_threshold = self.calculateQuietSun(fitsMap) * self.spot_intensity_threshold

        # New numpy way
        Logger.debug("[Tracking - identifySpots] Finding spot candidates...")
        coords_spot_pixels = np.argwhere(np.array(fitsMap.data) <= intensity_threshold)

        Logger.debug("[Tracking - identifySpots] Weighting sunspot candidates...")
        for i in range(0,len(coords_spot_pixels)):
            coord = coords_spot_pixels[i]
            weight = self.calculateWeight(fitsMap.data[coord[0],coord[1]], intensity_threshold)
            # add point multiple times for weighting.
            for k in range(0, weight):
                weighted_spot_coords.append([coord[1],coord[0]])

        Logger.debug("[Tracking - identifySpots] Saving...")
        weighted_spot_coords = np.array(weighted_spot_coords)
        # Check path exists
        self.path_man.getDir('dat',posix_path=self.numpy_save_dir)
        np.save(self.numpy_save_dir / (fitsPath.stem + ".npy"), weighted_spot_coords)
        Logger.debug("[Tracking - identifySpots] Done!")
        return weighted_spot_coords

    # Clustering with OPTICS
    def applyOPTICS(self, dataset, eps, minPts):
        Logger.debug("[Tracking - applyOPTICS] Applying optics, eps: {0} minpts: {1}...".format(eps, minPts))
        instance = optics(dataset,eps,minPts)
        instance.process()
        Logger.debug("[Tracking - applyOPTICS] Done!")
        return instance.get_clusters(), instance.get_ordering()

    def extractClusterCentres(self, clusters_, dataset, weighted=False):
        """
        Get the geometric centres of a list of cluster indices returned from the OPTICS clustering. If weighted,
        points are repeated in the list to apply a weighting to the average.
        :param clusters_:
        :param dataset:
        :param weighted:
        :return:
        """
        meanCoordsList = []
        for cluster in clusters_:
            clusterPoints = []
            for i in range(0, len(cluster)):
                clusterPoints.append(dataset[cluster[i]])
            clusterPointsArr = np.array(clusterPoints)
            centrePoint = [np.mean(clusterPointsArr[:,0]), np.mean(clusterPointsArr[:,1])]
            if weighted:
                for i in range(0,len(cluster)):
                    meanCoordsList.append(centrePoint)
            else:
                meanCoordsList.append(centrePoint)
        return meanCoordsList

    def getClusterCentresByIntensity(self, _clusters, coordinate_list, fits_map):
        """
        Get the centres of clusters generated by OPTICS by the location of the darkest point in the cluster. Always
        returns 1 value per cluster (the co-ordinate of the darkest point in that cluster)
        :param _clusters: A list of lists of integers that are indices of coordinate_list. Each sublist is a cluster
                            created by the OPTICS algorithm.
        :param coordinate_list: List of valid co-ordinates in fits_map.data to be used by _clusters.
        :param fits_map:
        :return:
        """
        darkest_coords_list = []
        for cluster in _clusters:
            pixel_values = []
            # Find all the pixel values for the elements in the cluster list
            for i in range(0, len(cluster)):
                coord = np.array(coordinate_list[cluster[i]])
                pixel_values.append(fits_map.data[coord[0]][coord[1]])
            index_of_min = np.argmin(pixel_values)
            darkest_coord = coordinate_list[cluster[index_of_min]]
            darkest_coords_list.append(darkest_coord)
        return darkest_coords_list



    def getSunspotGroups(self, data):
        # Do Clustering
        # Logger.log("Applying OPTICS...")
        clusters, spot_ordering = self.applyOPTICS(data,20,10)
        # Logger.log("Getting Centres...")
        centres = self.extractClusterCentres(clusters, data, weighted=True)

        # Get Clusters of clusters
        groups, group_ordering = self.applyOPTICS(centres,150,1)
        groupCentres = self.extractClusterCentres(groups, centres)

        # return number of sunspot groups
        return groupCentres

    def plotFigures(self, fitsMap, spot_snapshot_list, group_snapshot_list):
        fig = plt.figure()
        ax = plt.subplot(projection=fitsMap)
        im = fitsMap.rotate(angle = 180 * u.deg).plot()
        ax.set_autoscale_on(False)

        xc = []
        yc = []
        gxc = []
        gyc = []

        for i in range(0,len(spot_snapshot_list)):
            coord = spot_snapshot_list[i].centre
            xc.append(((coord[0]-2048.)/2.) * u.arcsec)
            yc.append(((coord[1]-2048.)/2.) * u.arcsec)

        for i in range(0,len(group_snapshot_list)):
            coord = group_snapshot_list[i].centre
            gxc.append(((coord[0]-2048.)/2.) * u.arcsec)
            gyc.append(((coord[1]-2048.)/2.) * u.arcsec)

        coords = SkyCoord(xc*u.arcsec,yc*u.arcsec,frame=fitsMap.coordinate_frame)
        groupCoords = SkyCoord(gxc*u.arcsec,gyc*u.arcsec,frame=fitsMap.coordinate_frame)
        p = ax.plot_coord(coords, 'x')
        p2 = ax.plot_coord(groupCoords, '.',color='red')

        filename = fitsMap.date.strftime('%Y-%m-%d_%H-%M-%S') + '.jpg'
        plt.savefig(str(self.figure_save_dir / filename), dpi=300)
        plt.close(fig)

    def plotGroupsOverTime(self, fitsPath, spotGroupList, fileprefix=''):
        fitsMap = sunpy.map.Map(str(fitsPath.resolve()))
        corrected_map = limbdark(fitsMap)
        fig = plt.figure()
        ax = plt.subplot(projection=fitsMap)
        im = corrected_map.rotate(angle = 180 * u.deg).plot()
        ax.set_autoscale_on(False)

        Logger.log("DEBUG: len(spotGroupList): " + str(len(spotGroupList)), Logger.LogLevel.debug)
        for i in range(0,len(spotGroupList)):
            xc = []
            yc = []
            for j in range(0,len(spotGroupList[i].history)):
                coord = spotGroupList[i].history[j].centre
                xc.append(((coord[0]-2048.)/2.) * u.arcsec)
                yc.append(((coord[1]-2048.)/2.) * u.arcsec)
            coords = SkyCoord(xc*u.arcsec,yc*u.arcsec,frame=fitsMap.coordinate_frame)
            p = ax.plot_coord(coords, SpotTools.get_marker(i, use_symbol=True),
                              color=SpotTools.get_colour(i), label=str(i))
        ax.legend(bbox_to_anchor=[1.3,1.1],ncol=2)
        filename = fileprefix + '_group_tracking_' + fitsMap.date.strftime('%Y-%m-%d_%H-%M-%S') + '.jpg'
        plt.savefig(str(self.figure_save_dir / filename), dpi=300)
        plt.close(fig)

    def getDarkestPoint(self, fitsMap, data, clusterPoints):
        '''Data is needed because it contains the redundant points added in the identifySpots() routine.'''
        pts = []
        for i in range(0,len(clusterPoints)):
            pts.append(fitsMap.data[data[clusterPoints[i]]])
        return data[clusterPoints[np.argmin(pts)]]

    def spotsAreClose(self, spot1, spot2, tolerance=None, giveXDist=False):
        '''
        Checks if two SpotSnapshot, GroupSnapshots, or SkyCoords are close to each other as defined by
        self.getGroupingTolerance. By default returns a bool, giveXDist makes function return a value for the distance.
        NOTE: Earlier on in the code numpy switches the X and Y meaning that the ydist is the xdist. This needs fixing
        but was only discovered after a whole lot of other stuff was written so I don't want to poke that hornets nest
        yet.
        :param spot1: SpotSnapshot, GroupSnapshot, or SkyCoord object for the first spot.
        :param spot2: SpotSnapshot, GroupSnapshot, or SkyCoord object for the second spot.
        :param tolerance: A real number value for the maximum acceptable squared separation in the x axis of two points.
                          This value is required if spot1 and spot 2 are SkyCoords.
        :param giveXDist: Set to true if a numerical output is desired.
        :return: Boolean or float, if giveXDist true.
        '''
        if tolerance is None:
            if (not isinstance(spot1,(SpotSnapshot, GroupSnapshot)) or
                not isinstance(spot2, (SpotSnapshot, GroupSnapshot))):
                raise ValueError("tolerance must be specified if spot1 or spot2 are not of type " +
                                 "SpotSnapshot or GroupSnapshot.")
            tolerance = self.getGroupingTolerance(spot1.timestamp, spot2.timestamp)

        if isinstance(spot1,(SkyCoord)):
            xdist = (spot1.Tx.value - spot2.Tx.value) ** 2
            ydist = (spot1.Ty.value - spot2.Ty.value) ** 2
        else:
            xdist = (spot1.centre_arcsec[0] - spot2.centre_arcsec[0]) ** 2
            ydist = (spot1.centre_arcsec[1] - spot2.centre_arcsec[1]) ** 2

        is_close = False

        if  (ydist < tolerance and xdist < tolerance/5):
            is_close = True

        # This is not a wrong. See header text
        return ydist if giveXDist else is_close


    def getSnapshots(self, fitsPath, plot_figures = False):
        '''
        Cluster dark pixels into sunspots, and then cluster sunspots into groups. Make data files for each of these for
        this image (snapshots).
        :param fitsPath: A Path object to where the fits files are kept
        :param spotEpsion: OPTICS epsilon parameter for spot clustering. "Radius" to search for adjacent points
        :param spotMinPts: OPTICS MinPts parameter for spot Clusterting. How many pixels must be in radiuis Epsilon to consider a core point?
        :param groupEpsilon: OPTICS epsilon parameter for group clustering.
        :param groupMinPts: OPTICS MinPts parameter for group clustering.
        :param plot_figures: Should figures showing the locations of the spots and the groups be made?
        :return: A list, first index of which is the list of spot snapshots in the image, second index is the group snapshots.
        '''
        # Data contianers for this file
        spotSnapshotList = []
        groupSnapshotList = []

        # Open file
        try:
            fitsMap = sunpy.map.Map(str(fitsPath.resolve()))
        except TypeError:
            Logger.log("[Tracking] Caught type error on fitsPath: {0}".format(str(fitsPath.resolve())))
            return [[],[]]

        Logger.debug("[Tracking - getSnapshots] Correcting for limb darkening...")
        corrected_map = limbdark(fitsMap)
        qsun_intensity = self.calculateQuietSun(corrected_map)
        data = self.identifySpots(fitsPath, corrected_map, self.useSavedData)

        # get clusters
        clusters, spot_ordering = self.applyOPTICS(data, self.spot_epsilon, self.spot_min_pts)
        centres = self.extractClusterCentres(clusters, data)
        #centres = self.getClusterCentresByIntensity(clusters, data, corrected_map)

        # Get Clusters of clusters
        weightedCentres = self.extractClusterCentres(clusters,data, weighted=True)
        #weightedCentres = self.getClusterCentresByIntensity(clusters, data, corrected_map)
        groups, group_ordering = self.applyOPTICS(weightedCentres, self.group_epsilon, self.group_min_pts)
        groupCentres = self.extractClusterCentres(groups, weightedCentres)
        #groupCentres = self.getClusterCentresByIntensity(groups, weightedCentres, corrected_map)
        if groupCentres == []:
            groupCentres = centres

        #if self.make_group_dendrogram:


        # Record Sunspots
        Logger.debug("[Tracking - getSnapshots] Finding Sunspots...")
        for j in range(0, len(centres)):
            spotshot = SpotSnapshot(filename=str(fitsPath.name),
                                    path = str(fitsPath.resolve()),
                                    timestamp=fitsMap.date,
                                    centre=centres[j],
                                    _size=len(clusters[j]))
            spotshot.darkestPoint = self.getDarkestPoint(fitsMap,data,clusters[j])
            spotshot.centre_arcsec = [(spotshot.centre[0]-fitsMap.reference_pixel[0].value)*fitsMap.scale[0].value,
                                      (spotshot.centre[1]-fitsMap.reference_pixel[1].value)*fitsMap.scale[1].value]
            spotshot.darkestPoint_arcsec = [(spotshot.darkestPoint[0]-fitsMap.reference_pixel[0].value)*fitsMap.scale[0].value,
                                            (spotshot.darkestPoint[0]-fitsMap.reference_pixel[0].value)*fitsMap.scale[0].value]
            spotSnapshotList.append(spotshot)

        # Record Groups
        Logger.debug("[Tracking - getSnapshots] Finding Sunspot Groups...")
        for j in range(0,len(groupCentres)):
            groupSnapshot = GroupSnapshot(filename=str(fitsPath.name),
                                          path = str(fitsPath.resolve()),
                                          timestamp=fitsMap.date,
                                          _centre=groupCentres[j],
                                          _qsun_intensity=qsun_intensity)
            if self.save_group_ordering:
                groupSnapshot.optics_ordering = spot_ordering
            groupSnapshot.centre_arcsec = [
                                    (groupSnapshot.centre[0]-fitsMap.reference_pixel[0].value)*fitsMap.scale[0].value,
                                    (groupSnapshot.centre[1]-fitsMap.reference_pixel[1].value)*fitsMap.scale[1].value]
            groupSnapshotList.append(groupSnapshot)

        # Match spots to groups
        Logger.debug("[Tracking - getSnapshots] Matching sunspots to groups...")
        for spot in spotSnapshotList:
            distances = []
            for j in range(0,len(groupSnapshotList)):
                sqrDist = (spot.centre[0] - groupSnapshotList[j].centre[0]) ** 2 +\
                          (spot.centre[1] - groupSnapshotList[j].centre[1]) ** 2
                distances.append([sqrDist,j])
            distances.sort(key= lambda dist: dist[0])
            closestGroupIndex = distances[0]

            # If near enough the centre of an existing group, join that. Else make a new one around at current position.
            if closestGroupIndex[0] < self.group_epsilon**2:
                closestGroup = groupSnapshotList[closestGroupIndex[1]]
                closestGroup.memberSpots.append(spot)
                spot.groupID = groupSnapshotList[closestGroupIndex[1]].id
            else:
                groupSnapshot = GroupSnapshot(filename=str(fitsPath.name),
                                              path = str(fitsPath.resolve()),
                                              timestamp=fitsMap.date,
                                              _centre=spot.centre,
                                              _qsun_intensity=qsun_intensity)
                groupSnapshot.centre_arcsec = [
                    (groupSnapshot.centre[0] - fitsMap.reference_pixel[0].value) * fitsMap.scale[0].value,
                    (groupSnapshot.centre[1] - fitsMap.reference_pixel[1].value) * fitsMap.scale[1].value]
                groupSnapshot.memberSpots.append(spot)
                spot.groupID = groupSnapshot.id
                groupSnapshotList.append(groupSnapshot)

        # Plot image and wait for input to move on
        if plot_figures:
            self.plotFigures(corrected_map,spotSnapshotList,groupSnapshotList)
            Logger.log("[Tracking - Plot] Number of spots: " + str(len(spotSnapshotList)), Logger.LogLevel.debug)
            Logger.log("[Tracking - Plot] Number of groups: " + str(len(groupSnapshotList)), Logger.LogLevel.debug)

        # Ensure everything is cleared. Probably unnecessary now.
        clusters.clear()
        centres.clear()
        groups.clear()
        groupCentres.clear()
        del data

        return [spotSnapshotList,groupSnapshotList]

    def divideTasks(self, dataLength, cpuCount):
        self.rank
        N = dataLength
        count = np.floor(N/cpuCount)
        remainder = N % cpuCount
        start = 0
        stop = 0

        # Get start and stop points for each MPI process
        if self.rank < remainder:
            start = self.rank * (count+1)
            stop = start + count
        else:
            start = (self.rank * count) + remainder
            stop = start + (count-1)

        return start, stop

    def getGroupingTolerance(self, time1, time2):
        '''
        Returns a value for the maximum distance a group can be from the expected position and still be classified as the
        same group. This value is scaled linearly based on the time difference between the two images, and on the
        observation that 1000 is a good tolerance for data separated by 1 hour. The threshold is clamped between 1000 and
        50000.
        '''
        time_delta = (time2-time1).total_seconds()
        threshold = 2000 * abs(time_delta / 3600)
        if threshold < 1000:
            threshold = 1000
        elif threshold > 50000:
            threshold = 50000
        return round(threshold)


    def createGroups(self):
        # Define vars
        dataPerImage = []

        # Get the start and stop for each process
        mpiStart, mpiStop = self.divideTasks(len(self.fits_files), self.size)

        mpiStart = mpiStart.astype(np.int64)    # Make sure that start and stop are ints
        mpiStop = mpiStop.astype(np.int64) + 1

        # Each process does their bit
        prog = PrintProgress(mpiStart, mpiStop, label="Node: {0} Processing: {1} - {2} Progress: ".format(
            self.rank, mpiStart, mpiStop-1))
        for i in range(mpiStart, mpiStop):
            Logger.debug("[Tracking - createGroups] Getting groups on fits file: {0}".format(self.fits_files[i].resolve()))
            dataPerImage.append(self.getSnapshots(self.fits_files[i], plot_figures=self.make_figures_per_fits))
            prog.update()

        # Collect data and sort
        self.comm.Barrier()
        results = self.comm.allgather(dataPerImage)
        dataPerImage.clear()

        # Sometimes results is put in a list as the lone entry, but not all the time(?) so try and deal with that
        try:
            Logger.log("/// DEBUG: Node {0} reports len(results) = {1} and len(results[1]) = {2}".format(
                self.rank, len(results), len(results[1])
            ), Logger.LogLevel.debug)
            for i in range(0,len(results)):
                dataPerImage += results[i]
        except:
            results = results[0]
            Logger.log("/// DEBUG: Node {0} reports len(results) = {1} and len(results[1]) = {2}".format(
                self.rank, len(results), len(results[1])
            ), Logger.LogLevel.debug)
            dataPerImage = results

        Logger.log("[Tracking] len(dataPerImage[0]) = {0}, type(dataPerImage[0]) = {1}, len(dataPerImage) = {2}"
                   .format(len(dataPerImage[0]), type(dataPerImage[0]), len(dataPerImage)) +
                   " type(dataPerImage[0][0] = {0}, type(dataPerImage[1][0] = {1}"
                   .format(type(dataPerImage[0][0]), type(dataPerImage[1][0])), Logger.LogLevel.debug)

        # Weed out bad values before sorting
        # First check for incorrect length of data lists (should be 2, 1 for spots, one for groups)
        bad_entries = []
        for data in dataPerImage:
            if len(data) != 2:
                Logger.log("[Tracking] Value in dataPerImage had a length that wasn't 2. ({0})".format(len(data)),
                           Logger.LogLevel.debug)
                bad_entries.append(data)
        Logger.log("[Tracking] There were {0} bad entries.".format(len(bad_entries)), Logger.LogLevel.debug)
        dataPerImage = [d for d in dataPerImage if len(d) == 2]

        # Then check for the length of the group list being 0
        bad_entries = []
        for data in dataPerImage:
            if len(data[1]) == 0:
                Logger.log("[Tracking] Found an element with a group list length of 0", Logger.LogLevel.debug)
                bad_entries.append(data)
        Logger.log("[Tracking] There were {0} bad entries.".format(len(bad_entries)), Logger.LogLevel.debug)
        dataPerImage = [d for d in dataPerImage if len(d[1]) > 0]

        try:
            dataPerImage.sort(key=lambda gp: gp[1][0].timestamp)
        except IndexError:
            Logger.log("[Tracking] Index out of range again. Skipping sort... :/", Logger.LogLevel.debug)
            Logger.log("[Tracking] len(dataPerImage[0]) = {0}, type(dataPerImage[0]) = {1}, len(dataPerImage) = {2}"
                       .format(len(dataPerImage[0]), type(dataPerImage[0]), len(dataPerImage)) +
                       " type(dataPerImage[0][0] = {0}, type(dataPerImage[1][0] = {1}"
                       .format(type(dataPerImage[0][0]), type(dataPerImage[1][0])), Logger.LogLevel.debug)
            #dataPerImage.sort(key=lambda gp: gp[0][0].timestamp)

        # Synchronise across nodes
        dataPerImage = self.comm.bcast(dataPerImage, root=0)

        # Save the group tracking data in case you crash
        Logger.log("[Tracking] Saving group tracking data...")
        filename = 'dataPerImage_{0}.gdat'.format(self.rank)
        group_data_save_path = self.group_data_save_dir / filename
        with group_data_save_path.open('wb') as file_object:
            pickle.dump(dataPerImage, file_object)
        Logger.log("[Tracking] Node {0} group tracking data saved!".format(self.rank))

        return dataPerImage

    def trackGroups(self, dataPerImage):
        '''
        Tracks spots across multiple images using MPI and produces a list of SpotData.SpotGroup() objects. This list
        may contain multiple copies of the same spot group identified by different processes and so needs stitchMPI to
        be run on its results to remove duplicates.

        :param dataPerImage: A list of sunspot and group snapshots as returned by createGroups()
        :return: a list of SpotGroup() objects, one for each spot group tracked across all images.
        '''
        # Reset MPI start/stop points.
        mpiStart, mpiStop = self.divideTasks(len(dataPerImage), self.size)
        mpiStart = mpiStart.astype(np.int64)    # Make sure that start and stop are ints
        mpiStop = mpiStop.astype(np.int64) + 1

        sunspotList = []
        sunspotGroupList = []

        prog_trk = Logger.PrintProgress(mpiStart, mpiStop, 
                                        label="[Tracking] Node {0} Tracking spot groups...".format(self.rank))
        for i in range(mpiStart, mpiStop):
            #skip last one because of i + 1
            if(i+1 >= len(dataPerImage)):
                break

            # Get image data
            imageDataCurrent = dataPerImage[i]
            groupDataCurrent = imageDataCurrent[1]

            try:
                imageDataNext = dataPerImage[i+1]
            except IndexError:
                Logger.log("i="+str(i), Logger.LogLevel.debug)
                Logger.log(dataPerImage, Logger.LogLevel.debug)
                raise IndexError

            groupDataNext = imageDataNext[1]

            for j in range(0,len(groupDataCurrent)):
                # Get the estimated position of the group in the next frame using differential rotation.
                groupSnapshot = groupDataCurrent[j]
                projectedPositon = solar_rotate_coordinate(groupSnapshot.getSkyCoord(), groupDataNext[0].timestamp)
                groupSnapshot.projected_next_location = projectedPositon
                linkedNextGroup = None
                distances = []
                tolerance = self.tracking_tolerance #self.getGroupingTolerance(groupSnapshot.timestamp, groupDataNext[0].timestamp)
                # If the spot is close enough to its estimated position, add it to the global SpotGroup class if one exists,
                # else make one and add it to it. The expected position is intentionally recorded here also, for use in the
                # next step.
                for k in range(0, len(groupDataNext)):
                    nextCoord = groupDataNext[k].getSkyCoord()
                    groupDataNext[k].projected_next_location = nextCoord
                    #sqrDist = self.spotsAreClose(projectedPositon, nextCoord, tolerance=tolerance, giveXDist=True)
                    sqrDist = ((groupSnapshot.getSkyCoord().Tx.value - nextCoord.Tx.value)**2
                               + (groupSnapshot.getSkyCoord().Ty.value - nextCoord.Ty.value)**2)
                    distances.append((sqrDist,k))
                # Find the closest group to the next expected location
                distances.sort(key=lambda dist: dist[0])
                # If the nearest group is in tolerance, make it the next likely group
                if(distances[0][0] <= tolerance):
                    linkedNextGroup = groupDataNext[distances[0][1]]
                if linkedNextGroup is not None:
                    if groupSnapshot.parent is None:
                        spotGroup = SpotGroup()
                        spotGroup.history.append(groupSnapshot)
                        groupSnapshot.parent = spotGroup
                        sunspotGroupList.append(spotGroup)
                    groupSnapshot.parent.history.append(linkedNextGroup)
                    linkedNextGroup.parent = groupSnapshot.parent

            prog_trk.update()

        # Sort and attempt Stitch
        print("[Node {0}] Performing self-stitch...".format(self.rank))
        sunspotGroupList = self.stitchMPI(sunspotGroupList)

        Logger.log("DEBUG: Node {0} reports {1} groups.".format(self.rank,len(sunspotGroupList)), Logger.LogLevel.debug)
        for i in range(0,len(sunspotGroupList)):
            Logger.log("DEBUG: [Node: {0}] [Element: {1}] {2}".format(self.rank, i, sunspotGroupList[i]),
                       Logger.LogLevel.debug)

        # Make a figure for the tracking post-stitching for all groups found by this node.
        if self.make_figures_per_node and len(sunspotGroupList) <= 120:
            self.plotGroupsOverTime(self.fits_files[0], sunspotGroupList, fileprefix='{0}_pre-stitch'.format(self.rank))
        elif self.make_figures_per_node and len(sunspotGroupList) > 120:
            print("DEBUG WARN: Too many spots, cannot display post-stitch figure for node {0}.".format(self.rank))

        # Collect results from all nodes
        results = self.comm.allgather(sunspotGroupList)
        Logger.debug("[Tracking] len(results) = {0} type(results[0]) = {1}".format(len(results), type(results[0])))
        sunspotGroupList.clear()
        for nodeResult in results:
            Logger.debug("[Tracking] type(nodeResult[0])".format(type(nodeResult[0])))
            # If the node result is a list of lists, go one step deeper to grab the SpotGroup objects.
            if isinstance(nodeResult[0], list):
                for instance in nodeResult:
                    sunspotGroupList.extend(instance)
            else:
                sunspotGroupList.extend(nodeResult)

        Logger.debug("DEBUG: Post AllGather Node {0} reports {1} groups.".format(self.rank, len(sunspotGroupList)))
        for i in range(0,len(sunspotGroupList)):
            Logger.log("DEBUG: [Node: {0}] [Element: {1}] {2}".format(self.rank, i, sunspotGroupList[i]),
                       Logger.LogLevel.debug)

        # Save the group tracking data in case you crash
        Logger.log("[Tracking] Saving group tracking data...")
        filename = 'sunspotGroupList_{0}.gdat'.format(self.rank)
        group_data_save_path = self.group_data_save_dir / filename
        with group_data_save_path.open('wb') as file_object:
            pickle.dump(sunspotGroupList, file_object)
        Logger.log("[Tracking] Node {0} group tracking data saved!".format(self.rank))

        return sunspotGroupList

    def stitchMPI(self, results, use_MPI=False):
        '''
        Checks for duplicate entries in 'results' from self.trackGroups().
        :param results: A list of SpotData.SpotGroup() objects found by trackGroups()
        :return: A list of SpotData.SpotGroup() objects with duplicates removed
        '''
        for i in range(0, len(results)):
            results[i].history.sort(key=lambda it: it.timestamp)
        results = [r for r in results if len(r.history) > 0]
        results.sort(key=lambda it: it.history[0].timestamp)
        Logger.debug("[Tracking - Stitching] Node {0} reports pre-stitch spot count of {1}".format(self.rank,
                                                                                                   len(results)))

        linked_spot_pairs = []
        sunspotGroupList = []

        # Declaring the start and stop points for MPI if used. If used, each process will be given a selection of groups
        # in the spot group list and the likely links combined at the end.
        if use_MPI:
            start, stop = self.divideTasks(len(results), self.size)
            start = start.astype(np.int64)  # Make sure that start and stop are ints
            stop = stop.astype(np.int64)
            label_ = "Node {0} Identifying stitch locations for groups {1} to {2}...".format(self.rank, start, stop)
        else:
            start = 0
            stop = len(results)
            label_ = "Node {0} Identifying stitch locations alone...".format(self.rank)


        prog_stitch_id = PrintProgress(start, stop, label=label_)
        for i in range(start, stop):
            try:
                linked_spot_pairs.append([results[i].id])
            except IndexError:
                Logger.log("[Tracking - Stitching]: i = ", i, Logger.LogLevel.debug)
                continue
            lastSnapshot = results[i].history[len(results[i].history)-1]
            likelyLinks = []
            # Compare the last entry in current spot's history to every entry in each other spot's history and look
            # for any close enough to be considered the same spot.
            for j in range(0, len(results)):
                # Skip if huge time difference, or if is itself.
                if not results[i].isSpotCloseInTime(results[j]): continue
                if results[i].id == results[j].id: continue

                # The 50 here is the maximum number of iterations of another history to look at.
                # it's a time saving feature.
                for k in range(0 ,np.min([len(results[j].history), 50])):
                    potentialLink = results[j].history[k]
                    group_projected_location = solar_rotate_coordinate(lastSnapshot.getSkyCoord(), potentialLink.timestamp)
                    next_location = potentialLink.getSkyCoord()
                    tolerance = self.tracking_tolerance #self.getGroupingTolerance(lastSnapshot.timestamp, potentialLink.timestamp)
                    #sqrDist = self.spotsAreClose(projectedPositon, nextCoord, tolerance=tolerance, giveXDist=True)
                    sqr_dist_proj = ((lastSnapshot.getSkyCoord().Tx.value - next_location.Tx.value)**2
                                    + (lastSnapshot.getSkyCoord().Ty.value - next_location.Ty.value)**2)
                    if sqr_dist_proj <= tolerance:
                        #sqr_dist_proj += ((lastSnapshot.getSkyCoord().Tx.value - next_location.Tx.value) ** 2 +
                        #                  (lastSnapshot.getSkyCoord().Ty.value - next_location.Ty.value) **2)
                        likelyLinks.append([sqr_dist_proj, results[j].id, k])
                        Logger.debug("[Tracking - Stitching] These two spots have snapshots within {0} pixels: \n{1}\n{2}".format(sqr_dist_proj, results[i], results[j]))
                        break
            likelyLinks.sort(key=lambda dist: dist[0])
            if len(likelyLinks) > 0:
                try:
                    index = linked_spot_pairs.index([results[i].id])
                    linked_spot_pairs[index].append([likelyLinks[0][1], likelyLinks[0][2]])
                except IndexError:
                    Logger.log("i = {0} len(linked_spot_pairs) = {1}, len(likelyLinks[0]) = {2}".format(
                        i, len(linked_spot_pairs), len(likelyLinks[0])), Logger.LogLevel.debug)
            prog_stitch_id.update()

        if use_MPI:
            linked_spot_pairs = self.comm.allgather(linked_spot_pairs)
        if len(linked_spot_pairs) == 1:
            linked_spot_pairs = linked_spot_pairs[0]

        # Stitch the spots together
        if self.rank == 0 or not use_MPI:
            badSpots = []
            prog_stitch_do = PrintProgress(0,len(linked_spot_pairs),label="Node {0} stitching spots...".format(self.rank))
            for i in range(len(linked_spot_pairs)-1,-1,-1):
                if len(linked_spot_pairs[i]) > 1:
                    Logger.log("[Tracking] linked_spot_pairs[i]: {0}".format(linked_spot_pairs[i]),
                               Logger.LogLevel.debug)
                    this_spot = SpotTools.first(elem for elem in results if elem.id == linked_spot_pairs[i][1][0])
                    new_spot_parent = SpotTools.first(elem for elem in results if elem.id == linked_spot_pairs[i][0])
                    try:
                        new_spot_parent.history.extend(this_spot.history)
                    except AttributeError as e:
                        Logger.log("[Tracking] Something bad happened: " + str(e), Logger.LogLevel.debug)
                        Logger.log("Attempting to skip...", Logger.LogLevel.debug)
                        continue
                    badSpots.append(this_spot.id)
                prog_stitch_do.update()

            # Remove small entries that are likely glitches
            for group in results:
                if len(group.history) < 3:
                    badSpots.append(group.id)

            results = [spot for spot in results if spot.id not in badSpots]
            Logger.log("[Tracking] Node {0} reports spot count of {1} before allgather".format(self.rank, len(results)),
                       Logger.LogLevel.debug)
        if use_MPI:
            results = self.comm.allgather(results)
            Logger.log("[Tracking] Node {0} reports final post-stitch spot count of {1}".format(self.rank, len(results)),
                       Logger.LogLevel.debug)
        return results

    def loadGrpData(self, rank):
        path = self.group_data_save_dir / 'dataPerImage_{0}.gdat'.format(rank)
        with path.open('rb') as f:
            dataPerImage = pickle.load(f)
        return dataPerImage

    def loadTrackingData(self, rank):
        path = self.group_data_save_dir / 'sunspotGroupList_{0}.gdat'.format(rank)
        with path.open('rb') as f:
            sunspotGroupList = pickle.load(f)
        return sunspotGroupList

    def trackSpots(self):
        sunspotGroupList = []
        # A little buffer to stagger starts
        time.sleep(self.rank)
        # ---------------------------------------
        # Apply functions: createGroups() -> trackGroups() -> stitchMPI()
        # Find the groups and spots in the images
        if not self.load_group_data:
            Logger.log("[Tracking] Identifying spots and groups...")
            dataPerImage = self.createGroups()
        else:
            Logger.log("[Tracking] Loading spot group data...")
            dataPerImage = self.loadGrpData(self.rank)

        # Track the groups across images
        if self.load_tracking_data:
            Logger.log("[Tracking] Loading tracking data...")
            groups = self.loadTrackingData(self.rank)
        else:
            Logger.log("[Tracking] Tracking groups...")
            groups = self.trackGroups(dataPerImage)

        # Prep for pre-merge per-node stitching if it is being done
        if self.do_per_node_pre_stitching:
            prog_dup = PrintProgress(0, len(groups), label="[Tracking] Removing duplicate spots...")
            for group in groups:
                group.removeDuplicateSnapshots() # important for Stitching!
                prog_dup.update()

        # Stitch together at the seams caused by MPI
        if self.do_per_node_pre_stitching:
            Logger.log("[Tracking] Stitching MPI seams...")
            groups = self.stitchMPI(groups,use_MPI=False)

        # Check groups is in the correct format
        if isinstance(groups[0], list):
            groups = groups[0]

        # Find snapshot where group is in the centre of the solar disk or at closest approach
        prog_cent = PrintProgress(0, len(groups), label="[Tracking] Finding spot centre times...")
        for group in groups:
            group.getCentreTime()
            prog_cent.update()

        sunspotGroupList = self.comm.bcast(groups)
        # ---------------------------------------

        # Make a pretty picture
        Logger.log('[Tracking] There were {0} sunspot groups found.'.format(str(len(sunspotGroupList))))
        if self.make_final_figure and self.rank == 0:
            self.plotGroupsOverTime(self.fits_files[0], sunspotGroupList)

        return sunspotGroupList

