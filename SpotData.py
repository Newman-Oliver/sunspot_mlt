# Science libraries
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import numpy as np

from Logger import PrintProgress
import Logger
import SpotTools

# Misc Libraries
import pickle
from pathlib import Path
import datetime
from functools import lru_cache


class Sunspot():
    """Data container for sunspots."""
    def __init__(self, id="", TRef=None, history=None):
        self.id = id
        self.TRef = TRef
        self.history = history

    def __contains__(self, item):
        return item in self.history


class SpotSnapshot():
    """Data container for a single sunspot image."""
    def __init__(self, _id=None, _groupID= None, filename="", timestamp=None, centre=None, _centre_arcsec=None, _size=0,
                 darkestPoint=None, _darkestPoint_arcsec=None, path=""):
        self.id = id(self) if _id is None else _id
        self.groupID = _groupID
        self.path = Path(path)
        self.filename = filename
        self.timestamp = timestamp if isinstance(timestamp, datetime.datetime) else datetime.datetime.strptime(
                                                                                    timestamp, '%Y-%m-%d_%H-%M-%S')
        self.centre = [] if centre is None else centre
        self.centre_arcsec = [] if _centre_arcsec is None else _centre_arcsec
        self.darkestPoint = [] if darkestPoint is None else darkestPoint
        self.darkestPoint_arscec = [] if _darkestPoint_arcsec is None else _darkestPoint_arcsec
        self.size = _size

    def __reduce__(self):
        return (self.__class__, (self.id, self.groupID, str(self.filename), self.timestamp, self.centre, self.centre_arcsec,
                                 self.size, self.darkestPoint, self.darkestPoint_arscec))

    def __eq__(self, other):
        return other.size == self.size

    def __lt__(self, other):
        return other.size < self.size

    def __le__(self, other):
        return other.size <= self.size

    def __gt__(self, other):
        return other.size > self.size

    def __ge__(self, other):
        return other.size >= self.size


class SpotGroup():
    """Data container for a group in a single image"""
    def __init__(self, _id=None, _centred_snapshot=None, _isCloseToCentre=False, _history=None):
        self.id = id(self) if _id is None else _id
        self.centred_snapshot = _centred_snapshot
        self.isCloseToCentre = _isCloseToCentre  # Does the group get to within approx. +/-100pix of the centre?
        self.minROI = [300,300]
        self.history = [] if _history is None else _history

    def __reduce__(self):
        return (self.__class__,(self.id, self.centred_snapshot, self.isCloseToCentre, self.history))

    def __str__(self):
        span = "x: {0} -> {1} y: {2} -> {3}".format(self.history[0].centre_arcsec[0],
                                                    self.history[len(self.history)-1].centre_arcsec[0],
                                                    self.history[0].centre_arcsec[1],
                                                    self.history[len(self.history) - 1].centre_arcsec[1])
        reachesCentre = "False" if not self.isCloseToCentre else str(self.centred_snapshot.timestamp)
        nextLoc = "X: {0} y: {1}".format(self.history[len(self.history) - 1].projected_next_location.Tx.value,
                                         self.history[len(self.history) - 1].projected_next_location.Ty.value)
        return "Group ID: {0} Spanning: {2} Next loc: {1}".format(str(self.id),nextLoc, span)

    def __contains__(self, item):
        return item in self.history

    def isConsecutiveSpot(self, other):
        return self.history[len(self.history)-1].timestamp < other.history[0].timestamp

    def isSpotCloseInTime(self,other):
        # TODO: make this less hard-coded
        return (abs(self.history[len(self.history)-1].timestamp
                    - other.history[0].timestamp).total_seconds()) // 3600 < 12

    def sortHistory(self):
        self.history.sort(key=lambda spot: spot.timestamp)

    def removeDuplicateSnapshots(self):
        """Removes duplicate entries in sunspot history."""
        dupes = []
        for i in range(0, len(self.history)):
            # Skip if this spot already marked as duplicate
            if self.history[i].id in dupes:
                continue
            for j in range(0, len(self.history)):
                # SKip yourself
                if i == j:
                    continue
                is_same = (self.history[i].centre == self.history[j].centre and
                           self.history[i].filename == self.history[j].filename and
                           self.history[i].timestamp == self.history[j].timestamp)
                if is_same:
                    dupes.append(self.history[j].id)
        self.history = [spot for spot in self.history if spot.id not in dupes]

    def getCentreTime(self):
        closest_approach = 999999999
        centre = [0,0]
        for i in range(0,len(self.history)):
            snapshot = self.history[i]
            sqrXDist = (snapshot.centre_arcsec[0] - centre[0])**2
            if sqrXDist < closest_approach:
                closest_approach = sqrXDist
                self.centred_snapshot = snapshot
                if sqrXDist < 50**2:
                    self.isCloseToCentre = True
        return self.centred_snapshot.timestamp


class GroupSnapshot():
    """Data container for a group across all images"""
    def __init__(self, _id=None, filename="", timestamp=None, _qsun_intensity=None, _centre=None, _centre_arcsec=None,
                 _projected_next_location=None, _memberSpots=None, _ROI=None, path="", _mlt_path=None, _ordering=None):
        self.id = _id if _id is None else id(self)
        self.parent = None
        self.path = Path(path) if isinstance(path, str) else None
        self.filename = filename
        self.timestamp = timestamp if isinstance(timestamp, datetime.datetime) else datetime.datetime.strptime(
                                                                                    timestamp, '%Y-%m-%d_%H-%M-%S')
        self.qsun_intensity = _qsun_intensity
        self.centre = [] if _centre is None else _centre
        self.centre_arcsec = [] if _centre_arcsec is None else _centre_arcsec
        if isinstance(_projected_next_location,list):
            self.projected_next_location = self.getSkyCoord(_projected_next_location)
        else:
            self.projected_next_location = _projected_next_location
        self.memberSpots = [] if _memberSpots is None else _memberSpots
        self.minROI = [300,300]
        self.ROI_path = _ROI
        self.mlt_path = _mlt_path
        self.optics_ordering = _ordering

    def __reduce__(self):
        nextCoords = None
        if self.projected_next_location is not None:
            nextCoords = [self.projected_next_location.Tx.value,self.projected_next_location.Ty.value]
        return (self.__class__, (self.id, str(self.filename), self.timestamp, self.qsun_intensity, self.centre,
                                 self.centre_arcsec, nextCoords, self.memberSpots, self.ROI_path, str(self.path),
                                 self.mlt_path, self.optics_ordering))

    def __str__(self):
        return "Sunspot group snapshot at time {0} and position {1} containing {2} umbras.".format(
                self.timestamp.strftime('%Y-%m-%d_%H-%M-%S'), str(self.centre), str(len(self.memberSpots)))

    def getSkyCoord(self, coord=None):
        if coord == None:
            if self.centre_arcsec is not None:
                coord = self.centre_arcsec
            else:
                return None
        elif isinstance(coord, SkyCoord):
            return coord
        return SkyCoord(coord[0]*u.arcsec, coord[1]*u.arcsec,
                        obstime=self.timestamp, frame=frames.Helioprojective)



class ROI():
    def __init__(self, _data=None, _qsun_int=None, _timestamp=None, _id=None, _pixel_scale=None,
                 _centre=None, _centre_arcsec=None):
        self.id = id(self) if _id is None else _id
        self.data = _data
        self.timestamp = _timestamp
        self.filename = _timestamp.strftime('%Y-%m-%d_%H-%M-%S') if _timestamp is not None else None
        self.qsun_intensity = _qsun_int
        self.pixel_scale = _pixel_scale
        self.centre = _centre
        self.centre_arcsec = _centre_arcsec

    def __reduce__(self):
        return (self.__class__, (self.data, self.qsun_intensity, self.timestamp, self.id, self.pixel_scale,
                                 self.centre, self.centre_arcsec))


class Cluster():
    """Object containing information on a single cluster within an MLT layer."""
    def __init__(self, _coordinates=None, _ellipse_parameters=None, _datetime=None, _threshold=None,
                 _threshold_ratio=None, _number=None):
        self.points = _coordinates
        self.datetime = _datetime
        self.threshold = _threshold
        self.threshold_ratio = _threshold_ratio
        self.centre = self.get_centre() if self.points is not None else None
        self.size = len(self.points) if self.points is not None else 0
        self.ellipse_parameters = _ellipse_parameters
        self.number = _number
        self.id = self.get_unique_id()

    # def __repr__(self):
    #     return "({0}) {1}".format(self.number, self.id)

    def __reduce__(self):
        return (self.__class__, (self.points, self.ellipse_parameters, self.datetime, self.threshold,
                                 self.threshold_ratio, self.number))

    def get_unique_id(self):
        return "{0},{1}#{2}#{3}".format(self.centre[0],self.centre[1],self.threshold_ratio,self.datetime)

    def get_summary_for_text(self):
        """Returns a representation of the cluster object that can be printed to a text file"""
        ellipse_center = None if self.ellipse_parameters is None else self.ellipse_parameters[0]
        ellipse_dim = None if self.ellipse_parameters is None else self.ellipse_parameters[1]
        ellipse_angle = None if self.ellipse_parameters is None else self.ellipse_parameters[2]
        return (str(self.datetime), self.threshold_ratio, ellipse_center, ellipse_dim, ellipse_angle, self.centre, self.size)

    def get_centre(self):
        """Calculate the centre by taking the mean x- and y- values of all constituent points."""
        centre = np.array([0,0])
        for point in self.points:
            centre = centre + np.array(point)
        return np.round(centre / len(self.points))


class Layer():
    """Object containing information on a single MLT layer"""
    def __init__(self, _threshold=None, _threshold_ratio=None, _mlt_clusters=None, _optics_ordering=None):
        self.threshold = _threshold
        self.threshold_ratio = _threshold_ratio
        self.mlt_clusters = _mlt_clusters
        self.optics_ordering = _optics_ordering

    def __reduce__(self):
        return (self.__class__, (self.threshold, self.threshold_ratio, self.mlt_clusters, self.optics_ordering))

class MLT_Layers():
    """Data object containing a list of layers for a given spot timestamp. Used to save MLT data without making the
    .dat files excessively big. """
    def __init__(self, _id=None, _timestamp=None, _layers=None):
        self.id = id(self) if _id is None else _id
        self.timestamp = _timestamp
        self.filename = self.timestamp.strftime('%Y-%m-%d_%H-%M-%S') if self.timestamp is not None else None
        self.layers = [] if _layers is None else _layers
        self.layer_thresholds = self.get_layer_thresholds()
        try:
            self.sort_layers()
        except:
            pass

    def __reduce__(self):
        return (self.__class__, (self.id, self.timestamp, self.layers))

    def get_layer_thresholds(self):
        layer_thresholds = []
        for layer in self.layers:
            if layer is None: continue
            if layer.threshold_ratio not in layer_thresholds:
                layer_thresholds.append(layer.threshold_ratio)
        return layer_thresholds

    def print_layers(self):
        Logger.debug("[Cluster - print_layers] "
                     + "Layer list threshold_ratios: {0}".format([l.threshold_ratio for l in self.layers]))

    def sort_layers(self):
        self.remove_null_layers()
        self.layers.sort(key=lambda l: l.threshold_ratio, reverse=True)

    def find_layer_by_threshold(self, _threshold_ratio):
        try:
            return SpotTools.first(l for l in self.layers if l.threshold_ratio == _threshold_ratio)
        except:
            Logger.log("[SpotData] Could not find layer!", Logger.LogLevel.verbose)
            return None

    def remove_null_layers(self):
        for l in self.layers:
            if l is None:
                self.layers.remove(l)

    def add_or_replace_layers(self, new_layer):
        """Add a new layer to the list of layers. If one with the same threshold_ratio value exists, replace it."""
        # Catch null layer
        if new_layer is None:
            Logger.log("[SpotData] A layer that was to be added to MLT_Layers object (date: {0}) was null!".format(
                self.timestamp.strftime('%Y-%m-%d_%H-%M-%S')
            ), Logger.LogLevel.verbose)
            return False

        # Attempt to add Layer to list
        layer = self.find_layer_by_threshold(new_layer.threshold_ratio)
        if layer is None:
            self.layers.append(new_layer)
        else:
            self.layers.remove(layer)
            self.layers.append(new_layer)

        # Check list for null layers. If present, remove them.
        self.remove_null_layers()

        # Update lists
        self.layer_thresholds = self.get_layer_thresholds()
        self.sort_layers()
        return True


class SpotData():
    cluster_cache_size = 256
    def __init__(self, _base_dir='/home/rig12/Work/TrackingData/', _config_parser=None):
        self.dirs = {}
        self.dirs['base'] = Path(_base_dir)
        try:
            self.loadPathDic()
            self.testDirectories()
        except FileNotFoundError:
            print("[SpotData] Path file not found. Could not load directory list.")

        # Settings that need access to the .ini files must be initialised with another value in case the config parser
        # is none (the case in an interactive terminal).
        self.config_parser = _config_parser
        global cluster_cache_size
        cluster_cache_size = 128 if self.config_parser is None else self.config_parser.getint('SpotData', 'cluster_cache_size')

    def getDir(self, key_string, posix_path=None):
        '''
        Manages requests for directories. If a path with the specified key exists: returns that path. If not, then it
        makes it and stores a reference to it in self.dirs.
        :param key_string: key to access the path from the dictionary. Can be anything, by convention a single lowercase
                            word
        :param posix_path: a pathlib path to the requested place to save to.
        :return: posix_path
        '''
        if key_string in self.dirs.keys():
            try:
                self.dirs[key_string].resolve()
                return self.dirs[key_string]
            except FileNotFoundError:
                print("[SpotData] Directory {0} in dir list but not resolvable - remaking...")

        if posix_path is None:
            return self.requestNewDir(key_string, self.dirs['base'] / key_string)
        else:
            return self.requestNewDir(key_string, posix_path)

    def requestNewDir(self, key_string, posix_path):
        '''
        Makes a new directory and adds it to the dictionary to keep track of them.
        :param key_string: key to access the path from the dictionary. Can be anything.
        :param posix_path: a pathlib path to the requested place to save to.
        :return: returns posix_path if successful.
        '''
        if not isinstance(posix_path, Path): posix_path = Path(posix_path)
        self.dirs[key_string] = posix_path
        print("[Dir Request] Dir request granted for {0}".format(key_string))
        if not posix_path.is_dir():
            try:
                posix_path.mkdir(parents=True)
                print("[Dir Request] Made new path '{0}'".format(str(posix_path.resolve())))
            except FileExistsError:
                print("Someone else made the path before me!")
            self.savePathDic()
        return posix_path

    def testDirectories(self):
        '''
        Tests the directory list to make sure all paths within it exist. If they do not it deletes them
        :return:
        '''
        remove_entries = []
        for value in self.dirs:
            try:
                self.dirs[value].resolve()
            except:
                print("[SpotData] Directory '{0}' could not be resolved, removing from dictionary.".format(
                    self.dirs[value]))
                remove_entries.append(value)
        for item in remove_entries:
            del self.dirs[item]

    def savePathDic(self):
        savepath = self.dirs['base'] / 'paths.cfg'
        with savepath.open('wb') as f:
            pickle.dump(self.dirs, f)
        return True

    def loadPathDic(self):
        loadpath = self.dirs['base'] / 'paths.cfg'
        if loadpath.stat().st_size > 0:
            with loadpath.open('rb') as f:
                self.dirs = pickle.load(f)
            return True
        else:
            return False

    def checkPath(self, _path, keepPosixPath = False):
        '''
        Checks to see if _path exists and returns the path as a string. If the path doesn't exist, it is made.
        :param _path: A str or pathlib Path to a directory.
        :return: A string representation of the string or path entered.
        '''
        if isinstance(_path, str):
            pyPath = Path(_path)
        elif isinstance(_path, Path):
            pyPath = _path
        else:
            raise ValueError("_path must be a string or pathlib Path.")

        if not pyPath.is_dir():
            pyPath.mkdir(parents=True)
            print("DEBUG: Making new path '{0}'...".format(str(pyPath.resolve())))

        if not keepPosixPath:
            return str(pyPath.resolve())
        else:
            return pyPath

    def saveROIData(self, _roi, _path):
        path = self.checkPath(_path, keepPosixPath=True)
        if isinstance(_roi, ROI):
            roiList = [_roi]
        else:
            roiList = _roi

        for roi in roiList:
            filename = path / (roi.timestamp.strftime('%Y-%m-%d_%H-%M-%S') + '.roi')
            with filename.open('wb') as f:
                pickle.dump(roi, f)
        return True

    def loadROIList(self, _path):
        path = self.checkPath(_path, keepPosixPath=True)
        files = [x for x in path.glob('*.roi') if x.is_file()]
        roiList = []
        for file in files:
            with file.open('rb') as f:
                roi = pickle.load(f, encoding='bytes')
                roiList.append(roi)
        return roiList

    def loadROI(self, _filename):
        """
        Loads a single ROI file and returns it
        :param _filename: filename in the roi directory of target roi
        :return: an ROI object.
        """
        _path = self.getDir('roi') / (_filename + '.roi')
        try:
            with _path.open('rb') as f:
                roi = pickle.load(f, encoding='bytes')
        except FileNotFoundError:
            _path_ext = self.getDir('roi_ext') / (_filename + '.roi')
            with _path_ext.open('rb') as f:
                roi = pickle.load(f, encoding='bytes')
        return roi

    def saveMLTData(self, _mlt, _path):
        path = self.checkPath(_path, keepPosixPath=True)
        if (isinstance(_mlt, MLT_Layers)):
            mlt_list = [_mlt]
        else:
            mlt_list = _mlt
        for mlt in mlt_list:
            filename = path / (mlt.timestamp.strftime('%Y-%m-%d_%H-%M-%S') + '.mlt')
            with filename.open('wb') as f:
                pickle.dump(mlt, f)
        return True

    def loadMLT(self, _filename):
        """
        Loads a single MLT file and returns it
        :param _filename: filename in the mlt directory of target mlt
        :return: an MLT_Layers object.
        """
        path = self.getDir('mlt') / (_filename + '.mlt')
        try:
            with path.open('rb') as f:
                mlt = pickle.load(f, encoding='bytes')
        except EOFError:
            Logger.log("[SpotData] An EOF error has occurred for file {0}.".format(str(path)))
            return None
        except FileNotFoundError:
            try:
                path = self.getDir('mlt_ext') / (_filename + '.mlt')
                with path.open('rb') as f:
                    mlt = pickle.load(f, encoding='bytes')
            except FileNotFoundError:
                Logger.log("[SpotData] Could not find mlt file {0}.".format(str(path)))
                return None
        return mlt

    def saveParameters(self, parameters, threshold_layer, spot_index, _path):
        path = self.checkPath(_path, keepPosixPath=True)
        filepath = path / '{0}_{1}.par'.format(spot_index, threshold_layer)
        with filepath.open('wb') as file_object:
            pickle.dump(parameters,file_object)
        return True

    def loadParameters(self, threshold_layer, spot_index):
        path = self.getDir('parameters') / ("{0}_{1}.par".format(spot_index, threshold_layer))
        try:
            with path.open('rb') as f:
                parameters = pickle.load(f, encoding='bytes')
        except FileNotFoundError:
            Logger.log("[SpotData] Could not find parameter file {0}".format(str(path)))
            return None
        return parameters

    def saveSpotData(self, sunspotGroupList, _path):
        path = self.checkPath(_path, keepPosixPath=True)
        for group in sunspotGroupList:
            filepath = path / '{0}.dat'.format(str(group.id))
            with filepath.open('wb') as file_object:
                pickle.dump(group,file_object)
        return True

    def loadSpotData(self, _path, ext_fits_dir=None, ask_input=True):
        Logger.log("Loading sunspots...")
        if ext_fits_dir is None:
            fits_base = self.getDir('fits_ext', posix_path=ext_fits_dir)
        else:
            fits_base = self.getDir('fits')
        try:
            # Look for .dat files in the dat directory.
            path = self.checkPath(_path, keepPosixPath=True)
            files = [x for x in path.glob('*.dat') if x.is_file()]
            # Return None if no data files
            if len(files) == 0:
                return None
            sunspotGroupList = []
            # Try to load the .dat files found and compile them into a list
            for file in files:
                with file.open('rb') as file_object:
                    group = pickle.load(file_object, encoding='bytes')
                    for hist in group.history:
                        try:
                            hist.path.resolve()
                        except:
                            hist.path = fits_base / hist.filename
                    sunspotGroupList.append(group)
            # Sort the list based on how long the history is (and thus how "complete" the spot data is).
            sunspotGroupList.sort(key=lambda x: len(x.history), reverse=True)

            # Try to find matching ROI files in the roi directory. Only checks for 0th indexed spot as that is the
            # longest and is likely to be the one being tracked.
            check_roi_file = self.check_roi_files(sunspotGroupList)
            if check_roi_file and ask_input:
                look_for_roi = input("More than 1% dat files have no ROI, attempt to match roi files to spot? (Y/N): ")
                if look_for_roi == 'Y' or look_for_roi == 'y':
                    self.associate_roi_files(sunspotGroupList)

            # Do the same for MLT files
            check_mlt_file = self.check_mlt_files(sunspotGroupList)
            if check_mlt_file and ask_input:
                look_for_mlt = input("More than 1% dat files have no MLT, attempt to match mlt files to spot? (Y/N): ")
                if look_for_mlt == 'Y' or look_for_mlt == 'y':
                    self.associate_mlt_files(sunspotGroupList)

            # Return the list!
            return sunspotGroupList
        except FileNotFoundError:
            print("[SpotData] Could not find .dat files.")
            return None

    def check_roi_files(self,sunspotGroupList, spot_index=0):
        """Returns the number of dat files that do not have an associated roi file"""
        counter = 0
        for group in sunspotGroupList[spot_index].history:
            if group.ROI_path is None:
                counter += 1
        Logger.log("[SpotData - ROI Check] {0} ({1}%) dat files do not have associated roi files.".format(
            counter,
            round((counter/len(sunspotGroupList[spot_index].history) * 100), 4)))
        # check how many roi actually exist and if it is a significant amount
        roiFiles = [x for x in self.getDir('roi').glob('*.roi') if x.is_file()]
        is_roi_count_nonzero = len(roiFiles) != len(sunspotGroupList[spot_index].history) - counter
        is_count_significant = counter >= len(sunspotGroupList[spot_index].history) * 0.01
        if is_count_significant and is_roi_count_nonzero:
            return True
        else:
            Logger.log("[SpotData - ROI Check] Could not find equivalent number of ROI files. Skipping.")
            return False

    def check_mlt_files(self, sunspotGroupList, spot_index=0):
        """Returns the number of dat files that do not have an associated mlt file"""
        counter = 0
        for group in sunspotGroupList[spot_index].history:
            if group.mlt_path is None:
                counter += 1
        Logger.log("[SpotData - MLT Check] {0} ({1}%) dat files do not have associated mlt files.".format(
            counter,
            round((counter / len(sunspotGroupList[spot_index].history) * 100), 4)))
        # check how many mlt actually exist and if it is a significant amount
        mltFiles = [x for x in self.getDir('mlt').glob('*.mlt') if x.is_file()]
        is_mlt_count_nonzero = len(mltFiles) != len(sunspotGroupList[spot_index].history) - counter
        is_count_significant = counter >= len(sunspotGroupList[spot_index].history) * 0.01
        if is_count_significant and is_mlt_count_nonzero:
            return True
        else:
            Logger.log("[SpotData - MLT Check] Could not find equivalent number of MLT files. Skipping.")
            return False

    def associate_roi_files(self, sunspotGroupList, spot_index=0):
        """Checks that all files in the roi folder are correctly associated with the dat files."""
        roiFiles = [x for x in self.getDir('roi').glob('*.roi') if x.is_file()]
        prog = PrintProgress(0, len(sunspotGroupList[spot_index].history), label='[SpotData] Matching ROIs... ')
        for group in sunspotGroupList[spot_index].history:
            for roi in roiFiles:
                if roi.name == group.timestamp.strftime('%Y-%m-%d_%H-%M-%S') + '.roi':
                    group.ROI_path = group.timestamp.strftime('%Y-%m-%d_%H-%M-%S')
                    break
            prog.update()

    def associate_mlt_files(self, sunspotGroupList, spot_index=0):
        """Checks that all files in the mlt folder are correctly associated with the dat files."""
        mltFiles = [x for x in self.getDir('mlt').glob('*.mlt') if x.is_file()]
        prog = PrintProgress(0, len(sunspotGroupList[spot_index].history), label='[SpotData] Matching MLTs... ')
        for group in sunspotGroupList[spot_index].history:
            for mlt in mltFiles:
                if mlt.name == group.timestamp.strftime('%Y-%m-%d_%H-%M-%S') + '.mlt':
                    group.mlt_path = group.timestamp.strftime('%Y-%m-%d_%H-%M-%S')
                    break
            prog.update()

    def mergeSpots(self, sunspot_group_list, parent_index, child_indices):
        """
        Function to merge two or more sunspots manually.
        :param parent_index: Index in sunspot_group_list of the spot to merge into
        :param child_indices: Indices of the spots to be merged, as a list or tuple.
        :param sunspot_group_list: List of all sunspot groups
        :return: list of sunspot groups with children groups merged into parent.
        """
        parent_group = sunspot_group_list[parent_index]
        child_groups = []
        for i in child_indices:
            child = sunspot_group_list[i]
            parent_group.history.extend(child.history)
        parent_group.sortHistory()
        parent_group.getCentreTime()
        pruned_list = [sunspot_group_list[i] for i in range(0, len(sunspot_group_list)) if i not in child_indices]
        return pruned_list

    def check_ROI_headers(self, sunspot_group):
        """
        Goes through the ROIs associated with a spot and checks to make sure they have a value for pixel_scale in
        their headers. Needed because I'm an idiot and have to fix this in post.
        :param sunspot_group: A SunspotGroup object (subscript of sunspot_group_list)
        :return:
        """
        files_fixed = 0
        prog = Logger.PrintProgress(0, len(sunspot_group.history),label="Check ROI progress...")
        for snapshot in sunspot_group.history:
            prog.update()
            if snapshot.ROI_path is None:
                Logger.debug("[Check ROI] Could not find ROI for spot {0}.".format(snapshot.filename))
                continue
            roi = self.loadROI(snapshot.ROI_path)

            # If the ROI doesn't have a required attribute, give it one so it'll load and then fix it.
            # Not sure why this needs it but it didn't need it when I ran the code at home. Maybe difference
            # between 3.5 and 3.6?
            try:
                check_scale = roi.pixel_scale is None
            except AttributeError:
                roi.__setattr__("pixel_scale", None)
            try:
                check_centre = roi.centre is None
            except AttributeError:
                roi.__setattr__("centre", None)
            try:
                check_centre_arcsec = roi.centre_arcsec is None
            except AttributeError:
                roi.__setattr__("centre_arcsec", None)

            fixes = 0
            if roi.pixel_scale is None:
                roi.pixel_scale = SpotTools.pixel_scale_from_centres(snapshot.centre, snapshot.centre_arcsec)
                fixes += 1
            if roi.centre is None:
                roi.centre = snapshot.centre
                fixes += 1
            if roi.centre_arcsec is None:
                roi.centre_arcsec = snapshot.centre_arcsec
                fixes +=1

            # Save fixed files
            if fixes > 0:
                self.saveROIData(roi, self.getDir('roi'))
                files_fixed += 1

        Logger.log("[Check ROI] {0} files fixed!".format(files_fixed))

    def export_clusters_to_text(self, cluster_list, thresholds, clusters_to_print):
        """
        Takes a list of clusters that have been tracked in MLT_Analysis and outputs their data in a plain text document.
        :param cluster_list:
        :return:
        """
        filepath = self.getDir("output_cluster_txts", posix_path=(self.getDir("output") / "cluster_txts"))
        for spot_index in clusters_to_print:
            spot_data = []
            filename = "cluster_{0}.txt".format(spot_index)
            for k in range(0, len(thresholds)):
                Logger.debug("[SpotData - export_clusters_to_text] Extracting data spot {0} threshold {1}".format(spot_index, k))
                try:
                    writeable_data = [self.get_cluster_from_id(c).get_summary_for_text() for c in cluster_list[k][spot_index]]
                except IndexError:
                    Logger.debug("[SpotData - export_clusters_to_text] - Index out of range. Skipping...")
                    continue
                spot_data.extend(writeable_data)
            Logger.debug("[SpotData - export_clusters_to_text] Saving data to {0}".format(str(filepath / filename)))
            np.savetxt(str(filepath / filename), spot_data,
                       fmt='%s', delimiter='\t',
                       header="Date | Threshold ratio | Ellipse Center | Major/Minor axis length | Major axis angle (rad) | Cluster Center (roi pix) | Size")
        Logger.debug("[SpotData - export_clusters_to_text] Done!")

    def export_bad_velocities(self, bad_velocities_data):
        """Save the parameters object for clusters where the velocity is too big."""
        pass

    @lru_cache(maxsize=cluster_cache_size)
    def get_cluster_from_id(self, id):
        """Finds the cluster object by matching the id value. ID's are stored as a string that uniquely identifies the
        cluster and the mlt file it exists in. The file is then opened and the cluster object returned."""
        #Logger.debug("[SpotData - get_cluster_from_id] Get cluster from id \'{0}\'".format(id))
        centre, threshold_ratio, filename = id.split('#', 3)
        #Logger.debug("[SpotData - get_cluster_from_id] Found data: centre: \'{0}\' threshold_ratio: \'{1}\' datetime: \'{2}\'".format(centre, threshold_ratio, filename))
        filename = SpotTools.parse_string_to_datetime(filename)
        mlt_layers = self.loadMLT(filename.strftime("%Y-%m-%d_%H-%M-%S"))
        layer = mlt_layers.find_layer_by_threshold(float(threshold_ratio))
        return SpotTools.first([cluster for cluster in layer.mlt_clusters if id == cluster.id])



