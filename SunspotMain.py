# My libraries
import Tracking
import SpotData
import SpotTools
import DifferentialRotation
import Logger
import DownloadVSO
import MLT
import MLT_Analysis
import ConfigWrapper

# Misc libraries
from pathlib import Path
from mpi4py import MPI
import warnings
import socket
import numpy as np
import json
import datetime
import sys

# Science Libraries
from astropy.utils.exceptions import AstropyWarning

class SunspotMain():
    def __init__(self, args):
        # Init MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # Init config parser
        self.config_parser = ConfigWrapper.ConfigWrapper() #configparser.ConfigParser()
        if len(args) == 0:
            self.config_parser.read('config.ini')
        else:
            self.config_parser.read(args[1])
        self.base_dir = Path(self.config_parser.get('Directories', 'base'))
        self.fits_ext = Path(self.config_parser.get('Directories', 'ext_fits'))

        # Init local directory and logging
        self.spot_data = SpotData.SpotData(_base_dir=self.base_dir, _config_parser=self.config_parser)
        Logger.init_logger(str(self.spot_data.getDir('logs')) + '/', "{0}_{1}.log".format(
                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), self.rank))
        Logger.debug("[Config] [{0}] \'{1}\' = {2}".format('Directories', 'base', self.base_dir))
        Logger.debug("[Config] [{0}] \'{1}\' = {2}".format('Directories', 'ext_fits', self.fits_ext))
        self.config_parser.logger_initiated = True

        # Vars
        self.sunspotGroupList = None
        self.chosen_spot = self.config_parser.getint('Options', 'chosen_spot')
        self.hostname = socket.gethostname()
        self.ask_input = self.config_parser.getboolean('Options', 'ask_input')
        Logger.log_level = Logger.LogLevel(self.config_parser.getint('Options', 'log_level'))
        self.start_date = datetime.datetime.strptime(self.config_parser.get('Options', 'start_date'),
                                                     '%Y-%m-%d_%H-%M-%S')
        self.end_date = datetime.datetime.strptime(self.config_parser.get('Options', 'end_date'),
                                                   '%Y-%m-%d_%H-%M-%S')
        self.thresholds = self.config_parser.get('Options', 'thresholds')
        if self.thresholds[0] == '[':
            self.thresholds = np.array(json.loads(self.thresholds))
        else:
            range_parameters = self.thresholds.replace('(','').replace(')','').split(',')
            # The strange looking maths here is to extract the correct decimal without the floating point errors.
            # No threshold ratio should be less than 1/100 of a percent, so converting to int and then dividing.
            range_parameters = [int(float(x)*10000)/10000 for x in range_parameters]
            self.thresholds = np.arange(range_parameters[0], range_parameters[1], range_parameters[2])
        Logger.debug("Thresholds: {0}".format(self.thresholds))
        Logger.debug("Thresolds type: {0}".format(type(self.thresholds[0])))

        # Jobs
        self.do_download = self.config_parser.getboolean('Jobs', 'do_download')
        self.do_tracking = self.config_parser.getboolean('Jobs', 'do_tracking')
        self.visualise_tracking = self.config_parser.getboolean('Jobs', 'visualise_tracking')
        self.try_stitch_dats = self.config_parser.getboolean('Jobs', 'try_stitch_dats')
        self.do_diff_rot = self.config_parser.getboolean('Jobs', 'do_diff_rot')
        self.do_mlt = self.config_parser.getboolean('Jobs', 'do_mlt')
        self.do_parameters = self.config_parser.getboolean('Jobs','do_parameters')
        self.do_angle_analysis = self.config_parser.getboolean('Jobs', 'do_angle_analysis')

        # Checks
        self.check_ROI_headers = self.config_parser.getboolean('Checks','check_ROI_headers')

        # Instantiate Modules
        self.tracker = Tracking.SpotTracker(path_manager=self.spot_data,_comm=self.comm,
                                            external_fits_dir=self.fits_ext, _config_parser=self.config_parser)
        self.diff_rot = DifferentialRotation.DiffRot(self.spot_data, self.comm, self.config_parser)

        self.downloader = DownloadVSO.Downloader(self.spot_data, self.config_parser, ext_fits_fir=str(self.fits_ext))
        self.mlt = MLT.MultiLevelThresholding(self.spot_data, self.comm, self.config_parser)
        self.mlt_analysis = MLT_Analysis.MLT_Analyser(self.spot_data, self.config_parser, self.comm)

        # Disable polluting astropy warnings
        warnings.simplefilter('ignore',category=AstropyWarning)

        # Only attempt download on head node
        if self.do_download:
            self.start_download()

        # Do NOT attempt processing on head node
        if not self.hostname.startswith('sl'):
            runtime_tot = Logger.Runtime(label='Total Runtime: ')
            self.start_processing()
            if self.rank == 0: runtime_tot.print()
        else:
            Logger.log("[SunspotMain] ERR: Cannot begin process - on head node!")

    def start_download(self):
        if self.hostname.startswith(('sl', 'pcrig', 'Ind', 'ind', 'Rich', 'rich', 'main', 'localhost')):
            runtime_dl = Logger.Runtime(label='Total Download Runtime: ')
            self.downloader.slowGetFiles()
            runtime_dl.print()
        else:
            Logger.log("[ERR] Unable to download as not on login node.")

    def start_processing(self):
        Logger.log("SunspotMain started.")

        # Attempt to load spots
        self.sunspotGroupList = self.spot_data.loadSpotData(self.spot_data.getDir('dat'), ask_input=self.ask_input)
        if not self.sunspotGroupList:
            Logger.log("No sunspot tracking data found.")
        else:
            Logger.log("Sunspots loaded!")

        # Perform checks if required.
        if self.check_ROI_headers and self.sunspotGroupList is not None:
            self.spot_data.check_ROI_headers(self.sunspotGroupList[self.chosen_spot])

        # Look for fits files in the fits dir and begin tracking.
        if self.do_tracking:
            self.run_tracking()

        # Stitch together dat files from tracking on multiple nodes without final stitching.
        if self.try_stitch_dats:
            self.sunspotGroupList = self.tracker.stitchMPI(self.sunspotGroupList)

        # Visualise sunspot groups
        if self.visualise_tracking and self.rank == 0:
            for i in range(0,min(len(self.sunspotGroupList),99)):
                mark = SpotTools.get_marker(i)
                Logger.log("[{0}] {1} {2} is {3}".format(str(i),SpotTools.get_colour(i),
                                                         mark, str(self.sunspotGroupList[i])),
                           Logger.LogLevel.verbose)
            self.tracker.plotGroupsOverTime(self.tracker.fits_files[0],self.sunspotGroupList[0:99])
            if len(self.sunspotGroupList) > 100:
                Logger.log("[SunspotMain - Visualise Tracking] Number of sunspot groups exceeds 100! {0} of {1} were not plotted".format(len(self.sunspotGroupList)-100, len(self.sunspotGroupList)))

        # Do the differential rotation.
        if self.do_diff_rot:
            Logger.log("Applying Differential Rotation...")
            self.diff_rot.doDiffRot(self.sunspotGroupList[self.chosen_spot],use_mpi=True)
            if self.rank == 0:
                self.spot_data.saveSpotData(self.sunspotGroupList, self.spot_data.getDir('dat'))

        # Do the MLT
        if self.do_mlt:
            self.mlt.run_thresholding_on_list(self.sunspotGroupList[self.chosen_spot], self.thresholds)
            if self.rank == 0:
                self.spot_data.saveSpotData(self.sunspotGroupList, self.spot_data.getDir('dat'))

        # Get the Parameter files for MLT_Analysis separately from the analysis
        if self.do_parameters:
            sunspot_group = self.sunspotGroupList[self.chosen_spot]
            start, stop = SpotTools.get_date_range_indices(sunspot_group, self.start_date, self.end_date)
            if start - self.mlt_analysis.velocity_stride < 0:
                start = 0
            else:
                start -= self.mlt_analysis.velocity_stride
            self.mlt_analysis.get_paramters(sunspot_group.history[start:stop],
                                            self.thresholds,
                                            range(0, self.mlt_analysis.number_of_clusters_to_track),
                                            calculate_missing=True)

        # Analyse the angles
        if self.do_angle_analysis:
            self.mlt_analysis.run(self.sunspotGroupList[self.chosen_spot],
                                  self.thresholds,
                                  self.start_date,
                                  self.end_date)

        # Final saves and cleanup before exiting
        Logger.debug("[SunspotMain] Final cluster cache info: {0}".format(self.spot_data.get_cluster_from_id.cache_info()))
        self.spot_data.savePathDic()
        if self.rank == 0:
            self.spot_data.saveSpotData(self.sunspotGroupList, self.spot_data.getDir('dat'))

    def run_tracking(self, **kwargs):
        self.sunspotGroupList = self.tracker.trackSpots()
        self.spot_data.saveSpotData(self.sunspotGroupList, self.spot_data.getDir('dat'))


if __name__ == "__main__":
    main = SunspotMain(sys.argv)
