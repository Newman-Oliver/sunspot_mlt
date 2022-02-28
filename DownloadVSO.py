from __future__ import print_function, division
import astropy.units as u
from sunpy.net import Fido, attrs as a
from Logger import PrintProgress
import Logger
import datetime

class Downloader:

    def __init__(self, path_man, _config_parser, ext_fits_fir=None):
        self.path_man = path_man
        self.config_parser = _config_parser
        if ext_fits_fir is None:
            self.fits_dir = self.path_man.getDir('fits')
        else:
            self.fits_dir = self.path_man.getDir('fits_ext',posix_path=ext_fits_fir)

    def search(self,time, instr, physobs, cadence):
        # time = a.Time('2014-09-05 00:00', '2014-09-05 03:59')
        # instr = a.vso.Instrument('HMI')
        # physobs = a.vso.Physobs('intensity')
        # cadence = a.vso.Sample(3600 * u.second)

        Logger.log("Searching for data matching the following parameters: " 
                   "\nTime: {0} to {1}\nInstrument: {2}\nPhysObs: {3}\nCadence: {4}s".format(
                    time.start, time.end, instr, physobs, cadence))

        result = Fido.search(time, instr, physobs, cadence)
        return result


    def getFiles(self):
        Logger.log("[DownloadVSO] Searching for fits files...")
        search_result = self.search()
        Logger.log("[DownloadVSO] {0} files found. Beginning download...".format(search_result.file_num))
        prog_down = PrintProgress.PrintProgress(0,1,label="Download progress: ")
        downloaded_files = Fido.fetch(search_result, path=str(self.fits_dir))
        prog_down.update()
        Logger.log("[DownloadVSO] Fetched files!")


    def slowGetFiles(self):
        Logger.log("[DownloadVSO] Searching for fits files...")
        time_start = datetime.datetime.strptime(self.config_parser.get('Options', 'start_date'),
                                                '%Y-%m-%d_%H-%M-%S')
        time_end = datetime.datetime.strptime(self.config_parser.get('Options', 'end_date'),
                                              '%Y-%m-%d_%H-%M-%S')
        instr = a.vso.Instrument(self.config_parser.get('Download','instrument'))
        physobs = a.vso.Physobs(self.config_parser.get('Download','phys_obs'))
        cadence = a.vso.Sample(self.config_parser.getint('Download','cadence') * u.second)
        time_step = (time_end - time_start)/100

        Logger.log("Searching for data matching the following parameters: "
                   "\nTime: {0} to {1}\nInstrument: {2}\nPhysObs: {3}\nCadence: {4}s".format(
                    time_start, time_end, instr, physobs, cadence))

        search_result = Fido.search(a.Time(time_start, time_end), instr, physobs, cadence)
        Logger.log("[DownloadVSO] {0} files found. Beginning download...".format(search_result.file_num))

        prog = PrintProgress(0,100,label="[DownloadVSO] Download progress...")
        failed_files = []
        for i in range(0,100):
            time = a.vso.Time(time_start + i * time_step,
                              time_start + (i +1) * time_step)
            result = Fido.search(time, instr, physobs, cadence)
            downloaded_files = Fido.fetch(result, path=str(self.fits_dir), progress=False)
            if len(downloaded_files) == 0:
                failed_files.append((time, instr, physobs, cadence))
                Logger.log("[DownloadVSO] I think I could not download file at time {0}. Adding to retry list".format(time),
                           Logger.LogLevel.debug)

            prog.update()

        files = [x for x in self.fits_dir.glob('*.fits') if x.is_file()]
        Logger.log("[DownloadVSO] {0} total files obtained. {1} failed to download".format(len(files), len(failed_files)))


    def hacky_download(self):
        pass






