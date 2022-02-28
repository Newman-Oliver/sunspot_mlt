# My Libraries
import SpotData
import Logger
import SpotTools

# Misc Libraries
from pathlib import Path

# Science Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import astropy.units as u
import sunpy.map
import sunpy.physics.differential_rotation
import limb_darkening as ld
import numpy as np

class DiffRot():
    def __init__(self, _path_man, _comm, _config_parser):
        self.path_man = _path_man
        self.config_parser = _config_parser
        self.use_external_directory  = self.config_parser.getboolean('DiffRot','use_external_directory')
        self.overwrite_existing = self.config_parser.getboolean('DiffRot','overwrite_existing')
        if not self.use_external_directory:
            self.roi_dir = self.path_man.getDir('roi')
        else:
            self.roi_dir = self.path_man.getDir('roi_ext', Path(self.config_parser.get('Directories','roi_ext')))
        self.output_dir = self.path_man.getDir('diff_rot')
        self.roi_size = self.config_parser.get_list_as_int('DiffRot', 'roi_size')

        # MPI
        self.comm = _comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def divide_tasks(self, dataLength, cpuCount):
        """Provides the start and stop indices for an MPI loop that uses cpuCount CPUs to iterate over a list of length
        dataLength"""
        N = dataLength
        count = np.floor(N/cpuCount)
        remainder = N % cpuCount

        # Get start and stop points for each MPI process
        if self.rank < remainder:
            start = self.rank * (count+1)
            stop = start + count
        else:
            start = (self.rank * count) + remainder
            stop = start + (count-1)

        return np.int64(start), np.int64(stop)

    def plotROI(self, fitsMap, datetime,  roi_box=None, drawBox=False):
        fig = plt.figure()
        ax = plt.subplot(projection = fitsMap)
        im = fitsMap.rotate(angle = 180 * u.deg).plot()
        ax.set_autoscale_on(False)

        if drawBox:
            rect = patches.Rectangle(roi_box[0],roi_box[1],roi_box[2],linewidth=1,edgecolor='r',fill=False)
            ax.add_patch(rect)

        filename = datetime.strftime('%Y-%m-%d_%H-%M-%S') + '.svg'
        plt.savefig(str((self.output_dir / filename)),dpi=300)
        plt.close(fig)
        return

    #@profile
    def doDiffRot(self, spotGroup, use_mpi=False):
        minROI = self.roi_size #spotGroup.getMinROI(set_=True)
        spotGroup.minROI = self.roi_size
        centreTime = spotGroup.getCentreTime()
        centred_snapshot = spotGroup.centred_snapshot
        Logger.log("[DiffRot] Rotating spots to time {0}".format(centreTime))
        if use_mpi:
            start, stop = self.divide_tasks(len(spotGroup.history), self.size)
        else:
            start = 0
            stop = len(spotGroup.history)

        prog = Logger.PrintProgress(start, stop,
                                    label='[DiffRot] CPU {0} creating ROIs for files {1} to {2}... '.format(
                                               self.rank, start, stop
                                           ))
        for i in range(start,stop):
            snapshot = spotGroup.history[i]
            # Check if there is already an ROI file there, if so and we aren't overriding, then skip.
            roi_exists = Path(self.roi_dir / (snapshot.timestamp.strftime('%Y-%m-%d_%H-%M-%S') + ".roi")).is_file()
            if roi_exists and not self.overwrite_existing:
                Logger.debug("[DiffRot] An ROI already exists for {0}. Skipping...".format(snapshot.timestamp))
                continue
            snapshot.minROI = self.roi_size
            try:
                Logger.debug("[DiffRot] Opening new fits file from main directory...")
                cmp = sunpy.map.Map(str(self.path_man.getDir('fits') / snapshot.filename))
            except ValueError:
                Logger.debug("[DiffRot] WRN: File Not found! Checking ext_fits dir (\'{0}\') ...".format(self.path_man.getDir('ext_fits') / snapshot.filename))
                cmp = sunpy.map.Map(str(self.path_man.getDir('fits_ext') / snapshot.filename))
            Logger.debug("[DiffRot] Applying limb darkening correction...")
            cmp = ld.limbdark(cmp)
            cmp.data[np.isnan(cmp.data)] = -1000
            #These need to be in pixels so that I can crop them out the data array
            roi_xMax = int(centred_snapshot.centre[1] + minROI[0])
            roi_xMin = int(centred_snapshot.centre[1] - minROI[0])
            roi_yMax = int(centred_snapshot.centre[0] + minROI[1])
            roi_yMin = int(centred_snapshot.centre[0] - minROI[1])
            Logger.debug("[DiffRot] Applying differential rotation...")
            cmp = sunpy.physics.differential_rotation.diffrot_map(cmp,centreTime)
            Logger.debug("[DiffRot] Cropping to {0}X{1} ROI...".format(minROI[1]*2, minROI[0]*2))
            roiData = cmp.data[roi_xMin:roi_xMax,roi_yMin:roi_yMax]
            pixel_scale = SpotTools.pixel_scale_from_centres(snapshot.centre, snapshot.centre_arcsec)
            Logger.debug("[DiffRot] Saving region...")
            roi = SpotData.ROI(roiData,
                               snapshot.qsun_intensity,
                               snapshot.timestamp,
                               _pixel_scale=pixel_scale)
            snapshot.ROI_path = snapshot.timestamp.strftime('%Y-%m-%d_%H-%M-%S')
            #self.plotROI(cmp, cmp.date, [(roi_xMin+2048,roi_yMin+2048), 2*minROI[0], 2*minROI[1]])
            self.path_man.saveROIData(roi, self.roi_dir)
            prog.update()




