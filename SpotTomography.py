#My modules
import ROIReader as ROI
import EllipseFit
import Contours
from Logger import PrintProgress
import SpotTools
import SpotData


#Misc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

#Clustering
from pyclustering.cluster.optics import optics

#Get ROI files
class Tomography():
    def __init__(self, _thresholds, _path_manager, _sunspot_group, OPTICS_eps = 20, OPTICS_minPts = 5 ):
        self.path_man = _path_manager
        self.roi_path = self.path_man.getDir('roi')
        self.output_path = self.path_man.getDir('tomography')
        #self.roiFiles = [x for x in self.roi_path.glob('*.roi') if x.is_file()]
        self.sunspot_group = _sunspot_group

        self.thresholds = _thresholds
        self.eps = OPTICS_eps
        self.minPts = OPTICS_minPts

    def applyOPTICS(self, dataset, eps, minPts):
        '''Apply the OPTICS clustering algorithm to the data and return the resulting clusters.'''
        instance = optics(dataset,eps,minPts)
        #print("Processing...");
        instance.process()
        return instance.get_clusters()

    def extractData(self, data):
        '''Extract a list of the points that contain data and their co-ordinates for passing into the OPTICS code'''
        (x,y) = np.shape(data)
        # Square the array
        if(x != y):
            if(y < x):
                x = y
            else:
                y = x
        extractedData = []
        for i in range(0,y):
            for j in range(0,x):
                if(data[j,i] > 0):
                    extractedData.append((j,i,data[j,i]))
        return np.array(extractedData)

    # Visualisation
    def plotFigure(self, tomo_data, tomo_clusters, cluster_perims, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for j in range(0, len(tomo_data)):
            clusters = tomo_clusters[j]
            # extractedData = tomo_data[j]
            perimData = cluster_perims[j]
            for k in range(0, len(tomo_clusters[j])):
                x = perimData[k][0]
                y = perimData[k][1]

                # Plot points from data
                ax.scatter(x, y, c=SpotTools.colourList[j])

                # Get ellipse fit
                elli = EllipseFit.fitEllipse_b2ac(np.transpose(x), np.transpose(y))
                centre = elli[0]
                phi = elli[2]
                axes = elli[1]

                # Plot it - ... in polar coords. oh god I'm going to have to change it all aren't I?
                R = np.arange(0, 2.0 * np.pi, 0.01)
                a, b = axes
                xx = centre[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
                yy = centre[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
                ax.plot(xx, yy, c=SpotTools.colourList[j])
        plt.savefig(str(self.output_path / (filename + '.png')))
        plt.close(fig)


    def getClustersFromROIs(self):
        prog_files = PrintProgress(0, len(self.sunspot_group.history), label='[SpotTomography] Overall progress: ')
        allClustersList = []
        for i in range(0, 1): #len(self.sunspot_group.history)
            # For each image, extract the data
            #(header, data) = ROI.ROIReader.load(str(self.roiFiles[i].resolve()))
            snapshot = self.sunspot_group.history[i]
            roi = self.path_man.loadROI("2014-09-07_07-44-06")#(snapshot.ROI)
            data = roi.data
            qsun_intensity = roi.qsun_intensity

            tomograms = []  # Whole picture thresholded
            tomo_data = []  # data above transcribed into [x,y,value] format
            tomo_clusters = []  # Grouped into clusters

            clusterList = []

            # Apply each threshold in thresholds to image. i.e. the "tomography" bit
            prog_tomo = PrintProgress(0, len(self.thresholds), label='  | Slicing image... ')
            for j in range(0,len(self.thresholds)):
                result = (data < self.thresholds[j] * qsun_intensity) * data
                # If the result array contains no non-zero values, skip it.
                if(result.max() == 0.0):
                    continue

                tomograms.append(result)
                # Get data into [x,y,value] format.
                extractedData = self.extractData(result)
                #extractedData = np.transpose(np.array([extractedData[:,0],extractedData[:,1]]))
                tomo_data.append(extractedData)
                # Cluster the data based on their cartesian coords
                clusters = self.applyOPTICS(extractedData,self.eps,self.minPts)
                tomo_clusters.append(clusters)

                # Get Perimeter of clusters
                perims = self.getPerimeters(tomo_data, tomo_clusters)

                # Make picture
                self.plotFigure(tomo_data, tomo_clusters, perims, roi.timestamp.strftime('%Y-%m-%d_%H-%M-%S'))

                # Make cluster file
                clust = SpotData.Cluster(self.thresholds[j],result, extractedData, clusters)
                clusterList.append(clust)
                prog_tomo.update()
            allClustersList.append(clusterList)
            prog_files.update()
        return allClustersList

    def getPerimeters(self, tomo_data, tomo_clusters):
        # Get perimeter
        cluster_perims = []
        prog_perim = PrintProgress(0, len(tomo_data), label='  | Getting next slice perimeters... ')
        for j in range(0, len(tomo_data)):
            slice_cluster_perims = []
            for k in range(0, len(tomo_clusters[j])):
                perimPoints = Contours.getClusterPerimeter(tomo_data[j][tomo_clusters[j][k]])
                slice_cluster_perims.append(perimPoints)
            cluster_perims.append((slice_cluster_perims))
            prog_perim.update()
        return cluster_perims


if __name__ == "__main__":
    path_man = SpotData.SpotData(_base_dir='/home/rig12/Work/sunspots/2014-09-05_16-hourly/')
    sunspot_group_list = path_man.loadSpotData(path_man.getDir('dat'))
    thresholds = np.array([0.5,0.4,0.3,0.25,0.2,0.15])*60000
    tm = Tomography(thresholds, path_man, sunspot_group_list[0])
    clusterlist = tm.getClustersFromROIs()
