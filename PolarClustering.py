#My modules
import ROIReader as ROI

#Misc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

#Clustering
from pyclustering.cluster.optics import optics


def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(r, phi)

def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return(x, y)

def applyOPTICS(dataset, eps, minPts):
    '''Apply the OPTICS clustering algorithm to the data and return the resulting clusters.'''
    instance = optics(dataset,eps,minPts)
    #print("Processing...");
    instance.process()
    return instance.get_clusters()

def extractPolData(data):
    '''Create a list of the polar coordinates and intensity at those coordinates.'''
    (x,y) = np.shape(data)
    # Data not being a square array is troublesome, so cut off edge data if too long. 
    if(x != y):
        if y < x:
            x = y
        else:
            y = x
    polData = []
    for i in range(0,y):
        for j in range (0,x):
            (r, phi) = cart2pol(j-(x/2),i-(y/2))
            polData.append((r, phi, data[j,i]))
    return np.array(polData)

roiBasePath = Path("/home/rig12/Work/sunspots/Single_ROI/roi/")
roiFiles = [x for x in roiBasePath.glob('*.roi') if x.is_file()]

colourList = ['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:pink', 'xkcd:brown', 'xkcd:red', 'xkcd:light blue', 'xkcd:teal',
              'xkcd:orange', 'xkcd:light green', 'xkcd:magenta', 'xkcd:yellow', 'xkcd:sky blue', 'xkcd:grey', 'xkcd:lime green', 'xkcd:light purple',
              'xkcd:violet', 'xkcd:dark green', 'xkcd:turquoise', 'xkcd:lavender', 'xkcd:dark blue', 'xkcd:tan', 'xkcd:cyan', 'xkcd:forest green',
              'xkcd:mauve', 'xkcd:dark purple', 'xkcd:bright green', 'xkcd:maroon', 'xkcd:olive', 'xkcd:salmon', 'xkcd:beige', 'xkcd:royal blue']

eps = 100
minPts = 3

for i in range(0,len(roiFiles)):
    (header, data) = ROI.ROIReader.load(str(roiFiles[i].resolve()))
    
    polData = extractPolData(data)
    #umbraData = polData[np.where(polData[:,2] < 30000)]
    umbraData = np.transpose(np.array([polData[:,0],polData[:,2]]))
    clusters = applyOPTICS(umbraData, eps, minPts)
    clusters.sort(key=len, reverse=True)
    #print("Clusters: " + str(clusters))
    print("Number of Clusters: " + str(len(clusters)))
    
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(111)
    
    for j in range(0, len(clusters) if len(clusters) <= 31 else 31):
        cartX, cartY = pol2cart(polData[clusters[j],0], polData[clusters[j],1])
        ax2d.scatter(cartY, cartX, c=colourList[j])
    plt.title('Epsilon = ' + str(eps) + ' clusters: ' + str(len(clusters)))
    plt.show()
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection='3d')
#    #ax.scatter(polData[:,0], polData[:,1], zs=polData[:,2], c='xkcd:slate')
#    for j in range(0, len(clusters) if len(clusters) <= 31 else 31):
#        ax.scatter(umbraData[clusters[j],0], umbraData[clusters[j],1], zs=umbraData[clusters[j],2], c=colourList[j])
#    ax.set_xlabel('Distance, r (pix)')
#    ax.set_ylabel('Angle, phi (rad)')
#    ax.set_zlabel('Intensity (arb.)')
#    plt.show()
    