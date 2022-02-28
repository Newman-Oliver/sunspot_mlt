import numpy as np
import SpotTools as tools
import Logger

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(r, phi)

def cart2polData(x, y, centre, value=None):
    if(len(x) != len(y)):
        raise Exception('Dimensions incompatible for co-ordinate conversion. len(x) != len(y)')
    polData = []
    for i in range(0, len(x)):
        (r, phi) = cart2pol(x[i] - centre[0], y[i] - centre[1])
        if value is not None:
            polData.append([r, phi, value[i]])
        else:
            polData.append([r, phi])
    return np.array(polData)

def reformat_perimeter(perimeter):
    return [list(x) for x in zip(*perimeter)]

def getPerimeter(data):
    """Take a blob and return a list of all pixels that lie on the perimeter. The perimeter is determined by checking
    to see if all of the point's 8 neighbours are also in the data set. If one or more is missing, then the point must
    lie on the edge of the cluster.
    :param data should be in the form [[x1, y1], [x2, y2], ...] where [] is a list, not a numpy array!
    :return perimeter in the same form as data."""

    # Make sure all the points are in native python lists. numpy is used in the for loop to do the matrix addition but
    # then must be converted back into python lists for all() to work.
    data = [x.tolist() for x in data]
    directions = [[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]
    perimeter = []

    for point in data:
        # Get the neighbours
        neighbours = [(np.array(x) + np.array(point)).tolist() for x in directions]
        # Try to find all neighbours in data. If not all points are there, this is a perimeter point.
        is_perimeter_point = not all(x in data for x in neighbours)
        if is_perimeter_point:
            perimeter.append(point)

    return perimeter

def getClusterPerimeter(data):
    # Take a blob and return it's perimeter
    # Data should be a 2d array with each item in a [x,y,value] format.
    x = data[:,0]
    y = data[:,1]
    centre = [np.mean(x), np.mean(y)]
    radius = 200 # distance to search
    #polData = cart2polData(x,y,centre,data[:,2])

    # TODO: The actual perimetery stuff
    # Scan outwards from the centre looking for furthest points in each direction within the same cluster
    angles = np.arange(0,360,1)
    clusterCoords = np.array(list(zip(x,y)))
    perim_coords = []

    for i in range(0, len(angles)):
        pointsOnLine = tools.getPointsOnLine(centre,radius,np.deg2rad(angles[i]))
        lineCoords = list(zip(pointsOnLine[0],pointsOnLine[1]))
        matchedPixels = []

        for j in range(0, len(lineCoords)):
            for k in range(0,len(clusterCoords)):
                if lineCoords[j][0] == clusterCoords[k][0] and lineCoords[j][1] == clusterCoords[k][1]:
                    #matchAt = np.where(lineCoords[j] in clusterCoords)
                    sqrDist = ((lineCoords[j][0] - centre[0]) ** 2 +
                                (lineCoords[j][1] - centre[1]) ** 2)
                    matchedPixels.append((lineCoords[j],sqrDist))

        # Sort the list by the 2nd index of each tuple (i.e. the distance)
        matchedPixels.sort(key=lambda dist: dist[1],reverse=True)
        if len(matchedPixels) != 0:
            #perim_coords.append(matchedPixels[0][0])
            # TODO: Uncomment code below and comment out line above. Hoping to fix issue of unnecessary points being
            #       added to the perimeter. Haven't tested as waiting for program to finish.
            # TODO: 2020/05/25 I don't now the state of this and at this point I'm too scared to ask.
            if(matchedPixels[0][0] not in perim_coords):
                perim_coords.append(matchedPixels[0][0])

    # perim_coords should at this point be a list of coords such that: [[0,0],[0,1],[2,1],...], so calling
    # zip(*perim_coords) will "unzip" them.
    return [list(x) for x in (zip(*perim_coords))]
