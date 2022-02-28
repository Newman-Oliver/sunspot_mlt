import numpy as np

class ROIReader():
    def load(path):
        data = np.loadtxt(path, skiprows=1)
        f = open(path, "r")
        line = f.readline()
        header = []
        for u in line.split():
            header.append(u)
            
        f.close()
        return (header, data)