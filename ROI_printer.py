import matplotlib.pyplot as plt
import SpotData


my_cmaps = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar',
            'Pastel1', 'Pastel2', 'Paired', 'Accent','PiYG', 'PRGn', 'BrBG',
            'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm',
            'bwr', 'seismic', 'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'viridis', 'plasma', 'inferno', 'magma'
            ]

def plot_all_cmaps(roiList, cmaps, dir):
    for roi in roiList:
        fig = plt.figure(figsize=(3,3),dpi=300)
        ax = plt.subplot()
        img = ax.imshow(roi.data, cmap='Greys_r')
        plt.xlim([450,150])
        plt.ylim([450,150])
        plt.tight_layout()
        plt.savefig(dir + str(roi.filename) + '.png')
        plt.close(fig)

if __name__ == "__main__":
    sd = SpotData.SpotData('/mnt/alpha/work/PhD/DataArchive/sunspots/2014-09-05_16-hourly/')
    roiList = sd.loadROIList(sd.getDir('roi'))
    plot_all_cmaps(roiList, my_cmaps, '/mnt/alpha/work/PhD/DataArchive/sunspots/2014-09-05_16-hourly/output/roi/')