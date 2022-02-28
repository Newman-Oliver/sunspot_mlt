#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:17:27 2019

@author: richard
"""

import SpotData
from Logger import PrintProgress

import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.dates as matdates
import cv2


sd = SpotData.SpotData('/mnt/alpha/work/PhD/DataArchive/sunspots/NOAA_11158-hourly')
cvd_colours = ['#7530a0', '#5da853', '#528ad5', '#ebcd28', '#6f6297', '#4e866d', '#2bbacc']

def blurry_coefficient(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def delta_blurry_coeff(array, stride=1):
    diffs = []
    for i in range(0, len(array)):
        if i < stride:
            diffs.append(0.)
            continue
        diffs.append(array[i] - array[i-stride])
    return diffs


def get_roi_params(snapshot_list, normalise=False):
    parameters = {}
    parameters['darkest_point'] = []
    parameters['brightest_point'] = []
    parameters['average_intensity'] = []
    parameters['darkest_delta'] = []
    parameters['blurriness'] = []
    parameters['blurred_difference_coeff'] = []
    parameters['time'] = []
    prog = PrintProgress(0, len(snapshot_list))
    for i in range(0, len(snapshot_list)):
        try:
            roi = sd.loadROI(snapshot_list[i].ROI_path)
        except:
            continue
        parameters['time'].append(matdates.date2num(roi.timestamp))
        parameters['darkest_point'].append(np.min(roi.data))
        parameters['brightest_point'].append(np.max(roi.data))
        parameters['average_intensity'].append(np.mean(roi.data))
        parameters['blurriness'].append(blurry_coefficient(roi.data))
        parameters['blurred_difference_coeff'].append(get_blurred_difference(roi.data))
        # if i >= 1:
        #     parameters['darkest_delta'].append(parameters['darkest_point'][i] - parameters['darkest_point'][i-1])
        prog.update()

    if normalise:
        parameters['darkest_point'] = (np.array(parameters['darkest_point'])
                                       - parameters['darkest_point'][0]).tolist()
        parameters['brightest_point'] = (np.array(parameters['brightest_point'])
                                         - parameters['brightest_point'][0]).tolist()
        parameters['average_intensity'] = (np.array(parameters['average_intensity'])
                                           - parameters['average_intensity'][0]).tolist()
    return parameters


def get_blurred_difference(data):
    """Get the difference in blurriness between the roi and one with a gaussian filter applied. if the ROI is blurry
        to start with, the difference between the two values should be small"""
    regular_blurriness = blurry_coefficient(data)
    gauss_img = cv2.blur(data, (10,10))
    blurriness_gauss = blurry_coefficient(gauss_img)
    return regular_blurriness - blurriness_gauss


def plot_darkest_intensity(parameters):
    fig = plt.figure(figsize=(10,5), dpi=90)
    plt.rcParams.update({'font.size': 10})
    ax = fig.add_subplot(111)

    # Plot
    ax.axhline(y=0, xmin=parameters['time'][0], xmax=parameters['time'][-1], c='k', lw=1)
    ax.plot_date(parameters['time'], parameters['darkest_point'], c=cvd_colours[0],
                 label='Darkest point', ms=((72. / fig.dpi) ** 2) * 5, linestyle='-')
    ax.plot_date(parameters['time'], parameters['brightest_point'], c=cvd_colours[1],
                 label='Brightest point', ms=((72. / fig.dpi) ** 2) * 5, linestyle='-')
    ax.plot_date(parameters['time'], parameters['average_intensity'], c=cvd_colours[2],
                 label='ROI Average', ms=((72. / fig.dpi) ** 2) * 5, linestyle='-')

    # Axes
    ax.set_ylabel("Delta Intensity (pixel value)")
    ax.set_xlabel("Time (day/month)")
    ax.set_xlim([parameters['time'][0], parameters['time'][-1]])
    date_formatter = matdates.DateFormatter('%d/%m')
    ax.xaxis.set_major_formatter(date_formatter)

    # Save and format figure
    leg = plt.legend()
    plt.tight_layout()
    plt.savefig(str(sd.getDir('output')) + '/intensity_vs_time.png')


def plot_blurriness(parameters):
    fig = plt.figure(figsize=(10,5.4), dpi=90)
    plt.rcParams.update({'font.size': 12})
    ax = fig.add_subplot(111)
    #plt.title("ROI Blurriness in NOAA 12158")

    delta_blurriness = delta_blurry_coeff(parameters['blurriness'], stride=1)

    ax.axhline(y=0, xmin=parameters['time'][0], xmax=parameters['time'][-1], c='k', lw=1)
    ax.plot_date(parameters['time'], parameters['blurriness'],
                 c=cvd_colours[0], label='Blurriness',
                 linestyle='-', ms=((72. / fig.dpi) ** 2) * 5)
    ax.set_yscale('log')
    ax.set_ylabel("Blurriness Coefficient", labelpad=6.)
    ax.set_xlabel("Time (day/month)")
    ax.set_xlim([parameters['time'][0], parameters['time'][-1]])
    ax.set_ylim([1.e4,2.e7])
    date_formatter = matdates.DateFormatter('%d/%m')
    ax.xaxis.set_major_formatter(date_formatter)

    ax2 = ax.twinx()
    ax2.plot_date(parameters['time'], delta_blurriness, c=cvd_colours[1], label='Blurriness Difference Coefficient',
                  linestyle='-', ms=((72. / fig.dpi) ** 2) * 5)
    ax2.set_ylabel("Running Difference (arb. units)", labelpad=6.)
    ax2.xaxis.set_major_formatter(date_formatter)
    ax2.set_ylim([-7.e6, 7.e6])

    # Switch the render order of the plots so plot 2 doesn't obscure plot 1
    ax.set_zorder(1)
    ax.set_frame_on(False)
    fig.tight_layout()
    plt.savefig(str(sd.getDir('output')) + '/blurriness.png')


def plot_blurriness_delta(parameters):
    fig = plt.figure(figsize=(8,4), dpi=120)
    plt.rcParams.update({'font.size': 16})
    ax = fig.add_subplot(111)

    # Get data
    delta_blurriness = delta_blurry_coeff(parameters['blurriness'], stride=1)
    delta_blurred_diff = delta_blurry_coeff(parameters['blurred_difference_coeff'], stride=1)

    #Plot Data
    ax.plot_date(parameters['time'], delta_blurriness, c=cvd_colours[0], label='Delta Blurriness',
                 linestyle='-', ms=((72. / fig.dpi) ** 2) * 5)

    #second y axis
    ax2 = ax.twinx()
    ax2.plot_date(parameters['time'], delta_blurred_diff, c=cvd_colours[1], label='Delta Blurred Difference',
                  linestyle='-', ms=((72. / fig.dpi) ** 2) * 5)

    # Axes labels
    ax.set_ylabel("Change in blurriness coefficient (arb. units)")
    ax2.set_ylabel("Change in blurred difference coefficient (arb. units)")
    ax.set_xlabel("Time (day/month)")

    #Save and format
    leg = plt.legend()
    plt.tight_layout()
    plt.savefig(str(sd.getDir('output')) + '/delta_blurriness_vs_time.png')



def plot_deltas(parameters):
    fig = plt.figure(figsize=(16, 9), dpi=90)

    plt.plot_date(parameters['time'], parameters['darkest_delta'], c='blue', label='Darkest delta')

    # Axes
    plt.ylabel("Intensity (pixel value)")
    plt.xlabel("Time (day/month)")
    # ax = plt.axes
    # date_formatter = matdates.DateFormatter('%d/%m')
    # ax.xaxis.set_major_formatter(date_formatter)

    # Save and format figure
    leg = plt.legend()
    plt.tight_layout()
    plt.savefig(str(sd.getDir('output')) + '/intensity_delta_vs_time.png')
    
def load_parameters(_path):
    path = _path / "intensity_parameters.dat"
    with path.open("rb") as f:
        parameters = pickle.load(f, encoding='bytes')
    return parameters
    

if __name__ == "__main__":
    spot_list = sd.loadSpotData(sd.getDir('dat'))
    print("Loaded!")
    parameters = get_roi_params(spot_list[0].history, normalise=True)
    #parameters = load_parameters(sd.getDir('experiment_data'))
    plot_darkest_intensity(parameters)
    #plot_deltas(parameters)

    print("Saving...")
    path = sd.getDir('experiment_data')
    filepath = path / 'intensity_parameters.dat'
    with filepath.open('wb') as file_object:
        pickle.dump(parameters, file_object)

    print("Attempting blurriness plot")
    plot_blurriness(parameters)
    print("Done!")
    
    
    
        
