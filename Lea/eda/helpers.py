
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.transform import resize as sk_resize

from scipy import ndimage
import cv2
  

def median_filter(data, kernel_size):
    """Manual implementation of median filter
       From https://www.researchgate.net/figure/Median-Filter-implementation-using-Python_fig9_332574579"""
    temp = []
    indexer = kernel_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range (len(data)):
        
        for j in range(len(data[0])):
            
            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(kernel_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
            
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def plot_lot(df1, lot, fig_size=(10, 10), filter_size=3, mfilter=False, filter_type='scipy',
             img_dims=[30, 30], resize=False):
    """
    Helper function to plot entire lot of wafers from df1
    
    :param lot: -> str | lotName that will be plotted e.g. 'lot1'
    :param fig_size: -> list [x,y] pixles to resize the image to
    :param filtersize: -> int to set median filter size
    :param mfilter -> bool | apply median filter if True
    :param filter_type -> str | method to use for median filtering ('scipy', 'cv', 'manual')
    :param img_dims: -> tuple (x,y) to adjust the overall figure size
    :param resize: -> bool | Resize the image to `img_dims` if True 
    """

    lot_df = df1[df1['lotName'] == lot]
    lot_df.reset_index(inplace=True)

    total_rows = len(lot_df.index)
    ax_cnt = 5
    
    print(f'{lot}')

    fig, axs = plt.subplots(ax_cnt, ax_cnt, figsize=fig_size)
    fig.tight_layout()
    
    # make a color map of fixed colors - blue passing die, fuchsia failing die
    cm_xkcd = colors.XKCD_COLORS.copy()
    cmap = colors.ListedColormap(
        [cm_xkcd['xkcd:white'], cm_xkcd['xkcd:azure'], cm_xkcd['xkcd:fuchsia']])

    # Nested for loops to loop through all digits and number of examples input for plotting
    for n_row in range(25):
        if n_row < total_rows:
            img = lot_df.waferMap[n_row]
            index = lot_df["index"][n_row]
            ftype = lot_df.failureType[n_row]
                
            if mfilter:
                if filter_type == 'scipy':
                    img = ndimage.median_filter(img, size=filter_size)
                elif filter_type == 'cv':
                    img = cv2.medianBlur(img, filter_size)
                elif filter_type == 'manual':
                    img = median_filter(img, filter_size)
            
            if resize:
                img = sk_resize(img, img_dims, 
                                order=0, preserve_range=True, anti_aliasing=False)
                
        else:
            img = np.zeros_like(lot_df.waferMap[0])
            index = ''
            ftype = ''

        # imshow to plot image in axs i,j location in plot
        i = n_row % ax_cnt
        j = int(n_row/ax_cnt)
        axs[i, j].imshow(img,
                         interpolation='none',
                         cmap=cmap,
                         vmin=0, vmax=2)
        axs[i, j].axis('off')

        # label the figure with the index# and defect classification [for future reference]
        axs[i, j].set_title(f'{index}\n{ftype}', fontsize=10)

    plt.show()
    
    
def filter_comparison(df, index, filter_type='scipy', filter_size=3,
                      img_dims=[30, 30], resize=False):
    """Helper function for looking at effect of median filter on one wafer map"""

    print(f"{df['lotName'].loc[index]}")
    print(f"{df['failureType'].loc[index]}")
    
    
    fig = plt.figure()

    # make a color map of fixed colors - blue passing die, fuchsia failing die
    cm_xkcd = colors.XKCD_COLORS.copy()
    #cmap = colors.ListedColormap(['white', 'blue', 'yellow'])
    cmap = colors.ListedColormap(
            [cm_xkcd['xkcd:white'], cm_xkcd['xkcd:azure'], cm_xkcd['xkcd:fuchsia']])

    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    
    ex = df['waferMap'].loc[index]
    if filter_type == 'scipy':
        img = ndimage.median_filter(ex, size=filter_size)
    elif filter_type == 'cv':
        img = cv2.medianBlur(ex, filter_size)
    elif filter_type == 'manual':
        img = median_filter(ex, filter_size)
        
    if resize:
        ex = sk_resize(ex, img_dims, 
                        order=0, preserve_range=True, anti_aliasing=False)
        img = sk_resize(img, img_dims, 
                        order=0, preserve_range=True, anti_aliasing=False)
        
    
    ax1.imshow(ex, cmap=cmap, vmin=0, vmax=2)
    ax1.set_axis_off()
    ax1.set_title('Original')
    ax2.imshow(img, cmap=cmap, vmin=0, vmax=2)
    ax2.set_axis_off()
    ax2.set_title('Filtered')
    
    plt.show()


def plot_list(df1, wafer_list, fig_size=(10, 10), filter_size=3, mfilter=False, filter_type='scipy',
              img_dims=[30, 30], resize=False):
    """
    Helper function to plot entire lot of wafers from df1
    
    :param lot: -> str | lotName that will be plotted e.g. 'lot1'
    :param fig_size: -> list [x,y] pixles to resize the image to
    :param filtersize: -> int to set median filter size
    :param mfilter -> bool | apply median filter if True
    :param filter_type -> str | method to use for median filtering ('scipy', 'cv', 'manual')
    :param img_dims: -> tuple (x,y) to adjust the overall figure size
    :param resize: -> bool | Resize the image to `img_dims` if True 
    """

    list_df = df1.iloc[wafer_list, :]
    list_df.reset_index(inplace=True)

    total_rows = len(list_df.index)
    ax_cnt = int(math.ceil(total_rows**(1/2)))


    fig, axs = plt.subplots(ax_cnt, ax_cnt, figsize=fig_size)
    fig.tight_layout()
    
    # make a color map of fixed colors - blue passing die, fuchsia failing die
    cm_xkcd = colors.XKCD_COLORS.copy()
    cmap = colors.ListedColormap(
        [cm_xkcd['xkcd:white'], cm_xkcd['xkcd:azure'], cm_xkcd['xkcd:fuchsia']])

    # Nested for loops to loop through all digits and number of examples input for plotting
    for n_row in range(ax_cnt**2):
        if n_row < total_rows:
            img = list_df.waferMap[n_row]
            index = list_df["index"][n_row]
            ftype = list_df.failureType[n_row]
                
            if mfilter:
                if filter_type == 'scipy':
                    img = ndimage.median_filter(img, size=filter_size)
                elif filter_type == 'cv':
                    img = cv2.medianBlur(img, filter_size)
                elif filter_type == 'manual':
                    img = median_filter(img, filter_size)
            
            if resize:
                img = sk_resize(img, img_dims, 
                                order=0, preserve_range=True, anti_aliasing=False)
                
        else:
            img = np.zeros_like(list_df.waferMap[0])
            index = ''
            ftype = ''

        # imshow to plot image in axs i,j location in plot
        i = n_row % ax_cnt
        j = int(n_row/ax_cnt)
        axs[i, j].imshow(img,
                         interpolation='none',
                         cmap=cmap, vmin=0, vmax=2)
        axs[i, j].axis('off')

        # label the figure with the index# and defect classification [for future reference]
        axs[i, j].set_title(f'{index}\n{ftype}', fontsize=10)

    plt.show()
