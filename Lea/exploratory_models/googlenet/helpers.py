
import pickle5 as pickle
import _pickle as cPickle
import gzip
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.metrics import confusion_matrix
from skimage.transform import resize as sk_resize
from scipy import ndimage
import random
from skimage.transform import rescale, resize, rotate


def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()

def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    object = pickle.load(file)
    file.close()

    return object


def plot_lot(df1, lot, fig_size=(10, 10), img_dims=[30, 30], resize=False, 
             filter_size=3, mfilter=False, vmax=2):
    """
    Helper function to plot entire lot of wafers from df1
    
    :param lot: -> str | lotName that will be plotted e.g. 'lot1'
    :param fig_size: -> list [x,y] pixles to resize the image to
    :param img_dims: -> tuple (x,y) to adjust the overall figure size
    :param resize: -> bool | Resize the image to `img_dims` if True 
    :param filtersize: -> int to set median filter size
    :param mfilter -> bool | apply median filter if True
    :param vmax -> int/float | max pixel value
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
                
            if resize:
                img = sk_resize(img, img_dims, 
                                order=0, preserve_range=True, anti_aliasing=False)
                
            if mfilter:
                img = ndimage.median_filter(img, size=filter_size)
                
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
                         vmin=0, vmax=vmax)
        axs[i, j].axis('off')

        # label the figure with the index# and defect classification [for future reference]
        axs[i, j].set_title(f'{index}\n{ftype}', fontsize=10)

    plt.show()
    
    
def plot_list(df1, wafer_list, fig_size=(10, 10), img_dims=[30, 30], resize=False, 
              filter_size=3, mfilter=False, vmax=2):
    """
    Helper function to plot entire lot of wafers from df1
    
    :param lot: -> str | lotName that will be plotted e.g. 'lot1'
    :param fig_size: -> list [x,y] pixles to resize the image to
    :param img_dims: -> tuple (x,y) to adjust the overall figure size
    :param resize: -> bool | Resize the image to `img_dims` if True 
    :param filtersize: -> int to set median filter size
    :param mfilter -> bool | apply median filter if True
    :param vmax -> int/float | max pixel value
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
                
            if resize:
                img = sk_resize(img, img_dims, 
                                order=0, preserve_range=True, anti_aliasing=False)
                
            if mfilter:
                img = ndimage.median_filter(img, size=filter_size)
                
        else:
            img = np.zeros_like(list_df.waferMap[0])
            index = ''
            ftype = ''

        # imshow to plot image in axs i,j location in plot
        i = n_row % ax_cnt
        j = int(n_row/ax_cnt)
        axs[i, j].imshow(img,
                         interpolation='none',
                         cmap=cmap, vmin=0, vmax=vmax)
        axs[i, j].axis('off')

        # label the figure with the index# and defect classification [for future reference]
        axs[i, j].set_title(f'{index}\n{ftype}', fontsize=10)

    plt.show()


def filter_comparison(df, index, filter_size=3, img_dims=[30, 30], resize=False, vmax=2):
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
    
    if resize:
        ex = sk_resize(ex, img_dims, 
                        order=0, preserve_range=True, anti_aliasing=False)
        img = sk_resize(img, img_dims, 
                        order=0, preserve_range=True, anti_aliasing=False)
        
    img = ndimage.median_filter(ex, size=filter_size)
        
    ax1.imshow(ex, cmap=cmap, vmin=0, vmax=vmax)
    ax1.set_axis_off()
    ax1.set_title('Original')
    ax2.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
    ax2.set_axis_off()
    ax2.set_title('Filtered')
    
    plt.show()

    
def defect_distribution(data, note=''):
    """Helper function to visualize distribution of defects
       Assumes none defects have been removed from data
       and data set has column failureType"""
    
    # count how many of each defect is present
    dist = data.groupby('failureType')['failureType'].count().sort_values()
    y = dist.tolist()
    x = dist.index.tolist()
    
    # bar plot
    plt.barh(x, y)
    xlim = math.ceil(max(y)*1.15)
    plt.xlim(0, xlim)
    plt.title(f'Failure Type Distribution\n({note})')

    for index, value in enumerate(y):
        plt.text(value, index,
                 str(value))

    plt.show()
    

def flip_rotate(df, col, defect, classLabel, labels, number, frac=25):
    """Helper function to produce number of new samples
       by randomly flipping and rotating
       
       :param df -> dataframe | source data
       :param col -> column containing wafer map
       :param defect -> str | failureType value
       :param classLabel -> int | classifyLabel value
       :param labels -> list | list of source data indices
       :param number -> int | number of new samples to generate
       :param frac -> int | out of 100, half the fraction of samples to be flipped
       
       Returns df of new samples"""
    
    new_df = pd.DataFrame()
    
    # how many to flip on direction
    f = math.ceil(random.randint(0, frac) / 100 * number)
    
    # how many to rotate
    r = number - 2*f
    
    # generate new flipped samples
    fliplr_list = random.choices(labels, k=f)
    for i in fliplr_list:
        img = df[col].loc[i]
        new_df = new_df.append({'ID':'A', 'failureType': defect, 'classifyLabels': classLabel, 
                                col: np.fliplr(img)}, ignore_index=True)
    
    flipud_list = random.choices(labels, k=f)
    for i in flipud_list:
        img = df[col].loc[i]
        new_df = new_df.append({'ID':'A', 'failureType': defect, 'classifyLabels': classLabel, 
                                col: np.flipud(img)}, ignore_index=True)
    
    # generate new rotated samples
    rotate_list = random.choices(labels, k=r)
    for i in rotate_list:
        img = df[col].loc[i]
        theta = random.randint(1, 359)
        new_df = new_df.append({'ID':'A', 'failureType': defect, 'classifyLabels': classLabel, 
                                col: rotate(img, theta)}, ignore_index=True)
    
    return new_df


def plot_confusion_matrix(y_test, y_pred, mode='classify', normalize=True, figsize=(7,5)):
    """Helper function for plotting confusion matrix of model results
       Modes: detect (none, defect), classify (all but none), all"""
    
    if mode == 'all':
        defects = ['N', 'L', 'EL', 'C', 'ER', 'S', 'R', 'NF', 'D']
    elif mode == 'detect':
        defects = ['None', 'Defect']
    elif mode == 'classify':
        defects = ['L', 'EL', 'C', 'ER', 'S', 'R', 'NF', 'D']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        f = sns.heatmap(cm, annot=True, xticklabels=defects, yticklabels=defects)
    
    else:
        cm = confusion_matrix(y_test, y_pred, normalize=None)
        f = sns.heatmap(cm, annot=True, xticklabels=defects, yticklabels=defects, fmt='d')
        
    f.set(xlabel='Predicted Label', ylabel='True Label')
