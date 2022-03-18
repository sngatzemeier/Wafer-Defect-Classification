
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
from skimage.util import img_as_ubyte
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


def plot_lot(df1, lot, fig_size=(10, 10), col='waferMap', cmap='viridis'):
    """
    Helper function to plot entire lot of wafers from df1.
    Lots must have >= 2 samples.
    
    :param lot: -> str | lotName that will be plotted e.g. 'lot1'
    :param fig_size: -> tuple | size of plot
    :param col: -> str | column that contains waferMap image
    :param cmap: -> str | color scheme to use
    """

    lot_df = df1[df1['lotName'] == lot]
    lot_df.reset_index(inplace=True)

    total_rows = len(lot_df.index)
    ax_cnt = 5
    
    print(f'{lot}')

    fig, axs = plt.subplots(ax_cnt, ax_cnt, figsize=fig_size)
    fig.tight_layout()

    # Nested for loops to loop through all digits and number of examples input for plotting
    for n_row in range(25):
        if n_row < total_rows:
            img = lot_df[col][n_row]
            index = lot_df["index"][n_row]
            ftype = lot_df.failureType[n_row]
                
        else:
            img = np.zeros_like(lot_df[col][0])
            index = ''
            ftype = ''

        # imshow to plot image in axs i,j location in plot
        i = n_row % ax_cnt
        j = int(n_row/ax_cnt)
        axs[i, j].imshow(img,
                         interpolation='none',
                         cmap=cmap)
        axs[i, j].axis('off')

        # label the figure with the index# and defect classification [for future reference]
        axs[i, j].set_title(f'{index}\n{ftype}', fontsize=10)

    plt.show()
    
    
def plot_list(df1, wafer_list, fig_size=(10, 10), col='waferMap', cmap='viridis', mode='index'):
    """
    Helper function to plot a list of indices from df1.
    Lists must have >= 2 samples.
    
    :param wafer_list: -> list | list of indices or ids to be plotted
    :param fig_size: -> tuple | size of plot
    :param col: -> str | column that contains waferMap image
    :param cmap: -> str | color scheme to use
    :param mode: -> str | 'index' or 'id'
    """

    if mode == 'index':
        index_list = wafer_list
    elif mode == 'id':
        index_list = [df1.index[df1.ID == i][0] for i in wafer_list]
    
    list_df = df1.loc[index_list, :]
    list_df.reset_index(inplace=True)

    total_rows = len(list_df.index)
    ax_cnt = int(math.ceil(total_rows**(1/2)))


    fig, axs = plt.subplots(ax_cnt, ax_cnt, figsize=fig_size)
    fig.tight_layout()

    # Nested for loops to loop through all digits and number of examples input for plotting
    for n_row in range(ax_cnt**2):
        if n_row < total_rows:
            img = list_df[col][n_row]
            index = list_df["ID"][n_row]
            ftype = list_df.failureType[n_row]
                
        else:
            img = np.zeros_like(list_df[col][0])
            index = ''
            ftype = ''

        # imshow to plot image in axs i,j location in plot
        i = n_row % ax_cnt
        j = int(n_row/ax_cnt)
        axs[i, j].imshow(img,
                         interpolation='none',
                         cmap=cmap)
        axs[i, j].axis('off')

        # label the figure with the index# and defect classification [for future reference]
        axs[i, j].set_title(f'{index}\n{ftype}', fontsize=10)

    plt.show()

    
def defect_distribution(data, note='', mode='classify'):
    """Helper function to visualize distribution of defects
       :param mode -> str | classify or detect"""
    
    if mode == 'classify':
        col = 'classifyLabels'
    elif mode == 'detect':
        col = 'detectLabels'
    
    # count how many of each defect is present
    dist = data.groupby(col)[col].count().sort_values()
    y = dist.tolist()
    
    if mode == 'classify':
        fail_dict = {8: 'none', 0: 'Loc', 1: 'Edge-Loc', 2: 'Center', 3: 'Edge-Ring', 
                     4: 'Scratch', 5: 'Random', 6: 'Near-full', 7: 'Donut'}
        indices = dist.index.tolist()
        x = [fail_dict[i] for i in indices]
    elif mode == 'detect':
        x = ['None', 'Defect']
      
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
       by randomly flipping and rotating.
       Assumes that all samples are the same class.
       
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
                                col: img_as_ubyte(rotate(img, theta))}, ignore_index=True)
    
    return new_df


def plot_confusion_matrix(y_test, y_pred, mode='classify', normalize=True, figsize=(7,5)):
    """Helper function for plotting confusion matrix of model results
       Modes: detect, classify, all
       For all, assumes that none is labeled as 8"""
    
    if mode == 'classify':
        defects = ['L', 'EL', 'C', 'ER', 'S', 'R', 'NF', 'D']
    elif mode == 'detect':
        defects = ['None', 'Defect']
    elif mode == 'all':
        defects = ['L', 'EL', 'C', 'ER', 'S', 'R', 'NF', 'D', 'N']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        f = sns.heatmap(cm, annot=True, xticklabels=defects, yticklabels=defects)
    
    else:
        cm = confusion_matrix(y_test, y_pred, normalize=None)
        f = sns.heatmap(cm, annot=True, xticklabels=defects, yticklabels=defects, fmt='d')
        
    f.set(xlabel='Predicted Label', ylabel='True Label')

    
def visualize_misclassified(test_data, y_test, y_pred, true_label, pred_label, n, 
                            figsize=(10, 10), col='waferMap', cmap='viridis'):
    """Helper function that visualizes a random n samples
       that are mispredicted as pred_label.
       Uses helper function plot_list to visualize samples.
       
       :param true_label -> int | true label of the sample
       :param pred_label -> int | label that the model mistakenly predicted
       :param n -> int | number of samples to visualize"""
    
    # collect indices
    mistakes = [i for i in range(len(y_test)) if (y_test[i] == true_label and y_pred[i] == pred_label)]
    
    # take a random n samples
    if n > len(mistakes):
        random_n = random.sample(mistakes, len(mistakes))
    else:
        random_n = random.sample(mistakes, n)
    
    # visualize using plot_list
    plot_list(test_data, random_n, figsize, col, cmap)
