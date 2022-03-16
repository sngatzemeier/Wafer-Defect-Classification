
### S3 WRAPPER FOR MODEL
# Source: https://gist.github.com/ramdesh/f00ec1f5d01f03114264e8f3d0c226e8
import s3fs
import zipfile
import tempfile
import numpy as np
from tensorflow import keras
from pathlib import Path
import logging
import os

AWS_ACCESS_KEY=os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME='wafer-capstone'

def get_s3fs():
  return s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)


def zipdir(path, ziph):
  # Zipfile hook to zip up model folders
  length = len(path) # Doing this to get rid of parent folders
  for root, dirs, files in os.walk(path):
    folder = root[length:] # We don't need parent folders! Why in the world does zipfile zip the whole tree??
    for file in files:
      ziph.write(os.path.join(root, file), os.path.join(folder, file))

            
def s3_save_keras_model(model, model_name):
  with tempfile.TemporaryDirectory() as tempdir:
    model.save(f"{tempdir}/{model_name}")
    # Zip it up first
    zipf = zipfile.ZipFile(f"{tempdir}/{model_name}.zip", "w", zipfile.ZIP_STORED)
    zipdir(f"{tempdir}/{model_name}", zipf)
    zipf.close()
    s3fs = get_s3fs()
    s3fs.put(f"{tempdir}/{model_name}.zip", f"{BUCKET_NAME}/models/{model_name}.zip")
    logging.info(f"Saved zipped model at path s3://{BUCKET_NAME}/models/{model_name}.zip")
 

def s3_get_keras_model(model_name: str) -> keras.Model:
  with tempfile.TemporaryDirectory() as tempdir:
    s3fs = get_s3fs()
    # Fetch and save the zip file to the temporary directory
    s3fs.get(f"{BUCKET_NAME}/models/{model_name}.zip", f"{tempdir}/{model_name}.zip")
    # Extract the model zip file within the temporary directory
    with zipfile.ZipFile(f"{tempdir}/{model_name}.zip") as zip_ref:
        zip_ref.extractall(f"{tempdir}/{model_name}")
    # Load the keras model from the temporary directory
    return keras.models.load_model(f"{tempdir}/{model_name}")


import logging
import boto3
from botocore.exceptions import ClientError
import os

def s3_upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


### VISUALIZATION
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
