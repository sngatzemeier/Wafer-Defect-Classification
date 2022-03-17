
# import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses, optimizers, callbacks, Model

import cv2
import numpy as np

# Source: https://towardsdatascience.com/understand-your-algorithm-with-grad-cam-d3b62fce353

def GradCam(model, img, layer_name, eps=1e-8):
    '''
    Creates a grad-cam heatmap given a model and a layer name contained with that model
    

    Args:
      model: tf model
      img: (img_width x img_width) numpy array
      layer_name: str


    Returns 
      uint8 numpy array with shape (img_height, img_width)

    '''

    img_array = np.expand_dims(img, axis=0)
    
    gradModel = Model(inputs=[model.inputs],
                      outputs=[model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(img_array, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, 0]
    
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)
    
    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
  
    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))

    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    
    # heatmap = (heatmap * 255).astype("uint8")
    # return the resulting heatmap to the calling function
    return heatmap


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))


def superimpose(img_bgr, cam, thresh=0.5, emphasize=False, img_wt=1, heatmap_wt=0.75):
    
    '''
    Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.
    *Modified from original function - uses cv2 to superimpose instead
    

    Args:
      img_bgr: (img_width x img_height x 3) numpy array
      cam: grad-cam heatmap, (img_width x img_width) numpy array
      threshold: float
      emphasize: boolean

    Returns 
      uint8 numpy array with shape (img_height, img_width, 3)

    '''
    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img_rgb = cv2.addWeighted(img_bgr, img_wt, heatmap, heatmap_wt, 0)
    
#     hif = 0.8
#     superimposed_img = heatmap * hif + img_bgr
#     superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
#     superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb

def generate_img_bgr(img, binary=False):
    """Helper function to properly format wafer images for superimposing
       
       Args:
       img: waferMap image, uint8 numpy array with shape (img_height, img_width) and values [0, 2]
       binary: boolean, true if thinned wafermap"""
    
    if binary:
        img2 = np.uint8(img*255)
        img_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        return img_bgr
    else:
        img2 = np.uint8(img/2*255)
        img_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        imagem = cv2.bitwise_not(img_bgr)
        return imagem
