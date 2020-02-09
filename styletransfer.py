import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import time
import functools

import tensorflow as tf

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

#content_path = '/static/content.jpg'
#style_path = '/static/style.jpg'

def load_img(path_to_img):
  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  return img

def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x

content_layers = ['block5_conv2',
                  'block3_conv2'
                  ] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def get_model():
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
  vgg.trainable = False 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  return models.Model(vgg.input, model_outputs)

def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)

def total_variation_loss(x):
    assert K.ndim(x) == 4
    img_nrows = x.shape[1]
    img_ncols = x.shape[2]
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def get_feature_representations(model, content_path, style_path):
  # Load our images in 
  content_image = load_and_process_img(content_path)
  style_image = load_and_process_img(style_path)
  
  # batch compute content and style features
  style_outputs = model(style_image)
  content_outputs = model(content_image)
  #changed style_layer[0] to style_layer and same with content_layer
  # Get the style and content feature representations from our model  
  style_features = [style_layer for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features, applied_percent):

  style_weight, content_weight = loss_weights
  
  # Feed our init image through our model. This will give us the content and 
  # style representations at our desired layers. Since we're using eager
  # our model is callable just like any other function!
  model_outputs = model(init_image)
  
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
  
  style_score = 0
  content_score = 0

  # Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
  weight_per_style_layer = 1.0 / float(num_style_layers)
  weights_for_style_layer = [1, .0001, .001, .0001, .1]
  iterator = 0
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    # style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    style_score += weights_for_style_layer[iterator] * get_style_loss(comb_style[0], target_style)
    iterator += 1
  #Variation or noise loss is commented as it didnt show much imporvement in the results, to use it just uncomment the three lines with variation score
  # variation_score = total_variation_loss(init_image)

  # Accumulate content losses from all layers 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  weights_for_content_layer = [.8, .2]
  iterator = 0
  for target_content, comb_content in zip(content_features, content_output_features):
    # content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
    content_score += weights_for_content_layer[iterator] * get_content_loss(comb_content[0], target_content)
    iterator += 1

  if applied_percent < 50:
    to_be_reduced = 50 - applied_percent
    style_weight_multiplier = 10 ** (-1 * to_be_reduced/float(10))
    style_weight *= style_weight_multiplier
  elif applied_percent == 50:
    style_weight_multipier = 1
  else:
    to_be_increased = applied_percent - 50
    style_weight_multiplier = 10 ** (-1 * to_be_increased/float(10))
    style_weight *= style_weight_multiplier

  variation_weight = 0.0001

  # variation_score *= variation_weight
  style_score *= style_weight
  content_score *= content_weight

  # Get total loss
  loss = style_score + content_score #+ variation_score
  return loss, style_score, content_score

def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss

def run_style_transfer(content_path, 
                       style_path,
                       num_iterations=100,
                       content_weight=1e5, 
                       style_weight=3e3,
                       applied_percent=50): 
  # We don't need to (or want to) train any layers of our model, so we set their
  # trainable to false. 
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False
  
  # Get the style and content feature representations (from our specified intermediate layers) 
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  # Set initial image
  init_image = load_and_process_img(content_path)
  init_image = tf.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  opt = tf.train.AdamOptimizer(learning_rate=8, beta1=0.99, epsilon=1e-1)

  # For displaying intermediate images 
  iter_count = 1
  
  # Store our best result
  best_loss, best_img = float('inf'), None
  
  # Create a nice config 
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features,
      'applied_percent':  applied_percent
  }
    
  # For displaying
  num_rows = 2
  num_cols = 5
  display_interval = num_iterations/(num_rows*num_cols)
  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  
  imgs = []
  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time() 
    
    if loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

    if i % display_interval== 0:
      start_time = time.time()
      
  return best_img, best_loss

def create_result(content_path, style_path, result_path):
  best, _ = run_style_transfer(content_path, 
                                      style_path, num_iterations=1000)
  best = Image.fromarray(best)
  best.save(result_path)


def imshow(img, title=None):
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display 
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)

def show_results(best_img, content_path, style_path, show_large_final=True):
  # plt.figure(figsize=(10, 5))
  content = load_img(content_path) 
  style = load_img(style_path)

  plt.subplot(1, 2, 1)
  imshow(content, 'Content Image')

  plt.subplot(1, 2, 2)
  imshow(style, 'Style Image')

  if show_large_final: 
    plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()


#show_results(best, content_path, style_path)

import cv2
def convert_to_original_color(content_img, stylized_img):
  content_img = deprocess_img(content_img)
  # stylized_img = deprocess_img(stylized_img)
  stylized_img = best
  cvt_type = cv2.COLOR_BGR2LAB
  inv_cvt_type = cv2.COLOR_LAB2BGR
  content_cvt = cv2.cvtColor(content_img, cvt_type)
  stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
  c1, _, _ = cv2.split(stylized_cvt)
  _, c2, c3 = cv2.split(content_cvt)
  merged = cv2.merge((c1, c2, c3))
  dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
  dst = np.expand_dims(dst, axis=0)
  imshow(dst)

#content_img = load_and_process_img(content_path)
#convert_to_original_color(content_img, best)



