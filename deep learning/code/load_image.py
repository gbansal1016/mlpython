# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 00:15:28 2017

@author: gbans6
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from setup import maybe_extract, train_test_files

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
 

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names
  
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels
  
def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels  

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
  
  
dir_name = 'C:\\Users\\gbans6\\gbansalmba\\mlearning\\mlpython\\data\\nminst\\'

cwd = os.getcwd()
os.chdir(dir_name)  

print("train:", os.getcwd())

train_filename,test_filename = train_test_files()
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)  
train_datasets = maybe_pickle(train_folders, 45000)

test_datasets = maybe_pickle(test_folders, 1800)

display(Image(filename=dir_name+"notMNIST_small/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png"))


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

#print('Training:', train_dataset.shape, train_labels.shape)
#print('Validation:', valid_dataset.shape, valid_labels.shape)  
#print('Testing:', test_dataset.shape, test_labels.shape)

pickle_file = 'notMNIST.pickle'

    
if (os.path.exists(dir_name+pickle_file)):
    print("skipping pickling of training and test set")
else:
    try:
        f = open(dir_name+pickle_file, 'wb')
        save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise 

logit = LogisticRegression()


(samples, width, height) = train_dataset.shape

train_set = np.reshape(train_dataset, (samples, width*height))

logit.fit(train_set[0:10000,:], train_labels[0:10000])

(samples, width, height) = test_dataset.shape

test_set = np.reshape(test_dataset, (samples, width*height))

pred = logit.predict(test_set[0:1000,:])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_labels[0:1000], pred)
print('accuracy', accuracy)


fig = plt.figure()
fig.set_size_inches(10, 2)

num_classes = len(train_datasets)
filters = np.ndarray(shape=(num_classes, width, height), dtype=np.float32)

for class_i in range(0,(num_classes)):
    filters[class_i, :, :] = logit.coef_.reshape(num_classes, width, height)[class_i]
    a = fig.add_subplot(1, 10, (class_i+1))
    a.set_title('class:%s' % (class_i+1))
    plt.imshow(filters[class_i])
    plt.axis('off')
    plt.show()
    
