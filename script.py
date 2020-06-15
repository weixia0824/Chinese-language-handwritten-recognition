def parse_label(label):
    result = label
    try:
        test = label[1]  # this will fail for chinese chars
        result = label[0]
    except:
        pass # uses 2 bytes for this char
    return result

import struct
import chardet
import functools

@functools.lru_cache()
def get_label_dict_casia(data_files):
  char_dict = {}
  for data_file in data_files:
    with open(data_file, 'rb') as f:
      while True:
        packed_length = f.read(4)
        if packed_length == b'':
          break
        length = struct.unpack("<I", packed_length)[0]
        raw_label = struct.unpack(">cc", f.read(2))
        width = struct.unpack("<H", f.read(2))[0]
        height = struct.unpack("<H", f.read(2))[0]
        photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))
        unencoded_label = raw_label[0] + raw_label[1]
        try:
          label = unencoded_label.decode("GB2312-80")
        except:
          label = unencoded_label.decode(chardet.detect(unencoded_label)["encoding"])
        if label not in char_dict.values():
          char_dict[len(char_dict)] = parse_label(label)
  return char_dict
      
def get_label_dict_hit(label_file):
  char_dict = {}
  with open(label_file, 'r', encoding='GB2312-80') as f:
    label_str = f.read()
    for i in range(len(label_str)):
        char_dict[int(i)] = label_str[i]
  return char_dict

import struct
import chardet
import numpy as np

def data_generator_casia(data_files_casia, label_dict, selected=None, transforms=[], return_missing=False):
  for data_file in data_files:
    selected_copy = selected.copy() if selected is not None else label_dict.keys()
    #print(data_file)
    with open(data_file, 'rb') as f:
      while True:
        if len(selected_copy) == 0:
          break
        # print(str(i) + " < " + str(limit))
        skipped = False
        packed_length = f.read(4)
        if packed_length == b'':
          break
        length = struct.unpack("<I", packed_length)[0]
        raw_label = struct.unpack(">cc", f.read(2))
        width = struct.unpack("<H", f.read(2))[0]
        height = struct.unpack("<H", f.read(2))[0]
        photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))
        unencoded_label = raw_label[0] + raw_label[1]
        try:
          label = unencoded_label.decode("GB2312-80")
        except:
          label = unencoded_label.decode(chardet.detect(unencoded_label)["encoding"])
        label = parse_label(label)

        if label not in selected_copy:
          continue

        while True:
          popped = selected_copy.pop(0)

          if popped != label:
            # missing value -> return -1 as label with empty image
            if return_missing:
              empty_image = np.zeros([64, 64, 3], dtype=np.uint8)
              empty_image.fill(255)
              yield (empty_image, -1)
          else:
            break

        image = np.array(photo_bytes, dtype='uint8').reshape(height, width)
        for transform in transforms:
          image = transform(image)
        
        key = list(label_dict.values()).index(label)
        yield (image, key)

def data_generator_hit(data_files, label_dict, selected=None, transforms=[], return_missing=False):
  for data_file in data_files:
    selected_copy = selected.copy() if selected is not None else label_dict.keys()
    with open(data_file, 'rb') as f:
      nChars = int(np.fromfile(f, dtype='int32', count=1)[0])
      nCharPixelHeight = int(np.fromfile(f, dtype='uint8', count=1)[0])
      nCharPixelWidth = int(np.fromfile(f, dtype='uint8', count=1)[0])
      range_number = nChars
      for n in range(range_number):
        character = label_dict[n]
        image = np.fromfile(f, dtype='uint8', count=nCharPixelWidth * nCharPixelHeight)
        if len(selected_copy) == 0:
          break
        if character in selected_copy:
          selected_copy.pop(0)
          image = image.reshape(nCharPixelWidth, nCharPixelHeight)
          for transform in transforms:
            image = transform(image)
          yield (image, n)

# DataProcessor : prepares data such that it can be easily used by different models
class DataProcessor:
  ''' Process image data into appropriate format for models, images need to be changed into numpy arrays '''

  def __init__(self, files, data_generator, label_dict, selected, transform=None, return_missing=False):
    '''
        Args:
            files: List of strings that are paths pointing to dataset files, each file should be in same format
            data_generator: Generator that loads the file and returns a tuple with the next image and label on each iteration
    '''
    self.generator = data_generator(files, label_dict, selected, transform, return_missing)
    self.label_dict = label_dict

  def get_label_dict(self):
    ''' transform label integer to text with this dictionary '''
    return self.label_dict
  
  def get_data(self):
    ''' returns quadruple in format: (train_images, train_labels, test_images, test_labels) '''
    return self.generator

import glob

def get_data(dataset):

    # Format dataset using our DataProcessor
    data_files_casia = glob.glob('casia/*')
    data_files_hit = glob.glob('hit/*_images')

    # Get parameters based on configuration
    data_files = data_files_casia if dataset == "CASIA" else data_files_hit
    data_generator = data_generator_casia if dataset == "CASIA" else data_generator_hit

    # Get labels
    label_dict = get_label_dict_casia(tuple(data_files_casia)) if dataset == "CASIA" else get_label_dict_hit("hit/labels.txt")

    return data_files, data_generator, label_dict

import cv2

def transform_func(image):

  resize_h, resize_w = (64,64)
  h, w = image.shape

  # resize if too small by adding white area
  top_bottom = int(np.floor((resize_h - h) / 2) if h < resize_h else 0)
  left_right = int(np.ceil((resize_w - w) / 2) if w < resize_w else 0)
  image = cv2.copyMakeBorder(image, top_bottom, top_bottom, left_right, left_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

  # resize if too large by scaling
  image = cv2.resize(image, (resize_h, resize_w))

  return image


import matplotlib.pyplot as plt 

def explore_data(data_files, data_generator, label_dict, selected, transform=None, rows=4):

    n_classes = len(selected)
    data_processor = DataProcessor(data_files, data_generator, label_dict, selected, transform, return_missing=False)
    data_processor_labels = data_processor.get_label_dict()
    data_limited = list(data_processor.get_data())
    print("Limited dataset length: " + str(len(data_limited)))
    num_of_images = n_classes * rows
    f = plt.figure(figsize=(18, rows))
    f.patch.set_facecolor('black')
    for index in range(0, num_of_images):
        subplot = f.add_subplot(rows, n_classes, index + 1)
        subplot.axis('off')
        img, lbl = data_limited[index]
        subplot.title.set_text(lbl)
        subplot.imshow(img.squeeze(), cmap='gray')

    data_limited = [(img, lbl) for img, lbl in data_limited if lbl != -1 and lbl < n_classes]
    # TODO: why need < 10 test???????
    print(set([lbl for img, lbl in data_limited]))
    print("Limited dataset without missing values length: " + str(len(data_limited)))

    return data_limited

# data_files, data_generator, label_dict = get_data('CASIA')
# print(len(label_dict))

# selected = list(label_dict.values())[180:190]
# processed_data = explore_data(data_files, data_generator, label_dict, selected, transform_func, 50)

# selected = list(label_dict.values())[180:190]
# selected

import matplotlib.pyplot as plt 

def prepare_data(data_files, data_generator, label_dict, selected, transform=None):

    data_processor = DataProcessor(data_files, data_generator, label_dict, selected, transform, return_missing=False)
    data_processor_labels = data_processor.get_label_dict()
    data_limited = list(data_processor.get_data())
    return data_limited

from sklearn.model_selection import train_test_split

def split(data_limited):
    images, labels = zip(*data_limited)
    images = np.array([i.flatten() for i in images])
    trainImg, testImg, trainLbl, testLbl = train_test_split(images, labels, test_size=0.25, random_state=22)

    print("Training size: " + str(len(trainImg)) + ", Test size: " + str(len(testImg)))
    return trainImg, testImg, trainLbl, testLbl 

from datetime import datetime

def time_action(action, *args):
  start = datetime.now()
  result = action(*args)
  duration = str((datetime.now() - start).total_seconds())
  duration_str = duration[:duration.index(".") + 3]
  return result, float(duration_str)

from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import random

def knn(trainImg, trainLbl, testImg, testLbl, n_neighbors=1):

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    nothing, fit_duration = time_action(model.fit, trainImg, trainLbl)
    acc, acc_duration = time_action(model.score, testImg, testLbl)
    print("Finished fit after: {0}s.".format(str(fit_duration)))
    print("Finished score '{0}%' after: {1}s.".format(str(acc),  str(acc_duration)))

    resize_h, resize_w = resize
    prediction_random = random.randint(0, n_classes)
    prediction, prediction_duration = time_action(model.predict, [trainImg[prediction_random]])
    plt.imshow(trainImg[prediction_random].reshape(resize_h, resize_w), cmap="gray")
    print("Finished single prediction '{0}' after: {1}s.".format(label_dict[prediction[0]], str(prediction_duration)))

    dump(model, dataset + '-knn-' + str(n_neighbors) + '.joblib') 

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump, load

def svm(trainImg, trainLbl, testImg, testLbl, kernel='rbf', C=1.0):

    svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C))
    nothing, fit_duration = time_action(svm.fit, trainImg, trainLbl)
    acc, acc_duration = time_action(svm.score, testImg, testLbl)

    print("Finished fit after: {0}s.".format(str(fit_duration)))
    print("Finished score '{0}%' after: {1}s.".format(str(acc),  str(acc_duration)))

    resize_h, resize_w = resize
    prediction_random = random.randint(0, n_classes)
    prediction, prediction_duration = time_action(svm.predict, [trainImg[prediction_random]])
    plt.imshow(trainImg[prediction_random].reshape(resize_h, resize_w), cmap="gray")
    print("Finished single prediction '{0}' after: {1}s.".format(label_dict[prediction[0]], str(prediction_duration)))

    dump(svm, dataset + '-svm-' + kernel + '-' + str(C) + '.joblib') 

from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import random

def rfc(trainImg, trainLbl, testImg, testLbl, n_jobs=-1, n_estimators=2000):

    rfc = RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators)
    nothing, fit_duration = time_action(rfc.fit, trainImg, trainLbl)
    acc, acc_duration = time_action(rfc.score, testImg, testLbl)
    print("Finished fit after: {0}s.".format(str(fit_duration)))
    print("Finished score '{0}%' after: {1}s.".format(str(acc),  str(acc_duration)))

    resize_h, resize_w = resize
    prediction_random = random.randint(0, n_classes)
    prediction, prediction_duration = time_action(rfc.predict, [trainImg[prediction_random]])
    plt.imshow(trainImg[prediction_random].reshape(resize_h, resize_w), cmap="gray")
    print("Finished single prediction '{0}' after: {1}s.".format(label_dict[prediction[0]], str(prediction_duration)))

    dump(rfc, dataset + '-randomforest-' + str(abs(n_jobs)) + '-' + str(n_estimators) + '.joblib') 

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras import backend as K
from joblib import dump, load
import random

def svm(trainImg, trainLbl, testImg, testLbl, learning_rate = 0.02, batch_size = 32, num_classes = 100, epochs = 5):

    np.random.seed(32)  # for reproducibility

    # input image dimensions
    img_rows, img_cols = resize

    # 1. Load data into train and test sets
    (x_train, y_train, x_test, y_test) = (trainImg, trainLbl, testImg, testLbl)

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
        
    # 3. "one-hot encoding"
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # 4. Define model architecture
    input_shape_ann = (img_rows * img_cols,)
    ANN = Sequential()
    ANN.name = 'ANN'
    ANN.add(Dense(512, activation='relu', input_shape=input_shape_ann))
    ANN.add(Dropout(0.2))
    ANN.add(Dense(512, activation='relu'))
    ANN.add(Dropout(0.2))
    ANN.add(Dense(num_classes, activation='softmax'))

    CNN = Sequential()
    CNN.name = 'CNN'
    CNN.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    CNN.add(Conv2D(64, (3, 3), activation='relu'))
    CNN.add(MaxPooling2D(pool_size=(2, 2)))
    CNN.add(Dropout(0.25))
    CNN.add(Flatten())
    CNN.add(Dense(256, activation='relu'))
    CNN.add(Dropout(0.5))
    CNN.add(Dense(num_classes, activation='softmax'))

    models = [CNN]  # ANN

    res = []

    for model in models:
        # 2. Preprocess input data
        if model.name == 'ANN':
            x_train = x_train.reshape(x_train.shape[0], input_shape_ann[0])
            x_test = x_test.reshape(x_test.shape[0], input_shape_ann[0])
        elif model.name == 'CNN':
            x_train = x_train.reshape(x_train.shape[0], *input_shape)
            x_test = x_test.reshape(x_test.shape[0], *input_shape)
            
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # x_train /= 255
        # x_test /= 255
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        model.summary()
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', str(score)) # [0]
        #print('Test accuracy:', str(score[1]))
        res.append(model)

    model_ann, model_cnn = res

    resize_h, resize_w = resize
    prediction_random = random.randint(0, len(x_test))
    prediction, prediction_duration = time_action(model_cnn.predict, np.array([x_test[prediction_random]]))
    print(prediction)
    prediction = np.where(prediction[0]==max(prediction[0]))[0]
    print("Finished single prediction '{0}' after: {1}s.".format(label_dict[prediction[0]], str(prediction_duration)))
    plot = plt.imshow(x_test[prediction_random].reshape(resize_h, resize_w), cmap="gray")

    dump(model_ann, dataset + '-ann-' + str(learning_rate) + '.joblib')
    dump(model_cnn, dataset + '-cnn-' + str(learning_rate) + '.joblib')


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize
from skimage import feature
from skimage.morphology import dilation
from skimage.morphology import square

def raw_pixel_intensities(images):
    return images

def preprocessed_pixel_intensities(images):
    images = feature.canny(images, sigma=2) # binarize included
    images = dilation(images, square(2))
    images = images / 255 # normalize
    # images = skeletonize(images)
    return images

def principal_component_analysis(images):
    images = StandardScaler().fit_transform(images)
    images = PCA(n_components=10).fit_transform(images)
    return images

def triple_loss_embeddings(images):
    return images

from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

param_grid = ParameterGrid({
    'dataset': [['CASIA', 172], ['HIT', 62]],#, 'CASIA'],
    'preprocessor': [
        raw_pixel_intensities, 
        preprocessed_pixel_intensities, 
        principal_component_analysis
    ],  # triple_loss_embeddings
    'classes': [3, 10],#[3, 10, 50, 100, 200]
    'model': [[KNeighborsClassifier(n_neighbors=1), 'knn'], [SVC(), 'svc'], [RandomForestClassifier(), 'random_forest']]
})
for params in param_grid:

    print(params)
    dataset_name = params['dataset'][0]
    start = params['dataset'][1]
    end = start + params['classes']
    preprocessor = params['preprocessor']
    model = params['model'][0]
    model_name = params['model'][1]

    data_files, data_generator, label_dict = get_data(dataset_name)
    preprocessor_name = str(preprocessor).split(' ')[1]
    classes = list(label_dict.values())[start:end]
    n_classes = str(len(classes))
    processed_data = prepare_data(data_files, data_generator, label_dict, classes, [transform_func, preprocessor])
    trainImg, testImg, trainLbl, testLbl = split(processed_data)
    # print(classes)

    _, fit_duration = time_action(model.fit, trainImg, trainLbl)
    acc, acc_duration = time_action(model.score, testImg, testLbl)
    # print("-- Finished fit after: {0}s.".format(str(fit_duration)))
    # print("-- Finished score '{0}%' after: {1}s.".format(str(acc),  str(acc_duration)))

    prediction_random = random.randint(start, end - 1)
    index_of_random_char_in_trainLbl = trainLbl.index(prediction_random)
    random_char = label_dict[prediction_random]
    prediction, prediction_duration = time_action(model.predict, [trainImg[index_of_random_char_in_trainLbl]])
    predicted_char = label_dict[prediction[0]]
    # print("-- Predicted '{0}' for '{1}' after: {2}s.".format(predicted_char, random_char, str(prediction_duration)))

    print("-- {0} - {1} - {2} - {3} - {4}".format(dataset_name, preprocessor_name, n_classes, model_name, acc))

    dump(model, params['dataset'][0] + '.joblib')