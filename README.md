# Automatic-Sleep-Stage-Classification-using-EEG-Data
Here we use EEG data to automatically classify sleep stages using neural networks built using Keras.

Data is acquired from the Sleep-EDF Database Extended. https://physionet.org/content/sleep-edfx/1.0.0/
This database contains 197 whole night sleep cycles with two files paired to each. First a PSG.edf file which contains the following information EEG (from Fpz-Cz and Pz-Oz electrode locations), EOG (horizontal), submental chin EMG, and an event marker. Second a Hypnogram.edf file which contains the 8 labels associated with each timestep. These labels include wake, REM sleep, stages 1 to 4, movement time, and ? or unlabeled data.

We are only interested in the EEG data of the PSG.edf file and the label data of the Hypnogram.edf file.

# Code Explanation

First pyedflib installation is necessary
```
pip install pyedflib
```
Here all necessary packages are imported
```  
  import numpy as np
  import pyedflib
  import glob
  import os
  
  from sklearn.model_selection import train_test_split
  
  import matplotlib.pyplot as plt
  import tensorflow as tf
  
  from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
  from tensorflow.keras.utils import plot_model
  from tensorflow.keras.utils import to_categorical
  
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Dropout

  # Keras RNN imports
  from tensorflow.keras.layers import LSTM
  from tensorflow.keras.layers import RNN, SimpleRNN
  
  # for data normalization
  from sklearn.preprocessing import MinMaxScaler
```


Here we get all files together, separating the hypnogram and PSG files, and re-ordering them so that they match and therefore the labelling matches
```
path = '/content/Samples' #location of data
c = glob.glob(os.path.join(path, '*Hypnogram.edf'))
d = glob.glob(os.path.join(path, '*PSG.edf'))
len_annot = len(glob.glob(os.path.join(path, '*Hypnogram.edf')))
len_signal = len(glob.glob(os.path.join(path, '*PSG.edf')))
c.sort()
d.sort()
```
Next all data is organized
```
X_data_total = np.zeros((0,3000,2))
Y_data_total = np.zeros(0)

for i in range(0, len_annot):
  #opening the files
  f = pyedflib.EdfReader(d[i]) #opening the signals file
  g = pyedflib.EdfReader(c[i]) #opening the annotations file
  n = f.signals_in_file #look at number of signals in file
  
  #reading in signals
  sigbufs = np.zeros((n-5, f.getNSamples()[0])) #allocating array for signal, taking the entire length of the signal, but only the first two rows because the rest are not EEG
  for i in np.arange(n-5):
    sigbufs[i, :] = f.readSignal(i) #reading in signal channel by channel

  #now reading the annotations in the hypnogram file, and formatting it in a way we want
  annots = g.readAnnotations() #reading annotations
  
  annots_norm = np.empty((len(annots)-1,len(annots[0]))) #pre-allocating an array for normalized data
  annots_norm[0:2,:] = np.asarray(annots[0:2],dtype =np.int64) #setting the first two rows of signals as an np array, so that I can perform calculations
  annots_str = np.asarray(annots[2]) #put the strings of classifications into a different array

  annots_norm[0,:] = annots_norm[0,:]/30 #based on the 1Hz samples, divide by 30 seconds to get 30s epoch, for both channels
  annots_norm[1,:] = annots_norm[1,:]/30

  # this chunk maps the annotations to an integer, 0 = wake, 1-4 = sleep stage 1-4, Rem = 5, not annotated (?) = 6
  annots_strtoclass_1 = np.char.replace(annots_str, ['Sleep stage W'], ['0'])
  annots_strtoclass_2 = np.char.replace(annots_strtoclass_1, ['Sleep stage 1'], ['1'])
  annots_strtoclass_3 = np.char.replace(annots_strtoclass_2, ['Sleep stage 2'], ['2'])
  annots_strtoclass_4 = np.char.replace(annots_strtoclass_3, ['Sleep stage 3'], ['3'])
  annots_strtoclass_5 = np.char.replace(annots_strtoclass_4, ['Sleep stage 4'], ['4'])
  annots_strtoclass_6 = np.char.replace(annots_strtoclass_5, ['Sleep stage R'], ['5'])
  annots_strtoclass_7 = np.char.replace(annots_strtoclass_6, ['Sleep stage ?'], ['6']) #may not have to worry about this actually
  annots_strtoclass_8 = np.char.replace(annots_strtoclass_7, ['Movement time'], ['7'])
  annots_strtoclass = annots_strtoclass_8.astype(np.int64)

  ## now to store all the information into an X_train and Y_train variable
  #setting counting variables
  count = 0
  k = 0 #need k because j resets every time i resets!
  # preallocating the data and label arrays
  X_data = np.zeros(shape = (int(sigbufs[0].size/3000), 3000, 2))
  Y_data = np.zeros(int(sigbufs[0].size/3000))
  

  #storing 3000 sample/30s (because eeg sampled at 100Hz) segments of the two channels into a nested array
  #also storing the labels to each array
  for i in range(annots_norm[0].size-1):
    for j in range(int(annots_norm[1,i])):
      X_data[k,:,:] = np.transpose(sigbufs[:,count:count+3000])
      Y_data[k] = annots_strtoclass[i]
      k = k+1
      count = count +3000
  X_data_total = np.append(X_data_total, X_data,axis = 0)
  Y_data_total = np.append(Y_data_total, Y_data)
  ```
Next, we are not interested in movement times or awake periods. The following removes this information.
```
#code for removing the movement times!
remove = np.where(Y_data_total == 7)
remove = np.array(remove)
X_data_total = np.delete(X_data_total, remove, axis = 0)
Y_data_total = np.delete(Y_data_total, remove)

#code for removing the awake periods!
remove1 = np.where(Y_data_total == 0)
remove1 = np.array(remove1)
X_data_total = np.delete(X_data_total, remove1, axis = 0)
Y_data_total = np.delete(Y_data_total, remove1)
```
Organization of data into training and testing sets
```
# Setting up training and testing sets
sleep_set = X_data_total # all of the input data
y1 = to_categorical(Y_data_total) #the labels
y = y1[:,1:6] #the labels

training_data, testing_data, training_labels, testing_labels = train_test_split(sleep_set, y, test_size=0.2)

dataset_train = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
dataset_test = tf.data.Dataset.from_tensor_slices((testing_data, testing_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 10000

dataset_train = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
dataset_test = dataset_test.batch(BATCH_SIZE)
```
# Testing of multiple NN types
Following are the architecture for the models we tested, these include a SimpleNN, SimpleRNN, GRU, LSTM, and 1D CNN. Note multiple lines are commented out. Through testing we increased and decreased the number of layers for each model to find optimal performance. Instead of removing this code completely we decided to include it here in order to better show our process.

First a simple Neural Network
```
model = tf.keras.Sequential([
    
    tf.keras.layers.Input(shape=(3000,2)),                        
    tf.keras.layers.Dense(1500, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# print model layers
model.summary()
# running of model
history = model.fit(dataset_train, epochs=10,validation_data=dataset_test)
```
While this model ran, the results were not good as illustrated below in Figure 1. The code above includes two layers but 3 and 1 layer models were tested as well as shown in the Figure 1.

![Figure 1: Results of SimpleNN](https://github.com/Newber0/Automatic-Sleep-Stage-Classification-using-EEG-Data/blob/main/Image-Results/SimpleNN%20Results.PNG)

This was primarily for proof of concept, ensuring the data was correctly processed and could be input into the model. The plotting code can be found in the [Full File.py](https://github.com/Newber0/Automatic-Sleep-Stage-Classification-using-EEG-Data/blob/main/Full%20file.py)

Next a Simple RNN.
```
RNN_model = tf.keras.Sequential(
[
    #   This is for an LSTM
    # tf.keras.layers.Embedding(input_dim=3000, output_dim=64),
    # tf.keras.layers.LSTM(128, return_sequences=False, recurrent_dropout=0.1, input_shape=(None,2)),
    # tf.keras.layers.LSTM(64, dropout=0.1),
  
    #   This is for a GRU
    # tf.keras.layers.GRU(128, return_sequences=False, recurrent_dropout=0.2, input_shape=(None,2)),
    # tf.keras.layers.GRU(64, dropout=0.1),
 
    #   This is for SimpleRNN
    tf.keras.layers.SimpleRNN(256, return_sequences=False, recurrent_dropout=0.2, input_shape=(3000,2)),
    tf.keras.layers.Dense(1500, activation='relu'),
    # tf.keras.layers.Dense(750, activation='relu'),
    # tf.keras.layers.Dense(300, activation='relu'),
    # tf.keras.layers.Dense(50, activation='relu'),
    # tf.keras.layers.Dense(25, activation='relu'),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(5, activation='softmax'),
])

RNN_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.000001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# summary of the model
RNN_model.summary()
```
These three models are well suited for time series data and so were tested. However, they are more effective at prediction of future events, and this is a problem of classification, therefore the models are not effective, as illustrated in Figure 2. While better than the SimpleNN, accuracy and loss were still very poor.

![Figure 2: SimpleRNN Results](https://github.com/Newber0/Automatic-Sleep-Stage-Classification-using-EEG-Data/blob/main/Image-Results/Simple%20RNN%20Results.PNG)

Finally a 1D CNN.
```
CNN_model = tf.keras.Sequential(
  [
      tf.keras.layers.Input(shape=(3000,2)),
      # tf.keras.layers.Reshape(input_shape=(2,3000), target_shape=(2,3000,1)),
   
                              
      tf.keras.layers.Conv1D(kernel_size=10, filters=50, activation='relu', padding='same', strides=2),
      tf.keras.layers.BatchNormalization(center=True, scale=False),
      tf.keras.layers.MaxPool1D(pool_size=(2), padding='same'),
      tf.keras.layers.Dropout(0.20),

      tf.keras.layers.Conv1D(kernel_size=10, filters=100, activation='relu', padding='same', strides=2),
      tf.keras.layers.BatchNormalization(center=True, scale=False),
      tf.keras.layers.MaxPool1D(pool_size=(2), padding='same'),
      tf.keras.layers.Dropout(0.20),

      #tf.keras.layers.Conv1D(kernel_size=10, filters=200, activation='relu', padding='same', strides=2),
      #tf.keras.layers.BatchNormalization(center=True, scale=False),
      #tf.keras.layers.MaxPool1D(pool_size=(2), padding='same'),
      #tf.keras.layers.Dropout(0.20),
   
      #tf.keras.layers.Conv1D(kernel_size=10, filters=400, activation='relu', padding='same', strides=2),
      #tf.keras.layers.BatchNormalization(center=True, scale=False),
      #tf.keras.layers.MaxPool1D(pool_size=(2), padding='same'),
      #tf.keras.layers.Dropout(0.20),   
   
      tf.keras.layers.Flatten(),
      # tf.keras.layers.Dense(1500, activation='relu'),   
      #tf.keras.layers.Dense(200, activation='relu'),
      tf.keras.layers.Dropout(0.20),
      tf.keras.layers.Dense(5, activation='softmax')

  ])

CNN_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# print model layers
CNN_model.summary()
```
Again multiple layers were tested and these lines have been left in the code. The final results from the optimized model can be found in Figure 3. 

![Figure 3:1D CNN Optimized Results](https://github.com/Newber0/Automatic-Sleep-Stage-Classification-using-EEG-Data/blob/main/Image-Results/Optimized%201D%20CNN.PNG)
