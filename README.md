# Automatic-Sleep-Stage-Classification-using-EEG-Data
Here we use EEG data to automatically classify sleep stages using neural networks built using Keras.

Data is acquired from the Sleep-EDF Database Extended. 
This data contains 
We are only interested in the EEG data of the PSG.edf file and the label data of the Hypnogram.edf file.

# Code Explanation
```
  # Here all necessary packages are imported
  
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

