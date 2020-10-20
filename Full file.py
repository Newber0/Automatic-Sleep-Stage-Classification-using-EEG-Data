import numpy as np
import pyedflib
import glob
import os

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tensorflow as tf

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import plot_model
from keras.utils import to_categorical
##
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#
# Keras RNN imports
from keras.layers import LSTM
from keras.layers import RNN, SimpleRNN

# for data normalization
from sklearn.preprocessing import MinMaxScaler
