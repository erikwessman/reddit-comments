#tensorflow uses python 3.5 - 3.8, wont work otherwise
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import save_model
from data_processing import get_x_data, get_y_data, get_vocab_len, get_seq_len

#only add the below code if you are using TF with a GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Trains a LSTM model with the data processed by
# 'data_processing.py'. Then saves that model
# in the 'saved_model' folder

x_data = get_x_data()
y_data = get_y_data()
vocab_len = get_vocab_len()
seq_length = get_seq_len()

n_patterns = len(x_data)
X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)
y = np_utils.to_categorical(y_data)

#create model and model layers
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = 'src/saved_model/model_weights_saved.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

model.fit(X, y, epochs=100, batch_size=32, callbacks=desired_callbacks)

filename = 'src/saved_model/model_weights_saved.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

save_model(model, 'src/saved_model')

print('Saved model to disk')