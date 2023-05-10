#############################  Authors  ##################################
# These notebooks are produced by 
# Abraham Bautista-Castillo (https://uh.edu/cbl/people/abraham-castillo.php) and 
# Ioannis A. Kakadiaris (https://uh.edu/cbl/people/about-director.php).

####################### License and copyright ############################

# All material belongs to the University of Houston (https://www.uh.edu/).

# All computer code is released under the University of Houston license.

# Permission is at this moment granted, free of charge, till 8/31/2023 
# to the Methodist Hospital Personnel working towards the Pilot AIM-AHEAD to 
# use the Software without limitation the rights to use, copy, modify, and 
# merge the Software.

# After 8/31/2023 all code derived from this code must be deleted and email 
# send to Prof. Kakadiaris (ioannisk@uh.edu)

# The above copyright and permission notice shall be included in all copies 
# or derivatives of the Software.
 
# Using this code should reference the publication associated with this work. Reference is available upon request to Prof. Kakadiaris

# Any distribution of this Software and associated documentation files must 
# be previously discussed with Prof. Kakadiaris and receive express written 
# authorization.

# THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR 
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, 
# ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

import os
import pandas as pd
import numpy as np
import tkinter as tk
import warnings
from functions import read_config, create_lstm, softmax
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter(action='ignore', category=FutureWarning)

env_file = 'UHCDSS-training-importance-environment.txt'

owd = os.getcwd()
os.chdir('..')
cwd = os.getcwd()

env_path =  cwd + '/Environment/' + env_file 
variables = read_config(env_path)

df_var=pd.DataFrame(variables, index=[0])

filename =  df_var['filename '].loc[0]
datafolder =  df_var['datafolder '].loc[0]
label = df_var['label '].loc[0]
test_size = df_var['test_size '].loc[0]
loss = df_var['loss_function '].loc[0]
optimizer = df_var['optimizer '].loc[0]
models_folder = df_var['models_folder '].loc[0]
path_checkpoint1 = models_folder + '/' + df_var['path_checkpoint1 '].loc[0]
path_checkpoint2 = models_folder + '/' + df_var['path_checkpoint2 '].loc[0]
att_module_scores = df_var['att_module_scores '].loc[0]

cwd = os.getcwd()
datafolder = cwd + '/'+ datafolder +'/'
models_folder = cwd + '/'+ models_folder +'/'

try:
    dataset = pd.read_csv(datafolder + filename)

except:
    dataset = pd.read_excel(datafolder + filename)

columns = list(dataset.columns)
Y = dataset[[label]].values
Y = Y.astype(float)
columns.remove(label)
X = dataset[columns].values
X = X.astype(float)
X = X[:, np.newaxis, :]

model = create_lstm(X)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

print('Model 01 created!')

model2 = create_lstm(X)
model2.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

print('Model 02 created!')

sss = StratifiedShuffleSplit(n_splits=2, random_state=1, test_size=test_size)
[K1, K2]  = sss.split(X,Y)

X_train = X[[K1[0]]]
Y_train = Y[[K1[0]]]
X_test  = X[[K1[1]]]
Y_test  = Y[[K1[1]]]

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint1,
                                      monitor='val_loss',
                                      verbose=0,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=30, 
                                        verbose=0)

callback_tensorboard = TensorBoard(log_dir='./LSTM_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.5,
                                       min_lr=1e-10,
                                       patience=10,
                                       verbose=0)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

history = model.fit(x = X_train,
                    y = Y_train,
                    epochs = 50,
                    steps_per_epoch = 4,
                    validation_data = (X_test,Y_test),
                    callbacks = callbacks, 
                    verbose=0)

print('Model Training 01: .......... DONE!')

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint2,
                                      monitor='val_loss',
                                      verbose=0,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=30, 
                                        verbose=0)

callback_tensorboard = TensorBoard(log_dir='./LSTM_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.5,
                                       min_lr=1e-10,
                                       patience=10,
                                       verbose=0)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

history = model2.fit(x = X_test,
                     y = Y_test,
                     epochs = 50,
                     steps_per_epoch = 4,
                     validation_data = (X_train,Y_train),
                     callbacks = callbacks, 
                     verbose=0)

print('Model Training 02: .......... DONE!')

weight_softmax = model.layers[2].get_weights()[0]
bias_softmax   = model.layers[2].get_weights()[1]

test_array = weight_softmax+bias_softmax
df_test = pd.DataFrame(test_array)

scaler = MinMaxScaler()
df_softmax = pd.DataFrame(softmax(df_test.mean(axis=0)))
scaled_data = scaler.fit_transform(df_softmax)
df_scaled = pd.DataFrame(scaled_data)
df_scaled.rename(columns={0: "Attention Score A"}, inplace=True)

weight_softmax2 = model2.layers[2].get_weights()[0]
bias_softmax2   = model2.layers[2].get_weights()[1]

test_array2 = weight_softmax2+bias_softmax2
df_test2 = pd.DataFrame(test_array2)

df_softmax2 = pd.DataFrame(softmax(df_test2.mean(axis=0)))
scaled_data2 = scaler.fit_transform(df_softmax2)
df_scaled2 = pd.DataFrame(scaled_data2)
df_scaled2.rename(columns={0: "Attention Score B"}, inplace=True)

val_scaled_avg = (df_scaled['Attention Score A'].values + df_scaled2['Attention Score B'].values)/2
df_scaled_avg = pd.DataFrame(val_scaled_avg)
df_scaled_avg.rename(columns={0: "Attention Score Avg"}, inplace=True)

kmeans = KMeans(n_clusters=4).fit(df_scaled_avg)
centroids = kmeans.cluster_centers_
root= tk.Tk()
canvas1 = tk.Canvas(root, width = 100, height = 100)
canvas1.pack()

label1 = tk.Label(root, text=centroids, justify = 'center')
canvas1.create_window(70, 50, window=label1)
labels = pd.DataFrame(kmeans.labels_)
df_scaled['Group'] = labels
df_scaled.insert(loc=0, column="Feature name", value=columns)
df_scaled.insert(loc=2, column='Attention Score B', value=df_scaled2['Attention Score B'].values)
df_scaled.insert(loc=3, column='Attention Score Avg', value=df_scaled_avg['Attention Score Avg'].values)
labels = pd.DataFrame(kmeans.labels_)
df_scaled['Group'] = labels
df_scaled.to_csv( datafolder + '/' + att_module_scores + '.csv', index=False)

print('Models stored in:', models_folder)
print('Attention Scores stored in:', datafolder)

os.chdir(owd)