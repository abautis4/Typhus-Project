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
from functions import read_config, create_lstm
from sklearn.metrics import confusion_matrix, accuracy_score

env_file = 'UHCDSS-classification-environment.txt'

owd = os.getcwd()
os.chdir('..')
cwd = os.getcwd()

env_path =  cwd + '/Environment/' + env_file 
variables = read_config(env_path)

df_var=pd.DataFrame(variables, index=[0])

filename =  df_var['filename '].loc[0]
datafolder =  df_var['datafolder '].loc[0]
label = df_var['label '].loc[0]
# test_size = df_var['test_size '].loc[0]
# loss = df_var['loss_function '].loc[0]
# optimizer = df_var['optimizer '].loc[0]
models_folder = df_var['models_folder '].loc[0]
model_file = models_folder + '/' + df_var['model '].loc[0]

datafolder = cwd + '/'+ datafolder +'/'
# models_folder = cwd + '/'+ models_folder +'/'

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
print('Model .... created !')

model.load_weights(model_file)
print('Model .... loaded !')

y_pred = model.predict(X)
y_pred[:,0] = np.round_(y_pred[:,0])
conf_mat = confusion_matrix(Y[:,0], y_pred[:,0])
print('Prediction .... DONE ! \n')
print('Confusion Matrix: ')
print(conf_mat)

print('\n Accuracy:', accuracy_score(Y[:,0], y_pred[:,0]))

os.chdir(owd)