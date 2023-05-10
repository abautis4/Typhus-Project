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
 
# Using this code should reference the publication associated with this work. 
# Reference is available upon request to Prof. Kakadiaris

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

import numpy as np
import tensorflow as tf

def read_config(filename):
    f = open(filename)
    config_dict = {}
    for lines in f:
        items = lines.split('= ', 1)
        config_dict[items[0]] = eval(items[1])
    return config_dict

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def create_lstm(X):
    inputs_functional = tf.keras.Input(shape=X.shape[1:], name="features")
    attention_dense   = tf.keras.layers.Dense(X.shape[2], activation='tanh', name='attention_dense')(inputs_functional)
    attention_softmax = tf.keras.layers.Dense(X.shape[2], activation='softmax', name="attention_softmax")(attention_dense)
    attention_mult    = tf.keras.layers.multiply([attention_softmax, inputs_functional], name='attention_output')
    lstm_01 = tf.keras.layers.LSTM(X.shape[2]*1, recurrent_dropout=0.3, name='LSTM')(attention_mult)
    hidden_dense_01   = tf.keras.layers.Dense(X.shape[2]*4, activation='relu', name='hidden_dense_01')(lstm_01)
    hidden_drop_01    = tf.keras.layers.Dropout(0.12, input_shape=(X.shape[2]*4,))(hidden_dense_01)
    hidden_dense_02   = tf.keras.layers.Dense(X.shape[2]*3, activation='relu', name='hidden_dense_02')(hidden_drop_01)
    hidden_drop_02    = tf.keras.layers.Dropout(0.12, input_shape=(X.shape[2]*3,))(hidden_dense_02)
    hidden_dense_03   = tf.keras.layers.Dense(X.shape[2]*2, activation='relu', name='hidden_dense_03')(hidden_drop_02)
    hidden_drop_03    = tf.keras.layers.Dropout(0.12, input_shape=(X.shape[2]*2,))(hidden_dense_03)
    hidden_dense_04   = tf.keras.layers.Dense(X.shape[2]*2, activation='relu', name='hidden_dense_04')(hidden_drop_03)
    hidden_drop_04    = tf.keras.layers.Dropout(0.12, input_shape=(X.shape[2]*2,))(hidden_dense_04)

    output_model = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(hidden_drop_04)
    model = tf.keras.Model(inputs=inputs_functional, outputs=[output_model])

    return model

def int2cat(df=None, column=None, low = None, high = None, sex = None, age = None):
 
    if low != None and column + ' - Low' not in df.columns:
        df[column + ' - Low'] = 0
    if high != None and column + ' - High' not in df.columns:
        df[column + ' - High'] = 0
    if column + ' - Normal' not in df.columns:
        df[column + ' - Normal'] = 0

    if age != None and sex != None:
        if low != None:
            df[column + ' - Low'].loc[(df['Sex']==sex)&
                                      (df['Age']>=age[0])&
                                      (df['Age']<age[1])&
                                      (df[column]<low)] = 1   
        if high != None:
            df[column + ' - High'].loc[(df['Sex']==sex)&
                                       (df['Age']>=age[0])&
                                       (df['Age']<age[1])&
                                       (df[column]>high)] = 1
            
        df[column + ' - Normal'].loc[(df['Sex']==sex)&
                                     (df['Age']>=age[0])&
                                     (df['Age']<age[1])&
                                     (df[column]>=low)&
                                     (df[column]<=high)] = 1
        
    if age == None and sex == None:
        if low != None:
            df[column + ' - Low'].loc[(df[column]<low)] = 1
            
        if high != None:
            df[column + ' - High'].loc[(df[column]>high)] = 1
        
        if low == None:
            df[column + ' - Normal'].loc[(df[column]<=high)] = 1
        
        if high == None:
            df[column + ' - Normal'].loc[(df[column]>=low)] = 1
        
        df[column + ' - Normal'].loc[(df[column]>=low)&
                                     (df[column]<=high)] = 1
                
    return df

def bin_encoding(df=None, column=None, zero=None, one=None):
    
    df[column] = df[column].replace(zero,0)
    df[column] = df[column].replace(one,1)
    
    return df