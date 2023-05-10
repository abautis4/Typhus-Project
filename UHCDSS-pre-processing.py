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
import warnings
from functions import read_config, int2cat, bin_encoding
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
# pd.options.mode.chained_assignment = None
pd.set_option("mode.chained_assignment", None)

env_file = 'UHCDSS-pre-processing-environment.txt'

owd = os.getcwd()
os.chdir('..')
cwd = os.getcwd()

env_path =  cwd + '/Environment/' + env_file 
variables = read_config(env_path)

df_var=pd.DataFrame(pd.Series(variables)).T

datafolder =  df_var['datafolder '].loc[0]
datafolder = cwd + '/'+ datafolder +'/'
filename =  df_var['filename '].loc[0]
label = df_var['label '].loc[0]
name_of_processed_file = df_var['name_of_processed_file '].loc[0]

try:
    dataset = pd.read_csv(datafolder + filename)

except:
    dataset = pd.read_excel(datafolder + filename)

columns = dataset.columns.to_list()

using_thresholds = df_var['using_thresholds '].loc[0]
if using_thresholds == 'Yes':
    thresholds_file = df_var['thresholds '].loc[0]
    thresholds_path = datafolder + '/' + thresholds_file
    try:
        thresholds = pd.read_csv(datafolder + thresholds_file)
    except:
        thresholds = pd.read_excel(datafolder + thresholds_file)

    for row in range(len(thresholds)):
        if pd.isnull(thresholds.loc[row, 'Age']) is False:
            [age1, age2] = thresholds.Age[row].split('-')
            age1 = float(age1)
            age2 = float(age2)
            thresholds['Age'][row] = [age1, age2]

    thresholds = thresholds.astype(object)
    thresholds = thresholds.where(pd.notnull(thresholds), None)

    for i in range(len(thresholds)):
        dataset = int2cat(df=dataset,
                          column=thresholds['Feature'][i], 
                          low=thresholds['Low'][i], 
                          high=thresholds['High'][i], 
                          sex=thresholds['Sex'][i], 
                          age=thresholds['Age'][i])
    
    features_to_drop = list(thresholds.Feature.unique())
    dataset = dataset.drop(features_to_drop, axis=1)
    print('Thresholds applied!')
if using_thresholds == 'No':
    print('No thresholds applied!')

create_new_column = df_var['create_new_column '].loc[0]
if create_new_column == 'Yes':
    new_column_name = df_var['new_column_name '].loc[0]
    new_column_operation = df_var['new_column_operation '].loc[0]
    dataset[new_column_name] = eval(new_column_operation)
    columns.append(new_column_name)
    print('Columns created!')
if create_new_column == 'No':
    print('No columns created!')

data_imputation = df_var['data_imputation '].loc[0]
if data_imputation == 'Yes':
    nan_columns = df_var['imputation_columns ']
    if nan_columns == 'All':
        for column in dataset.columns:
            dataset[column].fillna(dataset[column].mean(), inplace=True)
    else:    
        for column in nan_columns:
            dataset[column].fillna(dataset[column].mean(), inplace=True)
    print('Data imputed!')
if data_imputation == 'No':
    print('No data imputation!')

normalize = df_var['normalize '].loc[0]
if normalize == 'Yes':
    scaler = MinMaxScaler()
    columns_to_normalize = df_var['columns_to_normalize '].loc[0]
    if columns_to_normalize == 'All':
        for column in dataset.columns:
            if len(dataset[column].unique()) == 2:
                if column == 'Sex':
                    dataset = bin_encoding(df=dataset, column='Sex', zero='Female', one='Male')
                else:
                    dataset = bin_encoding(df=dataset, column=column, zero='No', one='Yes')
            if len(dataset[column].unique()) > 2:
                dataset[[column]] = scaler.fit_transform(dataset[[column]])
    print('Data normalized!')
if normalize == 'No':
    print('No normalization!')

dataset.to_csv(datafolder +  name_of_processed_file, index=False)
print('\n Processed file was stored in:', datafolder +  name_of_processed_file)

os.chdir(owd)