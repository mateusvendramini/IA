# the goal is to use create a knn model to predict if a person has an yearly income greater than 50k
#dataset "adult" from https://github.com/itdxer/adult-dataset-analysis
#apud http://mlr.cs.umass.edu/ml/datasets/Census+Income

import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import numbers

#matplotlib inline

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
CURRENT_DIR = os.path.abspath(os.path.dirname(__name__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')

TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'adult.data')
TEST_DATA_FILE = os.path.join(DATA_DIR, 'adult.test')

from collections import OrderedDict
#extracted from 
data_types = OrderedDict([
    ("age", "int"),
    ("workclass", "category"),
    ("final_weight", "int"),  # originally it was called fnlwgt
    ("education", "category"),
    ("education_num", "int"),
    ("marital_status", "category"),
    ("occupation", "category"),
    ("relationship", "category"),
    ("race", "category"),
    ("sex", "category"),
    ("capital_gain", "float"),  # required because of NaN values
    ("capital_loss", "int"),
    ("hours_per_week", "int"),
    ("native_country", "category"),
    ("income_class", "category"),
])
target_column = "income_class"

#reading data
def read_dataset(path):
    data = pd.read_csv(
        path,
        names=data_types,
        index_col=None,
        dtype=data_types,
        comment='|',  
    )
    #data = data.drop('final_weight', axis=1)
    return data

print("Reading data")
train_data = read_dataset(TRAIN_DATA_FILE)
test_data = read_dataset(TEST_DATA_FILE)
print("Train shape: {}".format(train_data.shape))
print("Test shape: {}".format(test_data.shape))
data = pd.concat([test_data, train_data])
print(data.describe(include='all'))

#problemas: Nans 
#capital gain -> nan = 0

#income class (target) tem 4 classes precisamos de 2

#colunas de interesse age, workclass, education o que é education nun?
#remoção de colunas education, only education_num. relationship keep marital_status

#mapear variação entre 0 e 1 para evitar distorções entre colunas
#age/ max
#capital gain, loss / max
#education_num


#dropar categorias que afetam pouco workclass, final_weight

def clean_dataset(data):
    # dropa colunas
    data = data.drop('final_weight', axis=1) # drops final_weight
    data = data.drop('workclass', axis=1) # drops workclass
    data = data.drop('education', axis=1) # drops education
    data = data.drop('relationship', axis=1) #drops  relationship
    
    #substituir nan pelo item mais comum (moda) em cada categoria
    values = {'age' : 39.0, 'education_num' : 10.0, 
              'capital_gain' : 1082.0, 'capital_loss' : 88.0} #'marital_status' : "Married-civ-spouse", occupation' : 'Prof-specialty',  'race' : 'White', 'sex' : 'Male', 'native_country' : 'United-States', 'income_class' : '<=50K.'
    data = data.fillna(value=values) 

    data['age'] = data['age']/90.0
    data['education_num'] = data['education_num']/16.0
    data['capital_gain'] = data['capital_gain']/99999.0
    data['capital_loss'] = data['capital_loss']/3800.0
    data['hours_per_week'] = data['hours_per_week']/99.0
    y = data['income_class']
    data = data.drop('income_class', axis=1) 
    #max age 90
    #max ed num 16
    #max capital_gain 100000
    #max capital_loss 3800
    #max hour week 100
    data = data.dropna() # for now
    return data, y


clean_test_data, clean_test_label = clean_dataset(test_data)
validation_data, validation_label = clean_dataset(train_data)
print(clean_test_data.describe(include='all'))
cat_arr = np.array(pd.Categorical(clean_test_data['marital_status']).categories)
print(cat_arr)

def distance(arr1, arr2):
    ''' Objetos com atributos age, education_num, marital_status, '''
    mse = 0
    for i in range(len(arr1)):
        if (isinstance(arr1[i], numbers.Number) and isinstance(arr2[i], numbers.Number)):
            mse += (arr1[i] - arr2[i]) ** 2
        else:
            mse += 1 if arr1[i] != arr2[i] else 0
    return mse

neigh = KNeighborsClassifier(n_neighbors=3, metric=distance, algorithm='brute')
neigh.fit(clean_test_data, clean_test_label)
a = validation_data.iloc[[14]]
print(neigh.predict(a))
print("predicting for line", a,"expected:", validation_label.iloc[[14]])

valid = neigh.predict(validation_data.head(500))
matrix = confusion_matrix(validation_label.head(500), valid)
print(matrix)
#agora vamos criar o knn

#print(pd.Categorical(clean_test_data['marital_status']).map())