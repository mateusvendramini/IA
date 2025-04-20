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
    data = data.drop('final_weight', axis=1) # drops final_weight
    data = data.drop('workclass', axis=1) # drops workclass
    data = data.drop('education', axis=1) # drops education
    data = data.drop('relationship', axis=1) #drops  relationship

    data['income_class'] = data.income_class.str.rstrip('.').astype('category')

    capital_mean = np.mean(data.capital_gain[data.capital_gain != 99999])
    data['capital_gain'] = data['capital_gain'].replace(99999, capital_mean)
    hours_per_week_mean = np.mean(data.hours_per_week[data.hours_per_week != 99])
    data['hours_per_week'] = data['hours_per_week'].replace(99, hours_per_week_mean)

    #data['workclass'] = data['workclass'].replace('?', 'Private')
    data['occupation'] = data['occupation'].replace('?', 'Prof-specialty')

    # condensa classe native_country
    data['native_country'] = data['native_country'].replace('?', 'United-States')
    data['native_country'] = data['native_country'].astype('category')
    mode = data['native_country'].cat.codes.mode()
    usa_map = lambda a : True if a == mode[0] else False

    native_usa = data['native_country'].cat.codes.map(usa_map)
    data = data.drop('native_country', axis=1)
    native_usa_df = pd.DataFrame(data={'native_usa': native_usa})
    data = pd.concat([data, native_usa_df], axis=1)

    data['marital_status'] = data['marital_status'].replace('?', 'Married-civ-spouse')
    #normaliza valores numéricos
    data['age'] = data['age']/90
    data['education_num'] = data['education_num']/16
    data['capital_gain'] = data['capital_gain']/41310.0
    data['capital_loss'] = data['capital_loss']/4356.0
    data['hours_per_week'] = data['hours_per_week']/98

    # one hot enconding 
    marital_oh = pd.get_dummies(data['marital_status'], dummy_na=False)
    data = data.drop('marital_status', axis=1)
    data = pd.concat([data, marital_oh], axis=1)

    occupation_oh = pd.get_dummies(data['occupation'])
    data = data.drop('occupation', axis=1)
    data = pd.concat([data, occupation_oh], axis=1)

    race_oh = pd.get_dummies(data['race'])
    data = data.drop('race', axis=1)
    data = pd.concat([data, race_oh], axis=1)

    sex_oh = pd.get_dummies(data['sex'])
    data = data.drop('sex',axis=1)
    data = pd.concat([data, sex_oh], axis=1)
    #drop duplicates 
    data = data.drop_duplicates()

    #saída 
    y = data['income_class']
    data = data.drop('income_class', axis=1)
    return data, y


clean_test_data, clean_test_label = clean_dataset(test_data)
validation_data, validation_label = clean_dataset(train_data)
#forces bool -> int where it applies

print(validation_data.dtypes)
print(validation_data.describe(include='all'))
print(validation_data.head(10))
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(validation_data, validation_label)
#a = validation_data.iloc[[14]]
#print(neigh.predict(a))
#print("predicting for line", a,"expected:", validation_label.iloc[[14]])

valid = neigh.predict(clean_test_data)
matrix = confusion_matrix(clean_test_label, valid)
print(matrix)
#agora vamos criar o knn

#print(pd.Categorical(clean_test_data['marital_status']).map())