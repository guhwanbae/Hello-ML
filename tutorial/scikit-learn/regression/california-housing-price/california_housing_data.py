import os
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

'''
Reference : Honds-on machine learning, Aurelien Geron, O'REILLY
'''

REPO_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master'

PROJECT_ROOT = os.path.join(Path.home(), 'workspace/machine-learning')
RESOURCE_DIR = os.path.join(PROJECT_ROOT, 'resource')
HOUSING_PATH = os.path.join(RESOURCE_DIR, 'california-housing-data')

HOUSING_URL = os.path.join(REPO_ROOT, 'datasets/housing/housing.tgz')

def fetch_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    '''
    Download california housing datasets from Aurelien Geron's repository.
    '''
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_data(housing_path=HOUSING_PATH):
    '''
    Return a pandas data frame structed by reading the csv file that contains california housing data.
    '''
    csv_path = os.path.join(housing_path, 'housing.csv')
    if not os.path.exists(csv_path):
        fetch_data()
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio=0.2):
    '''
    Return splitted data and test samples from the given pandas data frame.
    '''
    n_samples = len(data)
    adjust_median_income(data)
    # Stratified sampling makes training and whole dataset same distribution.
    split = StratifiedShuffleSplit(n_splits=1,
                                   test_size=test_ratio,
                                   random_state=42)
    # Generate indices to split data into training and test set.
    for train_indices, test_indices in split.split(data, data['income_cat']):
        strat_train_set = data.loc[train_indices]
        strat_test_set = data.loc[test_indices]
    # Remove a income_cat column after adjusting.
    for dataset in (strat_train_set, strat_test_set):
        dataset.drop('income_cat', axis=1, inplace=True)
    return strat_train_set, strat_test_set

def adjust_median_income(data):
    '''
    Adjust the median income category.
    '''
    data['income_cat'] = np.ceil(data['median_income'] / 1.5)
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)
    return data

from sklearn.base import BaseEstimator, TransformerMixin

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    '''
    Transformer that combines strongly related features.
    '''
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_idx, bedrooms_idx, population_idx, household_idx = 3, 4, 5, 6
        # Transform total rooms feature into rooms per household by combining.
        rooms_per_household = X[:, rooms_idx] / X[:, household_idx]
        population_per_household = X[:, population_idx] / X[:, household_idx]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_idx] / X[:, rooms_idx]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    Transformer that converts pandas data frame to numpy array. 
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class CategoricalOneHotEncoder:
    '''
    Transformer that outputs one-hot vectorized array.
    '''
    def __init__(self, dense_array_output=True):
        self.dense_array_output = dense_array_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        encoder = OneHotEncoder(categories='auto')
        onehot_vectorized = encoder.fit_transform(X)
        if self.dense_array_output:
            return onehot_vectorized.toarray()
        else:
            return onehot_vectorized

def split_sample_and_label(dataset):
    data = dataset.drop('median_house_value', axis=1)
    label = dataset['median_house_value'].copy()
    return data, label

def preprocess(housing):
    '''
    Return a preprocessed samples.
    '''
    housing_numerical_categories = housing.drop('ocean_proximity', axis=1)
    numerical_attributes = list(housing_numerical_categories)
    categorical_attributes = ['ocean_proximity']

    numerical_pipeline = Pipeline([
        ('selector', DataFrameSelector(numerical_attributes)),
        ('imputer', SimpleImputer(strategy='median')),
        ('attributes_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('selector', DataFrameSelector(categorical_attributes)),
        ('cat_encoder', CategoricalOneHotEncoder(dense_array_output=True)),
    ])
    
    full_pipeline = FeatureUnion(transformer_list=[
        ('numerical_pipeline', numerical_pipeline),
        ('categorical_pipeline', categorical_pipeline),
    ])
    
    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared