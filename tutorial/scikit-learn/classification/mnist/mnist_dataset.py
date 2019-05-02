from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(split=True):
    mnist_dataset = fetch_openml(name='mnist_784', version=1)
    data, labels = mnist_dataset['data'], mnist_dataset['target']
    
    labels = labels.astype('uint8')
    
    if split:
        return split_train_test(data, labels)
    else:
        return data, labels

def split_train_test(data, labels, n_test_samples=10000, shuffle=True):
    n_samples = len(data)
    n_train_samples = n_samples - n_test_samples
    
    train_data, train_labels = data[:n_train_samples], labels[:n_train_samples]
    test_data, test_labels = data[n_train_samples:], labels[n_train_samples:]
    
    if shuffle:
        shuffle_indice = np.random.permutation(n_train_samples)
        train_data = train_data[shuffle_indice]
        train_labels = train_labels[shuffle_indice]
    return train_data, train_labels, test_data, test_labels

def preprocess(train_data, test_data):
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data.astype('float64'))
    scaled_test_data = scaler.transform(test_data.astype('float64'))
    return scaled_train_data, scaled_test_data