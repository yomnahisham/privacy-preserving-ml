import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests
from io import StringIO

'''
    • Replace load_data with your own dataset loading logic if you have a custom dataset.
    • Adjust the epsilon parameter in the main function to vary the privacy level.
    • Run the script to see accuracy results on your chosen dataset.
'''

class LaplaceMechanism:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def laplace_mechanism(self, data, sensitivity):
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(loc=0.0, scale=scale, size=data.shape)
        return data + noise

def load_data(dataset_name):
    if dataset_name == 'iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
    elif dataset_name == 'wine':
        wine = load_wine()
        X = wine.data
        y = wine.target
    elif dataset_name == 'adult':
        #load the Adult Census Income dataset (example)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                        'hours_per_week', 'native_country', 'income']
        response = requests.get(url, verify=False)
        data_adult = pd.read_csv(StringIO(response.text), header=None, names=column_names, na_values=' ?')
        data_adult.dropna(inplace=True)

        #encode categorical variables
        label_encoders = {}
        for column in data_adult.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data_adult[column] = le.fit_transform(data_adult[column])
            label_encoders[column] = le

        X = data_adult.drop('income', axis=1)
        y = data_adult['income']
    elif dataset_name == 'custom':
        #replace with your custom dataset loading logic
        #ex.
        # data_custom = pd.read_csv('path_to_your_custom_dataset.csv')
        # X = data_custom.drop('target_column', axis=1)
        # y = data_custom['target_column']
        pass
    else:
        raise ValueError("Invalid dataset name. Please use 'iris', 'wine', 'adult', or provide your custom dataset.")
    
    return X, y

def main(dataset_name, epsilon=1.0):
    #load data
    X, y = load_data(dataset_name)
    
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #train KNN classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    
    #predict on test set
    y_pred = knn.predict(X_test)
    
    #evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on {dataset_name} dataset: {accuracy:.2f}')
    
    #apply Laplace mechanism
    laplace = LaplaceMechanism(epsilon=epsilon)
    X_test_private = laplace.laplace_mechanism(X_test, sensitivity=1.0)  # Adjust sensitivity based on dataset
    
    #re-evaluate accuracy after applying privacy mechanism
    y_pred_private = knn.predict(X_test_private)
    accuracy_private = accuracy_score(y_test, y_pred_private)
    print(f'Accuracy on {dataset_name} dataset with privacy: {accuracy_private:.2f}')

if __name__ == '__main__':
    dataset_name = 'adult'  #replace with 'iris', 'wine', 'adult', or your custom dataset name
    epsilon = 1.0  #adjust privacy parameter epsilon as needed
    main(dataset_name, epsilon)