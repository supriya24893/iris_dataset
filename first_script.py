import numpy as np
import scipy
import matplotlib
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# print ("Hello World")
#load data set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
feature_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] #feature names
iris_dataset=pd.read_csv(url,names=feature_names)
# data is clean
#looking at data
type(iris_dataset)
# print (type(iris_dataset))
# print(iris_dataset.shape) #shape determines number of columns and rows
# print(iris_dataset)
# print(iris_dataset.head(4)) #prints first 4 rows
# print(iris_dataset.describe()) #statistical summary
iris_dataset_array=np.array(iris_dataset)
# print(iris_dataset_array)
from feature_engineering import new_matrix
random_matrix=new_matrix(iris_dataset_array)
from feature_engineering import compute_accuracy
accuracy_score=compute_accuracy(random_matrix)





