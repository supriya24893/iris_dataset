import numpy as np
def new_matrix(original_array):
    n=original_array.shape[1]
    r=original_array.shape[0]
    new_column=np.zeros_like(original_array[:0])
    label_column=original_array[:,n-1]
    temp=original_array[:,0:n-1] #goes till n-1th column
    new_temp=np.append(original_array,new_column)
    final_matrix=np.append(new_temp,label_column)
    for x in range(0,r):
        print(final_matrix)
        if final_matrix[x,n-1]=='Iris-virginica':
            final_matrix[x,n-2]=np.random.uniform(1,3)
        if final_matrix[x,n-1]=='Iris-setosa':
            final_matrix[x,n-2]=np.random.uniform(0.8,2.5)
        if final_matrix[x,n-1]=='Iris-versicolor':
            final_matrix[x,n-2]=np.random.uniform(0.1,0.3)
    return final_matrix

def compute_accuracy(data_matrix):
    import sklearn
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # splitting the data
    n=data_matrix.shape[1]

    y = data_matrix[:, n]

    x = data_matrix[:, 0:n]

    # split data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

    lg = LogisticRegression()  # instance of the class logistic regression
    lg.fit(x_train, y_train)  # fits x and y data on the model
    y_pred = lg.predict(x_test)  # predicts y values for x
    result = accuracy_score(y_test, y_pred)
    print(result)
    return result