from joblib import load
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler

class Classifier_Test:
    def __init__(self):
        model = load('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/classifier.joblib')
        X, id = self.preprocess(pd.read_excel('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/testDatasetExample.xls')) #Chen's data here
        self.test(model, X)

    def preprocess(self, data):
        #adjust 999s
        col_list = data.columns
        for i in range(3, len(col_list)) :
            data[col_list[i]].replace(to_replace = 999, value = data[col_list[i]].mean(), inplace = True)
            data['Age'] = data['Age'] // 10

        #split id, X
        X = data.iloc[: , 1:]
        id = data.iloc[:, 0]

        #feature selection
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X.drop(columns = to_drop, axis = 1, inplace = True)

        sel = VarianceThreshold(threshold = (0.95 * (1 - 0.95)))
        X = sel.fit_transform(X)

        #minmax normalisation
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)

        return X_norm, id

    def test(self, model, X):
        print(model.predict(X))
    
def main():
    Classifier_Test()

if __name__ == '__main__':
    main()