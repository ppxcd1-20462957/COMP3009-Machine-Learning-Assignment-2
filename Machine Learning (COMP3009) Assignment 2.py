import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from keras import layers, models, optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential, load_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text


class Preprocess:
    def __init__(self, dataset):
        clean_data = self.data_cleaning_remove(dataset) #alternate between data_cleaning functions
        X, Y_PCR, Y_RFS = self.data_split(clean_data)
        X = self.feature_selection(X) #alternate between feature_selection and _dim_reduction functions
        Y_PCR = Y_PCR.iloc[:, 0].values
        Y_RFS = Y_RFS.iloc[:, 0].values
        X_norm = self.normalisation_zmean(X) #alternate between normalisation functions
        self.X_train_PCR, self.X_test_PCR, self.Y_train_PCR, self.Y_test_PCR = train_test_split(X_norm, Y_PCR, test_size = 0.2, random_state = 69)
        self.X_train_RFS, self.X_test_RFS, self.Y_train_RFS, self.Y_test_RFS = train_test_split(X_norm, Y_RFS, test_size = 0.2, random_state = 69)

    def data_cleaning_remove(self, data):
        data['pCR (outcome)'] = data['pCR (outcome)'].replace(999, np.nan)
        data = data.dropna()

        return data

    def data_cleaning_replace_mean(self, data):
        col_list = data.columns
        for i in range(3, len(col_list)):
            data[col_list[i]].replace(to_replace = 999, value = data[col_list[i]].median(), inplace = True)
            data['Age'] = round(data['Age'] / 20, 0)

        return data

    def data_cleaning_replace_median(self, data):
        col_list = data.columns
        for i in range(3,len(col_list)) :
            data[col_list[i]].replace(to_replace = 999, value = data[col_list[i]].mean(), inplace = True)
            data['Age'] = data['Age'] // 10

        return data

    def data_split(self, data):
        X = data.copy(deep = True)
        X.drop(columns = data.columns[:3], axis = 1, inplace = True)

        Y_PCR = data[['pCR (outcome)']].copy(deep = True)
        Y_RFS = data[['RelapseFreeSurvival (outcome)']].copy(deep = True)

        return X, Y_PCR, Y_RFS

    def feature_selection(self, X):
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X.drop(columns = to_drop, axis = 1,inplace = True)

        sel = VarianceThreshold(threshold = (0.95 * (1 - 0.95)))
        X = sel.fit_transform(X)
        
        return X

    def dim_reduction(self, X):
        pca = PCA(n_components = 33)
        X_PCA = pca.fit_transform(X)

        return X_PCA

    def normalisation_zmean(self, X):
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        return X_norm

    def normalisation_minmax(self, X):
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)

        return X_norm

#classification

class Classification_NeuralNetwork:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        folds = 5
        activation = 'relu'
        kf = StratifiedKFold(folds, shuffle = True, random_state = 42) 
        fold = 0
        accuracy_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = models.Sequential([
                                       layers.Dense(1, input_shape = (47, ), activation = activation),
                                       layers.Dropout(0.1),
                                       layers.Dense(1, activation = activation),
                                       layers.Dropout(0.1),
                                       layers.Dense(1, activation = activation),
                                       layers.Dropout(0.1),
                                       layers.Dense(1, activation = activation),
                                       layers.Dense(1, activation = 'sigmoid')
            ])

            model.compile(
                          loss = 'binary_crossentropy', 
                          optimizer = 'adam', 
                          metrics = ['accuracy']
            )

            model.fit(inputs[train], targets[train], epochs = 2000)
            scores = model.evaluate(inputs[test], targets[test], verbose = 0)
            accuracy_per_fold.append(scores[1] * 100)

        return '{}-Fold Cross-Validation Accuracy: {}%'.format(folds, np.round(sum(accuracy_per_fold) / folds), 2)

class Classification_XGBoost:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        folds = 5
        kf = StratifiedKFold(folds, shuffle = True, random_state = 42) 
        fold = 0
        accuracy_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = xgb.XGBClassifier(objective = 'binary:logistic', random_state = 42)
            model.fit(inputs[train], targets[train])
            Y_pred = model.predict(inputs[test])
            accuracy = accuracy_score(targets[test], Y_pred)
            accuracy_per_fold.append(accuracy * 100)

        return '{}-Fold Cross-Validation Accuracy: {}%'.format(folds, np.round(sum(accuracy_per_fold) / folds), 2)

class Classification_SVM:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        folds = 5
        kf = StratifiedKFold(folds, shuffle = True, random_state = 42) 
        fold = 0
        accuracy_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = SVC(
                kernel = 'poly',
                C = 1,
                degree = 5,
                coef0 = 0.001,
                gamma = 'auto'
            )

            model.fit(inputs[train], targets[train])
            Y_pred = model.predict(inputs[test])
            accuracy = accuracy_score(targets[test], Y_pred)
            accuracy_per_fold.append(accuracy * 100)

        return '{}-Fold Cross-Validation Accuracy: {}%'.format(folds, np.round(sum(accuracy_per_fold) / folds), 2)

class Classification_LogisticRegression:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        folds = 5
        kf = StratifiedKFold(folds, shuffle = True, random_state = 42) 
        fold = 0
        accuracy_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = LogisticRegression(
                penalty = 'l2',
                random_state = 42,
                solver = 'saga',
                max_iter = 100
            )

            model.fit(inputs[train], targets[train])
            Y_pred = model.predict(inputs[test])
            accuracy = accuracy_score(targets[test], Y_pred)
            accuracy_per_fold.append(accuracy * 100)

        return '{}-Fold Cross-Validation Accuracy: {}%'.format(folds, np.round(sum(accuracy_per_fold) / folds), 2)

class Classification_KNN:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        folds = 5
        kf = StratifiedKFold(folds, shuffle = True, random_state = 42) 
        fold = 0
        accuracy_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = KNeighborsClassifier(n_neighbors = 5)
                
            model.fit(inputs[train], targets[train])
            Y_pred = model.predict(inputs[test])
            accuracy = accuracy_score(targets[test], Y_pred)
            accuracy_per_fold.append(accuracy * 100)

        return '{}-Fold Cross-Validation Accuracy: {}%'.format(folds, np.round(sum(accuracy_per_fold) / folds), 2)

class Classification_DecisionTree:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        folds = 5
        kf = StratifiedKFold(folds, shuffle = True, random_state = 42) 
        fold = 0
        accuracy_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = DecisionTreeClassifier(max_depth = 2)
                
            model.fit(inputs[train], targets[train])
            Y_pred = model.predict(inputs[test])
            accuracy = accuracy_score(targets[test], Y_pred)
            accuracy_per_fold.append(accuracy * 100)

        return '{}-Fold Cross-Validation Accuracy: {}%'.format(folds, np.round(sum(accuracy_per_fold) / folds), 2)

class Classification_RandomForest:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.accuracy = self.train(X_train, Y_train, X_test, Y_test)

    def train(self, X_train, Y_train, X_test, Y_test):
        inputs = np.concatenate((X_train, X_test), axis = 0)
        targets = np.concatenate((Y_train, Y_test), axis = 0)
        
        folds = 5
        kf = StratifiedKFold(folds, shuffle = True, random_state = 42) 
        fold = 0
        accuracy_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = RandomForestClassifier(
                        n_estimators = 100, 
                        criterion = 'entropy', 
                        max_depth = 6, bootstrap = True, 
                        random_state = 0, 
                        max_samples = 0.85, 
                        min_samples_leaf = 3)
                
            model.fit(inputs[train], targets[train])
            Y_pred = model.predict(inputs[test])
            accuracy = accuracy_score(targets[test], Y_pred)
            accuracy_per_fold.append(accuracy * 100)

        return '{}-Fold Cross-Validation Accuracy: {}%'.format(folds, np.round(sum(accuracy_per_fold) / folds), 2)

class Run_Classifications:
    def __init__(self):
        dataset = pd.read_excel('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/trainDataset.xls')
        data = Preprocess(dataset)
        ann = Classification_NeuralNetwork(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        xgb = Classification_XGBoost(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        svm = Classification_SVM(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        lr = Classification_LogisticRegression(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        knn = Classification_KNN(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        dt = Classification_DecisionTree(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)
        rf = Classification_RandomForest(data.X_train_PCR, data.X_test_PCR, data.Y_train_PCR, data.Y_test_PCR)

        print('\nClassification: Neural Network ', ann.accuracy)
        print('Classification: XGBoost ', xgb.accuracy)
        print('Classification: SVC ', svm.accuracy)
        print('Classification: Logistic Regression ', lr.accuracy)
        print('Classification: KNN ', knn.accuracy)
        print('Classification: Decision Tree ', dt.accuracy)
        print('Classification: Random Forest ', rf.accuracy, '\n')

#regression

class Regression:
    pass

def main():
    Run_Classifications()

if __name__ == '__main__':
    main()