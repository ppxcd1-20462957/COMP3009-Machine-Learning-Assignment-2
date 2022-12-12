from joblib import load

class Regressor_Test:
    def __init__(self):
        model = load('D:/OneDrive/Academia/MSc Machine Learning in Science/Modules/COMP3009 Machine Learning/Submissions/Assignment 2/regressor.joblib') 

def main():
    Regressor_Test()

if __name__ == 'main':
    main()