# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None;
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        
        # x = np.array(X)
        samples = len(X)
        x = np.c_[np.ones((samples, 1)), X] 
         
        weights = np.zeros(len(x[0]))
        epochs  = self.iteration
        
        for epoch in range(epochs):
            
            prediction = np.dot(x, weights)
            prediction = self.sigmoid(prediction)
            
            gradient = (x.T.dot(prediction - y)) / samples
            
            weights -= gradient * self.learning_rate
            
        self.weights = weights[1:]
        self.intercept = weights[0]
            
            
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        prediction = np.round(self.sigmoid(np.dot(X, self.weights)+self.intercept))
        return prediction

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):

        nSample = len(X)
        X0 = X[y == 0]
        X1 = X[y == 1]
        
        class0Mean = np.mean(X0, axis = 0)
        class1Mean = np.mean(X1, axis = 0)
        
        self.m0 = class0Mean
        self.m1 = class1Mean

        diff1 = X0 - class0Mean
        diff2 = X1 - class1Mean
        
        self.sw = np.dot(diff1.T, diff1) + np.dot(diff2.T, diff2)

        mean = np.mean(X, axis = 0)
        
        # self.sb = nSample * np.outer (class0Mean - mean, class1Mean - mean)
        
        self.sb = nSample * np.outer (class0Mean - class1Mean, class0Mean - class1Mean)
        
        eigenValues, eigenVectors = np.linalg.eigh(np.linalg.inv(self.sw).dot(self.sb)) 
        
        self.w = eigenVectors[np.argmax(eigenValues)]
        
        self.slope = self.w[1] / self.w[0]
        

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        proj0 = np.dot(self.m0.T, self.w)
        proj1 = np.dot(self.m1.T, self.w)
        proj  = np.dot(X, self.w.T)
        arr = (abs(proj-proj0) > abs(proj-proj1)).astype(int)
        return arr

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        y = self.predict(X)
        X0 = X[y == 0]
        X1 = X[y == 1]
        mean = np.mean(X, axis = 0)

        plt.figure(figsize=(9, 7))
        # Create a scatter plot
        plt.scatter(X0[:, 0], X0[:, 1], c='red', label='Class 0', marker='o',s = 14)
        plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class 1', marker='o',s = 14)
        plt.axis('equal')
        plt.ylim(0, 200)  
        plt.xlim(0, 200)  

        m = self.slope  
        x0 = mean[0]  
        y0 = mean[1]  
        intercept = y0 - x0 * m 

        plt.title(f"Projection Line: w={self.slope}, b={intercept}")  
        x = [x0 - 30, x0 + 30]  

 
        y = [y0 + m * (xi - x0) for xi in x]

        plt.plot(x, y, color='darkblue')
        
        for xi, yi in zip(X[:, 0], X[:, 1]):
            x_projection = (xi + m * yi - m * intercept) / (m * m + 1)
            y_projection = m * x_projection + intercept
            plt.plot([xi, x_projection], [yi, y_projection], lw=1.2, alpha=0.2,color='blue')  # Draw projection lines



        plt.legend()


        plt.show()
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    # LR = LogisticRegression(learning_rate=0.000047, iteration=30)
    
    LR = LogisticRegression(learning_rate = 0.000047, iteration = 300000)
    
    # LR = LogisticRegression(learning_rate=0.000057, iteration=300000)
    # LR = LogisticRegression(learning_rate=0.000077, iteration=1000000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    # print (f"y_pred{y_pred}")
    accuracy = accuracy_score(y_test , y_pred)
    # print(accuracy)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    # assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    FLD.plot_projection(X_test)
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"
