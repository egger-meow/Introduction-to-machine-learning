# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv



class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        self.gradientLossMark = []
        
    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):
        # Compute closed-form solution.

        nSamples = len(X)
        
        x = np.c_[np.ones((nSamples, 1)), X]

        weights = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        self.closed_form_weights = weights[1:]
        self.closed_form_intercept = weights[0]
        # print(self.closed_form_weights)
        # print(self.closed_form_intercept)
        # self.closed_form_intercept = intercept

    # This function computes the gradient descent solution of linear regression.
    def gradient_descent_fit(self, X, y, lr, epochs, recorDot = False,dot = 20):
        # Compute the solution by gradient descent.        
        self.gradientLossMark = []
        
        nSamples = len(X)
        weights = np.random.rand(len(X[0])+1)
        
        x = np.c_[np.ones((nSamples, 1)), X]  
        # print (x[0])
        mark = int(epochs/dot)
        mark = 1 if mark == 0 else mark
        

        
        switch1, switch2 = True, True
        for epoch in range(epochs):
            
            for i in range(nSamples):
                randI = np.random.randint(nSamples)
                xi = x[randI:randI+1]
                yi = y[randI:randI+1]
                
                gradient = -2 * xi.T.dot(xi.dot(weights) - yi)
   
                
                weights += lr * gradient 
     
                # print(xi, yi) 
            


            
            if (recorDot and epoch % mark == 0):
                self.gradient_descent_weights = weights[1:].squeeze()
                self.gradient_descent_intercept = weights[0].squeeze()
                self.gradientLossMark.append((epoch,self.gradient_descent_evaluate(X, y)))

            
        # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept
        self.gradient_descent_weights = weights[1:].squeeze()
        self.gradient_descent_intercept = weights[0].squeeze()
        
        # print (weights)
        
        
    
    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):
      

        arr = (prediction - ground_truth) ** 2
        result = float(np.sum(arr)) / float(len (prediction))

        # Return the value.
        return result

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):
        y = X.dot(self.closed_form_weights) + self.closed_form_intercept
        y = np.round(y,0)
        # Return the prediction.
        return y


    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):
        y = X.dot(self.gradient_descent_weights) + self.gradient_descent_intercept
        # print(np.round(y,0))
        return np.round(y,0)
      
    
    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and return the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and return the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        # This function is finished for you.
        # print(self.gradient_descent_predict(X)[:10],y[:10])
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_learning_curve(self):
        x, y = zip(*self.gradientLossMark)
        # Add labels and a title
        # plt.scatter(x, y, c="b", s=4, label="learning curve")
        plt.plot(x, y, marker='o', linestyle='-', color = "red")

        plt.xlabel('epochs')
        plt.ylabel('MSELoss')
        plt.title('gradient desecnt learning curve')

        
        plt.grid()  
        plt.show()
        

# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    
    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    
    np.random.seed(47)

  
    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)


    LR.gradient_descent_fit(train_x, train_y, lr=0.00002, epochs=600, recorDot=True)
    # LR.gradient_descent_fit(train_x, train_y, lr=0.00001, epochs=1000, recorDot=True)
    
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")
    # exit()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    # print (closed_form_loss,gradient_descent_loss)
    # print(f"gradient_descent_loss: {gradient_descent_loss}",f"closed_form_loss: {closed_form_loss}")
    
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")
    
    LR.plot_learning_curve()