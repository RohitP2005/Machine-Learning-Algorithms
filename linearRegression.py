import numpy as np

#Linear Regression
class LinearRegression():
    def __init__(self,learnRate,interations):
        self.learnRate = learnRate
        self.iterations = interations
    
    # Function for model trinning
    def fitModel(self,X,Y):
        # m = No. of trainning rows
        # n = No. of features
        self.m,self.n = X.shape

        # Weight initialisation
        # Y = W.X + b 
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        self.gradientDescent()
        
    
    # Fucntion for gradient descent
    def gradientDescent(self):
        for i in range (self.iterations):
            y_predicted = self.predict(self.X)
            #gradients calculations
            dW = - (2*(self.X.T).dot(self.Y-y_predicted))/self.m
            db = -2*np.sum(self.Y - y_predicted)/self.m

            # update weights
            self.W = self.W - self.learnRate*dW
            self.b = self.b - self.learnRate*db
            return self

    # Fucntion to predict solution
    def predict(self,X):
        return X.dot(self.W) + self.b