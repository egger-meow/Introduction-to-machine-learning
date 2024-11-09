# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    l = len(y)
    count = np.bincount(y)
    p = count / l
    return 1 - np.sum(p ** 2)

# This function computes the entropy of a label array.
def entropy(y):
    
    l = len(y)

    count = np.bincount(y)
    p = count / l
    p = p[p > 0]
    return - np.sum(p * np.log(p))
        
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
ds = []
class Node():
    def __init__(self):

        self.xi = None
        
        self.distcrete = True
        self.dilimeter = 0
        
        self.childs = []
               
        self.leaf = None
    def addChild(self, c):
        self.childs.append(c)
        
    def setDelimeters(self, a):
        self.dilimeter = a

        
    def setXi(self, xi): 
        self.xi = int(xi)
            
         
    def setLeaf(self,cla):
        self.leaf = cla
        
    def findClass(self, X, dis = True):
        if self.leaf != None:
            return self.leaf
        # print(self.dilimeter)
        
        featureVal = X[self.xi]
        
        if ds[self.xi][0] and dis:
            # print(ds[self.xi][1], featureVal, self.childs)
            node = self.childs[ds[self.xi][1].index(featureVal)]
            return node.findClass(X)
        else:
            node1 = self.childs[0]
            node2 = self.childs[1]
            return node1.findClass(X,dis) if featureVal <= self.dilimeter else node2.findClass(X,dis)
        
    
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None, noDiscrete = False):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.root = None 
        self.theta = 0.07
        self.discreteArr = []
        self.DisAvailible = not noDiscrete
        
        self.recorder = []
    
    def generateTree(self, n:Node, depth, X:np.array):
        if (len(X) == 0):
            n.setLeaf(1)
            return n
        x = X[:,:-1]
        y = X[:,-1]
        if depth == self.max_depth or self.impurity(y) < self.theta:
            n.setLeaf(np.argmax(np.bincount(y)))
            return n
        
        i, discrete, cut = self.splitAttribute(X)
        
        branchesI = X[:,i]
        # print(branchesI[:10])
        n.setXi(i)
        self.recorder[i] += 1
        
        if discrete and self.DisAvailible:
            branches = np.unique(branchesI)
            for j in ds[i][1]:
                n.addChild(self.generateTree(Node(),depth + 1,X[branchesI == j]))
        else:
            n.setDelimeters([cut])
            n.addChild(self.generateTree(Node(),depth + 1,X[X[:,i] <= cut]))
            n.addChild(self.generateTree(Node(),depth + 1,X[X[:,i] > cut]))
        return n
        
    def splitAttribute(self, X):
        x = X[:,:-1]
        y = X[:,-1]
        attributes = len(x[0])
        minImp = 999999999 
        besti = 0
        discrete = True
        cut = -1
        
        for i in range (attributes):
            branches, counts = np.unique(X[:,i], return_counts=True)
            
            if not ds[i][0] or not self.DisAvailible:
    
                for j in branches:
                    X1 = X[X[:,i] <= j]
                    X2 = X[X[:,i] > j]
                    totalImpurity = (len(X1) / attributes) * self.impurity(X1[:,-1]) + (len(X2) / attributes) * self.impurity(X2[:,-1])
                    
                    if totalImpurity < minImp:
                        minImp, besti = totalImpurity, i
                        discrete = False
                        cut = j
                        
            else:
                totalImpurity = 0
                for j in range(len(branches)):
                    XT = X[X[:,i] == branches[j]]
                    totalImpurity += (counts[j]/attributes) * self.impurity(XT[:,-1])
                if totalImpurity < minImp:
                    minImp, besti = totalImpurity, i
                    discrete = True
           
        return besti, discrete, cut
        
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        x = np.c_[X,y]
        for i in range(len(X[0])):
            u = list(np.unique(X[:,i]))
            if len (u) < 6:
                ds.append([True,u])
            else:
                ds.append([False])
            self.recorder = [0]*len(X[0])
        self.root = self.generateTree(Node(), 0, x)
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        
        return [self.root.findClass(i,self.DisAvailible) for i in X]
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        categories = ['age', 'sex', 'cp', 'fbs', 'thalach', 'thal']
        values = self.recorder
        plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        plt.barh(categories, values,  height=0.7)


        plt.title('Feature importance')

        # Show plot
        # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
        # plt.tight_layout()  # Adjust layout to prevent cutoff of labels
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.alphas = []
        self.learners = []
      
    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        import numpy as np
        # y = np.array(y)
        
        nSample = len(X)
        yp = np.array(y.copy())
        yp[yp == 0] = -1
        # print(yp)
        weights = np.ones(nSample) / nSample

        
        for _ in range(self.n_estimators):
            
            weaker = DecisionTree(self.criterion,1)
            
            x = np.c_[X,y]
            
            temp = np.arange(nSample)
            
            # print(weights)
            t = np.random.choice(temp, size = nSample*15, p=weights)
            weightedX = np.array([x[i] for i in t])
            # print(weightedX[:3])
            
            weaker.fit(weightedX[:,:-1], weightedX[:,-1])
            # weaker.fit(weightedX)
            # weaker.fit(X, y)
            
            pred = np.array(weaker.predict(X))
            pred[pred == 0] = -1
            # print(accuracy_score(yp, pred))
            
            # print(pred[:10])
            
            err = np.sum(weights *(pred != yp))
            
            alpha = 0.5 * np.log((1-err) / (err + 0.0000000001))
            self.alphas.append(alpha)
            
            weights *= np.exp(-alpha * yp * pred) 
            weights /= np.sum(weights)
            
            self.learners.append(weaker)
            
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        nSample = len(X)
        pred = np.zeros(nSample)
        for alpha, weakLearner in zip(self.alphas, self.learners):
            p = np.array(weakLearner.predict(X))
            p[p == 0] = -1
            
            pred += alpha * p.astype(np.float64)

        # print(pred)
        pred[pred >= 0] = 1
        pred[pred < 0] = 0
        # pred[pred > 0.5] = 1
        # pred[pred <= 0.5] = 0
        return pred

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='gini', max_depth=15)
    tree.fit(X_train, y_train)
    tree.plot_feature_importance_img(None)

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=200)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))