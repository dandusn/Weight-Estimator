import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster.k_means_ import KMeans
import math

class regression():
    linier = LinearRegression()
    knn = neighbors.KNeighborsRegressor()
    data = []
    x = []
    y = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    kmeans = KMeans()


    def trainmodel5(self, filename):
        df = pd.read_csv(filename)
        df.loc[:, 'animal'].replace(['sheep', 'cow'], [1, 2], inplace=True)
        df = df.drop(df[df['scale'] == 0].index)
        x = df.loc[:, 'animal':'age']
        y = df.loc[:, 'weight']
        self.x = x
        self.y = y
        self.data = df
        self.spliter(x, y)

        self.linier.fit(self.x, self.y)
        n_neighbors = 3
        for i, weights in enumerate(['uniform', 'distance']):
            self.knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            self.knn.fit(self.x, self.y)
        self.kmeans = KMeans(n_clusters=5)
        self.kmeans.fit(self.x)

    def trainmodel7(self, filename):
        df = pd.read_csv(filename)
        df.loc[:, 'animal'].replace(['sheep', 'cow'], [1, 2], inplace=True)
        df = df.drop(df[df['scale'] == 0].index)
        x = df.loc[:, 'animal':'age']
        y = df.loc[:, 'weight']
        self.x = x
        self.y = y
        self.data = df
        self.spliter(x, y)

        self.linier.fit(self.x, self.y)
        n_neighbors = 3
        for i, weights in enumerate(['uniform', 'distance']):
            self.knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            self.knn.fit(self.x, self.y)

        self.kmeans = KMeans(n_clusters=7)
        well = self.data.drop(columns=["scale", "animal"])
        self.kmeans.fit(well)


    def drawcluster(self, title, labelx, labely):
        plt.title("Cluster " + title)
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.scatter(self.x.loc[:, 'age'], self.y, c=self.kmeans.labels_, cmap='rainbow')
        x = self.kmeans.cluster_centers_
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1], color='black')
        plt.show()

    def prediction(self, animal, scale, age, model):
        if animal == 'sheep':
            x = [1, scale, age]
        else:
            x = [2, scale, age]
        df = pd.DataFrame(columns=['animal', 'scale', 'age'])
        df.loc[0] = x
        y = model.predict(df)
        return y[0]

    def getcoef(self, model, y, ypred):
        # The coeficients
        print('With coefficients: \n', model.coef_)
        # Explained variance score: 1 is perfect prediction
        e = mean_absolute_error(self.y_train, ypred)
        print("mean absolut error = " + str(e))

    def spliter(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=56)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def drawregline(self, xlabel, ylabel, x, y, y1):
        plt.title("Sheep's " + xlabel + " vs " + ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(x, y, color='blue')
        plt.scatter(x, y1, color='red')
        #m, b = np.polyfit(x, y, 1)
        #plt.plot(x, y, 'yo', x, m*x+b, '--k')
        plt.show()


    def drawerror(self, x, xlabel, y, ylabel):
        plt.title("Sheep's " + xlabel + " vs " + ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(x, y, color='black')
        plt.axhline(y=10)
        plt.show()

    def correlationCoefficient(self, X, Y, n):
        sum_X = 0
        sum_Y = 0
        sum_XY = 0
        squareSum_X = 0
        squareSum_Y = 0

        i = 0
        while i < n:
            # sum of elements of array X.
            sum_X = sum_X + X[i]

            # sum of elements of array Y.
            sum_Y = sum_Y + Y[i]

            # sum of X[i] * Y[i].
            sum_XY = sum_XY + X[i] * Y[i]

            # sum of square of array elements.
            squareSum_X = squareSum_X + X[i] * X[i]
            squareSum_Y = squareSum_Y + Y[i] * Y[i]

            i = i + 1

        # use formula for calculating correlation
        # coefficient.
        corr = (float)(n * sum_XY - sum_X * sum_Y) / (float)(math.sqrt((n * squareSum_X - sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)))

        return corr

    def squarederror(self, y, ypred):
        sum = 0
        for i in range(len(y)):
            sum = sum + ((ypred[i]-y[i]) * (ypred[i]-y[i]))
        return sum

    def rsquare(self,y_or, ypred):
        ymean = 0
        for i in range(len(y_or)):
            ymean = ymean + (y_or[i])
        ymean = ymean/len(y_or)
        yml = [ymean for y in y_or]
        ser = self.squarederror(y_or, ypred)
        sem = self.squarederror(y_or, yml)
        return 1-(ser/sem)

'''
    rg = regression()
    
    rg.trainmodel7("../data/regression/regcow.csv")
    ypredlin = rg.linier.predict(rg.x)
    ypredknn = rg.knn.predict(rg.x)
    
    print("knn")
    for i in range(len(ypredknn)):
        print(ypredknn[i])
    print(rg.rsquare(rg.y,ypredknn))
    
    print("linier")
    for i in range(len(ypredlin)):
        print(ypredlin[i])
    print(rg.rsquare(rg.y, ypredlin))
'''


