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


    def trainmodel(self, filename):
        df = pd.read_csv(filename)
        df.loc[:, 'animal'].replace(['sheep', 'cow'], [1, 2], inplace=True)
        df = df.drop(df[df['scale'] == 0].index)
        x = df.loc[:, 'animal':'age']
        y = df.loc[:, 'weight']
        self.x = x
        self.y = y
        self.data = df
        self.spliter(x, y)

        self.linier.fit(self.x_train, self.y_train)
        n_neighbors = 3
        for i, weights in enumerate(['uniform', 'distance']):
            self.knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            self.knn.fit(self.x_train, self.y_train)
        self.kmeans = KMeans(n_clusters=n_neighbors)
        self.kmeans.fit(self.x_train)

    def trainmodelt(self, filename, t):
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
        n_neighbors = 7
        for i, weights in enumerate(['uniform', 'distance']):
            self.knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
            self.knn.fit(self.x, self.y)

        self.kmeans = KMeans(n_clusters=n_neighbors, random_state=t, max_iter=2000)
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

rg.trainmodel("../data/regression/regsheep.csv")
ypredlin = rg.linier.predict(rg.x)
ypredknn = rg.knn.predict(rg.x)

print(r2_score(rg.y, ypredlin))
print(r2_score(rg.y, ypredknn))

'''

'''


t = 1
avg = 0
while (t<100 and avg<0.9):
    t = t+1


    rg.trainmodelt("../data/regression/regcow.csv", t)


    label = rg.kmeans.labels_
    data = rg.data.as_matrix()

    for i in range(len(label)):
        print(label[i])

    kluster0 = []
    kluster1 = []
    kluster2 = []
    kluster3 = []
    kluster4 = []
    kluster5 = []
    kluster6 = []


    for i in range(len(label)):
        if label[i] == 0:
            kluster0.append(data[i])
        elif label[i] == 1:
            kluster1.append(data[i])
        elif label[i] == 2:
            kluster2.append(data[i])
        elif label[i] == 3:
            kluster3.append(data[i])
        elif label[i] == 4:
            kluster4.append(data[i])
        elif label[i] == 5:
            kluster5.append(data[i])
        elif label[i] == 6:
            kluster6.append(data[i])

    y0 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []

    ypred0 = []
    ypred1 = []
    ypred2 = []
    ypred3 = []
    ypred4 = []
    ypred5 = []
    ypred6 = []

    for i in range(len(kluster0)):
        y0.append(kluster0[i][3])
        ypred0.append(rg.prediction(kluster0[i][0], kluster0[i][1], kluster0[i][2], rg.knn))

    for i in range(len(kluster1)):
        y1.append(kluster1[i][3])
        ypred1.append(rg.prediction(kluster1[i][0], kluster1[i][1], kluster1[i][2], rg.knn))

    for i in range(len(kluster2)):
        y2.append(kluster2[i][3])
        ypred2.append(rg.prediction(kluster2[i][0], kluster2[i][1], kluster2[i][2], rg.knn))

    for i in range(len(kluster3)):
        y3.append(kluster3[i][3])
        ypred3.append(rg.prediction(kluster3[i][0], kluster3[i][1], kluster3[i][2], rg.knn))

    for i in range(len(kluster4)):
        y4.append(kluster4[i][3])
        ypred4.append(rg.prediction(kluster4[i][0], kluster4[i][1], kluster4[i][2], rg.knn))

    for i in range(len(kluster5)):
        y5.append(kluster5[i][3])
        ypred5.append(rg.prediction(kluster5[i][0], kluster5[i][1], kluster5[i][2], rg.knn))

    for i in range(len(kluster6)):
        y6.append(kluster6[i][3])
        ypred6.append(rg.prediction(kluster6[i][0], kluster6[i][1], kluster6[i][2], rg.knn))

    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0


    if(len(y0)!=0):
        a = rg.rsquare(y0, ypred0)
    if (len(y1) != 0):
        b = rg.rsquare(y1, ypred1)
    if (len(y2) != 0):
        c = rg.rsquare(y2, ypred2)
    if (len(y3) != 0):
        d = rg.rsquare(y3, ypred3)
    if (len(y4) != 0):
        e = rg.rsquare(y4, ypred4)
    if (len(y4) != 0):
        e = rg.rsquare(y4, ypred4)
    if (len(y4) != 0):
         e = rg.rsquare(y4, ypred4)
    if (len(y5) != 0):
         f = rg.rsquare(y4, ypred4)
    if (len(y6) != 0):
         g = rg.rsquare(y4, ypred4)


    avg = (a+b+c+d+e+f+g)/7
    print(a, b, c, d, e, f, g)
    print(avg)

rg.drawcluster("Cow", "Age(Month)", "Weight(Kg)")
'''
'''
xr = rg.x_train.loc[:, 'age']
ypredlin = rg.linier.predict(rg.x_train)
xr = xr.tolist()
rg.drawregline("Age(Month)", "Weight(Kg)", xr, rg.y_train, ypred)
er = []
row = rg.y_train.tolist()
for i in range(len(row)):
    er.append(math.fabs(ypred[i] - row[i])*100/row[i])
rg.drawerror(rg.y_train, "Weight(Kg)", er, "Error(%)")

'''