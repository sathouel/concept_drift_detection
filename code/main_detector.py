import numpy as np
from sklearn.svm import SVC


# Mapping : 0 - RAS, 1 - Warning, 2 - Detection
# driftlst : list of indexes of CD occurence
thr = 5

class MainDD:

    def __init__(self, baseClassifier=SVC(), detectorsSet):
        self.mainClassifer = baseClassifier
        self.detectorsSet = detectorsSet

    def fit(X, Y, driftlst):
        X_main, y_main = self.getDataset(X, Y, driftlst)
        self.mainClassifer.fit(X_main, y_main)

    def getDataset(X, Y, driftlst):
        X_sol, y_sol = [], []

        for i in range(Y.shape[0]):
            x = X[i,:]
            y = Y[i]
            detection, vector = self.detectorsSet.predict((x, y))
            if detection and i not in driftlst:
                X_sol.append(vector)
                y_sol.append(0)
                continue
            if abs(i - driftlst) <= thr :
                X_sol.append(vector)
                y_sol.append(1)

        return np.array(X_sol), np.array(y_sol)


    def predict(x, y):
        detection, vector = self.detectorsSet.predict((x, y))
        if detection:
            return self.mainClassifer.predict(np.array(vector))

        return 0







