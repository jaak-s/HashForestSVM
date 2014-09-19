import numpy as np
from sklearn.metrics.pairwise import pairwise_distances as pwdist
import math

class KernelPegasos:
    def __init__(self, gamma, lmbda, T):
        self.gamma = gamma
        self.lmbda = lmbda
        self.T = T
        
    def fit(self, X, y):
        N = X.shape[0]
        ## rbf kernel
        self.X = X
        self.y = y
        self.K = self.kernel(X)
        self.alpha = np.repeat(0.0, N)
        for t in range(1, self.T + 1):
            i = np.random.random_integers(0, N-1)
            if y[i] / self.lmbda / t * np.sum(self.alpha * y * self.K[:,i]) < 1:
                self.alpha[i] += 1
        pass
    
    def kernel(self, Xnew):
        return np.exp( - self.gamma * pwdist(Xnew, self.X)**2 )
    
    def predict(self, Xnew):
        Knew = self.kernel(Xnew)
        return 2*(Knew.dot(self.alpha * self.y) > 0 ) - 1
    
    def predictRaw(self, Xnew):
        Knew = self.kernel(Xnew)
        return Knew.dot(self.alpha * self.y)
    
def dot(x1, x2):
    d = 0.0
    for key, value in x1.iteritems():
        d += x2.get(key, 0) * value
    return d

class WeightVector:
    def __init__(self):
        self.scale = 1.0
        self.weights = dict()

    def scale_by(self, scaling_factor):
        if scaling_factor <= 0.0:
            self.scale = 1.0
            self.weights.clear()
            return

        if self.scale < 1e-8:
            for key, value in self.weights.iteritems():
                self.weights[key] = self.scale * value
            self.scale = 1.0

        self.scale *= scaling_factor

    def add(self, xi, scaler):
        for key, value in xi.iteritems():
            v = self.weights.get(key, 0)
            self.weights[key] = v + value * scaler / self.scale

    def inner(self, x):
        return dot(x, self.weights) * self.scale
    
    def __str__(self):
        return "[" + ", ".join(["%d => %.3f" % (k,v*self.scale) for k,v in self.weights.iteritems()]) + "]"


class LinearPegasos:
    def __init__(self, lmbda, T):
        self.lmbda = lmbda
        self.T = T

    def fit(self, X, y):
        self.w = WeightVector()
        N = len(X)
        ## linear kernel
        self.X = X
        self.y = y
        for t in range(1, self.T + 1):
            eta = 1.0/self.lmbda/t
            i = np.random.random_integers(0, N-1)
            if y[i] * self.w.inner(X[i]) < 1:
               self.w.scale_by(1 - 1.0/t)
               self.w.add(X[i], y[i]*eta)
            else:
               self.w.scale_by(1 - 1.0/t)
        pass
    
    def predict(self, Xnew):
        return [math.copysign(1,self.w.inner(x)) for x in Xnew]

    def decisionValues(self, Xnew):
        return [self.w.inner(x) for x in Xnew]



