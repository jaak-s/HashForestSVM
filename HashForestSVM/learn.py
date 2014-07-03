import numpy as np
from sklearn.metrics.pairwise import pairwise_distances as pwdist

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
    