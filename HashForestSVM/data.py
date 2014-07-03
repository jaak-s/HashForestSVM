import numpy as np
import matplotlib.pyplot as pp

from sklearn import metrics

import pylab as pl

def genXOR(n):
    n = n // 4
    means = np.array( [[2, -2], [-2, 2], [2, 2], [-2, -2]] )
    cov   = np.array( [[1, 0],[0, 1]] )
    X = np.vstack( [ np.random.multivariate_normal(m, cov, size=n) for m in means ] )
    y = np.concatenate( [np.repeat(-1, n*2), np.repeat(1, n*2)] )
    return (X, y)
        
def plotXOR(X, y):
    pp.plot(X[y==1,0], X[y==1,1], 'ro')
    pp.plot(X[y!=1,0], X[y!=1,1], 'bs')
    pp.grid()
    pp.axis("Equal")
    pp.show()

def prAUC(ytest, yprob):
    precision, recall, th = metrics.precision_recall_curve(ytest, yprob)
    return metrics.auc(recall, precision)

def prPlot(ytest, yprob, yprob2=None, method1="", method2=""):
    pl.clf()
    auc = prAUC(ytest, yprob)
    precision, recall, th = metrics.precision_recall_curve(ytest, yprob)
    pl.plot(recall, precision, label='Precision-Recall: %s (%0.3f)' % (method1, auc) )
    if yprob2 is not None:
        pr2, rc2, t2 = metrics.precision_recall_curve(ytest, yprob2)
        pl.plot(rc2, pr2, 'r', label='Precision-Recall: %s (%0.3f)' % (method2, prAUC(ytest, yprob2)) )
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall curve' % auc)
    pl.legend(loc="lower left")
    pl.show()
