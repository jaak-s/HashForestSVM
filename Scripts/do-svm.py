import numpy as np

import pylab as pl

from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics

from HashForestSVM import data, learn

n = 2000
X, y = data.genXOR(n)
Xtest, ytest = data.genXOR(2000)

clf = svm.SVC(gamma=0.1, probability=True)
clf.fit(X, y)

yhat = clf.predict(Xtest)
print("SVM.Accuracy: %1.4f" % np.mean( yhat==ytest ))

yprob = clf.predict_proba(Xtest)
precision, recall, th = metrics.precision_recall_curve(ytest, yprob[:,1])
auc = metrics.auc(recall, precision)

print("SVM.prAUC: %1.4f" % auc)

## Pegasos:
peg = learn.KernelPegasos(gamma=0.1, lmbda=1.0, T=10000)
peg.fit(X, y)
peg_yhat = peg.predict(Xtest)
peg_yprob = peg.predictRaw(Xtest)

print("Pegasos.Accuracy: %1.4f" % np.mean( peg_yhat==ytest ))
print("Pegasos.prAUC: %1.4f" % data.prAUC(ytest, peg_yprob) )

data.prPlot(ytest, peg_yprob, yprob[:,1], method1="Pegasos", method2="SVM" )
