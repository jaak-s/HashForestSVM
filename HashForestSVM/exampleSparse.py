import random
import math
from learn import WeightVector
from learn import LinearPegasos

def generateX(n):
    X = [{ i: random.random() for i in random.sample(range(100), 10) }
            for k in range(n)]
    return X

def generateW():
    w = WeightVector()
    w.add( { i: random.random()-0.5 for i in random.sample(range(100), 50) }, 1.0 )
    return w

def generateW2():
    w = WeightVector()
    for i in range(100):
        if i < 50:
            w.add({i: -1}, 1.0)
        else:
            w.add({i: 1}, 1.0)
    return w

def generateY(X, w):
    Y = [math.copysign(1,w.inner(x)) for x in X]
    return Y

def readY(filename):
    y = dict()
    with open(filename) as f:
        for line in f:
            comp, activity = line.split()
            y[int(comp)] = 2*int(activity) - 1
    return y

def readX(filename):
    X = dict()
    with open(filename) as f:
        for line in f:
            comp, inch, fp = line.split()
            X[int(comp)] = {int(i): 1 for i in fp.split(",")}
    return X

def run():
    Xtrain = generateX(4000)
    #w      = generateW()
    w      = generateW2()
    Ytrain = generateY(Xtrain, w)

    lp = LinearPegasos(1.0, 100000)
    lp.fit(Xtrain, Ytrain)

    ## predict
    Xtest = generateX(1000)
    Ytest = generateY(Xtest, w)
    Yhat = lp.predict(Xtest)
    acc = 0
    for i in range(len(Ytest)):
        if Ytest[i] == Yhat[i]:
            acc += 1
    acc = acc / float(len(Ytest))
    print("Accuracy: %.3f" % acc)

def run_P08684():
    X = readX("/home/jaak/Dropbox/two/chemogen/P08684/P08684.1000.compounds.txt")
    Y = readY("/home/jaak/Dropbox/two/chemogen/P08684/P08684.1000.train")

    ids = Y.keys()
    Ytrain = [Y[i] for i in ids]
    Xtrain = [X[i] for i in ids]

    lp = LinearPegasos(1.0, 100000)
    lp.fit(Xtrain, Ytrain)

    ## predict
    Yhat = lp.decisionValues(Xtrain)
    return ids, Yhat




