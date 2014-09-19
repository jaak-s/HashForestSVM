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


