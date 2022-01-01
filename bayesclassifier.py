from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import copy
import matplotlib.pyplot as plt
import seaborn as sn


def PDfBuilder(val , m, v):
    f = 0.0
    p = 0.0
    for k in range(len(m)):
        f = f + np.log(multivariate_normal.pdf(val[k], mean=m[k], cov=v[k]))
        p = p + np.log(multivariate_normal.cdf(val[k], mean=m[k], cov=v[k]))
    return f, np.exp(p)


def Naiive(x , ms, vs):
    ind = 0
    probs = np.array([0 for i in range(len(ms))], dtype ='f')
    vm, p =PDfBuilder(x , ms[0], vs[0])
    for i in range(0,len(ms)):
        v,p = PDfBuilder(x , ms[i], vs[i])

        probs[i] = p
        if v > vm:
            vm = v
            ind = i
  #  print('jus in case ', probs , ind)
 #   probs = probs/sum(probs)
    return ind, vm, probs



def Bayes(x , ms, vs):
    ind = 0
    probs = np.array([0 for i in range(len(ms))], dtype ='f')
    vm = multivariate_normal.pdf(x, mean=ms[0], cov=vs[0])
    for i in range(0,len(ms)):

        v = multivariate_normal.pdf(x, mean=ms[i], cov=vs[i])
        probs[i] = multivariate_normal.cdf(x, mean=ms[i], cov=vs[i])
        if v > vm:
            vm = v
            ind = i
  #  probs = probs/sum(probs)
    #print('jus in case ', probs , ind)
    return ind, vm, probs

def Evaluate(learning_method ,testdata, testlabel, m , v ,labs, target):
    e = 0

    c_mat = np.array([[0 for i in range(len(labs))] for j in range(len(labs))])
    c_confd = np.array([[0 for i in range(len(labs))] for j in range(len(labs))] , dtype = 'f')
    for b in range(len(testdata)):
        r = learning_method(testdata.iloc[b] , m , v)
        p = labs[r[0]]
        t = testlabel.iloc[b]
        c_mat[np.where(labs == p)[0][0] , np.where(labs == t)[0][0]] +=1
        c_confd[r[0]] =c_confd[r[0]]+  r[2]
        if p != t:
            e = e + 1

    b1 = pd.DataFrame(c_mat , index = labs)
    b1.columns = labs

    for i in range(len(c_confd)):
        c_confd[i] = c_confd[i]*100/sum(c_confd[i])

    b2 = pd.DataFrame(c_confd , index = labs)
    b2.columns = labs
    #b2 = pd.Series(c_confd, index = labs)
    ac = (len(testdata)-e)*100 / len(testdata)
    return b1,b2,ac

def PreData(path, ats):
    df = pd.read_csv(path, sep=',')
    df = pd.DataFrame(df)
    r1 = df.columns
    df.columns = ats
    df.loc[len(df.index)]= r1
    for i in range(5):
        df = df.sample(frac = 1)
    return df

def Parameters(path , attributes,labels,target, frc):

    df = PreData(path , attributes)
    train = df.iloc[0:int(frc * len(df))+1]
    test = df.drop(train.index)
    trainlabel = train[target]
    train_set = copy.deepcopy(train)
    del train_set[target]
    testlabel = test[target]
    test = test.drop([target], axis = 1)
    dataConditional = [train.loc[train[target] == x] for x in labels]
    for pf in dataConditional:
        del pf[target]
    means = [np.mean(dataConditional[i].astype(float)) for i in range(len(dataConditional))]
    covs = [np.cov(dataConditional[i].astype(float), rowvar=False) for i in range(len(dataConditional))]
    varha = [np.var(dataConditional[i].astype(float)) for i in range(len(dataConditional))]


    return [means, covs, varha , test, testlabel, train_set, trainlabel]



def K_Fold(k ,j, data , attributes,labels,target):
    division_length = int(len(data)/k)
    test = data.iloc[j* division_length:division_length * (j+1)-1]
    train = data.drop(test.index)
    testlabel = test[target]
    test = test.drop([target], axis = 1)
    dataConditional = [train.loc[train[target] == x] for x in labels]
    for pf in dataConditional:
        del pf[target]
    means = [np.mean(dataConditional[i].astype(float)) for i in range(len(dataConditional))]
    covs = [np.cov(dataConditional[i].astype(float), rowvar=False) for i in range(len(dataConditional))]
    varha = [np.var(dataConditional[i].astype(float)) for i in range(len(dataConditional))]
    return [means, covs, varha , test, testlabel]



def K_FoldCrossValidation(k, learning_method,path , attributes,labels,target, BayesM,showPlt = False):

    data = PreData(path , attributes)

    acu = 0.0
    conf = np.array([[0 for i in range(len(labels))] for j in range(len(labels))])
    b1 = pd.DataFrame(conf , index = labels)
    b1.columns = labels
    cond = np.array([[0 for i in range(len(labels))] for j in range(len(labels))])
    b2 = pd.DataFrame(cond , index = labels)
    b2.columns = labels
    for i in range(k):
        #print('iter: ',i)
        params =K_Fold(k ,i, data , attributes,labels,target)
        means = params[0]
        covs = params[1]
        varha = params[2]
        test = params[3]
        testlabel = params[4]
        #print(params)
        if BayesM == 0:
            b3 = Evaluate(learning_method , test , testlabel , means , covs, labels, target)
        else:
            b3 = Evaluate(learning_method , test , testlabel , means , varha, labels, target)
        b1 = b3[0] + b1
        b2 = b3[1] + b2
        if showPlt:
            plt.figure(figsize = (7,7))
            tit = str(i+1)+ "th iteration: Confusion Matrix with accuracy = "+ "{0:.2f}".format(b3[2])+ " percent"
            plt.title(tit)
            sn.heatmap(b3[0], annot=True ,cmap="Blues")

        acu = acu  + b3[2]
    return b1 ,b2/k, acu/k

def Analysis(path,attributes,labels,target, frc,learning_method ,whichbayes):

    params = Parameters(path , attributes,labels,target, frc)
    if whichbayes == 0:
        return (Evaluate(learning_method , params[3] , params[4] , params[0] , params[1],labels,target),
                Evaluate(learning_method , params[5] , params[6] , params[0] , params[1],labels,target))

    return (Evaluate(learning_method , params[3] , params[4] , params[0] , params[2],labels,target),
            Evaluate(learning_method , params[5] , params[6] , params[0] , params[2],labels,target))

def MatPlot(mat , tit):
    ax = plt.figure(figsize = (7,7))
    plt.title(tit)

    sn.heatmap(mat, annot=True ,cmap="Blues")
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()
    #name  = tit+'.png'
    #plt.savefig(name)

# Data
iris = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
labels = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
target = 'class'
frc = 0.7

# Bayes

ter1 , trr1 = Analysis(iris, attributes,labels,target, frc,Bayes ,0)
ter2, trr2 = Analysis(iris, attributes,labels,target, frc,Bayes ,0)

MatPlot(ter1[0] + trr1[0] + ter2[0] + trr2[0] , 'Bayes Confusion Matrix')
MatPlot((ter1[1] + trr1[1] + ter2[1] + trr2[1])/4, 'Bayes Confidence Matrix')
print('Iteration 1 on Test: Bayes: Accuracy is ',"{0:.2f}".format(ter1[2]))
print('Iteration 1 on Train: Bayes: Accuracy is ',"{0:.2f}".format(  trr1[2] ))
print('Iteration 2 on Test: Bayes: Accuracy is ',"{0:.2f}".format( ter2[2] ))
print('Iteration 2 on Train: Bayes: Accuracy is ',"{0:.2f}".format( trr2[2]))

print('Bayes: Average Accuracy is ',"{0:.2f}".format((ter1[2] + trr1[2] + ter2[2] + trr2[2])/4))

# Bayes 4_Fold

b,bx  ,ba = K_FoldCrossValidation(4, Bayes,iris , attributes,labels,target, 0)
MatPlot(b ,"4 fold Bayes Confusion Matrix" )
MatPlot(bx ,"4 fold Bayes Confidence Matrix" )
print('4 fold Bayes: Average Accuracy is ',"{0:.2f}".format(ba))



# diagonal covariance Bayes

ter1 , trr1 = Analysis(iris, attributes,labels,target, frc,Bayes ,1)
ter2, trr2 = Analysis(iris, attributes,labels,target, frc,Bayes ,1)

MatPlot(ter1[0] + trr1[0] + ter2[0] + trr2[0] , 'Bayes with diagonal covariance Confusion Matrix')
MatPlot((ter1[1] + trr1[1] + ter2[1] + trr2[1])/4, 'Bayes with diagonal covariance Confidence Matrix')
print('Iteration 1 on Test: Bayes with diagonal covariance: Accuracy is ',"{0:.2f}".format(ter1[2]))
print('Iteration 1 on Train: Bayes with diagonal covariance: Accuracy is ',"{0:.2f}".format(  trr1[2] ))
print('Iteration 2 on Test: Bayes with diagonal covariance: Accuracy is ',"{0:.2f}".format( ter2[2] ))
print('Iteration 2 on Train: Bayes with diagonal covariance: Accuracy is ',"{0:.2f}".format( trr2[2]))
print('Bayes with diagonal covariance: Average Accuracy is ',"{0:.2f}".format((ter1[2] + trr1[2] + ter2[2] + trr2[2])/4))

# 4-fold

b,bx  ,ba = K_FoldCrossValidation(4, Bayes,iris , attributes,labels,target, 1)
MatPlot(b ,"4 fold Bayes with diagonal covariance Confusion Matrix")
MatPlot(bx ,"4 fold Bayes with diagonal covariance Confidence Matrix" )
print('4 fold Bayes with diagonal covariance: Average Accuracy is ',"{0:.2f}".format(ba))

# Naiive Bayes

ter1 , trr1 = Analysis(iris, attributes,labels,target, frc,Naiive ,1)
ter2, trr2 = Analysis(iris, attributes,labels,target, frc,Naiive ,1)

MatPlot(ter1[0] + trr1[0] + ter2[0] + trr2[0] , 'Naiive Bayes Confusion Matrix')
MatPlot((ter1[1] + trr1[1] + ter2[1] + trr2[1])/4, 'Naiive Bayes Confidence Matrix')
print('Iteration 1 on Test: Naiive Bayes: Accuracy is ',"{0:.2f}".format(ter1[2]))
print('Iteration 1 on Train: Naiive Bayes: Accuracy is ',"{0:.2f}".format(  trr1[2] ))
print('Iteration 2 on Test: Naiive Bayes: Accuracy is ',"{0:.2f}".format( ter2[2] ))
print('Iteration 2 on Train: Naiive Bayes: Accuracy is ',"{0:.2f}".format( trr2[2]))
print('Naiive Bayes: Average Accuracy is ',"{0:.2f}".format((ter1[2] + trr1[2] + ter2[2] + trr2[2])/4))

# 4 4_Fold
b,bx  ,ba = K_FoldCrossValidation(4, Naiive,iris , attributes,labels,target, 1)
MatPlot(b , "4 fold Naiive Bayes Confusion Matrix")
MatPlot(bx , "4 fold Naiive Bayes Confidence")

print('4 fold Naiive Bayes: Average Accuracy is ',"{0:.2f}".format(ba))
