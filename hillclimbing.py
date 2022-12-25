# stochastic optimization for feature selection
from numpy import mean
from numpy.random import rand
from numpy.random import choice
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 导入FontProperties
font = FontProperties(fname="SimHei.ttf", size=14)  # 设置字体


# objective function
def objective(X, y, subset):
    # convert into column indexes
    ix = [i for i, x in enumerate(subset) if x]
    # check for now column (all False)
    if len(ix) == 0:
        return 0.0
    # select columns
    X_new = X[:, ix]
    # define model
    model = DecisionTreeClassifier()
    # evaluate model
    scores = cross_val_score(model, X_new, y, scoring='accuracy', cv=5, n_jobs=-1)
    # summarize scores
    result = mean(scores)
    return result, ix


# mutation operator
def mutate(solution, p_mutate):
    # make a copy
    child = solution.copy()
    for i in range(len(child)):
        # check for a mutation
        if rand() < p_mutate:
            # flip the inclusion
            child[i] = not child[i]
    return child

# hill climbing local search algorithm
def hillclimbing(X, y, objective, n_iter, p_mutate):
    # generate an initial point
    solution = choice([True, False], size=X.shape[1])
    # evaluate the initial point
    solution_eval, ix = objective(X, y, solution)
    iteration=[]
    accuracy=[]
    numberoffeature=[]
    # run the hill climb
    for i in range(n_iter):
        # take a step
        candidate = mutate(solution, p_mutate)
        # evaluate candidate point
        candidate_eval, ix = objective(X, y, candidate)
        # check if we should keep the new point
        if candidate_eval >= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidate_eval
        # report progress
        print('>%d f(%s) = %f' % (i + 1, len(ix), solution_eval))
        iteration.append(i+1)
        accuracy.append(solution_eval)
        numberoffeature.append(solution)
    return solution, solution_eval, iteration, accuracy, numberoffeature
def datapre():
    data = pd.read_csv('./GZSS_Feature.csv')
    # y = data.diagnosis  # M or B
    y = pd.read_csv('./GZSS_label.csv')
    y = y['1']
    # list = ['Unnamed: 32', 'id', 'diagnosis']
    # X = data.drop(list, axis=1)
    X = data
    return X, y, data
def datapre2():
    data = pd.read_csv('./Breast_Cancer_Wisconsin_DataSet.csv')
    y = data.diagnosis  # M or B
    # y = pd.read_csv('./GZSS_label.csv')
    # y = y['1']
    list = ['Unnamed: 32', 'id', 'diagnosis']
    X = data.drop(list, axis=1)
    return X, y
def plotlinechart(x,k1,k2):
    # 折线图
    # x = [5,7,11,17,19,25]#点的横坐标
    # k1 = [0.8222,0.918,0.9344,0.9262,0.9371,0.9353]#线1的纵坐标
    # k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
    plt.plot(x,k1,color ='gold')#s-:方形
    # plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
    plt.xlabel("迭代次数", fontproperties=font)#横坐标名字
    plt.ylabel("交叉验证准确度", fontproperties=font)#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.show()
if __name__=="__main__":
    # define dataset
    # X, y = make_classification(n_samples=10000, n_features=500, n_informative=10, n_redundant=490, random_state=1)
    # print(type(X),type(y))
    X, y = datapre2()
    X=np.array(X)
    y=np.array(y)
    # define the total iterations
    n_iter = 300
    # probability of including/excluding a column
    p_mut = 10.0 / 500.0
    # perform the hill climbing search
    subset, score, iteration, accuracy, numberoffeature = hillclimbing(X, y, objective, n_iter, p_mut)
    # convert into column indexes
    ix = [i for i, x in enumerate(subset) if x]
    print('Done!')
    print('Best: f(%d) = %f' % (len(ix), score))
    plotlinechart(iteration, accuracy, numberoffeature)
