from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE,RFECV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from pandas.core.frame import DataFrame
import pandas as pd
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.font_manager import FontProperties  # 导入FontProperties
import seaborn as sns
font = FontProperties(fname="SimHei.ttf", size=14)  # 设置字体
import time

def read2list(path):
    import csv
    with open(path, 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    return rows
def datapre_Breast(): #威斯康星州乳腺癌（诊断）数据集
    X = read2list('./Breast_Cancer_Wisconsin_DataSet.csv')
    header = X[0]
    del X[0]
    y = [a[1] for a in X]
    # print(y)
    for i in range(len(X)):
        del X[i][0]
        del X[i][0]
    # print(X)
    # X = np.array(X)
    X = DataFrame(X)
    for i in range(len(y)):
        if y[i]=='M':
            y[i]=0
        else:
            y[i]=1
    # y = np.array(y)
    y = DataFrame(y)
    header.pop(0)
    header.pop(0)
    header.pop()
    header=DataFrame(header)
    return X,y,header
def importance_rank():
    data = pd.read_csv('./Breast_Cancer_Wisconsin_DataSet.csv')
    y = data.diagnosis  # M or B
    list = ['Unnamed: 32', 'id', 'diagnosis']
    X = data.drop(list, axis=1)
    clf_rf_5 = RandomForestClassifier()
    clr_rf_5 = clf_rf_5.fit(X, y)
    importances = clr_rf_5.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_rf_5.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest

    plt.figure(figsize=(10, 8))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="g", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=60)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    return 1
def cross_validation():
    data = pd.read_csv('./GZSS_Feature.csv')
    # y = data.diagnosis  # M or B
    y= pd.read_csv('./GZSS_label.csv')
    y=y['1']
    # list = ['Unnamed: 32', 'id', 'diagnosis']
    # X = data.drop(list, axis=1)
    X=data
    print(X)
    print(y)
    tic = time.time()
    lr = RandomForestClassifier()
    selector = RFECV(estimator=lr, step=1,cv=5,scoring='accuracy').fit(X,y)
    toc = time.time()
    print(f"Done in {toc - tic:.3f}s")

    print('Optimal number of features :', selector.n_features_)
    print('Best features :', X.columns[selector.support_])
    X_best = pd.DataFrame(data, columns=X.columns[selector.support_])


    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_,color='green',label='RFE')
    plt.show()
    return X,X_best,y

def test(X_best,y):
    x_train, x_test, y_train, y_test = train_test_split(X_best, y, test_size=0.3, random_state=42)
    clf_rf = RandomForestClassifier(random_state=43)
    clr_rf = clf_rf.fit(x_train, y_train)
    ac = accuracy_score(y_test, clf_rf.predict(x_test))
    print('Accuracy is: %.3f%%' % (100 * ac))

    y_predt = clr_rf.predict(x_test)
    TN, FP, FN, TP = confusion_matrix(y_test, y_predt).ravel()
    Sen = float(TP) / (TP + FN)
    Spe = float(TN) / (FP + TN)
    npv = float(TN) / (TN + FN)
    ppv = float(TP) / (TP + FP)
    print("Sensitivity:%.3f%%" % (100 * Sen))
    print("Specificity:%.3f%%" % (100 * Spe))
    print("NPV:%.3f%%" % (100 * npv))
    print("PPV:%.3f%%" % (100 * ppv))
    print("TN, FP, FN, TP =", TN, FP, FN, TP)

    cm = confusion_matrix(y_test, clf_rf.predict(x_test))
    classes = ['患病', '健康']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    plt.title('RFE特征降维', fontproperties=font)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontproperties=font)
    plt.yticks(tick_marks, classes, fontproperties=font)
    thresh = cm.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, "{:}".format(cm[i, j]))  # 显示对应的数字
    plt.ylabel('实际值', fontproperties=font)
    plt.xlabel('预测值', fontproperties=font)
    plt.tight_layout()
    plt.show()
    return 1

if __name__=="__main__":
    X, X_best, y=cross_validation()
    test(X_best,y)
    test(X,y)
    # importance_rank()
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# ranking = rfe.ranking_.reshape(digits.images[0].shape)