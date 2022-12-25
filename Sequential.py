import time

from matplotlib import pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.font_manager import FontProperties  # 导入FontProperties
import numpy as np
from sklearn.model_selection import cross_val_score
font = FontProperties(fname="SimHei.ttf", size=14)  # 设置字体

def datapre():
    data = pd.read_csv('./GZSS_Feature.csv')
    # y = data.diagnosis  # M or B
    y = pd.read_csv('./GZSS_label.csv')
    y = y['1']
    # list = ['Unnamed: 32', 'id', 'diagnosis']
    # X = data.drop(list, axis=1)
    X = data
    return X,y,data
def test(X_best,y,name):
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
    plt.title(name, fontproperties=font)
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
def sfs_selection(X,y,kind):
    scores=[]
    best_score=0
    best_i=0
    model = RandomForestClassifier()
    for i in range(2,31):
        sfs = SequentialFeatureSelector(model, n_features_to_select=2,
                                                direction='forward').fit(X, y)
        X = pd.DataFrame(X, columns=X.columns[sfs.support_])
        score = cross_val_score(model, X, y, cv=5).mean()
        scores.append(score)
        if score>best_score:
            best_score=score
            best_i=i
    return scores,best_i

if __name__=="__main__":
    model=RandomForestClassifier()
    X, y, data = datapre()
    # scoresf,besti_f=sfs_selection(X,y,'forward')
    # scoresb, besti_b = sfs_selection(X, y, 'backward')
    # print("scoresf=",scoresf,'bestif=',besti_f)
    # print("scoresb=",scoresb,'bestib=',besti_b)
    tic_fwd = time.time()
    sfs_forward = SequentialFeatureSelector(model, n_features_to_select=14,
                                            direction='forward').fit(X, y)
    toc_fwd = time.time()

    tic_bwd = time.time()
    sfs_backward = SequentialFeatureSelector(model, n_features_to_select=14,
                                             direction='backward').fit(X, y)
    toc_bwd = time.time()

    X_bestf = pd.DataFrame(data, columns=X.columns[sfs_forward.support_])
    X_bestb = pd.DataFrame(data, columns=X.columns[sfs_backward.support_])

    # plt.plot(range(1, len(sfs_backward.grid_scores_) + 1), sfs_forward.grid_scores_, color='yellow',
    #          label='forward_search')
    test(X_bestf, y, '前向搜索特征降维')
    test(X_bestb, y, '后向搜索特征降维')

    # print("Features selected by forward sequential selection: "
    #       f"{feature_names[sfs_forward.get_support()]}")
    print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    # print("Features selected by backward sequential selection: "
    #       f"{feature_names[sfs_backward.get_support()]}")
    print(f"Done in {toc_bwd - tic_bwd:.3f}s")

    # Features selected by forward sequential selection: ['bmi' 's5']
    # Done in 2.177s
    # Features selected by backward sequential selection: ['bmi' 's5']
    # Done in 6.433s