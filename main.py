# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。
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

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 ⌘F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    classes = ['患病', '健康']
    # sum = 132+65+54+121
    confusion_matrix = np.array([(48, 11), (4,44)])
    confusion_matrix = np.around(confusion_matrix, 4)
    print(confusion_matrix)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    plt.title('前向搜索特征降维', fontproperties=font)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontproperties=font)
    plt.yticks(tick_marks, classes, fontproperties=font)

    thresh = confusion_matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, "{:}".format(confusion_matrix[i, j]))  # 显示对应的数字
    plt.ylabel('实际值', fontproperties=font)
    plt.xlabel('预测值', fontproperties=font)
    plt.tight_layout()
    plt.show()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
