from pylab import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
def y_update_scale_value(temp, position):
    result = temp
    return "{}%".format(float(result))
if __name__=="__main__":
    label = ["ACC", "Sensitivity", "Specificity", "NPV", "PPV"]
    Hillclimb = [81.48,79.67,86.45,76.45,81.38]
    PSO = [85.84,87.24,79.39,82.14,88.23]

    x = np.arange(len(label))  # x轴刻度标签位置
    width = 0.2  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    plt.figure(figsize=(10, 6))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_update_scale_value))  # 设置y1轴单位
    plt.bar(x - 0.5 * width, Hillclimb, width, label='HillClimbing')
    plt.bar(x + 0.5 * width, PSO, width, label='PSO')
    plt.ylabel('Scores')
    plt.title('Testing set')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=label, rotation=-15)
    plt.legend()
    plt.show()