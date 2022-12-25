from __future__ import print_function
from genetic_selection import GeneticSelectionCV
import numpy as np
from sklearn.neural_network import MLPRegressor
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

def datapre():
    data = pd.read_csv('./GZSS_Feature.csv')
    # y = data.diagnosis  # M or B
    y = pd.read_csv('./GZSS_label.csv')
    y = y['1']
    # list = ['Unnamed: 32', 'id', 'diagnosis']
    # X = data.drop(list, axis=1)
    X = data
    return X, y, data
def main():
    # 1.数据获取
    x, y, data = datapre()
    print(x.shape, y.shape)

    # 2.样本集划分和预处理
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    x_scale, y_scale = StandardScaler(), StandardScaler()
    x_train_scaled = x_scale.fit_transform(x_train)
    x_test_scaled = x_scale.transform(x_test)
    y_train_scaled = y_scale.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scale.transform(y_test.reshape(-1, 1))
    print(x_train_scaled.shape, y_train_scaled.shape)
    print(x_test_scaled.shape, y_test_scaled.shape)

    # 3. 优化超参数
    base, size = 30, 21
    wavelengths_save, wavelengths_size, r2_test_save, mse_test_save = [], [], [], []
    for hidden_size in range(base, base + size):
        print('隐含层神经元数量: ', hidden_size)
        estimator = MLPRegressor(hidden_layer_sizes=hidden_size,
                                 activation='relu',
                                 solver='adam',
                                 alpha=0.0001,
                                 batch_size='auto',
                                 learning_rate='constant',
                                 learning_rate_init=0.001,
                                 power_t=0.5,
                                 max_iter=1000,
                                 shuffle=True,
                                 random_state=1,
                                 tol=0.0001,
                                 verbose=False,
                                 warm_start=False,
                                 momentum=0.9,
                                 nesterovs_momentum=True,
                                 early_stopping=False,
                                 validation_fraction=0.1,
                                 beta_1=0.9, beta_2=0.999,
                                 epsilon=1e-08)

        selector = GeneticSelectionCV(estimator,
                                      cv=5,
                                      verbose=1,
                                      scoring="neg_mean_squared_error",
                                      max_features=5,
                                      n_population=200,
                                      crossover_proba=0.5,
                                      mutation_proba=0.2,
                                      n_generations=200,
                                      crossover_independent_proba=0.5,
                                      mutation_independent_proba=0.05,
                                      tournament_size=3,
                                      n_gen_no_change=10,
                                      caching=True,
                                      n_jobs=-1)
        selector = selector.fit(x_train_scaled, y_train_scaled.ravel())
        print('有效变量的数量：', selector.n_features_)
        print(np.array(selector.population_).shape)
        print(selector.generation_scores_)

        x_train_s, x_test_s = x_train_scaled[:, selector.support_], x_test_scaled[:, selector.support_]
        estimator.fit(x_train_s, y_train_scaled.ravel())

        # y_train_pred = estimator.predict(x_train_s)
        y_test_pred = estimator.predict(x_test_s)
        # y_train_pred = y_scale.inverse_transform(y_train_pred)
        y_test_pred = y_scale.inverse_transform(y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        wavelengths_save.append(list(selector.support_))
        wavelengths_size.append(selector.n_features_)
        r2_test_save.append(r2_test)
        mse_test_save.append(mse_test)
        print('决定系数：', r2_test, '均方误差：', mse_test)

    print('有效变量数量', wavelengths_size)

    # 4.保存过程数据
    dict_name = {'wavelengths_size': wavelengths_size, 'r2_test_save': r2_test_save,
                 'mse_test_save': mse_test_save, 'wavelengths_save': wavelengths_save}
    f = open('bpnn_ga.txt', 'w')
    f.write(str(dict_name))
    f.close()

    # 5.绘制曲线
    plt.figure(figsize=(6, 4), dpi=300)
    fonts = 8
    xx = np.arange(base, base + size)
    plt.plot(xx, r2_test_save, color='r', linewidth=2, label='r2')
    plt.plot(xx, mse_test_save, color='k', linewidth=2, label='mse')
    plt.xlabel('generation', fontsize=fonts)
    plt.ylabel('accuracy', fontsize=fonts)
    plt.grid(True)
    plt.legend(fontsize=fonts)
    plt.tight_layout(pad=0.3)
    plt.show()


if __name__ == "__main__":
    main()
