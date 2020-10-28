
#генетический на java

from sklearn import linear_model
import numpy as np
from skimage import measure
from sklearn import preprocessing
import matplotlib.pyplot as plt

# region preparing data
dataset = []
filepath = './Linear_tests/1.txt'
f = open(filepath)
M = int(f.readline())
N_train = int(f.readline())
traindata = []
i = 0
while i < N_train:
    traindata.append(list(map(int, f.readline().split())))
    i += 1

N_test = int(f.readline())
testdata = []
i = 0
while i < N_test:
    testdata.append(list(map(int, f.readline().split())))
    i += 1
f.close()

dataset = np.array(traindata)
testset = np.array(testdata)

X_train = preprocessing.normalize(dataset[:, :M])
y_train = dataset[:, M]

X_test = preprocessing.normalize(testset[:, :M])
y_test = testset[:, M]
#endregion

# region gradient descent
nrmse_test = []
loss_funcs = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
for loss_func in loss_funcs:
    rgrs = linear_model.SGDRegressor(loss=loss_func)
    rgrs.fit(X_train, y_train)
    y_pred_test = rgrs.predict(X_test)
    nrmse_test.append([measure.compare_nrmse(y_test, y_pred_test, norm_type="mean"), loss_func])

nrmse_test.sort()
best_loss_func = nrmse_test[0]


nrmse_test = []
penalties = ['l2', 'l1', 'elasticnet']
for penalty in penalties:
    rgrs = linear_model.SGDRegressor(penalty=penalty)
    rgrs.fit(X_train, y_train)
    y_pred_test = rgrs.predict(X_test)
    nrmse_test.append([measure.compare_nrmse(y_test, y_pred_test, norm_type="mean"), penalty])

nrmse_test.sort()
best_penalty = nrmse_test[0]


num_iterations = [*range(10, 300, 30)]
nrmse_train = []
nrmse_test = []
for num_iteration in num_iterations:
    rgrs = linear_model.SGDRegressor(max_iter=num_iteration)
    rgrs.fit(X_train, y_train)
    y_pred_train = rgrs.predict(X_train)
    y_pred_test = rgrs.predict(X_test)
    nrmse_train.append(measure.compare_nrmse(y_train, y_pred_train))
    nrmse_test.append([measure.compare_nrmse(y_test, y_pred_test), num_iteration])

arr = list(zip(*nrmse_test))
plt.plot(num_iterations, nrmse_train, 'r', num_iterations, arr[0], 'b')
plt.show()

nrmse_test.sort()
best_iter_max = nrmse_test[1]


learning_rates = ['constant', 'optimal', 'invscaling', 'adaptive']
nrmse_test = []
for learning_rate in learning_rates:
    rgrs = linear_model.SGDRegressor(learning_rate=learning_rate)
    rgrs.fit(X_train, y_train)
    y_pred_test = rgrs.predict(X_test)
    nrmse_test.append([measure.compare_nrmse(y_test, y_pred_test, norm_type="mean"), learning_rate])

nrmse_test.sort()
best_learning_rate = nrmse_test[0]
print("best loss function: ", best_loss_func)
print("best penalty: ", best_penalty)
print("best iter_max: ", best_iter_max)
print("best learning_rate: ", best_learning_rate)
#endregion

# !!!
# вместо pinv нужно использовать регуляризацию в svm ridge(sol..='svm', alpha=..) и пройтись по коэффициенту альфа
# calculate coefficients
b = np.linalg.pinv(X_train).dot(y_train)
# predict using coefficients
y_pred = X_test.dot(b)
nmrse = measure.compare_nrmse(y_test, y_pred, norm_type="mean")
print("Ошибка SVD: ", nmrse)

