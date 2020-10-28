from pandas import read_csv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import itertools

#region Constatns

M = 5
PI = 3.1415926535
E = 2.7182818284
MANHATTAN = 'manhattan'
EUCLIDEAN = 'euclidean'
CHEBYSHEV = 'chebyshev'
FIXED = 'fixed'
VARIABLE = 'variable'
UNIFORM = 'uniform'
TRIANGULAR = 'triangular'
EPANECHNIKOV = 'epanechnikov'
QUARTIC = 'quartic'
TRIWEIGHT = 'triweight'
TRICUBE = 'tricube'
GAUSSIAN = 'gaussian'
COSINE = 'cosine'
LOGISTIC = 'logistic'
SIGMOID = 'sigmoid'

#endregion

#region Distances
def manhattan(X, Xi):
    distance = 0
    i = 0
    while i < M:
        distance += abs(X[i]-Xi[i])
        i += 1
    return distance


def euclidean(X, Xi):
    distance = 0
    i = 0
    while i < M:
        distance += pow(X[i] - Xi[i], 2)
        i += 1
    return pow(distance, 0.5)


def chebyshev(X, Xi):
    distance = []
    i = 0
    while i < M:
        distance.append(abs(X[i] - Xi[i]))
        i += 1
    return max(distance)
#endregion

#region Kernels
def uniform(u):
    if -1 < u < 1:
        return 0.5
    else:
        return 0


def triangular(u):
    if -1 < u < 1:
        return 1-abs(u)
    else:
        return 0

def epanechnikov(u):
    if -1 < u < 1:
        return 3 * (1 - u*u) / 4
    else:
        return 0


def quartic(u):
    if -1 < u < 1:
        return 15 * pow((1 - u*u), 2) / 16
    else:
        return 0


def triweight(u):
    if -1 < u < 1:
        return 35 * pow((1 - u * u), 3) / 32
    else:
        return 0


def tricube(u):
    if -1 < u < 1:
        return 70 * pow((1 - abs(u * u * u)), 3) / 81
    else:
        return 0


def gaussian(u):
    return pow(E, -0.5 * u * u) / pow(2 * PI, 0.5)


def cosine_f(u):
    if -1 < u < 1:
        return PI * (E**((PI * u / 2)*1j)).real / 4
    else:
        return 0


def logistic(u):
    return 1 / (pow(E, u) + 2 + pow(E, -1 * u))


def sigmoid(u):
    return 2 / (PI * (pow(E, u) + pow(E, -1 * u)))

def avg(x):
    i = 0
    sum = 0
    while i < len(x):
        sum += x[i][M]
        i += 1
    return sum / len(x)
#endregion


def non_parametric_regression(n, M, features, request, distance, kernel_type, window_type, h):
    i = 0
    while i < n:
        features[i].append(distance(features[i], request))
        i += 1

    features.sort(key=lambda x: x[M + 1])
    if window_type == VARIABLE:
        h = features[h][M + 1]

    if h == 0:
        suitable = []
        i = 0
        while i < n:
            bool = 1
            j = 0
            while j < M:
                if features[i][j] != request[j]:
                    bool = 0
                    break
                j += 1
            if bool == 1:
                suitable.append(features[i])
            i += 1
        if len(suitable) != 0:
            # we got into existing point feature[index]
            return avg(suitable)
        else:
            return avg(features)

    # apply final function for every el in elements
    up = 0
    down = 0
    i = 0
    while i < n:
        yi = features[i][M]
        if kernel_type == UNIFORM:
            kernel = uniform
        elif kernel_type == TRIANGULAR:
            kernel = triangular
        elif kernel_type == EPANECHNIKOV:
            kernel = epanechnikov
        elif kernel_type == QUARTIC:
            kernel = quartic
        elif kernel_type == TRIWEIGHT:
            kernel = triweight
        elif kernel_type == TRICUBE:
            kernel = tricube
        elif kernel_type == GAUSSIAN:
            kernel = gaussian
        elif kernel_type == COSINE:
            kernel = cosine_f
        elif kernel_type == LOGISTIC:
            kernel = logistic
        elif kernel_type == SIGMOID:
            kernel = sigmoid

        kernel_value = kernel(features[i][M + 1] / h)
        up += yi * kernel_value
        down += kernel_value

        i += 1
    if down == 0 and window_type == FIXED:
        return 0
    if down == 0 and window_type == VARIABLE:
        return avg(features)

    result = up / down

    return result


def f_measure(p, window_type, encoding_type, distance, kernel):

    CM = loo(p, window_type, encoding_type, distance, kernel)
    K = 3

    measures = [[None for x in range(0)] for y in range(K)]

    overall_sum = 0
    TP = 0
    i = 0
    weight_precision = 0

    while i < K:
        TP += CM[i][i]
        sum_y = 0
        sum_x = sum(CM[i])
        row = 0
        while row < K:
            sum_y += CM[row][i]
            row += 1
        overall_sum += sum_y

        if CM[i][i] == 0:
            precision = 0
            recall = 0
            F1_score = 0
        else:
            precision = CM[i][i] / sum_x
            recall = CM[i][i] / sum_y
            F1_score = 2 * precision * recall / (precision + recall)
            weight_precision += CM[i][i] * sum_y / sum_x

        measures[i].append(sum_y)
        measures[i].append(precision)
        measures[i].append(recall)
        measures[i].append(F1_score)
        i += 1

    # macro F1
    if overall_sum == 0:
        weight_recall = 0
        weight_macro_F1 = 0
    else:
        weight_precision /= overall_sum
        weight_recall = TP / overall_sum
        if weight_precision == 0 and weight_recall == 0:
            weight_macro_F1 = 0
        else:
            weight_macro_F1 = 2 * weight_precision * weight_recall / (weight_precision + weight_recall)
    return weight_macro_F1


def get_values_for_onehot(length, train_set, x_test, distance, kernel, window_type, p, ys):
    pred_value1 = non_parametric_regression(length, M, train_set[0].tolist(), x_test, distance,
                                            kernel, window_type, p)
    actual_value1 = ys[0]

    pred_value2 = non_parametric_regression(length, M, train_set[1].tolist(), x_test, distance,
                                            kernel, window_type, p)
    actual_value2 = ys[1]

    pred_value3 = non_parametric_regression(length, M, train_set[2].tolist(), x_test, distance,
                                            kernel, window_type, p)
    actual_value3 = ys[2]

    p_value = max(pred_value1, pred_value2, pred_value3)
    if p_value == pred_value1:
        p_value = 1
    elif p_value == pred_value2:
        p_value = 2
    elif p_value == pred_value3:
        p_value = 3

    act_value = max(actual_value1, actual_value2, actual_value3)
    if act_value == actual_value1:
        act_value = 1
    elif act_value == actual_value2:
        act_value = 2
    elif act_value == actual_value3:
        act_value = 3

    return [p_value, act_value]

def get_values_for_naive(length, train_set, x_test, distance, kernel, window_type, p, ys):
    p_value = round(
        non_parametric_regression(length, M, train_set[0].tolist(), x_test, distance, kernel, window_type, p))
    act_value = ys
    return [p_value, act_value]


def loo(p, window_type, encoding_type, distance, kernel):
    cm = [[0 for x in range(3)] for y in range(3)]
    loo_i = 0
    while loo_i < len(x_norm):
        if encoding_type == get_values_for_onehot:
            x_train = np.array([x for i, x in enumerate(x_norm) if i.__index__() != loo_i])
            y_train1 = np.array([x for i, x in enumerate(y1) if i.__index__() != loo_i])
            y_train2 = np.array([x for i, x in enumerate(y2) if i.__index__() != loo_i])
            y_train3 = np.array([x for i, x in enumerate(y3) if i.__index__() != loo_i])
            x_test = x_norm[loo_i]
            ys = [y1[loo_i], y2[loo_i], y3[loo_i]]
            train_set = [np.concatenate((x_train, y_train1[:, None]), axis=1),
                         np.concatenate((x_train, y_train2[:, None]), axis=1),
                         np.concatenate((x_train, y_train3[:, None]), axis=1)]
        else:
            x_train = np.array([x for i, x in enumerate(x_norm) if i.__index__() != loo_i])
            y_train = np.array([x for i, x in enumerate(y) if i.__index__() != loo_i])
            train_set = [np.concatenate((x_train, y_train[:, None]), axis=1)]
            x_test = x_norm[loo_i]
            ys = y[loo_i]

        length = len(x_train)
        values = encoding_type(length, train_set, x_test, distance, kernel, window_type, p, ys)
        cm[int(values[0]) - 1][int(values[1]) - 1] += 1
        loo_i += 1

    return cm


dataset = read_csv("./dataset.csv")
array = dataset.values
x = array[:, :M]
y = array[:, M]


x_norm = preprocessing.normalize(x)

#region binary encoding
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = y.reshape(len(y), 1)
y_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

y1 = y_onehot_encoded[:, 0]
y2 = y_onehot_encoded[:, 1]
y3 = y_onehot_encoded[:, 2]
ys = [y1, y2, y3]
#endregion


distances = [manhattan, euclidean, chebyshev]
kernels = [UNIFORM, TRIANGULAR, EPANECHNIKOV, QUARTIC, TRIWEIGHT, TRICUBE, GAUSSIAN, COSINE, LOGISTIC, SIGMOID]
ks = [*range(2, int(pow(len(array), 0.5)) + 1)]

# for each class find the best combination of p, k and K
compare_table = [[], [], [], []]

for distance in distances:
    print('calculating...', distance.__name__)
    current_distances = []
    for pair in itertools.product(x_norm, repeat=2):
        current_distances.append(distance(*pair))
    hs = np.unique(np.array(current_distances))

    for kernel in kernels:
        for h in hs:
            # f-measure for one hot
            compare_table[0].append(
                [f_measure(h, FIXED, get_values_for_onehot, distance, kernel), distance.__name__, kernel, h])

            # f-measure for naive
            compare_table[1].append(
                [f_measure(h, FIXED, get_values_for_naive, distance, kernel), distance.__name__, kernel, h])
        for k in ks:
            # f-measure for one hot
            compare_table[2].append(
                [f_measure(k, VARIABLE, get_values_for_onehot, distance, kernel), distance.__name__, kernel, k])

            # f-measure for naive
            compare_table[3].append(
                [f_measure(k, VARIABLE, get_values_for_naive, distance, kernel), distance.__name__, kernel, k])

print('start sorting...')
compare_table[0].sort()
compare_table[1].sort()
compare_table[2].sort()
compare_table[3].sort()

best_h_for_one_hot_encoding = compare_table[0][len(compare_table[0]) - 1]
best_h_for_naive = compare_table[1][len(compare_table[1]) - 1]
best_k_for_one_hot_encoding = compare_table[2][len(compare_table[2]) - 1]
best_k_for_naive = compare_table[3][len(compare_table[3]) - 1]

print(best_h_for_one_hot_encoding)
print(best_h_for_naive)
print(best_k_for_one_hot_encoding)
print(best_k_for_naive)

f1_measure_for_naive_by_k = []
i = 0
k = []
while i < len(compare_table[3]):
    if compare_table[3][i][1] == best_k_for_naive[1] and compare_table[3][i][2] == best_k_for_naive[2]:
        f1_measure_for_naive_by_k.append(compare_table[3][i][0])
        k.append(compare_table[3][i][3])
    i += 1

plt.plot(k, f1_measure_for_naive_by_k)
plt.show()
print('f1 by k for naive is shown')

f1_measure_for_one_hot_by_k = []
i = 0
k = []
while i < len(compare_table[2]):
    if compare_table[2][i][1] == best_k_for_one_hot_encoding[1] and compare_table[2][i][2] == best_k_for_one_hot_encoding[2]:
        f1_measure_for_one_hot_by_k.append(compare_table[2][i][0])
        k.append(compare_table[2][i][3])
    i += 1

plt.plot(k, f1_measure_for_one_hot_by_k)
plt.show()
print('f1 by k for onehot is shown')
