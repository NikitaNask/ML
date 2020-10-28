import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas
from sklearn.metrics import f1_score

def SVM_linear(X_test, X_train, y_train, C):
    svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    return svc.predict(np.array(X_test).reshape(1, -1))

def SVM_rbf(X_test, X_train, y_train, C):
    rbf_svc = svm.SVC(kernel='rbf', C=C).fit(X_train, y_train)
    return rbf_svc.predict(np.array(X_test).reshape(1, -1))

def SVM_poly(X_test, X_train, y_train, C, degree):
    poly_svc = svm.SVC(kernel='poly', degree=degree, C=C).fit(X_train, y_train)
    return poly_svc.predict(np.array(X_test).reshape(1, -1))

def SVM_sigmiod(X_test, X_train, y_train, C):
    sigmoid_svc = svm.SVC(kernel='sigmoid', C=C).fit(X_train, y_train)
    return sigmoid_svc.predict(np.array(X_test).reshape(1, -1))



chips = pandas.read_csv('./chips.csv')
geyser = pandas.read_csv('./geyser.csv')
datasets = [chips, geyser]
for dataset in datasets:
    X = dataset.values[:, :2]
    y = dataset.values[:, 2]

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    svm_types = [SVM_linear, SVM_rbf, SVM_poly]


    Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
    results = []
    results_rbf = []
    results_sigmiod = []
    #for linear and rbf
    for C in Cs:
        y_pred = []
        y_pred_rbf = []
        y_pred_sigmiod = []
        loo_i = 0
        while loo_i < len(X):
            x_train = np.array([x for i, x in enumerate(X) if i.__index__() != loo_i])
            y_train = np.array([x for i, x in enumerate(y) if i.__index__() != loo_i])
            train_set = [np.concatenate((x_train, y_train[:, None]), axis=1)]
            x_test = X[loo_i]
            y_test = y[loo_i]

            y_pred.append(SVM_linear(x_test, x_train, y_train, C))
            y_pred_rbf.append(SVM_rbf(x_test, x_train, y_train, C))
            y_pred_sigmiod.append(SVM_sigmiod(x_test, x_train, y_train, C))

            loo_i += 1
        f1 = f1_score(y, y_pred)
        f1_rbf = f1_score(y, y_pred_rbf)
        f1_sigmiod = f1_score(y, y_pred_sigmiod)
        results.append([f1, C])
        results_rbf.append([f1_rbf, C])
        results_sigmiod.append([f1_sigmiod, C])
    results.sort()
    results_rbf.sort()
    results_sigmiod.sort()
    best_C_for_linear = results[-1]
    best_C_for_rbf = results_rbf[-1]
    best_C_for_sigmiod = results_sigmiod[-1]
    print('Best C for linear kernel: ', best_C_for_linear)
    print('Best C for radial based kernel: ', best_C_for_rbf)
    print('Best C for sigmiod kernel: ', best_C_for_sigmiod)

    #for polynomial kernel
    results = []
    degrees = [2, 3, 4, 5, 6]
    for degree in degrees:
        for C in Cs:
            y_pred = []
            loo_i = 0
            while loo_i < len(X):
                x_train = np.array([x for i, x in enumerate(X) if i.__index__() != loo_i])
                y_train = np.array([x for i, x in enumerate(y) if i.__index__() != loo_i])
                train_set = [np.concatenate((x_train, y_train[:, None]), axis=1)]
                x_test = X[loo_i]
                y_test = y[loo_i]

                y_pred.append(SVM_poly(x_test, x_train, y_train, C, degree))

                loo_i += 1
            f1 = f1_score(y, y_pred)
            results.append([f1, C, degree])
    results.sort()
    best_C_degree_for_polynomial = results[-1]
    print('Best C and degree for polynomial kernel: ', best_C_degree_for_polynomial)

    #plotting

    C_linear = best_C_for_linear[1]
    C_rbf = best_C_for_rbf[1]
    C_sigmoid = best_C_for_sigmiod[1]
    C_poly = best_C_degree_for_polynomial[1]
    degree_poly = best_C_degree_for_polynomial[2]

    h = .02  # step size in the mesh
    C = 1.0  # SVM regularization parameter

    svc = svm.SVC(kernel='linear', C=C_linear).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', C=C_rbf).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=degree_poly, C=C_poly).fit(X, y)
    sigmoid_svc = svm.SVC(kernel='sigmoid', C=C_sigmoid).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVM with linear kernel',
              'SVM with RBF kernel',
              'SVM with polynomial kernel',
              'SVM with sigmoid kernel']


    for i, clf in enumerate((svc, rbf_svc, poly_svc, sigmoid_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()
    print()
    print()
