import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

# parsing data
i = 0
dataset = []
ys = []
for dirname in os.listdir('messages'):
    dataset.append([])
    ys.append([])
    for filename in os.listdir('messages/' + dirname):
        (path, name) = os.path.split(filename)

        fp = open('messages/' + dirname + '/' + filename)
        sbj = fp.readline()
        fp.readline()
        text = fp.readline()
        fp.close()

        if re.search(r'spmsg', name) == None:
            dataset[i].append([sbj + text])
            ys[i].append(0)

        else:
            dataset[i].append([sbj + text])
            ys[i].append(1)
    i += 1


def NB(weight, result_type):
    y_pred = []
    j = 0
    k = 10
    while j < k:
        y_pred.append([])
        x_test = np.array(dataset[j]).reshape(-1)
        x_train = np.array([x for i, x in enumerate(dataset) if i.__index__() != j]).reshape(-1)
        y_train = np.array([x for i, x in enumerate(ys) if i.__index__() != j]).reshape(-1)

        vectorizer = CountVectorizer()
        counts = vectorizer.fit_transform(x_train).toarray()
        classifier = MultinomialNB(class_prior=[weight, 1 - weight])
        targets = y_train
        classifier.fit(counts, targets)
        count_pred = vectorizer.transform(x_test).toarray()
        if result_type == 'proba':
            y_pred[j].append(classifier.predict_proba(count_pred).tolist())
        else:
            y_pred[j].append(classifier.predict(count_pred).tolist())

        x_train = []
        y_train = []

        j += 1
    return y_pred

# ROC curve
y_pred_probabilities = NB(0.5, 'proba')
y_pred_probabilities = np.array(y_pred_probabilities).reshape(-1)[1::2]
y_act = np.array(ys).reshape(-1)

fpr, tpr, thresholds = roc_curve(y_act, y_pred_probabilities)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()


# F measure
prs = [0.00000000000000000001, 0.00000000000000001, 0.00000000001, 0.000000001, 0.000001, 0.00001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
accuracies = []
for pr in prs:
    y_pred = NB(pr, 'clf')
    y_pred = np.array(y_pred).reshape(-1)
    y_act = np.array(ys).reshape(-1)

    accuracies.append(f1_score(y_act, y_pred))

plt.plot(prs, accuracies)
plt.show()

