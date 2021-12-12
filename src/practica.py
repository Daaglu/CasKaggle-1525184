import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from matplotlib import pyplot as plt
from numpy import mean
from numpy import std
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, \
    average_precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import time


def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

dataset_rock = load_dataset('./Data/0.csv')
dataset_rock.columns = [
"reading 1 sensor 1", "reading 1 sensor 2","reading 1 sensor 3","reading 1 sensor 4","reading 1 sensor 5","reading 1 sensor 6","reading 1 sensor 7", "reading 1 sensor 8",
"reading 2 sensor 1", "reading 2 sensor 2","reading 2 sensor 3","reading 2 sensor 4","reading 2 sensor 5","reading 2 sensor 6","reading 2 sensor 7", "reading 2 sensor 8",
"reading 3 sensor 1", "reading 3 sensor 2","reading 3 sensor 3","reading 3 sensor 4","reading 3 sensor 5","reading 3 sensor 6","reading 3 sensor 7", "reading 3 sensor 8",
"reading 4 sensor 1", "reading 4 sensor 2","reading 4 sensor 3","reading 4 sensor 4","reading 4 sensor 5","reading 4 sensor 6","reading 4 sensor 7", "reading 4 sensor 8",
"reading 5 sensor 1", "reading 5 sensor 2","reading 5 sensor 3","reading 5 sensor 4","reading 5 sensor 5","reading 5 sensor 6","reading 5 sensor 7", "reading 5 sensor 8",
"reading 6 sensor 1", "reading 6 sensor 2","reading 6 sensor 3","reading 6 sensor 4","reading 6 sensor 5","reading 6 sensor 6","reading 6 sensor 7", "reading 6 sensor 8",
"reading 7 sensor 1", "reading 7 sensor 2","reading 7 sensor 3","reading 7 sensor 4","reading 7 sensor 5","reading 7 sensor 6","reading 7 sensor 7", "reading 7 sensor 8",
"reading 8 sensor 1", "reading 8 sensor 2","reading 8 sensor 3","reading 8 sensor 4","reading 8 sensor 5","reading 8 sensor 6","reading 8 sensor 7", "reading 8 sensor 8",
"result"]
dataset_scissors = load_dataset('./Data/1.csv')
dataset_scissors.columns = [
"reading 1 sensor 1", "reading 1 sensor 2","reading 1 sensor 3","reading 1 sensor 4","reading 1 sensor 5","reading 1 sensor 6","reading 1 sensor 7", "reading 1 sensor 8",
"reading 2 sensor 1", "reading 2 sensor 2","reading 2 sensor 3","reading 2 sensor 4","reading 2 sensor 5","reading 2 sensor 6","reading 2 sensor 7", "reading 2 sensor 8",
"reading 3 sensor 1", "reading 3 sensor 2","reading 3 sensor 3","reading 3 sensor 4","reading 3 sensor 5","reading 3 sensor 6","reading 3 sensor 7", "reading 3 sensor 8",
"reading 4 sensor 1", "reading 4 sensor 2","reading 4 sensor 3","reading 4 sensor 4","reading 4 sensor 5","reading 4 sensor 6","reading 4 sensor 7", "reading 4 sensor 8",
"reading 5 sensor 1", "reading 5 sensor 2","reading 5 sensor 3","reading 5 sensor 4","reading 5 sensor 5","reading 5 sensor 6","reading 5 sensor 7", "reading 5 sensor 8",
"reading 6 sensor 1", "reading 6 sensor 2","reading 6 sensor 3","reading 6 sensor 4","reading 6 sensor 5","reading 6 sensor 6","reading 6 sensor 7", "reading 6 sensor 8",
"reading 7 sensor 1", "reading 7 sensor 2","reading 7 sensor 3","reading 7 sensor 4","reading 7 sensor 5","reading 7 sensor 6","reading 7 sensor 7", "reading 7 sensor 8",
"reading 8 sensor 1", "reading 8 sensor 2","reading 8 sensor 3","reading 8 sensor 4","reading 8 sensor 5","reading 8 sensor 6","reading 8 sensor 7", "reading 8 sensor 8",
"result"]
dataset_paper = load_dataset('./Data/2.csv')
dataset_paper.columns = [
"reading 1 sensor 1", "reading 1 sensor 2","reading 1 sensor 3","reading 1 sensor 4","reading 1 sensor 5","reading 1 sensor 6","reading 1 sensor 7", "reading 1 sensor 8",
"reading 2 sensor 1", "reading 2 sensor 2","reading 2 sensor 3","reading 2 sensor 4","reading 2 sensor 5","reading 2 sensor 6","reading 2 sensor 7", "reading 2 sensor 8",
"reading 3 sensor 1", "reading 3 sensor 2","reading 3 sensor 3","reading 3 sensor 4","reading 3 sensor 5","reading 3 sensor 6","reading 3 sensor 7", "reading 3 sensor 8",
"reading 4 sensor 1", "reading 4 sensor 2","reading 4 sensor 3","reading 4 sensor 4","reading 4 sensor 5","reading 4 sensor 6","reading 4 sensor 7", "reading 4 sensor 8",
"reading 5 sensor 1", "reading 5 sensor 2","reading 5 sensor 3","reading 5 sensor 4","reading 5 sensor 5","reading 5 sensor 6","reading 5 sensor 7", "reading 5 sensor 8",
"reading 6 sensor 1", "reading 6 sensor 2","reading 6 sensor 3","reading 6 sensor 4","reading 6 sensor 5","reading 6 sensor 6","reading 6 sensor 7", "reading 6 sensor 8",
"reading 7 sensor 1", "reading 7 sensor 2","reading 7 sensor 3","reading 7 sensor 4","reading 7 sensor 5","reading 7 sensor 6","reading 7 sensor 7", "reading 7 sensor 8",
"reading 8 sensor 1", "reading 8 sensor 2","reading 8 sensor 3","reading 8 sensor 4","reading 8 sensor 5","reading 8 sensor 6","reading 8 sensor 7", "reading 8 sensor 8",
"result"]
dataset_ok = load_dataset('./Data/3.csv')
dataset_ok.columns = [
"reading 1 sensor 1", "reading 1 sensor 2","reading 1 sensor 3","reading 1 sensor 4","reading 1 sensor 5","reading 1 sensor 6","reading 1 sensor 7", "reading 1 sensor 8",
"reading 2 sensor 1", "reading 2 sensor 2","reading 2 sensor 3","reading 2 sensor 4","reading 2 sensor 5","reading 2 sensor 6","reading 2 sensor 7", "reading 2 sensor 8",
"reading 3 sensor 1", "reading 3 sensor 2","reading 3 sensor 3","reading 3 sensor 4","reading 3 sensor 5","reading 3 sensor 6","reading 3 sensor 7", "reading 3 sensor 8",
"reading 4 sensor 1", "reading 4 sensor 2","reading 4 sensor 3","reading 4 sensor 4","reading 4 sensor 5","reading 4 sensor 6","reading 4 sensor 7", "reading 4 sensor 8",
"reading 5 sensor 1", "reading 5 sensor 2","reading 5 sensor 3","reading 5 sensor 4","reading 5 sensor 5","reading 5 sensor 6","reading 5 sensor 7", "reading 5 sensor 8",
"reading 6 sensor 1", "reading 6 sensor 2","reading 6 sensor 3","reading 6 sensor 4","reading 6 sensor 5","reading 6 sensor 6","reading 6 sensor 7", "reading 6 sensor 8",
"reading 7 sensor 1", "reading 7 sensor 2","reading 7 sensor 3","reading 7 sensor 4","reading 7 sensor 5","reading 7 sensor 6","reading 7 sensor 7", "reading 7 sensor 8",
"reading 8 sensor 1", "reading 8 sensor 2","reading 8 sensor 3","reading 8 sensor 4","reading 8 sensor 5","reading 8 sensor 6","reading 8 sensor 7", "reading 8 sensor 8",
"result"]

dataset = dataset_rock.append(dataset_scissors)
dataset = dataset.append(dataset_paper)
dataset = dataset.append(dataset_ok)

print("Dimensionalitat de la BBDD de Roca:", dataset_rock.shape)
print("Dimensionalitat de la BBDD de Tissores:", dataset_scissors.shape)
print("Dimensionalitat de la BBDD de Paper:", dataset_paper.shape)
print("Dimensionalitat de la BBDD de Ok:", dataset_ok.shape)
print("Dimensionalitat de la BBDD:", dataset.shape)

print(dataset.isnull().values.any())
print(dataset.isnull().sum().sum())

data_mean, data_std = mean(dataset.values), std(dataset.values)
print(data_mean, data_std)

data = dataset.values

x = data[:, :64]
y = data[:, 64]

print(x)
print(y)

particions = [0.3, 0.5, 0.7, 0.8]
kernel_lst = ['linear', 'poly', 'rbf', 'sigmoid']

for kern in kernel_lst:
    inici = time.time()
    print("---------- KERNEL:", kern, " ---------- ")
    for part in particions:
        x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=part)

        logireg = LogisticRegression(C=1.0, fit_intercept=True, penalty='l2', max_iter = 1000)
        logireg.fit(x_t, y_t)

        print("Correct classification Logistic ", part, "% of the data: ", logireg.score(x_v, y_v))
        fi = time.time()
        total = fi - inici
        print("Ha tardat: ", total, " segons")
plt.show()

print("Decision tree")
for part in particions:
    inici = time.time()
    x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=part)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_t, y_t)
    cv = cross_val_score(clf, x_v, y_v, cv=10)
    total = 0
    for v in cv:
        total += v
    print("Mean accuracy with ", part, "% of the data: ", round(total/10, 4))
    fi = time.time()
    total = fi - inici
    print("Ha tardat: ", total, " segons")
    plot_tree(clf)
probs = clf.predict_proba(x_v)

total_classes = 4
precision = {}
recall = {}
average_precision = {}
plt.figure()
for i in range(total_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
    average_precision[i] = average_precision_score(y_v == i, probs[:, i])

    plt.plot(recall[i], precision[i],
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")

fpr = {}
tpr = {}
roc_auc = {}
for i in range(total_classes):
    fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(total_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
plt.legend()

print("KNN")

x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=0.7)

neigh_array = range(1, 10)

for n in neigh_array:
    inici = time.time()
    print("---------- num neighbors = ", n, " ---------- ");

    model = KNeighborsClassifier(n_neighbors=n)

    model.fit(x_t,y_t)

    predictions = model.predict(x_t)

    acc = accuracy_score(y_t, predictions)
    prec = precision_score(y_t, predictions, average='micro')
    rec = recall_score(y_t, predictions, average='micro')
    conf_mat = confusion_matrix(y_t, predictions).T
    fi = time.time()
    total = fi - inici
    print(f'Accuracy:{acc}')
    print(f'Precision:{prec}')
    print(f'Recall:{rec}')
    print("Ha tardat: ", total, " segons")