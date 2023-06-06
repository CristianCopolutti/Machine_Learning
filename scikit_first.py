from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

#le classi per i target sono:
# [0,1,2] dove 
# 0 = Iris-setosa
# 1 = Iris-versicolor
# 2 = Iris

#stratify = y --> il metodo train_test_split restituisce sottoinsiemi di addestramento e di test aventi le stesse proporzioni, in termini di etichette delle classi, del dataset di input
#quindi vuol dire che ci saranno un numero uguale di etichette per dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#standardizzo le caratteristiche
sc = StandardScaler()
sc.fit(X_train)
X_train_scaled = sc.transform(X_train)
X_test_scaled = sc.transform(X_test)

#addestro modello perceptron
ppn = Perceptron(eta0=0.01, random_state=1)
ppn.fit(X_train_scaled, y_train)

y_pred = ppn.predict(X_test_scaled)
print('Esempi classificati male: %d' % (y_test != y_pred).sum())

#calcoliamo accuracy modello
print('Accuracy; %3f' % accuracy_score(y_test, y_pred))



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #imposto proprietà
    markers = ('s', 'x', 'o', '^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #caratteristica 1 minima e caratteristica 1 max
    #X[:,0] --> prendo la prima colonna
    x1_min, x1_max = X[:,0].min() -1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min() -1, X[:,1].max()+1
    #la funzione
    xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)