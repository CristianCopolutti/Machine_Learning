import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

#s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
#print('URL:', s)

df = pd.read_csv('iris.csv', header=None)
print(df.tail())


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate --> tasso di apprendimento
    n_iter : int
        Iterazioni sul dataset
    random_state : int
        generatore numero casuali per inizializazzione casuale del peso


    Attributi
    ---------
    w: 1d-array
        pesi dopo il fit
    errors: lista
        numero di classificazioni errate in ogni epoca
    """

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit dati di addestramento
        Parametri:
        X : shape = [num_esempi, num_caratteristiche]
        y : array, shape = [num_esempi]
            valori target

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) #inizializzo in modo casuale i pesi
        self.errors_ = []

        for _ in range(self.n_iter): 
            errors = 0
            for xi, target in zip(X, y): #cicla su tutti gli esempi del dataset
                #aggiornmento dei pesi su tutti gli esempi
                update = self.eta * (target - self.predict(xi))
                self.w[1:] = self.w[1:] + update * xi
                self.w[0] = self.w[0] + update
                errors = errors + int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0] #self.w[0] bias unit

    def predict(self, X):
        return np.where(self.net_input(X)>= 0.0, 1, -1)

#estraggo le prime 100 etichette delle classi che corrispondono ai 50 fiori Iris-setosa e ai 50 fiori Iris-versicolor
#converto le due etichette delle classi in due etichette intere: 1 (versicolo) -1 (setosa)

y = df.iloc[0:100, 4].values #selezione le prime 100 righe e la 4 colonna
y = np.where(y == 'Iris-setosa', -1, 1)

#estraggo la lunghezza di sepalo e petalo
X = df.iloc[0:100, [0, 2]].values

#disegno il grafico
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=50)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Epoche')
plt.ylabel('Numero di aggiornamenti')
plt.show()