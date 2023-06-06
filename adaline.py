import num as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv', header=None)

class AdalineGD(object):
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    #metodo per l'addestramento
    def fit(self, X, y): #X forma [n_esempi, n_caratteristiche], y: valori target
         #creo un generatore di numeri casuali con seme random_state (inzializzato a 1--> stessa sequenza di numeri casuali)
         rgen = np.random.RandomState(self.random_state) 
         #X.shape -> restituisce numero righe e colonne del dataset (TUPLA)
         #X.shape[0] -> restituisce il numero di righe 
         #X.shape[1] -> restituisce il numero di colonne
         #creo un vettore dei pesi, inizialmente numeri casuali: numero di colonne (caratteristiche) + 1 (per bias iniziale)
         self.w_ = rgen.normal(loc=0.0, scale=1.0, size = 1 + X.shape[1])
         self.cost_ = []

        #in questa porzione di codice vado ad aggiornare i pesi
         for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        
            return self

    #quello che faccio in questa funzione non è altro che il calcolo
    #questa funzione restituisce un numero --> net_input (sul foglio y_in)
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

y = df.iloc[0:100, 4].values #selezione le prime 100 righe e la 4 colonna
y = np.where(y == 'Iris-setosa', -1, 1)

#estraggo la lunghezza di sepalo e petalo
X = df.iloc[0:100, [0, 2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_)+ 1), np.log10(ada1.cost_), marker='o')

        
