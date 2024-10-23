import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, lr:float, epochs:int, seed):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        randomize = np.random.RandomState(seed)
        self.w_ = randomize.normal(loc=0, scale=0.01, size=3)
        self.errors_ = []

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)

    def fit(self, X, y):
        # X es un array de muestras, que contienen a su vez características, la forma va a ser [100,2], este perceptron solo sirve para etiqueta binaria
        # Y son las etiquetas verdaderas forma [100,], sus vallores van a ser 1 o -1

        # Lo que tiene que hacer esto es iterar épocas veces y por cada una procesa muestras, calcula errores y actualiza pesos
        # procesar muestras: funcion prediict y calcular delta

        for i in range(0,self.epochs):
            print(f'Epoch {i+1}')
            errors_epoch = 0
            for x1, target in zip(X, y):
                delta = self.lr*(target-self.predict(x1))
                self.w_[1:] += delta*x1
                self.w_[0] += delta

                if delta != 0:
                    errors_epoch += 1

            self.errors_.append(errors_epoch)
            print(f'errors { errors_epoch }')


X=iris.data[:100,[0,2]]
y = list(map(lambda x: -1 if x == 0 else 1, iris.target[:100]))
print(y)

newPerceptron = Perceptron(lr=0.1, epochs=10, seed=1)
newPerceptron.fit(X,list(y))

print(newPerceptron.predict([[3.1, 1.2]]))
print(newPerceptron.predict([[5.1, 3.2]]))

#inventar foto setosa y versicolor y invocar predecir observando grafico longitud sepalo y petalo
