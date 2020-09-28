# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from IPython.display import clear_output


# %%
class Dense():

    def __init__(self, n_neurons):
        self.activation_function = sigmoid
        self.d_activation_function = d_sigmoid
        self.neurons = np.zeros(n_neurons)
    
    def __str__(self):
        return "Camada densa com " + str(self.neurons.shape[0]) + " neuronios e função de ativação " + str(self.activation_function)


class Sequential():

    def __init__(self, input_size):
        self.input_size = 4
        self.weights = []
        self.bias = []
        self.layers = []

    def add_layer(self, camada):
        self.layers.append(camada)
        pass

    def summary(self):
        for layer in self.layers:
            print("------------------------------------------------------------------------------------------------")
            print(layer)

    def compile(self):
        self.weights.append(np.random.rand( self.input_size, self.layers[0].neurons.shape[0] ))
        self.bias.append(np.random.rand(self.layers[0].neurons.shape[0]))
        self.error_function = erro_quadratico
        self.d_error_function = d_erro_quadratico
        i = 0
        for layer in range(len(self.layers) - 1):
            self.weights.append( np.random.rand( self.layers[i].neurons.shape[0], self.layers[i + 1].neurons.shape[0] ) )
            self.bias.append( np.random.rand(self.layers[i + 1].neurons.shape[0]) )
            i+=1
        
        print("Pesos:")
        for weights in self.weights:
            print(weights.shape)
        print("Bias:")
        for bia in self.bias:
            print(bia.shape)
    
    def feedforward(self, X):
        self.layers[0].neurons = self.layers[0].activation_function(( X @ self.weights[0]) + self.bias[0])
        for i in range(1, len(self.layers)):
            self.layers[i].neurons = self.layers[i].activation_function((self.layers[i - 1].neurons @ self.weights[i]) + self.bias[i])  
        return self.layers[-1].neurons

    def backpropagation(self, X, Y, lr):
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                erro = self.d_error_function(self.layers[i].neurons, Y)
            else:
                erro = (self.weights[i + 1] @ delta.T).T

            delta = erro * self.layers[i].d_activation_function(self.layers[i].neurons)
            
            if i == 0:
                dw = delta.T @ X
            else:
                dw = delta.T @ self.layers[i - 1].neurons

            self.weights[i] -= lr * dw.T
            self.bias[i] -= lr * np.sum(delta, axis=0)

    def predict(self, X):
        self.feedforward(X)
        return self.layers[-1].neurons

    def fit(self, X, Y, X_val, Y_val, epochs=10, lr=0.01):
        """
        Função de treino. Faz o treino sobre os conjuntos X e Y por epochs épocas. Após cada época o modelo               calcula o erro em cima dos conjuntos de validação. Após finalizar o treino, o modelo retorna os pesos que         tiveram o menor erro.
        """
        menor_erro = 100_000
        for i in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, Y, lr)
            erro = sum(sum(self.error_function(self.predict(X_val), Y_val))/Y_val.shape[0])/Y_val.shape[1]
            if erro < menor_erro:
                menor_erro = erro
                melhores_pesos = self.weights
        return melhores_pesos


# %%
def sigmoid(x):
    return 1/(1+np.e**-x)

def d_sigmoid(sig_x):
    return sig_x*(1-sig_x)

def d_erro_quadratico(Y, D):
    return (2*Y)-(2*D)

def erro_quadratico(Y, D):
    return (D - Y)**2


# %%
# Pré processamento
dataset = pd.read_csv("iris.csv", header=None)

X_train = pd.concat([dataset[0:30], dataset[50:80], dataset[100:130]]).sample(frac=1)
X_val = pd.concat([dataset[30:40], dataset[80:90], dataset[130:140]]).sample(frac=1)
X_test = pd.concat([dataset[40:50], dataset[90:100], dataset[140:150]]).sample(frac=1)

Y_train = X_train.pop(4)
Y_val = X_val.pop(4)
Y_test = X_test.pop(4)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)


Y_train = np.array([ [1, 0, 0] if y == 'Iris-setosa' else [0, 1, 0] if y == 'Iris-versicolor' else [0, 0, 1] for y in Y_train])
Y_val = np.array([ [1, 0, 0] if y == 'Iris-setosa' else [0, 1, 0] if y == 'Iris-versicolor' else [0, 0, 1] for y in Y_val])
Y_test = np.array([ [1, 0, 0] if y == 'Iris-setosa' else [0, 1, 0] if y == 'Iris-versicolor' else [0, 0, 1] for y in Y_test])


# %%
# Criação das redes
nn1 = Sequential(4)
nn1.add_layer(Dense(3))
nn1.compile()
nn2 = Sequential(4)
nn2.add_layer(Dense(10))
nn2.add_layer(Dense(3))
nn2.compile()
nn3 = Sequential(4)
nn3.add_layer(Dense(5))
nn3.add_layer(Dense(5))
nn3.add_layer(Dense(3))
nn3.compile()


# %%
# Treino
nn1.weights = nn1.fit(X_train, Y_train, X_val, Y_val, 10_000, lr=0.01)
nn2.weights = nn2.fit(X_train, Y_train, X_val, Y_val, 10_000, lr=0.01)
nn3.weights = nn3.fit(X_train, Y_train, X_val, Y_val, 10_000, lr=0.01)


# %%
# Cálculo das métricas
vp1, vn1, fp1, fn1 = [], [], [], []
vp2, vn2, fp2, fn2 = [], [], [], []
vp3, vn3, fp3, fn3 = [], [], [], []

for c in range(3):
    vp1.append(0)
    vn1.append(0)
    fp1.append(0)
    fn1.append(0)
    vp2.append(0)
    vn2.append(0)
    fp2.append(0)
    fn2.append(0)
    vp3.append(0)
    vn3.append(0)
    fp3.append(0)
    fn3.append(0)

output_nn1 = nn1.predict(X_test)
output_nn2 = nn2.predict(X_test)
output_nn3 = nn3.predict(X_test)

output_nn1 = np.array([ [1, 0, 0] if n.argmax() == 0 else [0, 1, 0] if n.argmax() == 1 else [0, 0, 1] for n in output_nn1])

output_nn2 = np.array([ [1, 0, 0] if n.argmax() == 0 else [0, 1, 0] if n.argmax() == 1 else [0, 0, 1] for n in output_nn2])

output_nn3 = np.array([ [1, 0, 0] if n.argmax() == 0 else [0, 1, 0] if n.argmax() == 1 else [0, 0, 1] for n in output_nn3])

a1 = 0
a2 = 0
a3 = 0

for predict_nn1, predict_nn2, predict_nn3, expected in zip(output_nn1, output_nn2, output_nn3, Y_test):
    for n in range(expected.shape[0]):

        if predict_nn1[n] == expected[n]:
            if expected[n] == 0:
                vn1[n]+=1
            else:
                vp1[n]+=1
        else:
            if expected[n] == 0:
                fp1[n]+=1
            else:
                fn1[n]+=1

        if predict_nn2[n] == expected[n]:
            if expected[n] == 0:
                vn2[n]+=1
            else:
                vp2[n]+=1
        else:
            if expected[n] == 0:
                fp2[n]+=1
            else:
                fn2[n]+=1

        if predict_nn3[n] == expected[n]:
            if expected[n] == 0:
                vn3[n]+=1
            else:
                vp3[n]+=1
        else:
            if expected[n] == 0:
                fp3[n]+=1
            else:
                fn3[n]+=1



vn1 = np.array(vn1)
vp1 = np.array(vp1)
fn1 = np.array(fn1)
fp1 = np.array(fp1) 

vn2 = np.array(vn2)
vp2 = np.array(vp2)
fn2 = np.array(fn2)
fp2 = np.array(fp2) 

vn3 = np.array(vn3)
vp3 = np.array(vp3)
fn3 = np.array(fn3)
fp3 = np.array(fp3)

precision1 = vp1/(vp1+fp1)
recall1 = vp1/(vp1+fn1)
accuracy1 = sum(vp1)/sum(vp1 + fn1)

precision2 = vp2/(vp2+fp2)
recall2 = vp2/(vp2+fn2)
accuracy2 = sum(vp2)/sum(vp2 + fn2)

precision3 = vp3/(vp3+fp3)
recall3 = vp3/(vp3+fn3)
accuracy3 = sum(vp3)/sum(vp3 + fn3)


# %%
print('Precision 1 camada: ', precision1)
print('Recall 1 camada: ', recall1)
print('Accuracy 1 camada: ', accuracy1)

print('Precision 2 camadas: ', precision2)
print('Recall 2 camadas: ', recall2)
print('Accuracy 2 camadas: ', accuracy2)

print('Precision 3 camadas: ', precision3)
print('Recall 3 camadas: ', recall3)
print('Accuracy 3 camadas: ', accuracy3)


