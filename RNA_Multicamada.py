import cupy as np
import pandas as pd
import time


class Dense():

    def __init__(self, n_neurons):
        self.activation_function = sigmoid
        self.d_activation_function = d_sigmoid
        self.neurons = np.zeros(n_neurons)
    
    def __str__(self):
        return "Camada densa com " + str(self.neurons.shape[0]) + " neuronios e função de ativação " + str(self.activation_function)


class Sequential():

    def __init__(self, input_size):
        self.input_size = input_size
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

    def compile(self, error_function, d_error_function):
        self.weights.append(np.random.rand( self.input_size, self.layers[0].neurons.shape[0] ) - 0.5)
        self.bias.append(np.random.rand(self.layers[0].neurons.shape[0]))
        self.error_function = erro_quadratico
        self.d_error_function = d_erro_quadratico
        i = 0
        for layer in range(len(self.layers) - 1):
            self.weights.append( np.random.rand( self.layers[i].neurons.shape[0], self.layers[i + 1].neurons.shape[0] ) - 0.5)
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

    def backpropagation(self, X, Y, lr, show_info=False):
        for i in reversed(range(len(self.weights))):

            if i == len(self.weights) - 1:
                erro = self.d_error_function(self.layers[-1].neurons, Y)
            else:
                # print('pesos', self.weights[i + 1].shape)
                # print('delta.T',delta.T.shape)
                # print('(self.weights[i + 1] @ delta.T).T', erro.shape)
                erro = (self.weights[i + 1] @ delta.T).T
                # input()

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

    def fit(self, X, Y, X_val=[], Y_val=[], epochs=10, lr=0.01, stop_after=-1):
        """
        Função de treino. Faz o treino sobre os conjuntos X e Y por epochs épocas. Após cada época o modelo calcula o erro em cima dos conjuntos de validação. Após finalizar o treino, o modelo retorna os pesos que tiveram o menor erro.
        """
        menor_erro = 100_000
        if stop_after < 0:
            stop_after = epochs
        if len(X_val) == 0:
            X_val = X
            Y_val = Y

        start = time.time()
        no_improvement = 0
        lr = lr/X.shape[0]
        for i in range(epochs):
            self.feedforward(X)
            

            if i % 30 == 0:
                self.backpropagation(X, Y, lr, True)
            else:
                self.backpropagation(X, Y, lr, False)

            validation = self.predict(X_val)

            erro = np.mean(self.error_function(validation, Y_val))

            if erro < menor_erro:
                no_improvement = 0
                melhor_iteracao = i
                menor_erro = erro
                melhores_pesos = self.weights
                melhores_bias = self.bias
            else:
                no_improvement+=1
                if no_improvement == stop_after:
                    break

            if i % 30 == 0:
                accuracy = 0
                for val in range(validation.shape[0]):
                    if validation[val].argmax() == Y_val[val].argmax():
                        accuracy+=1
                accuracy = accuracy/Y_val.shape[0]
                complete = str((i/epochs)*100)
                acc = str(accuracy * 100)
                print(complete+"% completo")
                print("Melhor Iteração:", melhor_iteracao)
                print("Menor Erro", menor_erro)
                print("Acurácia até agora:", acc + "%")
                

        complete = str((i/epochs)*100)
        acc = str(accuracy * 100)
        print(complete+"% completo")
        print("Melhor Iteração:", melhor_iteracao)
        print("Menor Erro", menor_erro)
        print("Acurácia até agora:", acc + "%")
        
        return melhores_pesos, melhores_bias


def sigmoid(x):
    return 1/(1+np.e**-x)

def d_sigmoid(sig_x):
    return sig_x*(1-sig_x)

def d_erro_quadratico(Y, D):
    return 2*(Y - D)

def erro_quadratico(Y, D):
    return (D - Y)**2

def erro_absoluto(Y, D):
    return np.abs(D - Y)

def d_erro_absoluto(Y, D):
    return 


dataset = pd.read_csv("dataset/MNIST.csv", header=None,)
X_train = dataset.sample(frac=0.70)
X_val = dataset.sample(frac=0.20)
X_test = dataset.sample(frac=0.10)
Y_train = X_train.pop(784)
Y_val = X_val.pop(784)
Y_test = X_test.pop(784)
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
aaaa = {'zero': 0, 'um': 1, 'dois': 2, 'tres': 3, 'quatro': 4, 'cinco': 5, 'seis': 6, 'sete': 7, 'oito': 8, 'nove': 9}

Y_train = np.array([aaaa[num] for num in Y_train])
Y_val = np.array([aaaa[num] for num in Y_val])
Y_test = np.array([aaaa[num] for num in Y_test])

new_Y_train = []
for ans in Y_train:
    vec = [0,0,0,0,0,0,0,0,0,0]
    vec[ans.get()] += 1
    new_Y_train.append(vec)
Y_train = np.array(new_Y_train)

new_Y_val = []
for ans in Y_val:
    vec = [0,0,0,0,0,0,0,0,0,0]
    vec[ans.get()] += 1
    new_Y_val.append(vec)
Y_val = np.array(new_Y_val)

new_Y_test = []
for ans in Y_test:
    vec = [0,0,0,0,0,0,0,0,0,0]
    vec[ans.get()] += 1
    new_Y_test.append(vec)
Y_test = np.array(new_Y_test)


# Configs que deram certo:
# 30, 25, 20, 10, lr=1, epochs=10_000, 97% depois de 2 fits
# 30, 25, 20, 15, 10, lr=1, epochs=10_000, 96.15% depois de 1 fit (melhor rate parece)
# 20, 20, 20, 10, lr=1, epochs=10_000, 95.8% depois de 1 fit
# 35, 30, 25, 20, 15, 10, lr=1, epochs=10_000, 94.39% depois de 1 fit (Parece que melhora com mais treino)
# 35, 30, 20, 20, 15, 10, lr=1, epochs=10_000, 94.39% depoisa de 1 fit (Parece que melhora com mais treino)
nn1 = Sequential(784)
nn1.add_layer(Dense(30))
nn1.add_layer(Dense(25))
nn1.add_layer(Dense(10))
nn1.compile(erro_quadratico, d_erro_quadratico)


nn1.weights, nn1.bias = nn1.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val, epochs=50_000, lr=1, stop_after=1000)


vp1, vn1, fp1, fn1 = [], [], [], []

for c in range(Y_test.shape[1]):
    vp1.append(0)
    vn1.append(0)
    fp1.append(0)
    fn1.append(0)

output_nn1 = nn1.predict(X_test)
prediction = []
for kick in output_nn1:
    prediction.append([0,0,0,0,0,0,0,0,0,0])
    prediction[-1][kick.argmax().get()]+=1 

for predict_nn1, expected in zip(prediction, Y_test):
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



vn1 = np.array(vn1)
vp1 = np.array(vp1)
fn1 = np.array(fn1)
fp1 = np.array(fp1) 


precision1 = vp1/(vp1+fp1)
recall1 = vp1/(vp1+fn1)
accuracy1 = sum(vp1)/sum(vp1 + fn1)


a = nn1.predict(X_train)


print('Precision: ', precision1)
print('Recall: ', recall1)
print('Accuracy: ', accuracy1)


a = [max(b) for b in a] 


a


