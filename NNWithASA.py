import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

def f(x, cache, epsilon):
    cache += x**2
    adjusted_learning_rate = learning_rate / (np.sqrt(cache) + epsilon)
    return x * adjusted_learning_rate

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,0,1,1]]).T
np.random.seed(1)
pesos0 = 2*np.random.random((3,1)) - 1

cache = np.zeros_like(X)

learning_rate = 2.5
epsilon = 1e-8

errors = []

for iter in range(500):
    capa0 = X
    capa1 = sigmoid(np.dot(capa0, pesos0))

    capa1_error = y - capa1
    errors.append(np.mean(np.abs(capa1_error))) #  Calculo y guardo error en cada iteración

    capa1_delta = capa1_error * sigmoid(capa1, True)

    f_X = f(capa0, cache, epsilon)

    pesos0 += np.dot(f_X.T, capa1_delta)

print("Resultado esperado:")
print(y)
print("Resultado luego del entrenamiento:")
print(capa1)

# Plot el error
plt.plot(errors)
plt.title('Error')
plt.xlabel('Iteration')
plt.ylabel('Mean Absolute Error')
plt.show()
