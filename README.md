
# Implementación de una Red Neuronal de 2 capas utilizando Ajuste Sináptico Adaptativo

 Vamos a utilizar únicamente la libreria numpy que nos permite utilizar funciones matemáticas para operar con vectores y matrices.
```
import numpy as np
```

Elegimos para esta implementación la funcion de activación sigmoide ya que está acotada, es facilmente derivable y es monotona.

```
def sigmoide(x, deriv=False):   
    if deriv:
        return x*(1-x)       
    else:
        return 1/(1+np.exp(-x))  #su deridava. Esto nos va a dar la pendiente para poder minimizar el coste total, el error.
```
Definimos la función f(x) para aplicar el Ajuste Sináptico Adaptativo. La variable `cache` ahora almacena la suma de los cuadrados de las entradas, y la variable `epsilon` se utiliza para evitar la división por cero. Dentro de la función `f()`, actualizamos la cache con las entradas al cuadrado y calculamos la tasa de aprendizaje ajustada en función de la cache y los hiperparámetros. Finalmente, la matriz de entrada `x` se escala mediante la tasa de aprendizaje ajustada y se devuelve. Esta matriz de entrada escalada se utiliza para actualizar los pesos en el bucle principal.
```
# Hiperparametros  
learning_rate = 2.5  
epsilon = 1e-8. 

def f(x, cache, epsilon):  

    cache += x**2  # suma de los cuadrados del input
  
    # Calculo de tasa de aprendizaje ajustada 
    adjusted_learning_rate = learning_rate / (np.sqrt(cache) + epsilon)  
  
    # Aplico ASA  
    return x * adjusted_learning_rate
```
Datos para entrenar la red neuronal
Definimos 2 vectores, uno con datos de entrada y otro los de salida, es decir los resultados esperados. 
```
# Vectores de entrada
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
# Valores de salida
y = np.array([[0,0,1,1]]).T  #T hace referencia a matriz traspuesta. Esto nos servirá para poder realizar operaciones matemáticas 
```

La libreria cuenta con un generador de numeros pseudoaleatorios.
Inicializamos el generador con una semilla establecida para
realizar el calculo.
```
np.random.seed(1)
```
`pesos0` será la primer capa de pesos que conecta la capa 0 con la capa 1 

`pesos0` debe ser un array de dimension (3,1) ya que estos pesos deben ser multiplicados 

por la misma cantidad de entradas, con valores al azar con promedio 0. 

Iniciamos los pesos 0 con valores aleatorios centrados alrededor de 0 para evitar sesgos.
```
pesos0 = 2*np.random.random((3,1)) - 1
```
Creamos una matriz del mismo tamaño y tipo de datos que la matriz de entrada X, pero con todos los elementos inicializados en cero.
```
cache = np.zeros_like(X)
```
Defino una tasa de aprendizaje inicial que luego será modificada bajo el modelo ASA. El valor definido para Epsilon es para evitar la división por 0 en el calculo de la tasa de aprendizaje.
```
learning_rate = 2.5
epsilon = 1e-8
```
Creamos tambíen una lista vacía para luego ir agregando los errores en cada iteración para visualizarlo luego.
```
errors = []
```

```
# Iteramos 500 veces
for iter in range(500):
  
    capa0 = X
    capa1 = sigmoide(np.dot(capa0, pesos0))  #dot permite hacer producto interno entre los datos de entrada y sus pesos

    # Cuanto error se ha cometido?
    capa1_error = y - capa1   # por cada iteración calculo el error cometido, resto el valor de salida deseado con el obtenido activado
    errors.append(np.mean(np.abs(capa1_error))) # Calculo y almaceno el error en cada iteración
    
    # Multiplicamos el error por la pendiende de la funcion sigmoide en los valores de capa1, para poder ir disminuyendo el error. 
    # La función de perdida MSE se encuentra implicita en el calculo 
    capa1_delta = capa1_error * sigmoide(capa1, True)

    
    # Computamos la funcion f(X) para el input matriz X
    f_X = f(capa0, cache, epsilon)
    # Actualizamos los pesos0
    pesos0 += np.dot(f_X.T, capa1_delta)
   
```
Imprimimos resultado:
```
print ("Restulado esperado:")
print (y)
print ("Restulado luego del entrenamiento:")
print (capa1)
```

<img width="403" alt="Captura de pantalla 2023-05-02 a la(s) 16 38 15" src="https://user-images.githubusercontent.com/24212744/235770213-adf538f6-e4a2-42ba-a829-1a747d00807c.png">

Resultado en plot que muestra como el error absoluto tiende a disminuir con cada iteración:

```
plt.plot(errors)
plt.title('Error')
plt.xlabel('Iteration')
plt.ylabel('Mean Absolute Error')
plt.show()

```
<img width="612" alt="Captura de pantalla 2023-05-02 a la(s) 21 06 24" src="https://user-images.githubusercontent.com/24212744/235810524-a98b1ecc-7799-4527-957e-8b48ce4e3273.png">

