# Importar librerias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns


#Importar los datos de los archivos .csv almacenados
test = pd.read_csv('/Users/Cedee/OneDrive/Documentos/Python/test.csv')
train = pd.read_csv('/Users/Cedee/OneDrive/Documentos/Python/train.csv')

print(test.head())
print(train.head())

#-----Entendiendo la data-------
# Cantidad de datos: train (filas: 891, col: 13 ) | test (filas: 418, col: 12 )
print('Cantidad de datos:')
print(train.shape)
print(test.shape)
# Tipos de datos: numericos, flotantes, objetos
print('Tipos de datos:')
print(train.info())
print(test.info())
# Datos faltantes: train (Age: 177, cabin 687, Embarked 2) | test (Age: 86, fare: 1, cabin 327)
print('Datos faltantes:')
print(pd.isnull(train).sum())
print(pd.isnull(test).sum())
# Estadisticas: la media y el valor estandar de ambos archivos son razonables 
# Asumimos que las relaciones deben funcionar para los dos archivos
print('Estadísticas del dataset:')
print(train.describe())
print(test.describe())

#-----Procesamiento de la data-----

#Cambiar de datos tipo objeto a datos numericos
train['Sex'].replace(['female','male'],[0,1],inplace=True)
test['Sex'].replace(['female','male'],[0,1],inplace=True)
train['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)
test['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)

#se calcula la media de la col Age para poder promediar los datos (Se puede descomentar para visualisar las medias)
#print(train["Age"].mean())
#print(test["Age"].mean())
promedio = 30 #promedio entre ambos archivos

#Se le asigna el valor promedio resultada de las medias a los datos faltantes en Age
train['Age'] = train['Age'].replace(np.nan, promedio)
test['Age'] = test['Age'].replace(np.nan, promedio)

#se crean varios grupos de acuerdo a las edades
# rangos: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100
rango = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']
train['Age'] = pd.cut(train['Age'], rango, labels = names)
test['Age'] = pd.cut(test['Age'], rango, labels = names)

#Eliminar columnas y filas que no son necesarias para el analisis (incluyendo datos perdidos)
train.drop(['Cabin'], axis = 1, inplace=True)
test.drop(['Cabin'], axis = 1, inplace=True)
train = train.drop(['PassengerId','Name','Ticket'], axis=1)
test = test.drop(['Name','Ticket'], axis=1)
train.dropna(axis=0, how='any', inplace=True)
test.dropna(axis=0, how='any', inplace=True)

#------------Probando presicion de modelos para la optima solucion-------------
X = np.array(train.drop(['Survived'], 1))
y = np.array(train['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Algoritmos de machine learning (Vectores de soporte) Modelo con mejor precision 
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print('Precisión Soporte de Vectores:')
print(svc.score(X_train, y_train))

#Algoritmos de machine learning (K neighbors) no tiene tanta precision
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print('Precisión K neighbors:')
print(knn.score(X_train, y_train))

#---------- Prediccion usando los modelos---------------------

ids = test['PassengerId']

#Vectores de soporte
prediccion_svc = svc.predict(test.drop('PassengerId', axis=1))
out_svc = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_svc })
print('Predicción Soporte de Vectores:')
print(out_svc.head())

##K neighbors
prediccion_knn = knn.predict(test.drop('PassengerId', axis=1))
out_knn = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_knn })
print('Predicción Vecinos más Cercanos:')
print(out_knn.head())

#------- implementacion de graficos para visualizacion de datos-----------

sns.histplot(data=train,x = 'Age',bins=80,kde=True)
plt.show()

train.hist(bins=20)
plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.countplot(data=train,x='Sex',palette='mako')
plt.subplot(1,2,2)
sns.countplot(data=train,x='Sex',hue='Survived',palette='mako')
plt.tight_layout()
plt.show()
