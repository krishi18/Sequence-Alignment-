import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from random import randint
import random
from sklearn.metrics import mean_absolute_error as mae
df=pd.read_csv(r'C:\Users\Krishi Thiruppathi\Desktop\Bank_Personal_Loan_Modelling.csv')
df.drop(['Age','ID','ZIP Code'],axis=1,inplace=True)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#%%
import tensorflow.keras as keras

#%%
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(16, input_shape=(10,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.drop(['CreditCard'],axis=1),df['CreditCard'],test_size = 0.1)
model.fit(x_train,y_train,epochs=100)
ans = (model.predict(x_test)>0.5).astype(int)  

#%%

def initial_pop(size_mlp):
    activation = ['identity','logistic', 'tanh', 'relu']
    solver = ['lbfgs','sgd', 'adam']
    popln =  np.array([[random.choice(activation), random.choice(solver), randint(2,100),randint(2,50)]])
    for i in range(0, size_mlp-1):
        popln = np.append(popln, [[random.choice(activation), random.choice(solver), randint(2,50),randint(2,50)]], axis=0)
    return popln

def crossover(par_1, par_2):
    child = [par_1[0], par_2[1], par_1[2], par_2[3]]    
    return child

def mutation(child, prob_mut):
    child_ = np.copy(child)
    for c in range(0, len(child_)):
        if np.random.rand() >= prob_mut:
            k = randint(2,3)
            child_[c,k] = int(child_[c,k]) + randint(1, 4)
    return child_


def fitness_func(popln, x_train, y_train, x_test, y_test): 
    fitness = []
    j = 0
    for w in popln:
        clf = MLPClassifier(learning_rate_init=0.09, activation=w[0], solver = w[1], alpha=1e-5, hidden_layer_sizes=(int(w[2]), int(w[3])),  max_iter=1000, n_iter_no_change=80)

        try:
            clf.fit(x_train, y_train)
            f = accuracy_score(clf.predict(x_test), y_test)
            

            fitness.append([f, clf, w])
        except:
            pass
    return fitness


def genetic_alg(x_train, y_train, x_test, y_test, num_epochs = 10, size_mlp=10, prob_mut=0.8):
    popln = initial_pop(size_mlp)
    fitness = fitness_func(popln, x_train, y_train, x_test, y_test)
    pop_fitness_sort = np.array(list(reversed(sorted(fitness,key=lambda x: x[0]))))

    for j in range(0, num_epochs):
        length = len(pop_fitness_sort)
        parent_1 = pop_fitness_sort[:,2][:length//2]
        parent_2 = pop_fitness_sort[:,2][length//2:]

        child_1 = [crossover(parent_1[i], parent_2[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = [crossover(parent_2[i], parent_1[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = mutation(child_2, prob_mut)
        
        fitness_child_1 = fitness_func(child_1,x_train, y_train, x_test, y_test)
        fitness_child_2 = fitness_func(child_2, x_train, y_train, x_test, y_test)
        pop_fitness_sort = np.concatenate((pop_fitness_sort, fitness_child_1, fitness_child_2))
        sort = np.array(list(reversed(sorted(pop_fitness_sort,key=lambda x: x[0]))))
        
        pop_fitness_sort = sort[0:size_mlp, :]
        fittest = sort[0][1]
        
    return fittest

from sklearn.metrics import accuracy_score
result = genetic_alg(x_train, y_train, x_test, y_test, num_epochs = 10, size_mlp=20, prob_mut=0.9)
print(accuracy_score(result.predict(x_test), y_test))