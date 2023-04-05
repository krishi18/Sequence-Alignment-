import pandas as pd
import numpy as np
df=pd.read_csv(r'C:\Users\Krishi Thiruppathi\Desktop\Bank_Personal_Loan_Modelling.csv')
print(df.head())
df.drop(['Age','ID','ZIP Code'],axis=1,inplace=True)
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
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
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
import random 

def fitness_function(model, x, y):
    y_pred = model.predict(x)
    return accuracy_score(y, y_pred.round())

def initialize_population(input_size, output_size, hidden_size, n_pop):
    population = []
    for i in range(n_pop):
        model = Sequential([
            Dense(hidden_size, input_shape=(input_size,), activation='relu'),
            Dense(output_size, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        population.append(model)
    return population

def cultural_algorithm(x, y, n_pop, n_gen, p_c, p_m):
    input_size = x.shape[1]
    output_size = 1
    hidden_size = 64
    population = initialize_population(input_size, output_size, hidden_size, n_pop)
    fitness = [fitness_function(model, x, y) for model in population]
    for gen in range(n_gen):
        # Sort the population by fitness
        sorted_pop = [x for _,x in sorted(zip(fitness, population), key=lambda pair: pair[0], reverse=True)]
        sorted_fit = sorted(fitness, reverse=True)

        # Perform cultural transmission
        elite = sorted_pop[0]
        for i in range(1, n_pop):
            if np.random.rand() < p_c:
                population[i].set_weights(elite.get_weights())
        
        # Perform mutation
        for i in range(1, n_pop):
            if np.random.rand() < p_m:
                weights = population[i].get_weights()
                for j in range(len(weights)):
                    for k in range(len(weights[j])):
                        if np.random.rand() < 0.5:
                            weights[j][k] += np.random.normal(0, 0.1)
                        else:
                            weights[j][k] -= np.random.normal(0, 0.1)
                population[i].set_weights(weights)

        # Evaluate the fitness of each individual
        fitness = [fitness_function(model, x, y) for model in population]

    best_model = sorted_pop[0]
    return best_model, fitness_function(best_model, x, y)

# Example usage
from sklearn.datasets import make_classification
x, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

best_model, accuracy = cultural_algorithm(x, y, n_pop=10, n_gen=50, p_c=0.8, p_m=0.1)
print("Accuracy on test set:", accuracy)