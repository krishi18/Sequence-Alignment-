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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score 

def ant_colony_optimization(x_train, y_train, x_test, y_test, num_ants=20, num_epochs=10, evap_rate=0.1, alpha=1, beta=1):
    num_features = x_train.shape[1]
    num_classes = len(np.unique(y_train))

    # initialize pheromone matrix
    pheromone = np.ones((num_features, num_classes)) / (num_features * num_classes)

    # initialize best weights and accuracy
    best_weights = None
    best_accuracy = 0

    for epoch in range(num_epochs):
        ant_weights = []
        ant_accuracies = []
        for ant in range(num_ants):
            # initialize weights for this ant
            weights = np.random.uniform(-1, 1, (num_features, num_classes))
            for feature in range(num_features):
                # calculate probability distribution for selecting class
                prob_dist = np.exp(alpha * pheromone[feature]) / np.sum(np.exp(alpha * pheromone[feature]))
                selected_class = np.random.choice(num_classes, p=prob_dist)

                # update weights for this feature based on selected class
                weights[feature, selected_class] += np.random.normal(0, beta)

            # create neural network with current weights
            clf = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', random_state=0)
            clf.coefs_ = [weights]

            # calculate accuracy of neural network on test set
            y_pred = clf.predict(x_test)
            accuracy=print('Accuracy:',accuracy_score(y_test, y_pred))

            # store weights and accuracy for this ant
            ant_weights.append(weights)
            ant_accuracies.append(accuracy)
            if accuracy > best_accuracy:
               best_weights = weights
               best_accuracy = accuracy

       # update pheromone matrix
        for feature in range(num_features):
           for class_ in range(num_classes):
               delta_pheromone = 0
               for ant in range(num_ants):
                   if np.argmax(ant_weights[ant][feature]) == class_:
                       delta_pheromone += ant_accuracies[ant]
               pheromone[feature, class_] = (1 - evap_rate) * pheromone[feature, class_] + delta_pheromone

   # create neural network with best weights and return it
    clf = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', random_state=0)
    clf.coefs_ = [best_weights]
    print(accuracy)


        
