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


                                         
