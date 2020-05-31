
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

df = pd.read_csv('/root/workshop/wines.csv')

y = df['Class']

print(df)
y_cat = pd.get_dummies(y)

X = df.drop('Class' , axis=1)

model  =  Sequential()

# spliting the dataset
x_train, x_test, y_train, y_test = train_test_split(X,y_cat,test_size=0.1,random_state=20)


model.add(Dense(units=64 , input_shape=(13,), 
                activation='relu'))

model.add(Dense(units=3, activation='softmax'))


print(model.summary())

model.compile(optimizer=Adam(),  
              loss='categorical_crossentropy',
             metrics=['accuracy']
             )

ep = 20

model.fit(x_train,y_train, epochs=ep)

accuracy = model.evaluate(x_test, y_test, verbose=0)
accuracy = accuracy[1]*100
print(accuracy)
 
import os
os.system("sudo touch /root/permdata/accuracy.txt")
os.system("echo {} > /root/permdata/accuracy.txt".format(accuracy))


model.save('/root/permdata/multiclassDL.h5')

