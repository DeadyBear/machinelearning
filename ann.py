# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:11:43 2021

@author: Mueez
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf


#importing dataset
import pyodbc
# Some other example server values are
# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port
server = 'VCSSQL02\SQL14'
database = 'SPARS_BI_Temp'
username = 'DEVUser'
password = 'DEVUser'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()



cursor.execute("select D.ItemID, count(D.SalesInvoiceNo)NoofSalesinvoice, sum(S.ShippingCharges) shippingcharges, sum(S.Servicecharges)Servicecharges,sum(S.MerchandiseTotal)MerchandiseTotal, sum(S.TotalAmount)Totalamount,  SUM(ExtPrice)ExtPrice,SUM(InvoicedQty)SOLD, AVG(D.UnitCost) UnitCost,AVG(D.UnitPrice)UnitPrice,d.Discontinued   from BI_SalesInvoiceDetail D inner join BI_SalesInvoice S ON D.SalesInvoiceNo = S.SalesInvoiceNo group by D.ItemID,D.Discontinued order by D.ItemID ")
dataset = cursor.fetchall()
print(dataset[-1])
##dataset = [tuple(x) for x in dataset]
dataset = np.array(dataset)


X= dataset[:,1:-1].astype('float32')
Y = dataset[:,[-1]].astype('float32')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.layers import LSTM
from keras.layers import Dropout

ann = tf.keras.models.Sequential()

ann.add(LSTM(units = 50, return_sequences = True, input_shape = (None, 9)))
ann.add(Dropout(0.2))


ann.add(LSTM(units = 50, return_sequences = True))
ann.add(Dropout(0.2))


ann.add(LSTM(units = 50, return_sequences = True))
ann.add(Dropout(0.2))


ann.add(LSTM(units = 50))
ann.add(Dropout(0.2))


#ann.add(tf.keras.layers.Dense(units =9,activation='relu'))
#ann.add(tf.keras.layers.Dense(units=9,activation='relu'))

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

ann.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics=['accuracy'])
X_train=np.array(X_train)
y_train=np.array(y_train)

ann.fit(X_train,y_train,batch_size=25,epochs=500)

y_pred = ann.predict(X_test)
y_boolean = (y_pred>0.5)
y_new=np.append(y_pred,y_boolean,axis=1)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

print(ann.predict(sc.transform([[48,	3441.75,	177.00,	203897.23,	207515.98,	45440.96,	52,	393.5391,	909.28]])) > 0.5)

# from keras.models import model_from_json

# ann.save("E:\machine learning\P16-Recurrent-Neural-Networks\Part 3 - Recurrent Neural Networks\savedmodel")
  
# from tensorflow import keras  
# model = keras.models.load_model("E:\machine learning\P16-Recurrent-Neural-Networks\Part 3 - Recurrent Neural Networks\savedmodel")
# print(ann.predict(sc.transform([[39,	508.07,	222.00,	4632.76,	5362.83,	431.69,	43,	25.00,	10.3643]])) > 0.5)
          
#  # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)))

#Tunning the ANN
# import sklearn as skl
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense
# def build_classifier(optimizer):
#     classifier = tf.keras.models.Sequential()



















    
#     classifier.add(tf.keras.layers.Dense(units =9,activation='relu'))
#     classifier.add(tf.keras.layers.Dense(units=9,activation='relu'))
#     classifier.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#     classifier.compile(optimizer = optimizer ,loss = 'binary_crossentropy',metrics=['accuracy'])
#     return classifier

# classifier = KerasClassifier(build_fn=build_classifier)
# parameters = {'batch_size' : [25,32],
#               'nb_epoch':[100,500],
#                'optimizer':['adam','rmsprop']
#               }

# grid_search= skl.model_selection.GridSearchCV(estimator = classifier, param_grid=parameters,scoring='accuracy',cv=10)
# grid_search=grid_search(X_train,y_train)
# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_

