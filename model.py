import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#load csv file
df = pd.read_csv("IRIS.csv")
df.head()
print(df.head())

#select the dependent and independent variable
X =df[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']]
y= df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#isntantiate the model
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
#fit the model
y_pred = classifier.predict(X_test)
classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("Accuracy :", acc)

#make a pickle file of the model
import pickle
pickle.dump(classifier, open('model.pkl','wb'))