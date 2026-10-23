import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
digits = load_digits()

data = pd.DataFrame(digits.data)
data['TARGET CLASS'] = digits.target

data.head()

from sklearn.model_selection import train_test_split
y=data.pop('TARGET CLASS')
x = data
x_train,x_test,Y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)
from sklearn.tree import DecisionTreeClassifier

clf_model = DecisionTreeClassifier()
clf_model.fit(x_train,Y_train)

from sklearn.metrics import accuracy_score
y_pred=clf_model.predict(x_test)
print(accuracy_score(y_test,y_pred))

