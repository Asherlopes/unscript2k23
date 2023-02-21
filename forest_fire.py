#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("df2.csv")
data = np.array(data)

X = data[1:, 0:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('float')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
# log_reg = LogisticRegression()

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# log_reg.fit(X_train, y_train)

inputt=[float(x) for x in "0.275 0.481 0.787 0.0219 0.0375".split(' ')]
final=[np.array(inputt)]

b = classifier.predict(final)
print(b)


pickle.dump(classifier,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


