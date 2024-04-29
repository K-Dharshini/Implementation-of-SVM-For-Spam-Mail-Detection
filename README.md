# EX09 - Implementation-of-SVM-For-Spam-Mail-Detection

## Aim:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
   
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import chardet
2. Read the dataset
3. Import SVC from sklearn
4. Fit the data in the model and run the algorithm
   
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: DHARSHINI K
RegisterNumber:  212223230047
*/
import chardet
file = 'spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv("spam.csv", encoding='Windows=1252')

data.head()

data.info()

data.isnull().sum()

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)   
```

## Output:
# result
![image](https://github.com/K-Dharshini/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139334830/782a6e97-0c99-464f-9fd6-e0df993b639a)

# data.head()
![image](https://github.com/K-Dharshini/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139334830/57c94928-5711-4a11-b3f0-4bda1f8e7bd6)

# data.info()
![image](https://github.com/K-Dharshini/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139334830/2e2b6df0-8484-44d3-9e22-38fdfd5898b7)

# data.isnull().sum()
![image](https://github.com/K-Dharshini/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139334830/397edae7-d720-49e9-b459-53c3f08d6012)

# y_pred
![image](https://github.com/K-Dharshini/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139334830/2eb36aaa-e905-49ea-8350-70d5fc708631)

# accuracy
![image](https://github.com/K-Dharshini/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139334830/cf613813-3cd1-4c45-bfaf-58a5b02916b4)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
