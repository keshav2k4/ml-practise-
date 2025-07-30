import numpy as np 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#load the dataset
data = np.genfromtxt('/Users/keshavpurohit/Desktop/keshav/keshav /data sceince practise/practise 1/custmores.csv', delimiter=',', skip_header=1)
X = data[:, [0, 1]]  # Assuming 'Age' is in column 0 and 'Salary' is in column 1
y = data[:, 2]  # Assuming 'Purchased' is in column 2       

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

#train the model using decision tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train , y_train)

#make  predictions on test set 
y_pred= model.predict(X_test)

#check accuracy of the model
acc  = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc * 100:.2f}%")

#predictions and classification report