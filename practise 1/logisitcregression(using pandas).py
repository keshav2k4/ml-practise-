import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler #feature scaling done here 
from sklearn.desiciontree #importting the model 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#step1 to load and get info aboout the dataset
df = pd.read_csv('/Users/keshavpurohit/Desktop/keshav/keshav /data sceince practise/practise 1/custmores.csv')
#print(df.head())
#print(df.info())
#print(df.describe())

#STEP 2 data preprocessing 
x = df[['Age' , 'Salary']]
y = df['Purchased']
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#step 3 feature scaling
sc =  StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)
'''print(df.head(x_train_scaled))
print(df.head(x_test_scaled))'''

#step 4 model training 

model = LogisticRegression()

model.fit(x_train_scaled, y_train)
predictions = model.predict(x_test_scaled)
print(predictions)

#print("Weights (coefficients):", model.coef_)
#print("Bias (intercept):", model.intercept_)

#visulization of confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()