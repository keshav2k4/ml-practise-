import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df  = pd.read_csv('/Users/keshavpurohit/Desktop/keshav/keshav /data sceince practise/practise 1/custmores.csv')
X = data[['Age', 'Salary']]   # Features
y = data['Purchased']         # Target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: Train Decision Tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)


#step 5: Make predictions
y_pred = model.predict(X_test)

# -----------------------------
# STEP 6: Evaluate the model
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# STEP 7: Visualize the Decision Tree
# -----------------------------
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=['Age', 'Salary'], class_names=['Not Purchased', 'Purchased'], filled=True)
plt.title("Decision Tree Classifier")
plt.show()
