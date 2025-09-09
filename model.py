import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Data Cleaning and Preprocessing
# Drop customerID
df = df.drop('customerID', axis=1)

# Handle missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert binary categorical variables
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

# One-hot encode other categorical variables
df = pd.get_dummies(df, drop_first=True)

# 3. Define Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
# We use RandomForestClassifier, a powerful and popular model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the Model (Optional but good practice)
preds = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, preds):.4f}")

# 7. Save the Trained Model
joblib.dump(model, 'churn_model.pkl')
print("Model trained and saved as churn_model.pkl")