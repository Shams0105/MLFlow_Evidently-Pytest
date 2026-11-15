
# Import the Iris dataset from sklearn
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()

# Separate features and target
X = iris.data
y = iris.target

# Display dataset information
print("Dataset loaded successfully!")

print("Features:", iris.feature_names)
print("Target classes:", iris.target_names)

print("Total records:", len(X))
print("Number of features:", X.shape[1])

print("\nFirst 5 records:")
print(X[:5])

print("\nFirst 5 target values:")
print(y[:5])


# STEP 5: Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt

# Convert dataset into a DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)

# Add target column
df["species"] = iris.target_names[y]

# Display first 5 records
print("\nFirst 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Count each flower species
print("\nSpecies Distribution:")
print(df["species"].value_counts())

# Visualize species distribution
df["species"].value_counts().plot(kind="bar")

plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

# STEP 6: Split the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 7: Train the ML model

from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# STEP 7.3: Evaluate the model

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Predict using testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=iris.target_names))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# STEP 8: MLflow Experiment Tracking

import mlflow
import mlflow.sklearn

# Set experiment name
mlflow.set_experiment("Iris_Classification")

# Start MLflow run
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("algorithm", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("test_size", 0.2)

    # Log model accuracy
    mlflow.log_metric("accuracy", accuracy)

    # Log trained model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model"
    )

    print("\nMLflow tracking completed!")
    print("Experiment: Iris_Classification")
    print("Accuracy logged:", accuracy)
