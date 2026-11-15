
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Divide data into reference and current datasets
reference_data, current_data = train_test_split(
    df,
    test_size=0.5,
    random_state=42
)

# Introduce artificial data drift
current_data = current_data.copy()
current_data["sepal length (cm)"] += 1.0

print("Reference dataset shape:", reference_data.shape)
print("Current dataset shape:", current_data.shape)

print("\nReference data:")
print(reference_data.head())

print("\nCurrent data:")
print(current_data.head())

# STEP 2: Evidently Data Drift Monitoring

from evidently import Report
from evidently.presets import DataDriftPreset

# Create a data drift report
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

# Compare reference and current datasets
result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save report as HTML
result.save_html("data_drift_report.html")

print("\nEvidently monitoring completed!")
print("Report saved as data_drift_report.html")
