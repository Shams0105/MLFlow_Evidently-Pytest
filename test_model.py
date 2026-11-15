
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_iris_model():
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def test_model_accuracy():
    model, X_test, y_test = train_iris_model()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.8


def test_prediction_count():
    model, X_test, y_test = train_iris_model()

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_prediction_classes():
    model, X_test, y_test = train_iris_model()

    predictions = model.predict(X_test)

    assert set(predictions).issubset({0, 1, 2})
