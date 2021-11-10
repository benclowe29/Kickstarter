import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV


# Load data
X = np.loadtxt('input_5D.txt')
y = np.loadtxt('labels.txt')

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create pipeline
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('rf', RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    ))]
)

# List parameters and values to be tuned with RandomizedSearch
params = {
    'rf__n_estimators': range(120, 170, 10),
    'rf__max_depth': range(16, 27, 1)
}

# Instantiate RF model with parameters already tuned
rs_rf = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=params,
    cv=5,
    n_iter=100,
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

# Fit model to training data
rs_rf.fit(X_train, y_train)
print("Best training accuracy after tuning:", rs_rf.best_score_)
print("Optimal parameters:", rs_rf.best_params_)
print("Test accuracy:", rs_rf.score(X_test, y_test))

# Plot confusion matrix
y_pred = rs_rf.best_estimator_.predict(X_test)
y_pred = np.array(y_pred, dtype='int')
y_test = np.array(y_test, dtype='int')

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# Print classification report
print("CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=['failure', 'success']))
