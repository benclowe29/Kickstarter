import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


X = np.loadtxt('input_5D.txt')
y = np.loadtxt('labels.txt', dtype='int')
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create model pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        random_state=42,
        n_jobs=-1,
        verbosity=1
    ))]
)

# List parameters and values to be tuned with RandomizedSearch
params = {
    'xgb__n_estimators': range(20, 55, 5),
    'xgb__learning_rate': [0.15, 0.1, 0.05, 0.01],
    'xgb__max_depth': range(22, 33, 1),
    'xgb__min_child_weight': range(4, 9, 1),
    'xgb__eta': np.arange(0.2, 1.0, 0.1),
    'xgb__subsample': range(0, 3, 1),
    'xgb__colsample_bytree': range(0, 3, 1)
}

# Build model from pipeline for RandomizedSearch
rs_xgb = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=params,
    cv=5,
    n_iter=100,
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

# Fit RandomizedSearch
rs_xgb.fit(X_train, y_train)

# Examine best accuracy score from training
print("Best training accuracy after tuning:", rs_xgb.best_score_)

# Find parameters which produced best score above
print("Optimal parameters:", rs_xgb.best_params_)

# Use best model to make predictions for test data - Test accuracy reported below
print("Test accuracy:", rs_xgb.score(X_test, y_test))
