import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import Sequential

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV


X = np.loadtxt('input_10D.txt')
y = np.loadtxt('labels.txt')
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Function to create model
def create_model(units=128, activation="relu", lr=0.01, opt=Adam):

    model = Sequential()
    model.add(Dense(units=units, input_dim=10, activation=activation))
    model.add(Dense(units=(units/2), activation=activation))
    model.add(Dense(units=(units/4), activation=activation))
    model.add(Dense(units=(units/8), activation=activation))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer=opt(learning_rate=lr),
                  metrics=["accuracy"])
    return model


model = KerasClassifier(build_fn=create_model)
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', model)]
)

params = {
    'model__units': [256, 128, 64, 32],
    'model__batch_size': [256, 128, 64, 32],
    'model__epochs': [40, 50, 60, 70, 80, 90, 100],
    'model__activation': ['relu', 'elu', 'selu', 'sigmoid'],
    'model__lr': [0.0001, 0.001, 0.01, 0.1]
}

rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=params,
    cv=5,
    n_iter=10,
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

rs.fit(X_train, y_train)

print("Best training accuracy score:", rs.best_score_)
print("Best model parameters:", rs.best_params_)
print("Test accuracy score:", rs.score(X_test, y_test))