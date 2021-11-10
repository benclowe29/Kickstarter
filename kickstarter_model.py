import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import spacy

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.layers import Dense, Embedding, Input, Bidirectional, LSTM, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from xgboost import XGBClassifier

# Load first file
df = pd.read_csv('data/Kickstarter.csv')

# Loop through 8 files and concat to original DF
for i in range(8):
  df_ = pd.read_csv(f'data/Kickstarter00{i+1}.csv')
  df = pd.concat([df, df_])


def clean_data(df):

  # Removing duplicate entries then set 'id' as index
  df.drop_duplicates(subset='id', inplace=True)
  df.set_index('id', inplace=True)

  # Drop columns with 99% null values
  df.drop(columns=['friends', 'is_backing', 'is_starred', 'permissions'], inplace=True)

  # Drop rows where state is not 'successful' or 'failed'.  We are looking at binary outcomes
  df = df[(df['state'] == 'successful')|(df['state'] == 'failed')]

  # Dropping high cardinality, redundant, and uninteresting columns
  df = df.drop(columns=['country_displayable_name', 'creator', 'currency_symbol', 'name', 'photo', 'profile', 'source_url', 'urls', 'usd_type'])

  # Dropping columns with only 1 unique value
  df = df.drop(columns=['disable_communication', 'is_starrable'])

  # Dropping leaky columns and currency exchange columns
  df = df.drop(columns=['converted_pledged_amount', 'currency', 'currency_trailing_code', 'current_currency', 'fx_rate', 'pledged', 'static_usd_rate', 'usd_exchange_rate', 'usd_pledged'])

  # Creating 'campaign_length' feature
  df['campaign_length'] = df['deadline'] - df['launched_at']

  # Dropping columns which can't be tinkered by user
  df.drop(columns=['country', 'created_at', 'deadline', 'launched_at', 'state_changed_at', 'spotlight', 'location', 'slug', 'backers_count'], inplace=True)

  # Pull the category names out and store in a list
  dict_list = []
  for entry in df['category']:
    category = json.loads(entry)
    dict_list.append(category['name'])

  # Create new category column with just the category and not dictionaries
  df['cat'] = dict_list

  # Drop old category
  df.drop(columns='category', inplace=True)

  # Create 'word_count' feature
  description_lengths = [len(description.split()) for description in df['blurb']]
  df['word_count'] = description_lengths

  # Make 'staff_pick' column integers
  df['staff_pick'] = df['staff_pick'].astype('int64')

  # Re-order columns
  df = df[['blurb', 'cat', 'word_count', 'campaign_length', 'goal', 'staff_pick', 'state']]
  
  return df


df = clean_data(df)

cat_dict = {}
for i, cat in enumerate(df['cat'].unique()):
    cat_dict[cat] = i
df['cat'] = df['cat'].map(cat_dict)

# Pull out target variable
y = df['state']

# Convert target variable to numeric labels
y = y.map({'successful': 1, 'failed': 0})

# Creating Feature Matrix by dropping target variable
X = df.drop(columns='state')


def clean_text(text):
    """
    Accepts a single text document and performs several regex substitutions in order to clean the document. 
    
    Parameters
    ----------
    text: string or object 
    
    Returns
    -------
    text: string or object
    """
    
    # order of operations - apply the expression from top to bottom
    non_alpha = '[^a-zA-Z]'
    multi_white_spaces = "[ ]{2,}"
    single_letter_words = '(\s[a-zA-Z]\s)'
    
    text = re.sub(non_alpha, ' ', text)
    text = re.sub(single_letter_words, ' ', text)
    text = re.sub(single_letter_words, ' ', text)
    text = re.sub(multi_white_spaces, " ", text)
    
    
    # apply case normalization 
    return text.lower().lstrip().rstrip()


def tokenize(document):
    """
    Takes a doc and returns a string of lemmas after removing stop words.
    """
    
    doc = nlp(document)
    
    tokens = [token.lemma_.strip() for token in doc if (token.is_stop != True) and (token.is_punct != True) and (len(token) > 2)]
    return ' '.join(tokens)


X_clean = [clean_text(text) for text in X['blurb']]
nlp = spacy.load("en_core_web_md")
X_token = [tokenize(text) for text in X_clean]


def get_word_vectors(docs):
    """
    This serves as both our tokenizer and vectorizer. 
    Returns a list of word vectors, i.e. our doc-term matrix
    """
    return [nlp(doc).vector for doc in docs.split()]


X_vect = []
for i, text in enumerate(X_token):
  X_vect.append(get_word_vectors(text))
  if i % 100 == 0:
    print(i)

X_vect = np.array(X_vect)
X_vect_np = []
for arr in X_vect:
    X_vect_np.append(np.array(arr))

X_vect_np = np.array(X_vect_np)



pca = PCA(n_components=5)
X_vect_5D = pca.fit_transform(X_vect_test)

X_vect_5D.shape

X_vect_5D_test = []
for arr in X_vect_5D:
    X_vect_5D_test.append(np.expand_dims(arr, axis=1))

X_vect_5D_test = np.array(X_vect_5D_test)

X_vect_5D_test.shape

X_meta = df.drop(columns=['blurb', 'state'])

X_meta.shape

X_meta = np.array(X_meta)

X_meta_test = []
for arr in X_meta:
    X_meta_test.append(np.expand_dims(arr, axis=1))

X_meta_test = np.array(X_meta_test)

X_meta_test.shape

early_stop = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=8,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


nlp_input = Input(shape=(3, 1))
meta_input = Input(shape=(5, 1))

forward_layer = LSTM(128, return_sequences=True)
backward_layer = LSTM(128, activation='relu', return_sequences=True, go_backwards=True)

nlp_out = Bidirectional(forward_layer, backward_layer=backward_layer)(nlp_input)
meta_out = Dense(256, input_dim=5, activation='relu')(meta_input)

concat = concatenate([nlp_out, meta_out], axis=1)
classifier = Dense(128, activation='relu')(concat)
classifier = Dense(64, activation='relu')(classifier)
classifier = Dense(32, activation='relu')(classifier)
classifier = Dense(16, activation='relu')(classifier)
classifier = Dense(8, activation='relu')(classifier)

output = Dense(1, activation='sigmoid')(classifier)

model = Model(inputs=[nlp_input, meta_input], outputs=output)

model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x=[X_vect_5D_test, X_test_scaled],
    y=y,
    validation_split=0.2,
    shuffle=True,
    batch_size=8,
    epochs=50,
    class_weight={0: 0.34, 1: 0.66},
    workers=-1,
    callbacks=early_stop
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

model2 = Sequential()

model2.add(Dense(128, input_dim=5, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(16, activation='relu'))
model2.add(Dense(8, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model2.fit(
    x=X_test_2,
    y=y,
    validation_split=0.2,
    shuffle=True,
    batch_size=8,
    epochs=50,
    class_weight={0: 0.34, 1: 0.66},
    workers=-1,
    callbacks=early_stop
)

X_test_ = X.drop(columns='blurb')

X_test_.head(2)

X_test_['campaign_length'] = (X_test_['campaign_length'] - X_test_['campaign_length'].min()) / (X_test_['campaign_length'].max() - X_test_['campaign_length'].min())

X_test_['cat'] = (X_test_['cat'] - X_test_['cat'].min()) / (X_test_['cat'].max() - X_test_['cat'].min())

X_test_['word_count'] = (X_test_['word_count'] - X_test_['word_count'].min()) / (X_test_['word_count'].max() - X_test_['word_count'].min())

X_test_['goal'] = (X_test_['goal'] - X_test_['goal'].min()) / (X_test_['goal'].max() - X_test_['goal'].min())

X_test_ = np.array(X_test_)

X_test_.shape

X_test_scaled = []
for arr in X_test_:
    X_test_scaled.append(np.expand_dims(arr, axis=1))

X_test_scaled = np.array(X_test_scaled)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=8,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

model3 = Sequential()

# model3.add(Dense(512, input_dim=5, activation='relu'))
# model3.add(Dense(256, activation='relu'))
# model3.add(Dense(128, activation='relu'))
model3.add(Dense(64, activation='relu', input_dim=5))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(16, activation='relu'))
model3.add(Dense(8, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model3.fit(
    x=X_meta,
    y=y,
    validation_split=0.2,
    shuffle=True,
    batch_size=16,
    epochs=50,
    class_weight={0: 0.34, 1: 0.66},
    workers=-1,
    callbacks=early_stop
)

def create_model(units=128, activation= "relu", lr=0.001, opt=Adam):

  model = Sequential()

  model.add(Dense(units=units, input_dim=5, activation=activation))
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
    ('scaler', StandardScaler()),
    ('model', model)]
)

params = {
    'model__units': [256, 128, 64, 32],
    'model__batch_size': [256, 128, 64, 32],
    'model__epochs': [40, 50, 60, 70, 80, 90, 100],
    'model__activation': ['relu', 'elu', 'selu', 'sigmoid'],
    'model__lr': [0.0001, 0.001, 0.01]
}

train_X, test_X, train_y, test_y = train_test_split(X_test_, y)

rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=params,
    cv=5,
    n_iter=10,
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

rs.fit(train_X, train_y)

rs.best_score_

rs.best_params_

rs.score(test_X, test_y)

pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        random_state=42,
        n_jobs=-1,
        verbosity=1
    ))]
)

params2 = {
    'xgb__n_estimators': range(30, 75, 10),
    'xgb__learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'xgb__max_depth': range(20, 39, 1),
    'xgb__min_child_weight': range(4, 9, 1),
    'xgb__eta': np.arange(0.1, 0.5, 0.1),
    'xgb__subsample': range(0, 3, 1),
    'xgb__colsample_bytree': range(0, 3, 1)
}

rs_xgb = RandomizedSearchCV(
    estimator=pipe2,
    param_distributions=params2,
    cv=5,
    n_iter=100,
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

rs_xgb.fit(train_X, train_y)

rs_xgb.best_score_

rs_xgb.best_params_

rs_xgb.score(test_X, test_y)