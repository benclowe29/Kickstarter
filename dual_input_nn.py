import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import spacy

from sklearn.model_selection import train_test_split

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Embedding, Input, Bidirectional, LSTM, concatenate


# Load first file
df = pd.read_csv('Kickstarter.csv')

# Loop through 8 files and concat to original DF
for i in range(8):
  df_ = pd.read_csv(f'Kickstarter00{i+1}.csv')
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

df.head(2)

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

def get_word_vectors(doc):
    """
    This serves as both our tokenizer and vectorizer. 
    Returns a list of word vectors, i.e. our doc-term matrix
    """
    return nlp(doc).vector

X_vect = []
for i, text in enumerate(X_token):
  X_vect.append(get_word_vectors(text))
  if i % 100 == 0:
    print(i)

X_vect = np.array(X_vect)
print(X_vect.shape)

X_vect_test = []
for arr in X_vect:
    X_vect_test.append(np.expand_dims(arr, axis=1))

X_vect_test = np.array(X_vect_test)
print(X_vect_test.shape)

X_meta = df.drop(columns=['blurb', 'state'])
print("X_meta shape:", X_meta.shape)
X_meta = np.array(X_meta)

X_meta_test = []
for arr in X_meta:
    X_meta_test.append(np.expand_dims(arr, axis=1))
X_meta_test = np.array(X_meta_test)
print("X_meta_test shape:", X_meta_test.shape)


# Build NN
nlp_input = Input(shape=(300, 1))
meta_input = Input(shape=(5, 1))

forward_layer = LSTM(128, return_sequences=True)
backward_layer = LSTM(128, activation='relu', return_sequences=True, go_backwards=True)

meta_out = Bidirectional(forward_layer, backward_layer=backward_layer)(meta_input)
nlp_out = Bidirectional(forward_layer, backward_layer=backward_layer)(nlp_input)

concat = concatenate([nlp_out, meta_out], axis=1)
classifier = Dense(32, activation='relu')(concat)
output = Dense(1, activation='sigmoid')(classifier)

model = Model(inputs=[nlp_input , meta_input], outputs=[output])

model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy'] 
)

model.fit(
    x=[X_vect_test, X_meta_test],
    y=y,
    validation_split=0.2,
    shuffle=True,
    batch_size=32,
    epochs=10,
    class_weight={0: 0.34, 1: 0.66},
    workers=-1,
    
)