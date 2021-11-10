import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

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


# Cleaning our dataframe
df = clean_data(df)
print(df.head(5))

# Encoding 'categories'
cat_dict = {}
for i, cat in enumerate(df['cat'].unique()):
    cat_dict[cat] = i

df['cat'] = df['cat'].map(cat_dict)

# Pull out target variable
y = df['state']

# Establish baseline accuracy of 66.5% 'successful'
baseline_accuracy = y.value_counts(normalize=True)[0]
print('Baseline Accuracy:', baseline_accuracy)

# Convert target variable to numeric labels
y = y.map({'successful': 1, 'failed': 0})

# Creating Feature Matrix by dropping target variable
X = df.drop(columns='state')

### BUILD RF MODEL

# dropping text blurb for RF model
X_rf = X.drop(columns=['blurb'])

# split data
X_train, X_test, y_train, y_test = train_test_split(X_rf, y, test_size=0.2, random_state=42)

# Instantiate RF model with parameters already tuned
model = RandomForestClassifier(
          random_state=42,
          n_estimators=140,
          class_weight={0:0.335, 1:0.665},
          max_depth=20,
          max_features=5,
          min_samples_leaf=5,
          min_samples_split=7
)

# Fit model to training data
model.fit(X_train, y_train)
print("Training Accuracy: ", model.score(X_train, y_train))
print("Test Accuracy: ", model.score(X_test, y_test))

# Plot confusion matrix
print(plot_confusion_matrix(model, X_test, y_test, values_format = '.0f', display_labels=['failure','success']))

# Print classification report
print(classification_report(y_test, model.predict(X_test), target_names = ['failure','success']))

# Display feature importances
importances = model.feature_importances_
features = X_train.columns
print(pd.Series(importances, index=features).sort_values().tail(10).plot(kind='barh'))