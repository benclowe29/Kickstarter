import pandas as pd
import numpy as np
import json
import re
import spacy
from sklearn.decomposition import PCA

# Load first file
campaign_df = pd.read_csv('data/Kickstarter.csv')

# Loop through 8 files and concat to original DF
for i in range(8):
    df_ = pd.read_csv(f'data/Kickstarter00{i+1}.csv')
    campaign_df = pd.concat([campaign_df, df_])


def clean_data(df):

    # Removing duplicate entries then set 'id' as index
    df.drop_duplicates(subset='id', inplace=True)
    df.set_index('id', inplace=True)

    # Drop columns with 99% null values
    df.drop(columns=['friends', 'is_backing', 'is_starred', 'permissions'], inplace=True)

    # Drop rows where state is not 'successful' or 'failed'.  We are looking at binary outcomes
    df = df[(df['state'] == 'successful') | (df['state'] == 'failed')]

    # Dropping high cardinality, redundant, and uninteresting columns
    df = df.drop(columns=['country_displayable_name', 'creator', 'currency_symbol', 'name', 'photo', 'profile',
                          'source_url', 'urls', 'usd_type'])

    # Dropping columns with only 1 unique value
    df = df.drop(columns=['disable_communication', 'is_starrable'])

    # Dropping leaky columns and currency exchange columns
    df = df.drop(columns=['converted_pledged_amount', 'currency', 'currency_trailing_code', 'current_currency',
                          'fx_rate', 'pledged', 'static_usd_rate', 'usd_exchange_rate', 'usd_pledged'])

    # Creating 'campaign_length' feature
    df['campaign_length'] = df['deadline'] - df['launched_at']

    # Dropping columns which can't be tinkered by user
    df.drop(columns=['country', 'created_at', 'deadline', 'launched_at', 'state_changed_at', 'spotlight', 'location',
                     'slug', 'backers_count'], inplace=True)

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


campaign_df = clean_data(campaign_df)

cat_dict = {}
for i, cat in enumerate(campaign_df['cat'].unique()):
    cat_dict[cat] = i
campaign_df['cat'] = campaign_df['cat'].map(cat_dict)

# Pull out target variable
y = campaign_df['state']

# Convert target variable to numeric labels
y = y.map({'successful': 1, 'failed': 0})

# Creating Feature Matrix by dropping target variable
X = campaign_df.drop(columns='state')
X.shape


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


X_vec = []
for i, text in enumerate(X_token):
    X_vec.append(get_word_vectors(text))
    if i % 100 == 0:
        print(i)
X_vec = np.array(X_vec)

pca = PCA(n_components=5)
X_vec_ = pca.fit_transform(X_vec)

X_meta = campaign_df.drop(columns=['blurb', 'state'])
X_meta = np.array(X_meta)

input_10D = np.concatenate((X_vec_, X_meta), axis=1)

# The below saves processed data into .txt files to be passed around between the different model files as needed
# I have commented them out since those files are provided, but this is how they were created
print("10D input (for NLP + metadata) shape:", input_10D.shape)
# np.savetxt('input_10D.txt', input_10D)

print("5D input (metadata only for XGB and RF) shape:", X_meta.shape)
# np.savetxt('input_5D.txt', X_meta)

print("Label shape:", y.shape)
# np.savetxt('labels.txt', y)
