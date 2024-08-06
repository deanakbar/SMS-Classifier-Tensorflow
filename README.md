# SMS-Classifier-Tensorflow

This project demonstrates how to build an SMS spam classifier using TensorFlow. The classifier identifies whether an SMS message is spam (unwanted) or not spam (legitimate), and demonstrates how to preprocess text data, build a neural network model, and train it to classify SMS messages as spam or ham using TensorFlow.

## Project Setup

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)
```

### 2.Load and Preprocess Data 

```import pandas as pd

df = pd.read_csv("/content/gdrive/MyDrive/SMS Classifier/spam.csv", encoding='ISO-8859-1')
df = df[['v1', 'v2']]  # Select relevant columns

# Check for null data
df.info()

# Encode labels
mapping_category = {
    'ham': 0,
    'spam': 1
}
df['v1'] = df['v1'].apply(lambda x: mapping_category.get(x, -1))

# Remove duplicates
df = df.drop_duplicates(keep='first')

# Clean text data
import string
import nltk

nltk.download('punkt')

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['v2'] = df['v2'].apply(remove_punctuation)
df['v2'] = df['v2'].str.lower()

# Tokenize and count unique words
from nltk.tokenize import word_tokenize
from collections import Counter

all_text = ' '.join(df['v2'])
words = word_tokenize(all_text.lower())
unique_words = set(words)
total_unique_words = len(unique_words)

print(f"Total unique words: {total_unique_words}")
longest_word = max(words, key=len)
print(f"Longest word: {longest_word}")

def count_words(sentence):
    return len(word_tokenize(sentence))

longest_sentence = ""
max_word_count = 0

for sentence in df['v2']:
    word_count = count_words(sentence)
    if word_count > max_word_count:
        longest_sentence = sentence
        max_word_count = word_count

print(f"The longest sentence is: \"{longest_sentence}\"")
print(f"Number of words in the longest sentence: {max_word_count}")
```
### 3.Tokenize Text Data
```
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Tokenize text data
max_words_name = 10000
max_len_name = 200

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)

# Tokenize training data
name_tokenizer = Tokenizer(num_words=max_words_name, oov_token='<OOV>')
name_tokenizer.fit_on_texts(X_train)
name_sequences_train = name_tokenizer.texts_to_sequences(X_train)
name_padded_train = pad_sequences(name_sequences_train, maxlen=max_len_name, padding='post')

# Tokenize testing data
name_sequences_test = name_tokenizer.texts_to_sequences(X_test)
name_padded_test = pad_sequences(name_sequences_test, maxlen=max_len_name, padding='post')

word_index = name_tokenizer.word_index
VOCAB_SIZE = len(word_index)
```
