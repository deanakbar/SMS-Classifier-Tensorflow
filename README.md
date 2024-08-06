<!DOCTYPE html>
<html>
<head>
    <title>SMS Classifier using TensorFlow</title>
</head>
<body>
    <h1>SMS Classifier using TensorFlow</h1>
    <p>This project demonstrates how to build an SMS spam classifier using TensorFlow. The classifier identifies whether an SMS message is spam (unwanted) or not spam (legitimate), and demonstrates how to preprocess text data, build a neural network model, and train it to classify SMS messages as spam or ham using TensorFlow.</p>

    <h2>Project Setup</h2>
    <h3>1. Mount Google Drive</h3>
    <pre><code>from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)
</code></pre>

    <h3>2. Load and Preprocess Data</h3>
    <pre><code>import pandas as pd
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
</code></pre>

    <h3>3. Tokenize Text Data</h3>
    <pre><code>from tensorflow.keras.preprocessing.text import Tokenizer
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
</code></pre>

    <h3>4. Build and Train the Model</h3>
    <pre><code>from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense

# Define model architecture
desc_input = Input(shape=(max_len_name,), name='desc_input')
desc_embedding = Embedding(input_dim=max_words_name, output_dim=64, input_length=max_len_name)(desc_input)
lstm_output = Bidirectional(LSTM(64))(desc_embedding)
output = Dense(1, activation='sigmoid')(lstm_output)

model = Model(inputs=desc_input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(name_padded_train, y_train, epochs=10, batch_size=32, validation_data=(name_padded_test, y_test))
</code></pre>

    <h3>5. Save the Model and Tokenizer</h3>
    <pre><code>import pickle

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(name_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the model architecture
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights('model.weights.h5')
</code></pre>

    <h3>6. Load and Use the Model</h3>
    <pre><code>import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

max_len_desc = 200

# Load model architecture
with open('/content/gdrive/MyDrive/SMS Classifier/after add test data/model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load model weights
loaded_model.load_weights('/content/gdrive/MyDrive/SMS Classifier/after add test data/model.weights.h5')

# Load tokenizer
with open('/content/gdrive/MyDrive/SMS Classifier/after add test data/tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Predict new text
new_text = "Your number won 5000 dollar, go to our website now"
new_text_sequence = loaded_tokenizer.texts_to_sequences([new_text])
new_text_padded = pad_sequences(new_text_sequence, maxlen=max_len_desc, padding='post')

class_category = {
    0: 'Ham',
    1: 'Spam'
}

predictions = loaded_model.predict(new_text_padded)
predicted_class = np.round(predictions).astype(int)[0][0]
predicted_class = class_category[predicted_class]
print(f'For input: "{new_text}" the predicted class is: {predicted_class}')
</code></pre>

</body>
</html>
