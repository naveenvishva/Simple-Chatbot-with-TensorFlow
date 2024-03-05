"""import tensorflow as tf
from tensorflow import keras
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import json
import tflearn
import random
from sklearn.preprocessing import LabelEncoder
nltk.download('punkt')


with open('in.json') as file:
    data = json.load(file)
    
words = []
labels = []
docs = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import nltk
import json
import random
from sklearn.preprocessing import LabelEncoder

# Download the required NLTK resource
nltk.download('punkt')

with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs = []
patterns_labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern.lower())
        words.extend(wrds)
        docs.append(wrds)
        patterns_labels.append((wrds, intent['tag']))

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Prepare training data
random.shuffle(patterns_labels)

training_data = []
training_labels = []

for pattern, label in patterns_labels:
    bag = [1 if word in pattern else 0 for word in words]
    training_data.append(bag)
    training_labels.append(labels.index(label))

training_data = np.array(training_data)
training_labels = np.array(training_labels)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Dense(8, input_shape=(len(words),), activation='relu'),
    keras.layers.Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model or load the saved model
load_saved_model = True
if load_saved_model:
    loaded_model = keras.models.load_model('chatbot_model.h5')
else:
    model.fit(training_data, training_labels, epochs=100, batch_size=8)
    # Save the trained model
    model.save('chatbot_model.h5')

# Define a function to generate a response from the chatbot
def generate_response(text):
    tokenized_text = nltk.word_tokenize(text.lower())
    bag_of_words = [1 if word in tokenized_text else 0 for word in words]
    input_data = np.array([bag_of_words])
    predictions = loaded_model.predict(input_data)
    predicted_label_index = np.argmax(predictions)
    predicted_label = labels[predicted_label_index]
    for intent in data['intents']:
        if intent['tag'] == predicted_label:
            response = random.choice(intent['responses'])
            return response

# User input loop
while True:
    user_input = input('You: ')
    response = generate_response(user_input)
    print('Bot:', response)
