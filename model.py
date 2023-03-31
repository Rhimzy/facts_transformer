import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import nltk
nltk.download('punkt')

# Load the data from a text file with UTF-8 encoding and error handling
with open("./phy.txt", 'r', encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Tokenize the text into sentences
sentences = nltk.sent_tokenize(text)

# Initialize a dictionary to map facts to their respective sentences
fact_sentences = {}

# Iterate over each sentence and group them into corresponding facts based on punctuation and capitalization
current_fact = ""
for sentence in sentences:
    if sentence[-1] in ["?", ".", "!"]:
        current_fact = sentence
        fact_sentences[current_fact] = []
    else:
        fact_sentences[current_fact].append(sentence)

# Convert the grouped facts into a list of strings
data = [' '.join(sentences) for sentences in fact_sentences.values()]

# Clean the data by removing any leading/trailing white space characters and converting to lowercase
data = [fact.strip().lower() for fact in data]

# Create a CountVectorizer object to convert the text data to a matrix of word counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)

# Create a Multinomial Naive Bayes model and train it on the data
model = MultinomialNB()
model.fit(X, np.arange(len(data)))

# Predict a new fact by converting it to a feature vector and using the trained model to predict the label
new_fact = input('Enter a fact: ')
new_fact_cleaned = new_fact.strip().lower()
new_fact_vectorized = vectorizer.transform([new_fact_cleaned])
predicted_index = model.predict(new_fact_vectorized)[0]

# Print the predicted fact
print('Predicted fact:', data[predicted_index])