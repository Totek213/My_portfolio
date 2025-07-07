from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample training data (text + sentiment label)
training_texts = [
    "I love this product!",
    "This is amazing.",
    "Absolutely fantastic experience.",
    "I hate it.",
    "Terrible service.",
    "Worst thing ever."
]
labels = ["positive", "positive", "positive", "negative", "negative", "negative"]

# Create a pipeline: vectorizer + Naive Bayes classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(training_texts, labels)

# Test the model
test_texts = [
    "I really love it!",
    "This was the worst day.",
    "Fantastic performance.",
    "Not good at all."
]

# Make predictions
predictions = model.predict(test_texts)

# Output the results
for text, sentiment in zip(test_texts, predictions):
    print(f"Text: \"{text}\" => Sentiment: {sentiment}")
