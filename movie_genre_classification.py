import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample movie data
movies = {
    'Description': [
        'hero fights villains and saves city',
        'funny comedy scenes and jokes',
        'ghost haunts old house at night',
        'romantic love story between couple',
        'action packed fight sequences'
    ],
    'Genre': [
        'Action',
        'Comedy',
        'Horror',
        'Romance',
        'Action'
    ]
}

df = pd.DataFrame(movies)

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Description'])

y = df['Genre']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Predict genre for a new movie description
new_movie = ["hero saves people with powerful fights"]
new_movie_vector = vectorizer.transform(new_movie)

prediction = model.predict(new_movie_vector)

print("Predicted Genre:", prediction[0])