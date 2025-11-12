import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the data from a CSV file
df = pd.read_csv(r"C:\Users\Shreelakshmi G Bhat\Desktop\IMP Asorted stuffs\Internships\Internship1-Neviton\hotel_review_rating\hotel_review_rating\tripadvisor_hotel_reviews.csv")

# Display the first few rows of the DataFrame
print(df.head())

# Extract reviews and ratings using iloc
X = df.iloc[:, 0]  # Assuming the first column contains the reviews
y = df.iloc[:, 1]  # Assuming the second column contains the ratings

# Vectorize the text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Function to predict rating for a new review
def predict_rating(review):
    review_vectorized = vectorizer.transform([review])
    predicted_rating = model.predict(review_vectorized)
    return predicted_rating[0]

# Example usage
new_review = input("Enter review:")
predicted_rating = predict_rating(new_review)
print(f'Predicted Rating: {predicted_rating}')
