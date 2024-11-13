import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



df = pd.read_csv('emails.csv')

# Split the dataset into features and labels
X = df['text']
y = df['label']

# Create a CountVectorizer to convert text to word vectors
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # Total number of documents
    numWords = len(trainMatrix[0])    # Total number of unique words (features)

    pAbusive = sum(trainCategory) / float(numTrainDocs)  # Probability of spam
    p0Num = np.zeros(numWords)  # Initialize counts for non-spam
    p1Num = np.zeros(numWords)  # Initialize counts for spam
    p0Denom = 0.0  # Total words in non-spam
    p1Denom = 0.0  # Total words in spam

    for i in range(numTrainDocs):  # Loop through each document
        if trainCategory[i] == 1:  # If the document is spam
            p1Num += trainMatrix[i]  # Add word counts for spam
            p1Denom += sum(trainMatrix[i])  # Total words in spam
        else:  # If the document is not spam
            p0Num += trainMatrix[i]  # Add word counts for non-spam
            p0Denom += sum(trainMatrix[i])  # Total words in non-spam

    # Apply Laplace smoothing and convert to log probabilities
    p1Vect = np.log((p1Num + 1) / (p1Denom + numWords))  # Log probabilities for spam
    p0Vect = np.log((p0Num + 1) / (p0Denom + numWords))  # Log probabilities for non-spam

    return p0Vect, p1Vect, pAbusive  # Return word probabilities and spam probability

# Train the model
p0Vect, p1Vect, pAbusive = trainNB0(X_train.toarray(), y_train)

def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Classify * p1Vect) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vect) + np.log(1.0 - pClass1)
    return 1 if p1 > p0 else 0

# Classify a new email
def classify_email(email):
    email_vectorized = vectorizer.transform([email]).toarray()  # Vectorize the new email
    prediction = classifyNB(email_vectorized[0], p0Vect, p1Vect, pAbusive)  # Classify the email
    return "Spam" if prediction == 1 else "Not Spam"

# Example of classifying a new email
new_email = "Claim your free gift now! Don't miss out!"
print(f'The email: "{new_email}" is classified as: {classify_email(new_email)}')
