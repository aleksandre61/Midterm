 task5
import numpy as np
import re
from sklearn.linear_model import LogisticRegression

spam_data = np.genfromtxt('spam-data.csv', delimiter=',', skip_header=1)
X = spam_data[:, :-1]  # Features
y = spam_data[:, -1]  # Labels (0 for ham, 1 for spam)

# logistic regression model building, training
model = LogisticRegression()
model.fit(X, y)

# extracting email features
def extract_email_features(email_text):
    num_words = len(email_text.split())
    num_links = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text))
    num_capitalized_words = len(re.findall(r'\b[A-Z][A-Z]+\b', email_text))
    spam_words = ['win', 'prize', 'lottery', 'offer', 'discount', 'free', 'promotion', 'opportunity']
    num_spam_words = sum(word.lower() in email_text.lower() for word in spam_words)
    return [num_words, num_links, num_capitalized_words, num_spam_words]

# Checking emails for spam
with open('emails.txt', 'r') as f:
    email_contents = f.read().split('\n\n')

for email in email_contents:
    if email:
        email_features = extract_email_features(email)
        prediction = model.predict([email_features])
        if prediction[0] == 0:
            print(f"Email: {email[:50]}...\nClassification: Ham (non-spam)\n")
        else:
            print(f"Email: {email[:50]}...\nClassification: Spam\n")

# searching the feature importances
feature_importances = model.coef_[0]
feature_names = ['Number of Words', 'Number of Links', 'Number of Capitalized Words', 'Number of Spam Words']

# to print the feature importances
for name, importance in zip(feature_names, feature_importances):
    print(f"{name}: {importance:.2f}")
