# task3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load data
data = pd.read_csv('spam-data.csv')

X = data.drop(columns=['Class'])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialize the logistic regression model and train data
model = LogisticRegression()
model.fit(X_train, y_train)

test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Test Accuracy: {test_accuracy:.4f}")

num_words = 45
num_links = 1
num_capitalized = 10
num_spam_words = 0

# Create a feature vector for the test email
test_email_features = [[num_words, num_links, num_capitalized, num_spam_words]]

# Step 4: Test the Email Using the Trained Model
is_spam_prediction = model.predict(test_email_features)[0]

if is_spam_prediction == 1:
    print("The email is predicted to be spam.")
else:
    print("The email is predicted not to be spam.")


    # result:
   #  Model Test Accuracy: 0.9655
# C:\Users\user\anaconda3\Lib\site-packages\sklearn\base.py:439: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
  warnings.warn(
# The email is predicted not to be spam.
