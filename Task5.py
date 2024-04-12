import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the data from spam-data.csv
data = pd.read_csv('spam-data.csv')

# Extract features and target variable
X = data[['Number of Words', 'Number of Links', 'Number of Capitalized Words', 'Number of Spam Words']]
y = data['Class']

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the data
model.fit(X, y)

# Parse emails.txt and extract features
emails = []
with open('emails.txt', 'r') as file:
    email = ''
    for line in file:
        if line.startswith('Subject:'):
            if email:
                emails.append(email)
            email = line
        else:
            email += line
    if email:
        emails.append(email)

# Extract features from emails
email_features = []
for email in emails:
    num_words = len(email.split())
    num_links = email.count('http')
    num_capitalized = sum(1 for word in email.split() if word.isupper())
    num_spam_words = sum(1 for word in email.split() if word.lower() in ['win', 'free', 'gift', 'prize', 'exclusive'])
    email_features.append([num_words, num_links, num_capitalized, num_spam_words])

# Convert email features to DataFrame
email_df = pd.DataFrame(email_features, columns=['Number of Words', 'Number of Links', 'Number of Capitalized Words', 'Number of Spam Words'])

# Predict if emails are spam or not
predictions = model.predict(email_df)

# Print the prediction results
for i, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Email {i+1} is classified as spam.")
    else:
        print(f"Email {i+1} is classified as not spam.")

# Analyze the importance of features
feature_importance = model.coef_[0]
feature_names = X.columns

print("\nFeature Importance:")
for importance, feature in zip(feature_importance, feature_names):
    print(f"{feature}: {importance}")