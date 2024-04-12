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

# Extract features from the first email in emails.txt
email_features = [59, 7, 0, 3]  # Manually extracted features: [Number of Words, Number of Links, Number of Capitalized Words, Number of Spam Words]

# Create a DataFrame with the email features and specify the column names
email_df = pd.DataFrame([email_features], columns=['Number of Words', 'Number of Links', 'Number of Capitalized Words', 'Number of Spam Words'])

# Predict if the email is spam or not
prediction = model.predict(email_df)

# Print the prediction result
if prediction[0] == 1:
    print("The email is classified as spam.")
else:
    print("The email is classified as not spam.")