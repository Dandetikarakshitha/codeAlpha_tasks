# Sentiment Analysis on Amazon Reviews

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load Dataset
df = pd.read_csv("amazon_reviews.csv")

df = df[['reviews.text', 'reviews.rating']]

df.dropna(inplace=True)

df.columns = ['Review', 'Rating']

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply Sentiment Analysis
df['Sentiment'] = df['Review'].apply(get_sentiment)

# Display First 10 Records
print("\nFirst 10 Reviews:")
print(df[['Review', 'Sentiment']].head(10))

print("\nSentiment Distribution:")
print(df['Sentiment'].value_counts())

# Bar Chart
plt.figure(figsize=(6,4))
sns.countplot(x='Sentiment', data=df)
plt.title("Sentiment Distribution of Amazon Reviews")
plt.show()

# Pie Chart
df['Sentiment'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(6,6)
)

plt.title("Sentiment Percentage")
plt.ylabel("")
plt.show()

# Save Output
df.to_csv("sentiment_analysis_output.csv", index=False)

print("\nProject Completed Successfully!")
print("Output saved as sentiment_analysis_output.csv")