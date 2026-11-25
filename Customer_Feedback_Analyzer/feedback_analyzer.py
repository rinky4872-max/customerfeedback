
import pandas as pd
import re
from collections import Counter
from textblob import TextBlob

# --------------------------------
# BLOCK 1: LOAD AND INSPECT DATA
# --------------------------------

df = pd.read_csv(
    r"Customer_Feedback_Analyzer\customer_reviews.csv"
)

print("Original Data:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Check required columns
required_columns = ["review_id", "customer_review", "rating"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Handle missing review text
df["customer_review"] = df["customer_review"].fillna("").astype(str)

# Remove empty reviews
df = df[df["customer_review"].str.strip() != ""].copy()

# --------------------------------
# BLOCK 2: CLEAN REVIEW TEXT
# --------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["cleaned_review"] = df["customer_review"].apply(clean_text)

# --------------------------------
# BLOCK 3: NLP TOKENIZATION
# --------------------------------

stop_words = {
    "the", "is", "was", "were", "a", "an",
    "and", "it", "to", "of", "for", "with",
    "very", "this", "that", "my", "i", "we",
    "our", "have", "has", "had", "yet"
}

def tokenize_text(text):

    words = re.findall(r"\b[a-z]+\b", text)

    # Preserve negation words
    tokens = [
        word for word in words
        if word not in stop_words
    ]

    return tokens


df["tokens"] = df["cleaned_review"].apply(tokenize_text)

# --------------------------------
# BLOCK 4: SENTIMENT ANALYSIS
# --------------------------------

def analyze_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        return "Positive"

    elif polarity < -0.1:
        return "Negative"

    else:
        return "Neutral"


df["sentiment"] = df["cleaned_review"].apply(analyze_sentiment)

# --------------------------------
# BLOCK 5: COMPARE WITH RATINGS
# --------------------------------

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

def rating_sentiment(rating):

    if pd.isna(rating):
        return "Unknown"

    if rating >= 4:
        return "Positive"

    elif rating <= 2:
        return "Negative"

    elif rating == 3:
        return "Neutral"

    return "Unknown"


df["rating_sentiment"] = df["rating"].apply(rating_sentiment)

# --------------------------------
# BLOCK 6: COMPLAINT CATEGORIES
# --------------------------------

def complaint_category(text):

    text = text.lower()

    if any(word in text for word in
           ["delivery", "shipping", "late", "delay"]):
        return "Delivery Issues"

    elif any(word in text for word in
             ["refund", "payment", "money"]):
        return "Refund/Payment Issues"

    elif any(word in text for word in
             ["quality", "broken", "damaged", "defective"]):
        return "Product Quality"

    elif any(word in text for word in
             ["packaging", "package"]):
        return "Packaging"

    elif any(word in text for word in
             ["service", "support", "staff", "customer"]):
        return "Customer Service"

    else:
        return "Other"


df["complaint_category"] = df["customer_review"].apply(
    complaint_category
)

# --------------------------------
# BLOCK 7: SENTIMENT COUNTS
# --------------------------------

sentiment_counts = df["sentiment"].value_counts()

total_reviews = len(df)

print("\nSENTIMENT ANALYSIS")

for sentiment in ["Positive", "Negative", "Neutral"]:

    count = sentiment_counts.get(sentiment, 0)

    percentage = (
        count / total_reviews * 100
        if total_reviews > 0 else 0
    )

    print(f"{sentiment}: {count} ({percentage:.2f}%)")

# --------------------------------
# BLOCK 8: FREQUENT KEYWORDS
# --------------------------------

all_words = []

for tokens in df["tokens"]:
    all_words.extend(tokens)

keyword_counts = Counter(all_words)

print("\nTOP 10 KEYWORDS")

for word, count in keyword_counts.most_common(10):
    print(word, ":", count)

# --------------------------------
# BLOCK 9: COMPLAINT SUMMARY
# --------------------------------

print("\nCOMPLAINT CATEGORIES")

print(df["complaint_category"].value_counts())

# --------------------------------
# BLOCK 10: MANUAL REVIEW CHECK
# --------------------------------

print("\nMANUAL REVIEW CHECK - FIRST 5 REVIEWS")

manual_check = df.head(5)[
    ["customer_review", "rating_sentiment", "sentiment"]
]

print(manual_check.to_string(index=False))

# Compare rating and text sentiment
df["sentiment_match"] = (
    df["sentiment"] == df["rating_sentiment"]
)

print("\nRating and Text Sentiment Agreement:")
print(df["sentiment_match"].value_counts())

# --------------------------------
# BLOCK 11: EXPORT CSV REPORT
# --------------------------------

df.to_csv("analysis_report.csv", index=False)

print("\nAnalysis report exported successfully!")

# --------------------------------
# BLOCK 12: BUSINESS SUMMARY
# --------------------------------

negative_count = sentiment_counts.get("Negative", 0)
positive_count = sentiment_counts.get("Positive", 0)
neutral_count = sentiment_counts.get("Neutral", 0)

negative_percentage = (
    negative_count / total_reviews * 100
    if total_reviews > 0 else 0
)

top_complaints = df["complaint_category"].value_counts()

summary = f"""
CUSTOMER FEEDBACK ANALYSIS SUMMARY

Total Reviews: {total_reviews}

Positive Reviews: {positive_count}
Negative Reviews: {negative_count}
Neutral Reviews: {neutral_count}

Negative Percentage: {negative_percentage:.2f}%

Top Keywords:
{keyword_counts.most_common(10)}

Complaint Categories:
{top_complaints.to_string()}

UNCERTAINTY:
TextBlob uses general-purpose sentiment rules.
It may misunderstand sarcasm, negation, and context.
Star ratings may not always match review sentiment.

MANUAL CHECK:
The first five reviews were displayed for manual
inspection. Verify their predicted sentiment against
the actual meaning before trusting the results.
"""

with open("summary.txt", "w", encoding="utf-8") as file:
    file.write(summary)

print("\nBUSINESS SUMMARY")
print(summary)
print("Summary saved successfully!")
