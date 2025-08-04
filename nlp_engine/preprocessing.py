import re
import string
import nltk
import emoji
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from textblob import TextBlob
from contractions import fix as expand_contractions

# Download required resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Custom empathy-preserving stopwords (remove words like “not”, “no” from default stopwords)
safe_stopwords = stop_words - {"no", "not", "don’t", "won’t", "never"}

# Optional offensive words list to be masked
offensive_words = {"damn", "hell", "stupid", "idiot", "moron"}

# POS tagging helper
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def clean_text_(text):
    text = text.lower()
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Expand contractions (e.g., can't → cannot)
    text = expand_contractions(text)
    # Remove URLs, emails, mentions
    text = re.sub(r"http\S+|www\S+|@\S+|\S+@\S+", "", text)
    # Remove emojis and emoticons
    text = emoji.replace_emoji(text, replace='')
    # Remove digits and punctuation
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Spelling correction (lightweight)
    text = str(TextBlob(text).correct())
    # tokenize and POS tagging
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    # Lemmatize with POS and filter
    cleaned_tokens = []
    for word, tag in pos_tags:
        if word in offensive_words:
            cleaned_tokens.append("[masked]")
        elif word not in safe_stopwords:
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            cleaned_tokens.append(lemma)
    # Rejoin
    cleaned_text = " ".join(cleaned_tokens)
    return cleaned_text.strip()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)      # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    return text.strip()

