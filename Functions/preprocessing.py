from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import re
import nltk 

# download Punkt Sentence Tokenizer
nltk.download('punkt')
# download stopwords
nltk.download('stopwords')

english_stopwords = stopwords.words('english')
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens]
    text = ' '.join(stemmed)
    text = ' '.join([word for word in text.split() if word not in english_stopwords])
    return text