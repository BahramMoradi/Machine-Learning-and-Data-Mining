# exercise 2.1.4

import numpy as np
from tmgsimple import TmgSimple
"""
# Generate text matrix with help of simple class TmgSimple
tm = TmgSimple(filename='../Data/textDocs.txt', stopwords_filename='../Data/stopWords.txt', stem=True)

# Extract variables representing data
X = tm.get_matrix(sort=True)
attributeNames = tm.get_words(sort=True)

# Display the result
print attributeNames
print X
"""
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.snowball import SnowballStemmer
import nltk
#print ' '.join(SnowballStemmer.languages)
#nltk.download()
stemmer=SnowballStemmer("english",ignore_stopwords=False)
f=open('../Data/textDocs.txt')
docs=f.read()
words=TreebankWordTokenizer().tokenize(docs)
stem_word=[stemmer.stem(word) for word in words]
print stem_word


