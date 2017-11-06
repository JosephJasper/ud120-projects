###Stop words###
from nltk.corpus import stopwords
sw = stopwords.words("english")
print sw[0]
print sw[10]
print len(sw)

###Stemmer###
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
print stemmer.stem("responsiveness")
print stemmer.stem("unresponsive")