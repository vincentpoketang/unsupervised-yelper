"""
set up the "pipeline" so that we have the topic model and then test a 
smaller selection of reviews on it

To get relevant scentences:
> model based on noun phrases
> have a dict mapping the noun phrases to sentences they came from
> then get the top noun phrases for a topic and use their sentences for our 
topic summary
"""


import pickle
import random

raw_data = pickle.load(open("list-of-reviews.p", "rb"))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

count_vect = CountVectorizer(stop_words=set(stopwords.words('english')))
train_counts = count_vect.fit_transform(random.sample(raw_data, 30000))
test_counts = count_vect.transform(random.sample(raw_data, 100))

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
test_tfidf = tfidf_transformer.transform(test_counts)

dtm = train_tfidf
dtm_test = test_tfidf

vocab = count_vect.get_feature_names()

# pickle.dump(dtm, open("term-doc-matrix.p", "wb"))
# dtm = pickle.load(open("term-doc-matrix.p", "rb"))
    
import numpy as np  # a conventional alias
import sklearn.feature_extraction.text as text

from sklearn import decomposition

num_topics = 20
num_top_words = 20

nmf = decomposition.NMF(n_components=num_topics, random_state=1)

# this next step may take some time
doctopic = nmf.fit_transform(dtm)
doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)


# print words associated with topics
topic_words = []
for topic in nmf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])

"""
print(topic_words)
print(word_idx)
"""

print()
print()
for t in range(len(topic_words)):
   print("Topic {}: {}".format(t, ' '.join(topic_words[t][:10])))
   

result = nmf.transform(dtm_test)

"""
novel_names = []

for fn in filenames:
    basename = os.path.basename(fn)
    name, ext = os.path.splitext(basename)
    name = name.rstrip('0123456789')
    novel_names.append(name)

# turn this into an array so we can use NumPy functions
novel_names = np.asarray(novel_names)
doctopic_orig = doctopic.copy()

# use method described in preprocessing section
num_groups = len(set(novel_names))

doctopic_grouped = np.zeros((num_groups, num_topics))

for i, name in enumerate(sorted(set(novel_names))):
    doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)

doctopic = doctopic_grouped
"""
    
"""
throw out short reviews
throw out words used in less than 20 reviews

look for bi-grams and tri-grams that are noun phrases: more accurate for this type of work

use the topics of a document as features of the document and compare it to ratings
"""
