# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:20:15 2016

@author: Dooshkukakoo
"""

import pickle

raw_data = pickle.load(open("list-of-reviews.p", "rb"))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

count_vect = CountVectorizer(stop_words=set(stopwords.words('english')))
train_counts = count_vect.fit_transform(raw_data[:20])
test_counts = count_vect.transform(raw_data[500:520])

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
import matplotlib.pyplot as plt

from sklearn import decomposition

num_topics = 5
num_top_words = 3
docnames = []
for review in raw_data[:20]:
    docnames.append(review[:3])
nmf = decomposition.NMF(n_components=num_topics, random_state=1)

# this next step may take some time
doctopic = nmf.fit_transform(dtm)

# print words associated with topics
topic_words = []

for topic in nmf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    print(np.argsort(topic)[::-1])
    topic_words.append([vocab[i] for i in word_idx])

doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

print(topic_words)
print(word_idx)

print()
print()
for t in range(len(topic_words)):
   print("Topic {}: {}".format(topic_words[t][:3], ' '.join(topic_words[t][:3])))
   

print (nmf.transform(dtm_test))

N, K = doctopic.shape  # N documents, K topics

ind = np.arange(N)  # the x-axis locations for the novels

width = 0.5  # the width of the bars

plots = []

height_cumulative = np.zeros(N)

for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if k == 0:
        p = plt.bar(ind, doctopic[:, k], width, color=color)
    else:
        p = plt.bar(ind, doctopic[:, k], width, bottom=height_cumulative, color=color)
    height_cumulative += doctopic[:, k]
    plots.append(p)

plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1

plt.ylabel('Topics')


plt.title('Topics in businesses')


plt.xticks(ind+width/2, docnames)

plt.yticks(np.arange(0, 1, 10))

topic_labels = ['Topic {}'.format(topic_words[k][:3]) for k in range(K)]

# see http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend for details
# on making a legend in matplotlib
plt.legend([p[0] for p in plots], topic_labels)

plt.show()

#for review in range(20):
#    counts = np.zeroes(num_topics)
#    for 
#    
#num_top_words = 3
#
#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
#
#for t in range(num_topics):
#    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
#    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
#    plt.xticks([])  # remove x-axis markings ('ticks')
#    plt.yticks([]) # remove y-axis markings ('ticks')
#    plt.title('Topic #{}'.format(t))
#    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
#    top_words_idx = top_words_idx[:num_top_words]
#    top_words = mallet_vocab[top_words_idx]
#    top_words_shares = word_topic[top_words_idx, t]
#    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
#        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base*share)
#
#plt.tight_layout()