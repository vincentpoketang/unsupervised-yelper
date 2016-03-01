import pickle
import random
import re

def lemmatize_docs(random_seed, number_of_docs):
    print("Start lemmatization...")
    
    from nltk.stem.wordnet import WordNetLemmatizer
    
    random.seed(random_seed)
    
    raw_data = pickle.load(open("pickles/list-of-reviews.p", "rb"))
    documents = random.sample(raw_data, number_of_docs)

    lemmatized_docs = []
    
    wnl = WordNetLemmatizer()
    for i in documents:
        tokens = re.sub("(^ )|( $)+", "", re.sub("(\s|\.|\?|,|;|:)+", " ", i.lower())).split(" ")
        lemmatized_doc = ""
        for j in tokens:
            lemmatized_doc += wnl.lemmatize(j) + " "
        lemmatized_docs.append(lemmatized_doc)
    
    pickle.dump(lemmatized_docs, open("pickles/lemmatized-docs.p", "wb"))
    print("Lemmatization complete.")
    
    return lemmatized_docs

if __name__ == "__main__":
    lemmatize_docs(1532525625823, 30000)