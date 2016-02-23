import json
import re

with open('yelp_academic_dataset_review.json') as reviews_file:
    num_reviews = 0
    num_words = 0
    vocab = set()
    
    for line in reviews_file:
        review = json.loads(line)
        review_words = re.sub("(^ )|( $)+", "", re.sub("(\s|\.|\?|,|;|:)+", " ", review["text"].lower())).split(" ")        
        
        num_reviews += 1
        num_words += len(review_words)
        vocab.update(review_words)
        

    print("Reviews: " + str(num_reviews))
    print("Vocabulary Size: " + str(len(vocab)))
    print("Average length: " + str(num_words/num_reviews))
