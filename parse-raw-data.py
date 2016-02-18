import json
import pickle
from collections import defaultdict


busi_to_rev = defaultdict(list)
reviews = []

with open("yelp_dataset_challenge_academic_dataset\yelp_academic_dataset_review.json") as fp:    
    for line in fp:
        review = json.loads(line)
        
        reviews.append(review["text"])
        busi_to_rev[review["business_id"]].append(review["text"])

pickle.dump(reviews, open("list-of-reviews.p", "wb"))
pickle.dump(busi_to_rev, open("dict-of-business-to-reviews.p", "wb"))
