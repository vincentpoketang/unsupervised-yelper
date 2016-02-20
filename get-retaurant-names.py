import json
import pickle

business_names = []

with open("yelp_dataset_challenge_academic_dataset\yelp_academic_dataset_business.json") as fp:    
    for line in fp:
        business = json.loads(line)
        business_names.append(business["name"])

pickle.dump(business_names, open("list-of-business-names.p", "wb"))