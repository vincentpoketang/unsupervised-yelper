import json

with open('yelp_academic_dataset_review.json') as reviews_file:
    num_reviews = 0
    num_words = 0
    for line in reviews_file:
        num_reviews += 1
        review = json.loads(line)
        num_words += len(review["text"].split())

    print("Reviews: " + str(num_reviews))
    print("Words: " + str(num_words))
    print("Average length: " + str(num_words/num_reviews))
