import json
from pprint import pprint

with open('yelp_academic_dataset_review.json') as data_file:
	counter = 0
	for line in data_file:
		counter+= 1
	
	print(counter)
