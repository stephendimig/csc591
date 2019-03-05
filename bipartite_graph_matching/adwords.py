import pandas as pd
import argparse
import heapq
from enum import Enum
import math
import random
from copy import deepcopy
import numpy as np
from tabulate import tabulate

# Algorithm Type Enumeration
class AlgorithmType(Enum):
    GREEDY = 1
    MSVV = 2
    BALANCE = 3
	# Converts an enumerated value to a string
    def __str__(self):
        val2str = {1: "GREEDY", 2: "MSVV", 3: "BALANCE"}
        return val2str[self.value]

	# Converts a string to an enumerated value. Will throw an excetion if the string is not a recognized enumeration.
    @classmethod
    def fromString(cls, str):
        str2val = {"GREEDY": AlgorithmType.GREEDY,
                   "MSVV": AlgorithmType.MSVV,
                   "BALANCE": AlgorithmType.BALANCE}
        if str.upper() in str2val.keys():
            return str2val[str.upper()]
        else:
            raise Exception('Error: not  a valid algorithm. str={0}'.format(str))

#
# Class: Bid
# Description: A small plain old data  class for a bid
#
class Bid(object):
	# Constructor
	def __init__(self, keyword, bid_value):
		self.bid_value = bid_value
		self.keyword = keyword

	# Outputs contents of class as a string
	def __str__(self):
		return "(keyword={}; bid_value={})".format(self.keyword, self.bid_value)

#
# Class: Advertiser
# Description: A class that represents an advitiser with a budget, balance, and bids.
#
class Advertiser(object):
	# Constructor
	def __init__(self, row):
		self.bids = {}
		self.name = row['Advertiser']
		self.budget = row['Budget']
		self.balance = self.budget
		self.bids[row['Keyword']] = Bid(row['Keyword'], row['Bid Value'])

	# This method updates the bids when the advitiser already exists
	def update(self, row):
		self.bids[row['Keyword']] = Bid(row['Keyword'], row['Bid Value'])

	# This method will return the value of the bid for a keyword
	def get_bid_value(self,  keyword):
		bid = self.bids[keyword]
		return bid.bid_value

	# This method will either accept or reject a bid. Bids are rejected if the bid
	# price exceeds the current value. A True is returned if the bid is accepted, a False
	# is returned otherwise.
	def accept_bid(self,  keyword):
		retval = False
		bid = self.bids[keyword].bid_value
		if bid <= self.balance:
			self.balance -= bid
			retval = True
		return retval

	# This method resets the balance to the budget price.
	def reset(self):
		self.balance = self.budget

	# Returns a string representation of the class.
	def __str__(self):
		return "name={}; budget={}; balance={}; bids={}".format(self.name, self.budget, self.balance, [str(bid) for bid in self.bids.values()])

# This method implements a comparator for the greedy algorithm when pushing a bid onto the heap. It is negative
# because we are using a max heap.
def greedy(keyword, advertiser):
	return -advertiser.get_bid_value(keyword)

# This method implements a comparator for the balanced algorithm when pushing a bid onto the heap.
def balance(keyword, advertiser):
	return -advertiser.balance

# This method implements a comparator for the msvv algorithm when pushing a bid onto the heap. Notice it is a hybrid
# approach that weights the bid (like greedy) with the balance.
def msvv(keyword, advertiser):
	b = advertiser.get_bid_value(keyword)
	xu = 1.0 - float(advertiser.balance/advertiser.budget)
	scale = 1 - math.exp(xu - 1)
	return -(b * scale)

# This method initializes the words_dict and advitisers_dict structures.
# words_dict - is a mapping from a word to a max heap of advertisers that have bid on the word
# advertisers_dict - is a mapping from an index to an advertiser.
def init(df, algorithm):
	advertisers_dict = {}
	words_dict = {}
	for index, row in df.iterrows():
		advertiser = None
		h = None
		keyword = row['Keyword']

		if row['Advertiser'] in advertisers_dict.keys():
			advertiser = advertisers_dict[row['Advertiser']]
			advertiser.update(row)
		else:
			advertiser = Advertiser(row)
			advertisers_dict[row['Advertiser']] = advertiser

		if keyword in words_dict.keys():
			h = words_dict[keyword]
		else:
			h = []
			words_dict[keyword] = h
		heapq.heappush(h, (algorithm(keyword, advertiser), advertiser))
	return (words_dict, advertisers_dict)

# Main method
if __name__ == '__main__':
	NUMBER_OF_PERMUTATIONS = 100

	#  Initializations
	algorithm_dict = {AlgorithmType.GREEDY: greedy, AlgorithmType.BALANCE: balance, AlgorithmType.MSVV: msvv}
	revenue = [0.0 for i in range(0, NUMBER_OF_PERMUTATIONS)]
	random.seed(0)

	# Command line argument processing
	c_args = argparse.ArgumentParser(description='')
	c_args.add_argument('algo', help='Algorithm type')
	c_args.add_argument('--queries', help='Optional query file', default='queries.txt')
	c_args.add_argument('--bidders', help='Optional file for bidders', default='bidder_dataset.csv')
	cli_args = c_args.parse_args()
	queries_file = cli_args.queries
	bidders_file = cli_args.bidders

	# This code proesses the algorithm and exits if it is not greedy, balanced, or msvv
	try:
		algo_type =  AlgorithmType.fromString(cli_args.algo)
	except Exception as e:
		print("Error: e={}".format(str(e)))
		exit(-1)
	algorithm = algorithm_dict[algo_type]

	# Read from bidders and queries files
	df = pd.read_csv(bidders_file)
	with open(queries_file, 'r') as f:
		queries_arr = [line.rstrip('\n') for line in f]

	OPT = sum([val for val in df['Budget'] if not pd.isna(val)])
	# Loop over 100 permutations
	for i in range(0, NUMBER_OF_PERMUTATIONS):
		# Initialize the heaps and dictionaries
		words_dict, advertisers_dict = init(df, algorithm)

		# Deep copy so we avoid changing the original data
		new_array = deepcopy(queries_arr)

		# Shuffle the array
		random.shuffle(new_array)

		# Now process each query in the query file.
		for query in new_array:
			if query in words_dict.keys():
				h = words_dict[query]

				# Processing a max heap. Keep popping while the bid is not accepted.
				while h:
					target = heapq.heappop(h)[1]
					if True == target.accept_bid(query):
						# Update the revenue for this iteration.
						revenue[i] += target.get_bid_value(query)

						# If the bid was accepted we can push it back on the heap  with a new weight
						heapq.heappush(h, (algorithm(query, target), target))
						break

	mean_revenue = np.mean(revenue)
	temp_df = pd.DataFrame(columns=['Algorithm', 'Mean Revenue', 'Competitive Ratio'])
	temp_df.loc[-1] = [cli_args.algo, mean_revenue,  float(mean_revenue/OPT)]
	print tabulate(temp_df, headers=temp_df.columns.values.tolist(), showindex=False, tablefmt='psql')
