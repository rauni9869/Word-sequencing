import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT PERFORM ANY FILE IO IN YOUR CODE

# DO NOT CHANGE THE NAME OF THE METHOD my_fit or my_predict BELOW
# IT WILL BE INVOKED BY THE EVALUATION SCRIPT
# CHANGING THE NAME WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc
def calculate_bigrams(word):
    return [word[i:i+2] for i in range(len(word)-1)]

def generating_bigrams(words):
    bigram_dict = {}
    all_bigrams = set()
    for word in words:
        bigrams = sorted(set(calculate_bigrams(word)))[:5]
        bigram_tuple = tuple(bigrams)
        if bigram_tuple in bigram_dict:
            bigram_dict[bigram_tuple].append(word)
        else:
            bigram_dict[bigram_tuple] = [word]
        all_bigrams.update(bigrams)
    return bigram_dict, sorted(all_bigrams)

class Tree:
    def __init__(self):
        self.model = None
        self.encoder = LabelEncoder()
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, words):
        bigram_dict, all_bigrams = generating_bigrams(words)
        self.mlb.fit([all_bigrams])
        X = self.mlb.transform(list(bigram_dict.keys()))
        y = [word_list[0] for word_list in bigram_dict.values()]
        self.encoder.fit(y)
        y_encoded = self.encoder.transform(y)
        self.model = DecisionTreeClassifier(criterion='entropy')
        self.model.fit(X, y_encoded)
    
    def predict(self, bigrams):
        bigram_tuple = tuple(bigrams)
        bigram_vector = self.mlb.transform([bigram_tuple])
        y_pred = self.model.predict(bigram_vector)
        guess_list = [self.encoder.inverse_transform([y])[0] for y in y_pred]
        return guess_list

################################
# Non Editable Region Starting #
################################
def my_fit( words ):
################################
#  Non Editable Region Ending  #
################################

	# Do not perform any file IO in your code
	# Use this method to train your model using the word list provided
	X_training = words
	model = Tree()
	model.fit(X_training)
	return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( model, bigram_list ):
################################
#  Non Editable Region Ending  #
################################
	
	# Do not perform any file IO in your code
	# Use this method to predict on a test bigram_list
	# Ensure that you return a list even if making a single guess
	guess_list = model.predict(bigram_list)
	return guess_list					# Return guess(es) as a list
