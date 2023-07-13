import itertools
import math
import random
import warnings

import numpy as np

import utils

float_type = np.single


def cumulative_sum(elements):
    # in Python 3.11 return list(itertools.accumulate(elements, initial=0))
    return [0] + list(itertools.accumulate(elements))


def adjacent_pairs(elements):
    # I should be able to use itertools.pairwise(elements), but it does not work
    return zip(elements, elements[1:])


def first(pair):
    return pair[0]


def second(pair):
    return pair[1]


def sort_using_scores(elements, scores, increasing=False):
    zipped = list(zip(elements, scores))
    zipped.sort(key=second, reverse=increasing)
    # print(f"In sort_using_scores, increasing={increasing}, and the zipped scores are \n{second(unzip(zipped))}")
    return [first(pair) for pair in zipped]


def concat(lists):
    return sum(lists, [])


def group(iterable, key_function=lambda x:x):
    """returns a dictionary, where the key is the key computed here"""

    precomputed_keys = [key_function(item) for item in iterable]
    catalog = dict()

    def add_to_catalog(item, item_key):
        for catalog_key in catalog:
            if catalog_key == item_key:
                catalog[catalog_key].append(item)
                return
        catalog[item_key] = [item]

    for item, item_key in zip(iterable, precomputed_keys):
        add_to_catalog(item, item_key)

    return catalog


def group_unhashable(iterable, key=lambda x: x, custom_equals=lambda x, y: x == y):
    """returns a list of pairs, where the key is the key computed here"""
    key_function = key

    precomputed_keys = [key_function(item) for item in iterable]
    catalog = []  # list of pairs

    def add_to_catalog(item_key, item):
        for (catalog_key, catalog_list) in catalog:
            if custom_equals(item_key, catalog_key):
                catalog_list.append(item)
                return
        catalog.append((item_key, [item]))

    for item_key, item in zip(precomputed_keys, iterable):
        add_to_catalog(item_key, item)

    return catalog


#def remove_duplicates(iterable, key=lambda x: x):
#    return [cluster[0] for cluster in group(iterable, key).values()]


def remove_duplicates(iterable, key=lambda x:x):
    seen_keys_set = set()
    seen_items = list()
    for item in iterable:
        item_key = key(item)
        if item_key not in seen_keys_set:
            seen_keys_set.add(item_key)
            seen_items.append(item)
    return seen_items

def remove_duplicates_unhashable(iterable, key=lambda x: x, custom_equals=lambda x, y: x == y):
    return [group_items[0] for (group_key, group_items) in group_unhashable(iterable, key, custom_equals)]


def rows_in_matrix(input_matrix):
    return input_matrix.shape[0]


def unzip(zipped):
    if (len(zipped) == 0):
        return []

    group_amount = len(zipped[0])

    def get_nth_group(n):
        return [elem[n] for elem in zipped]

    return tuple(get_nth_group(n) for n in range(group_amount))


def identity_matrix(n):
    """Creates a MATRIX as an Identity, with INTEGER values"""
    #This function exists because I used to use matrices
    return np.identity(n, dtype=float_type)


def zero_matrix(n):
    return np.zeros((n, n), dtype=float_type)


def array_to_row_vector(array):
    return np.reshape(array, (1, array.shape[0]))


def matrix_to_array(matrix):
    """assumes the matrix is a single row"""
    return np.array(matrix, type=float_type).squeeze()


def column_matrix_to_row_vector(matrix):
    """You should just use np.transpose"""
    return np.transpose(matrix)


def matrix_to_scalar(matrix):
    return matrix[0, 0]


def one_hot_encoding(value, card):
    """Note that this returns all zeros if value is None"""
    result = np.zeros(card, dtype=float_type)
    if value is not None:
        result[value] = 1.0
    return result


def from_hot_encoding(hot_encoding):
    """determines what the original value was"""
    """Note: None signifies that there were no 1s found (valid), and math.nan signifies that there were multiple (invalid)"""

    result = None
    for (i, value) in enumerate(hot_encoding):
        if value:
            if result is None:
                result = i
            else:  # the result had been found already!
                return math.nan
    return result


def from_hot_encoding_old(hot_encoding):
    result = None
    for (i, value) in enumerate(hot_encoding):
        if value:
            if result is None:
                result = value
            else:  # the result had been found already!
                return math.nan
    return result


def weighted_sum_of_columns(weights, matrix):
    return np.sum(matrix @ np.diag(weights), axis=1)


def negate_list(iterable):
    return [item * (-1) for item in iterable]


def to_column_vector(row_vector):
    return row_vector.reshape((row_vector.shape[0], 1))


# BIN area

"""    def reset_combinations(self):
        def two_features(which_a, a_value, which_b, b_value):
            as_list = [None]*self.search_space.amount_ofnp._features
            as_list[which_a] = a_value
            as_list[which_b] = b_value
            return tuple(as_list)

        pairs_of_vars = itertools.combinations(range(self.search_space.dimensions), 2)
        pairwise_features = [two_features(first, first_value, second, second_value)
                                for (first, second) in pairs_of_vars
                                for first_value in range(self.search_space.cardinalities[first])
                                for second_value in range(self.search_space.cardinalities[second])]

        pairwise_features_matrix = self.search_space.get_population_as_one_hot_matrix(pairwise_features)
        single_features_matrix = np.identity(self.search_space.amount_of_features, dtype="int32")

        #print(f"The pairs of variables are {list(pairs_of_vars)}")
        #print(f"The pairwise features are {pairwise_features}")
        #print(f"The one hot encoded pairwise features matrix is \n{pairwise_features_matrix}")
        #print(f"The single features matrix is {single_features_matrix}")

        self.combinations = np.vstack((single_features_matrix, pairwise_features_matrix)) #strange that vstack needs a tuple
 """


def binomial_coeff(n, k):
    return math.factorial(n)/(math.factorial(n-k)*math.factorial(k))


def sigmoid(x):
    return 1/(1+math.exp(-1*x))

def stop_for_every_warning(func):
    warnings.filterwarnings("error")
    try:
        func()
    except RuntimeWarning:
        print("A warning was raised!")



def sample_index_with_weights(probabilities):
    return random.choices(range(len(probabilities)), probabilities, k=1)[0]


def sample_from_grid_of_weights(probabilities):
    rows = len(probabilities)
    flattened_probabilities = utils.concat(probabilities)
    aggregated_index = sample_index_with_weights(flattened_probabilities)
    return divmod(aggregated_index, rows)


def reattempt_until(generator, condition_to_satisfy):
    while True:
        candidate = generator()
        if condition_to_satisfy(candidate):
            return candidate