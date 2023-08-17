import copy
import functools
import itertools
import math
import operator
import random
import traceback
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


def third(triplet):
    return triplet[2]


def sort_using_scores(elements, scores, increasing=False):
    zipped = list(zip(elements, scores))
    zipped.sort(key=second, reverse=increasing)
    # print(f"In sort_using_scores, increasing={increasing}, and the zipped scores are \n{second(unzip(zipped))}")
    return [first(pair) for pair in zipped]


def concat(lists):
    return sum(lists, [])


def group(iterable, key_function=lambda x: x):
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


# def remove_duplicates(iterable, key=lambda x: x):
#    return [cluster[0] for cluster in group(iterable, key).values()]


def remove_duplicates(iterable, key=lambda x: x):
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
    # This function exists because I used to use matrices
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

    if np.max(hot_encoding) == 0:
        return None

    if np.sum(hot_encoding) > 1.0:
        return math.nan
    else:
        for (i, value) in enumerate(hot_encoding):
            if value:
                return i


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
    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))


def sigmoid(x):
    return 1 / (1 + math.exp(-1 * x))


def stop_for_every_warning(func):
    warnings.filterwarnings("error")
    try:
        func()
    except RuntimeWarning:
        print("A warning was raised!")
        traceback.print_exc()


def sample_index_with_weights(probabilities):
    return random.choices(range(len(probabilities)), probabilities, k=1)[0]


def chi_squared(observed, expected):
    return ((observed - expected) ** 2) / expected


def sample_from_grid_of_weights(probabilities):
    rows = len(probabilities)
    flattened_probabilities = concat(probabilities)
    aggregated_index = sample_index_with_weights(flattened_probabilities)
    return divmod(aggregated_index, rows)


def reattempt_until(generator, condition_to_satisfy):
    while True:
        candidate = generator()
        if condition_to_satisfy(candidate):
            return candidate


def as_row_matrix(array_input):
    return np.reshape(array_input, (1, array_input.shape[0]))

def as_column_matrix(array_input):
    return np.reshape(array_input, (array_input.shape[0], 1))


def row_wise_self_outer_product(input_matrix):
    return np.einsum('ij,ik->ijk', input_matrix, input_matrix, optimize=True).reshape(input_matrix.shape[0], -1)


def flat_outer_product(input_array):
    return np.outer(input_array, input_array).ravel()


def weighted_sum(a, weight_a, b, weight_b):
    return a * weight_a + b * weight_b


def arithmetic_weighted_average(a, weight_a, b, weight_b):
    return weighted_sum(a, weight_a, b, weight_b) / (weight_a + weight_b)


def geometric_weighted_average(a, weight_a, b, weight_b):
    return ((a ** weight_a) * (b ** weight_b)) ** (1.0 / (weight_a + weight_b))


def harmonic_weighted_average(a, weight_a, b, weight_b):
    return (weight_a + weight_b) / ((weight_a / a) + (weight_b / b))


def remap(x, starting_rage, ending_range):
    (starting_min, starting_max) = starting_rage
    (ending_min, ending_max) = ending_range

    in_zero_one_range = (x - starting_min) / (starting_max - starting_min)
    return in_zero_one_range * (ending_max - ending_min) + ending_min


def remap_array_in_zero_one(input_array):
    """remaps the values in the given array to be between 0 and 1"""
    min_value = np.min(input_array)
    max_value = np.max(input_array)

    if (min_value == max_value):
        return np.ones_like(input_array)/2  # all 0.5!

    return (input_array - min_value) / (max_value - min_value)

def remap_array_in_zero_one_ammortise(input_array):
    amount_of_items = len(input_array)
    positions = np.arange(start=0, stop=amount_of_items, dtype = float) / (amount_of_items-1)

    zipped = sorted(zip(input_array, positions), key=utils.first)
    _, rearranged_positions = unzip(zipped)
    return np.array(rearranged_positions)


def remap_array_in_zero_one_ignore_zeros(input_array):

    def is_zero(x):
        return x < 0.01
    min_value = np.min([datapoint for datapoint in input_array if not is_zero(datapoint)])
    max_value = np.max(input_array)

    if (min_value == max_value):
        return input_array / min_value  # should be all ones! TODO perhaps these should be all 0.5?

    def remap(original):
        return (original-min_value) / (max_value-min_value)

    return np.array([0.0 if is_zero(datapoint) else remap(datapoint) for datapoint in input_array])


def remap_second_value(input_list):
    (original, to_remap) = unzip(input_list)
    return [(from_original, remapped) for (from_original, remapped)
            in zip(original, remap_array_in_zero_one(to_remap))]


def boost_range(x):
    """the input array is in [0, 1], and the result will have values lower than 0.5 lowered, greater than 0.5 increased"""
    return (3 * x ** 2) - (2 * x ** 3)  # this is the solution to integral([x*(1-x)]^k) for k=1


def min_max(array_of_values):
    if (len(array_of_values) == 0):
        raise Exception("utils.min_max was called on an empty array")
    return (np.min(array_of_values), np.max(array_of_values))

def get_min_max_of_arrays(arrays):
    """returns the min and max of concat(arrays), without actually contanenating them"""
    min_maxes = [min_max(single_array) for single_array in arrays if len(single_array)>0]
    if len(min_maxes) == 0:
        raise("utils.get_min_max_of_arrays received either an empty lists or all the arrays are empty")
    (mins, maxes) = unzip(min_maxes)
    return (np.min(mins), np.max(maxes))


def nth_power_flat_outer_product(array: np.ndarray, n: int):
    current_result = array.copy()
    if (n<1):
        raise Exception("in nth_power_flat_outer_product, n<1")

    for _ in range(n-1):
        current_result = np.outer(current_result, array).ravel()

    return current_result


def row_wise_nth_power_self_outer_product(input_matrix, n:int):
    current_result = input_matrix.copy()
    if (n < 1):
        raise Exception("in nth_power_flat_outer_product, n<1")

    for _ in range(n - 1):
        current_result = np.einsum('ij,ik->ijk', current_result, input_matrix, optimize=True).reshape(input_matrix.shape[0], -1)

    return current_result





def product(iterable):
    return functools.reduce(operator.mul, iterable)


def weighted_sum_of_rows(matrix: np.ndarray, weights: np.ndarray):
    return np.einsum('ij,i->j', matrix, weights)


def divide_arrays_safely(numerator, deniminator):
    return np.divide(numerator, deniminator, out=np.zeros_like(numerator), where=deniminator != 0.0)




