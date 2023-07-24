import HotEncoding
import SearchSpace
import utils
import numpy as np


class VariateModels:
    # features: a list of featuresH
    # correlation matrices

    def get_feature_presence_matrix(self, candidate_matrix, featureH_pool):
        """returns a matrix in which element i, j is 1 if candidate i contains feature j"""
        feature_matrix = np.transpose(np.array(featureH_pool))
        positive_when_absent = (1 - candidate_matrix) @ feature_matrix
        return 1 - np.minimum(positive_when_absent, 1)


    def get_fitness_unfitness_scores(self, feature_presence_matrix, fitness_list):
        """returns the univariate model according to the fitness"""
        count_for_each_feature = np.sum(feature_presence_matrix, axis=0)
        sum_of_fitness_for_each_feature = np.sum(feature_presence_matrix * utils.to_column_vector(fitness_list),
                                                 axis=0)

        average_fitnesses = np.array([total / float(count) if count > 0 else 0.0 for total, count in
                                      zip(sum_of_fitness_for_each_feature, count_for_each_feature)])
        # the scores are unnormalised, but they get normalised when we change the range anyway
        fitness_scores = utils.remap_array_in_zero_one(average_fitnesses)
        fitness_scores = utils.boost_range(fitness_scores)
        return (fitness_scores, 1-fitness_scores)

    def get_popularity_unpopularity_scores(self, feature_presence_matrix, expected_amounts):
        """returns the univariate model according to the popularity"""
        observed_amounts = np.sum(feature_presence_matrix, axis=0)

        chi_squared_and_is_popular = [(utils.chi_squared(observed, expected),
                                       observed > expected)
                                        for (observed, expected) in zip(observed_amounts, expected_amounts)]

        (popularity, unpopularity) = utils.unzip([(x_2, 0.0) if is_popular else (0.0, x_2)
                                     for (x_2, is_popular)
                                     in chi_squared_and_is_popular])

        return (utils.remap_array_in_zero_one(popularity),
                utils.remap_array_in_zero_one(unpopularity))


    def get_cooccurrence_matrix_for_fitness_unfitness(self, feature_presence_matrix, fitness_list):
        row_wise_outer_product = utils.row_wise_self_outer_product(feature_presence_matrix)
        weighted_sum_by_fitness = np.sum(row_wise_outer_product * utils.to_column_vector(fitness_list),
                                                 axis=0)
        count_for_each_pair_of_features = np.sum(row_wise_outer_product, axis=0)
        average_fitnesses = np.array([total / float(count) if count > 0 else 0.0 for total, count in
                                      zip(weighted_sum_by_fitness, count_for_each_pair_of_features)])
        # the scores are unnormalised, but they get normalised when we change the range anyway
        cooccurrence_matrix = utils.remap_array_in_zero_one(average_fitnesses)
        cooccurrence_matrix = utils.boost_range(cooccurrence_matrix)

        amount_of_features = feature_presence_matrix.shape[1]
        cooccurrence_matrix.reshape((amount_of_features, amount_of_features))
        return (cooccurrence_matrix, 1-cooccurrence_matrix)

    # TODO 24 / 7
    #    Calculate fitness relevance by using chi-squared(fitness, average)
    #    filter all features in all organised_by_weight categories, together rather than group-wise
    #    implement bivariate models, should be easy






    def __init__(self, search_space: SearchSpace.SearchSpace):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(search_space)