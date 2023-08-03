import random

import Version_B.VariateModels
import utils
import SearchSpace
import HotEncoding
import numpy as np


class Sampler:
    features: list
    search_space: SearchSpace.SearchSpace
    hot_encoder: HotEncoding.HotEncoder

    feature_detector: Version_B.VariateModels.FeatureDetector

    validator: Version_B.VariateModels.CandidateValidator

    def __init__(self, search_space, features, validator):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.features = features

        self.feature_detector = Version_B.VariateModels.FeatureDetector(self.search_space, self.features)
        self.validator = validator

        #  To be set during training
        self.bivariate_matrix = None

    def train_for_fitness(self, candidateC_pool, fitness_array: np.ndarray):
        variate_model_generator = Version_B.VariateModels.VariateModels(self.search_space)
        feature_presence_matrix = self.feature_detector.get_feature_presence_matrix_from_candidates(candidateC_pool)
        self.bivariate_matrix = variate_model_generator. \
            get_bivariate_fitness_qualities(feature_presence_matrix, fitness_array)

    def train_for_novelty(self, candidateC_pool):
        variate_model_generator = Version_B.VariateModels.VariateModels(self.search_space)
        feature_presence_matrix = self.feature_detector.get_feature_presence_matrix_from_candidates(candidateC_pool)
        self.bivariate_matrix = variate_model_generator. \
            get_bivariate_popularity_qualities(feature_presence_matrix)

        # as this will measure popularity, we "invert it" to measure unpopularity
        self.bivariate_matrix = 1 - self.bivariate_matrix

    def train_for_uniformity(self):
        """Basically the values are 1 when the trivial features can coexist"""
        amount_of_features = len(self.features)
        self.bivariate_matrix = 1-self.validator.clash_matrix.reshape((amount_of_features, -1))

    def get_starting_pseudo_candidate(self):
        distribution_grid = np.ndarray.tolist(self.bivariate_matrix)

        def get_tentative_result():
            (index_for_feature_a, index_for_feature_b) = list(utils.sample_from_grid_of_weights(distribution_grid))
            return HotEncoding.merge_features(self.features[index_for_feature_a], self.features[index_for_feature_b])

        while True:
            tentative_result = get_tentative_result()
            if self.validator.is_candidate_valid(tentative_result):
                return tentative_result

    def get_feature_presence_vector(self, pseudo_candidateH):
        return self.feature_detector \
            .get_feature_presence_matrix_from_candidate_matrix(utils.as_row_matrix(pseudo_candidateH))

    def get_present_features(self, pseudo_candidateH):
        feature_presence_vector = self.get_feature_presence_vector(pseudo_candidateH)
        return [index for (index, presence_value) in enumerate(feature_presence_vector)
                if presence_value > 0.5]

    def specialise_unsafe(self, pseudo_candidateH):
        feature_presence_vector = self.get_feature_presence_vector(pseudo_candidateH)

        distribution = (feature_presence_vector @ self.bivariate_matrix).ravel()
        # we remove the features that are already present
        distribution *= (1.0 - feature_presence_vector.ravel())

        # then we sample from the distribution to obtain a new feature to add
        new_feature_index = utils.sample_index_with_weights(distribution)
        new_feature = self.features[new_feature_index]
        return HotEncoding.merge_features(pseudo_candidateH, new_feature)

    def specialise(self, pseudo_candidateH: np.ndarray):
        """this was decommissioned because it would get stuck!"""
        before_specialisation = pseudo_candidateH.copy()

        attempts = 0
        while True:
            if attempts > 10:
                print("So many attempts!")
            attempted_specialisation = self.specialise_unsafe(before_specialisation)
            if self.validator.is_candidate_valid(attempted_specialisation):
                return attempted_specialisation
            attempts += 1


class ESTEEM_Sampler:
    fit_sampler: Sampler
    novelty_sampler: Sampler
    uniform_sampler: Sampler
    unfit_feature_detector: Version_B.VariateModels.FeatureDetector
    validator: Version_B.VariateModels.CandidateValidator

    search_space: SearchSpace.SearchSpace
    hot_encoder: HotEncoding.HotEncoder

    importance_of_novelty: float

    def __init__(self, search_space, fit_features, unfit_features, unpopular_features,
                 importance_of_novelty=0.5):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.fit_features = fit_features
        self.unfit_features = unfit_features
        self.unpopular_features = unpopular_features

        self.validator = Version_B.VariateModels.CandidateValidator(self.search_space)
        self.fit_sampler = Sampler(self.search_space, self.fit_features, self.validator)
        self.novelty_sampler = Sampler(self.search_space, self.unpopular_features, self.validator)
        self.uniform_sampler = Sampler(self.search_space, self.hot_encoder.get_hot_encoded_trivial_features(), self.validator)
        self.unfit_feature_detector = Version_B.VariateModels.FeatureDetector(self.search_space, unfit_features)

        self.importance_of_randomness = 0.0
        self.importance_of_novelty = importance_of_novelty

    def train(self, candidateC_pool, fitness_list):
        """uses the given data to train its models"""
        fitness_array = np.array(fitness_list)
        self.fit_sampler.train_for_fitness(candidateC_pool, fitness_array)
        self.novelty_sampler.train_for_novelty(candidateC_pool)
        self.uniform_sampler.train_for_uniformity()

    def candidate_is_complete(self, candidateH):
        """returns true when the input is a fully fileld candidate"""
        combinatorial_candidate = self.hot_encoder.candidate_from_hot_encoding(candidateH)
        amount_of_nones = sum([1 if val is None else 0 for val in combinatorial_candidate.values])
        return amount_of_nones == 0

    def contains_worst_features(self, candidateH):
        """returns true when the candidate contains features recognised by the unfit feature detector"""
        return self.unfit_feature_detector.candidateH_contains_any_features(candidateH)

    @property
    def model_of_choice(self):
        """returns a model, which can be the fit one or the novelty one, randomly"""
        if random.random() < self.importance_of_randomness:
            return self.uniform_sampler
        elif random.random() < self.importance_of_novelty:
            return self.novelty_sampler
        else:
            return self.fit_sampler

    def sample(self):
        """ Generates a new candidate by using the many models within"""

        # General process:

        """
           current_solution = generate a starting solution
           
           repeat until current_solution is complete:
                tentative_specialisation = (a model).specialise(current_solution)
                
                if (tentative_specialisation is valid 
                    and does not contain any really bad features
                    current_solution = tentative_specialisation
                    
            return current_solution
        """

        current_state = self.model_of_choice.get_starting_pseudo_candidate()

        attempts = 0
        too_many_attempts = self.search_space.total_cardinality
        while True:
            # the uniform sampler helps prevent getting stuck in "invalidity basins"
            self.importance_of_randomness = attempts / too_many_attempts
            if self.candidate_is_complete(current_state):
                break
            tentative_specialisation = self.model_of_choice.specialise_unsafe(current_state)
            if self.validator.is_candidate_valid(tentative_specialisation) and \
                    not self.contains_worst_features(tentative_specialisation):
                current_state = tentative_specialisation

            attempts += 1

        return self.hot_encoder.candidate_from_hot_encoding(current_state)
