import random

from Version_B import VariateModels
import utils
import SearchSpace
import HotEncoding
from Version_B.PopulationSamplePrecomputedData import PopulationSamplePrecomputedData, \
    PopulationSampleWithFeaturesPrecomputedData
import numpy as np


class SingleObjectiveSampler:
    features: list[SearchSpace.Feature]
    search_space: SearchSpace.SearchSpace
    hot_encoder = HotEncoding.HotEncoder
    feature_detector: VariateModels.FeatureDetector

    between_feature_clash_matrix: np.ndarray

    def __init__(self, search_space: SearchSpace.SearchSpace,
                 features: list[SearchSpace.Feature]):
        self.search_space = search_space
        self.features = features
        self.hot_encoder = HotEncoding.HotEncoder(search_space)
        self.feature_detector = VariateModels.FeatureDetector(self.search_space, features)

        self.between_feature_clash_matrix = VariateModels.get_between_feature_shouldnt_merge_matrix(search_space, features)

        #  To be set during training
        self.bivariate_matrix = None

    def train_for_fitness(self, training_data: PopulationSamplePrecomputedData):
        variate_model_generator = VariateModels.VariateModels(self.search_space)
        training_data_with_features = PopulationSampleWithFeaturesPrecomputedData(training_data, self.features)
        self.bivariate_matrix = variate_model_generator. \
            get_bivariate_fitness_qualities(training_data_with_features.feature_presence_matrix,
                                            training_data.fitness_array)

    def train_for_novelty(self, training_data: PopulationSamplePrecomputedData):
        variate_model_generator = VariateModels.VariateModels(self.search_space)
        training_data_with_features = PopulationSampleWithFeaturesPrecomputedData(training_data, self.features)
        self.bivariate_matrix = variate_model_generator. \
            get_bivariate_popularity_qualities(training_data_with_features.feature_presence_matrix)

        # as this will measure popularity, we "invert it" to measure unpopularity
        self.bivariate_matrix = 1 - self.bivariate_matrix

    def train_for_uniformity(self):
        """Basically the values are 1 when the trivial features can coexist"""
        """assumes that features are exactly the trivial features, in order of their hot encoding positions"""
        clash_matrix = VariateModels.CandidateValidator.get_search_space_clash_matrix(self.search_space)
        self.bivariate_matrix = 1 - clash_matrix

    def get_starting_pseudo_candidate(self) -> SearchSpace.Feature:
        distribution_grid = np.ndarray.tolist(self.bivariate_matrix)

        def get_tentative_result():
            (index_for_feature_a, index_for_feature_b) = list(utils.sample_from_grid_of_weights(distribution_grid))
            feature_a = self.features[index_for_feature_a]
            feature_b = self.features[index_for_feature_b]

            return SearchSpace.merge_two_features(feature_a, feature_b)

        while True:
            tentative_result = get_tentative_result()
            if self.search_space.feature_is_valid(tentative_result):
                return tentative_result

    def get_feature_presence_vector_in_conglomerate(self, conglomerate: SearchSpace.Feature) -> np.ndarray:
        hot_encoded_conglomerate = self.hot_encoder.feature_to_hot_encoding(conglomerate)
        return self.feature_detector.get_feature_presence_from_candidateH(hot_encoded_conglomerate)

    def feature_clash_vector(self, feature_presence_vector: np.ndarray) -> np.ndarray:
        """returns a vector where, for each feature, it has a 1 if there's a clash with the input, 0 if it's fine"""
        wind_sum = utils.weighted_sum_of_columns(feature_presence_vector, self.between_feature_clash_matrix).ravel()
        return np.minimum(wind_sum, 1)




    def specialise_unsafe(self, conglomerate: SearchSpace.Feature) -> SearchSpace.Feature:
        feature_presence_vector = self.get_feature_presence_vector_in_conglomerate(conglomerate)
        distribution = (feature_presence_vector @ self.bivariate_matrix).ravel()

        # we remove the features that are already present
        distribution *= (1.0 - feature_presence_vector.ravel())
        exlusion_vector = self.feature_clash_vector(feature_presence_vector)
        distribution *= (1.0 - exlusion_vector)

        if np.sum(distribution) <= 0.0:  # happens when no features are present, or edge cases
            new_feature = random.choice(self.features)
        else:
            new_feature = random.choices(self.features, weights=distribution, k=1)[0]

        return SearchSpace.merge_two_features(conglomerate, new_feature)


class Sampler:
    wanted_feature_sampler: SingleObjectiveSampler
    novelty_sampler: SingleObjectiveSampler
    uniform_sampler: SingleObjectiveSampler
    unwanted_feature_detector: VariateModels.FeatureDetector

    search_space: SearchSpace.SearchSpace
    importance_of_novelty: float

    def __init__(self, search_space: SearchSpace.SearchSpace,
                 wanted_features: list[SearchSpace.Feature],
                 unwanted_features: list[SearchSpace.Feature],
                 unpopular_features: list[SearchSpace.Feature],
                 importance_of_novelty=0.5):
        self.search_space = search_space
        self.wanted_features = wanted_features
        self.unwanted_features = unwanted_features
        self.unpopular_features = unpopular_features

        def get_micro_sampler(features):
            return SingleObjectiveSampler(self.search_space, features)

        self.wanted_feature_sampler = get_micro_sampler(wanted_features)
        self.novelty_sampler = get_micro_sampler(unpopular_features)
        self.uniform_sampler = get_micro_sampler(self.search_space.get_all_trivial_features())
        self.unwanted_feature_detector = VariateModels.FeatureDetector(self.search_space, unwanted_features)

        self.importance_of_randomness = 0.0
        self.importance_of_novelty = importance_of_novelty


    # TODO make a new constructor, which will take into consideration the fact that some problems might have constraints
    # specifically, we should not be generating the constraints, but only the parameters, and then let the problem
    # add the constraints. Perhaps I should make a wrapper class, with a function called sample_for_problem, having the
    # parameter, uses constraints: bool

    def train(self, training_data: PopulationSamplePrecomputedData):
        """uses the given data to train its models"""
        self.wanted_feature_sampler.train_for_fitness(training_data)
        self.novelty_sampler.train_for_novelty(training_data)
        self.uniform_sampler.train_for_uniformity()

    def conglomerate_is_complete(self, conglomerate: SearchSpace.Feature):
        return self.search_space.feature_is_complete(conglomerate)

    def conglomerate_is_valid(self, conglomerate: SearchSpace.Feature):
        return self.search_space.feature_is_valid(conglomerate)

    def contains_worst_features(self, conglomerate: SearchSpace.Feature):
        return self.unwanted_feature_detector.feature_contains_any_features(conglomerate)

    @property
    def model_of_choice(self):
        """returns a model, which can be the fit one or the novelty one, randomly"""
        if random.random() < self.importance_of_randomness:
            return self.uniform_sampler
        elif random.random() < self.importance_of_novelty:
            return self.novelty_sampler
        else:
            return self.wanted_feature_sampler

    def maybe_without_some_subfeatures(self, conglomerate: SearchSpace.Feature):
        def with_erased_vars(feature: SearchSpace.Feature, amount_to_remove: int) -> SearchSpace.Feature:
            present_vars = feature.var_vals
            return SearchSpace.Feature(random.sample(present_vars, k=len(present_vars) - amount_to_remove))

        if random.random() < self.importance_of_randomness:
            amount_of_holes = random.randrange(1, 4)
            return with_erased_vars(conglomerate, amount_of_holes)
        else:
            return conglomerate

    def sample(self):
        """ Generates a new candidate by using the many models within"""

        accumulator: SearchSpace.Feature = self.model_of_choice.get_starting_pseudo_candidate()

        attempts = 0
        too_many_attempts = self.search_space.total_cardinality * 2
        while not self.conglomerate_is_complete(accumulator):
            self.importance_of_randomness = (attempts / too_many_attempts)
            tentative_specialisation = self.model_of_choice.specialise_unsafe(accumulator)
            if self.conglomerate_is_valid(tentative_specialisation):
                if not (attempts < too_many_attempts and self.contains_worst_features(tentative_specialisation)):
                    accumulator = tentative_specialisation
            attempts += 1

        return self.search_space.feature_to_candidate(accumulator)
