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
    hot_encoded_features: list[np.ndarray]
    search_space: SearchSpace.SearchSpace
    hot_encoder: HotEncoding.HotEncoder

    feature_detector: VariateModels.FeatureDetector

    validator: VariateModels.CandidateValidator

    def __init__(self, search_space: SearchSpace.SearchSpace,
                 features: list[SearchSpace.Feature],
                 validator: VariateModels.CandidateValidator):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.features = features
        self.hot_encoded_features = [HotEncoding.hot_encode_feature(feature, self.search_space) for feature in features]

        self.feature_detector = VariateModels.FeatureDetector(self.search_space, self.hot_encoded_features)
        self.validator = validator

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
        amount_of_features = len(self.hot_encoded_features)
        self.bivariate_matrix = 1 - self.validator.clash_matrix.reshape((amount_of_features, -1))

    def get_starting_pseudo_candidate(self) -> SearchSpace.Feature:
        distribution_grid = np.ndarray.tolist(self.bivariate_matrix)

        def get_tentative_result():
            (index_for_feature_a, index_for_feature_b) = list(utils.sample_from_grid_of_weights(distribution_grid))
            return HotEncoding.merge_features(self.hot_encoded_features[index_for_feature_a],
                                              self.hot_encoded_features[index_for_feature_b])

        while True:
            tentative_result = get_tentative_result()
            if self.validator.is_hot_encoded_candidate_valid(tentative_result):
                return tentative_result

    def get_feature_presence_vector(self, pseudo_candidate: np.ndarray) -> np.ndarray:
        return self.feature_detector \
            .get_feature_presence_matrix_from_candidate_matrix(utils.as_row_matrix(pseudo_candidate))

    def get_present_features(self, pseudo_candidate: np.ndarray):
        feature_presence_vector = self.get_feature_presence_vector(pseudo_candidate)
        return [index for (index, presence_value) in enumerate(feature_presence_vector)
                if presence_value > 0.5]

    def specialise_unsafe(self, pseudo_candidate: np.ndarray):
        feature_presence_vector = self.get_feature_presence_vector(pseudo_candidate)
        distribution = (feature_presence_vector @ self.bivariate_matrix).ravel()

        # we remove the features that are already present
        distribution *= (1.0 - feature_presence_vector.ravel())

        new_feature_index = None
        if np.sum(distribution) <= 0.0:  # happens when no features are present, or edge cases
            new_feature_index = random.randrange(len(self.hot_encoded_features))
        else:
            new_feature_index = utils.sample_index_with_weights(distribution)

        new_feature = self.hot_encoded_features[new_feature_index]
        return HotEncoding.merge_features(pseudo_candidate, new_feature)

    def specialise(self, pseudo_candidate: np.ndarray):
        """this was decommissioned because it would get stuck!"""
        before_specialisation = pseudo_candidate.copy()

        attempts = 0
        while True:
            if attempts > 10:
                print("So many attempts!")
            attempted_specialisation = self.specialise_unsafe(before_specialisation)
            if self.validator.is_hot_encoded_candidate_valid(attempted_specialisation):
                return attempted_specialisation
            attempts += 1


class Sampler:
    wanted_feature_sampler: SingleObjectiveSampler
    novelty_sampler: SingleObjectiveSampler
    uniform_sampler: SingleObjectiveSampler
    unwanted_feature_detector: VariateModels.FeatureDetector
    validator: VariateModels.CandidateValidator

    search_space: SearchSpace.SearchSpace
    hot_encoder: HotEncoding.HotEncoder

    importance_of_novelty: float

    def __init__(self, search_space: SearchSpace.SearchSpace,
                 wanted_features: list[SearchSpace.Feature],
                 unwanted_features: list[SearchSpace.Feature],
                 unpopular_features: list[SearchSpace.Feature],
                 importance_of_novelty=0.5):
        self.search_space = search_space
        self.hot_encoder = HotEncoding.HotEncoder(self.search_space)
        self.wanted_features = wanted_features
        self.unwanted_features = unwanted_features
        self.unpopular_features = unpopular_features

        self.validator = VariateModels.CandidateValidator(self.search_space)

        def get_micro_sampler(features):
            return SingleObjectiveSampler(self.search_space, features, self.validator)

        self.wanted_feature_sampler = get_micro_sampler(wanted_features)
        self.novelty_sampler = get_micro_sampler(unwanted_features)
        self.uniform_sampler = get_micro_sampler(self.search_space.get_all_trivial_features())
        self.unwanted_feature_detector = VariateModels.FeatureDetector(self.search_space, unwanted_features)

        self.importance_of_randomness = 0.0
        self.importance_of_novelty = importance_of_novelty

    def train(self, training_data: PopulationSamplePrecomputedData):
        """uses the given data to train its models"""
        self.wanted_feature_sampler.train_for_fitness(training_data)
        self.novelty_sampler.train_for_novelty(training_data)
        self.uniform_sampler.train_for_uniformity()

    def candidate_is_complete(self, candidate: np.ndarray):
        """returns true when the input is a fully filled candidate"""
        combinatorial_candidate = self.hot_encoder.candidate_from_hot_encoding(candidate)
        amount_of_nones = sum([1 if val is None else 0 for val in combinatorial_candidate.values])
        return amount_of_nones == 0

    def contains_worst_features(self, candidateH):
        """returns true when the candidate contains features recognised by the unfit feature detector"""
        return self.unwanted_feature_detector.candidateH_contains_any_features(candidateH)

    @property
    def model_of_choice(self):
        """returns a model, which can be the fit one or the novelty one, randomly"""
        if random.random() < self.importance_of_randomness:
            return self.uniform_sampler
        elif random.random() < self.importance_of_novelty:
            return self.novelty_sampler
        else:
            return self.wanted_feature_sampler

    def erase_random_values(self, candidate: np.ndarray, amount: int):
        result = candidate.copy()
        where_can_remove = [index for index in range(len(result)) if result[index] == 1.0]
        if amount >= len(where_can_remove):
            return candidate
        indices_to_reset = random.choices(where_can_remove, k=amount)
        for i in indices_to_reset:
            result[i]=0.0
        return result


    def maybe_without_some_subfeatures(self, candidate: np.ndarray):
        if random.random() < 0.3:
            amount_of_holes = random.randrange(1, 4)
            return self.erase_random_values(candidate, amount_of_holes)
        else:
            return candidate

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

        current_state: np.ndarray = self.model_of_choice.get_starting_pseudo_candidate()

        attempts = 0
        too_many_attempts = self.search_space.total_cardinality * 2
        while True:
            # the uniform sampler helps prevent getting stuck in "invalidity basins"
            self.importance_of_randomness = attempts / too_many_attempts  # impatience grows with the amount of attempts
            if self.candidate_is_complete(current_state):
                break
            if attempts < too_many_attempts:
                concurrent_state = self.maybe_without_some_subfeatures(current_state)
            tentative_specialisation = self.model_of_choice.specialise_unsafe(current_state)
            if self.validator.is_hot_encoded_candidate_valid(tentative_specialisation) and \
                    not (attempts > too_many_attempts and self.contains_worst_features(tentative_specialisation)):
                current_state = tentative_specialisation

            attempts += 1

        return self.hot_encoder.candidate_from_hot_encoding(current_state)



    # TODO: somewhere the tentative candidate is being converted into an erroneous hot encoded form
