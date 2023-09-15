
import itertools
from enum import Enum, auto

import utils
from MinerLayer import MinerLayer
from Version_D import MeasurableCriterion
from Version_D.Miner.LayerMixer import ParentPairIterator, TotalSearchIterator, GreedyHeuristicIterator, \
    StochasticIterator
from SearchSpace import SearchSpace
from Version_D import Feature


class IterationParameters:
    generated_feature_amount: int
    kept_feature_amount: int
    mixing_iterator: ParentPairIterator
    criteria_and_weights: MeasurableCriterion.LayerScoringCriteria

    TOTAL_SEARCH = TotalSearchIterator()
    HEURISTIC_SEARCH = GreedyHeuristicIterator()
    STOCHASTIC_SEARCH = StochasticIterator()

    def __init__(self, generated_feature_amount: int,
                 kept_feature_amount: int,
                 mixing_iterator: ParentPairIterator,
                 criteria_and_weights: MeasurableCriterion.LayerScoringCriteria):
        self.generated_feature_amount = generated_feature_amount
        self.kept_feature_amount = kept_feature_amount
        self.mixing_iterator = mixing_iterator
        self.criteria_and_weights = criteria_and_weights


    def __repr__(self):
        return (f"IterationParameters:\n"
                f"{self.generated_feature_amount = }\n"
                f"{self.kept_feature_amount = },"
                f"{self.mixing_iterator = },"
                f"{self.criteria_and_weights = }")


class SearchMethod(Enum):
    TOTAL_SEARCH = auto()
    HEURISTIC_SEARCH = auto()
    STOCHASTIC_SEARCH = auto()

    def __str__(self):
        return ["Total Search", "Heuristic", "Stochastic"][self.value-1]

    def to_code(self):
        return ["Tot", "Heu", "Sto"][self.value-1]


class Proportionality(Enum):
    FIXED = auto()
    ENTIRE_SEARCH_SPACE = auto()
    PROBLEM_PARAMETERS = auto()

    def __str__(self):
        return ["Fixed", "∝ Search space", "∝ Problem"][self.value-1]

    def to_code(self):
        return ["Fix", "Pro", "SSp"][self.value-1]


class Thoroughness(Enum):
    LEAST = auto()
    KINDA = auto()
    AVERAGE = auto()
    VERY = auto()
    MOST = auto()

    def __str__(self):
        return ["Least", "Kinda", "Average", "Very", "Most"][self.value-1]

    def to_code(self):
        return ["0", "1", "2", "3", "4"][self.value-1]


class CriteriaStart(Enum):
    ALWAYS = auto()
    FROM_MIDPOINT = auto()
    ONLY_LAST = auto()

    def __str__(self):
        return ["Always", "From Midpoint", "Only Last"][self.value-1]

    def to_code(self):
        return ["Alws", "MidP", "Last"][self.value-1]



def get_iteration_parameters(search_space: SearchSpace,
                             weight: int,
                             search_method: SearchMethod,
                             proportionality: Proportionality,
                             thoroughness: Thoroughness,
                             criteria_start: CriteriaStart):
    pass




class IterationParameterFactory:
    search_space: SearchSpace
    criteria_and_weights: MeasurableCriterion.LayerScoringCriteria

    def __init__(self, search_space: SearchSpace, criteria_and_weights: MeasurableCriterion.LayerScoringCriteria):
        self.search_space = search_space
        self.criteria_and_weights = criteria_and_weights


    @property
    def only_explainability_criterion(self):
        def is_explainability(criterion_and_weight: (MeasurableCriterion, float)):
            criterion, weight = criterion_and_weight
            return isinstance(criterion, MeasurableCriterion.ExplainabilityCriterion) and weight != 0
        explainabilities = [criterion for criterion, weight in self.criteria_and_weights
                            if is_explainability(criterion)]

        if explainabilities:
            return explainabilities[0]
        else:
            raise Exception("Error in Miner.py: attempting to extract the explainability MeasurableCriterion, but none were found"
                            f"(They are {self.criteria_and_weights})")

    def get_total_amount_of_features_in_iteration(self, weight: int):
        return sum([utils.product(which_vars) for which_vars in
                    itertools.combinations(self.search_space.cardinalities, weight)])

    def get_total_search_iteration(self, weight: int):
        total_amount = self.get_total_amount_of_features_in_iteration(weight)
        return IterationParameters(generated_feature_amount=total_amount,
                                   kept_feature_amount=total_amount,
                                   mixing_iterator=IterationParameters.TOTAL_SEARCH,
                                   criteria_and_weights=[])


    def get_greedy_search_iteration(self,
                                    proportion_to_generate: float,
                                    proportion_to_keep: float,
                                    only_explainability: bool,
                                    weight: int,
                                    explainability_only = False):
        total_amount = self.get_total_amount_of_features_in_iteration(weight)
        criteria_and_weights_to_use = self.only_explainability_criterion if only_explainability else self.criteria_and_weights

        to_generate = int(total_amount*proportion_to_generate)
        to_keep = int(total_amount*proportion_to_keep)

        return IterationParameters(generated_feature_amount=to_generate,
                                   kept_feature_amount=to_keep,
                                   mixing_iterator=IterationParameters.HEURISTIC_SEARCH,
                                   criteria_and_weights=criteria_and_weights_to_use)




