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

    def to_code(self):
        return (f"gen={self.generated_feature_amount}|"
                f"kept={self.kept_feature_amount}|"
                f"mix={self.mixing_iterator.to_code()}|"
                f"cw={self.criteria_and_weights}")


class SearchMethod(Enum):
    TOTAL_SEARCH = auto()
    HEURISTIC_SEARCH = auto()
    STOCHASTIC_SEARCH = auto()

    def __str__(self):
        return ["Total Search", "Heuristic", "Stochastic"][self.value - 1]

    def to_code(self):
        return ["Tot", "Heu", "Sto"][self.value - 1]


class Proportionality(Enum):
    FIXED = auto()
    ENTIRE_SEARCH_SPACE = auto()
    PROBLEM_PARAMETERS = auto()

    def __str__(self):
        return ["Fixed", "∝ Search space", "∝ Problem"][self.value - 1]

    def to_code(self):
        return ["Fix", "Pro", "SSp"][self.value - 1]


class Thoroughness(Enum):
    LEAST = auto()
    KINDA = auto()
    AVERAGE = auto()
    VERY = auto()
    MOST = auto()

    def __str__(self):
        return ["Least", "Kinda", "Average", "Very", "Most"][self.value - 1]

    def to_code(self):
        return ["0", "1", "2", "3", "4"][self.value - 1]


class CriteriaStart(Enum):
    ALWAYS = auto()
    FROM_MIDPOINT = auto()
    ONLY_LAST = auto()

    def __str__(self):
        return ["Always", "From Midpoint", "Only Last"][self.value - 1]

    def to_code(self):
        return ["Alws", "MidP", "Last"][self.value - 1]


def get_total_amount_of_features_in_iteration(search_space: SearchSpace, weight: int):
    return sum([utils.product(which_vars) for which_vars in
                itertools.combinations(search_space.cardinalities, weight)])


def only_explainability_criterion(criteria_and_weights):
    def is_explainability(criterion_and_weight: (MeasurableCriterion, float)):
        criterion, weight = criterion_and_weight
        return isinstance(criterion, MeasurableCriterion.ExplainabilityCriterion) and weight != 0

    explainabilities = [criterion for criterion, weight in criteria_and_weights
                        if is_explainability(criterion)]

    if explainabilities:
        return explainabilities[0]
    else:
        raise Exception(
            "Error in Miner.py: attempting to extract the explainability MeasurableCriterion, but none were found"
            f"(They are {criteria_and_weights})")


def get_iteration_parameters(search_space: SearchSpace,
                             weight: int,
                             explored_depth: int,
                             search_method: SearchMethod,
                             criteria_and_weights: MeasurableCriterion.LayerScoringCriteria,
                             proportionality: Proportionality,
                             thoroughness: Thoroughness,
                             criteria_start: CriteriaStart):
    total = get_total_amount_of_features_in_iteration(search_space, weight)
    problem_coefficient = search_space.total_cardinality * weight * utils.binomial_coeff(explored_depth, weight)

    generated_feature_amount = None
    kept_feature_amount = None

    generated_and_kept_map = {
        (Proportionality.FIXED, Thoroughness.LEAST): (20, 10),
        (Proportionality.FIXED, Thoroughness.KINDA): (100, 50),
        (Proportionality.FIXED, Thoroughness.AVERAGE): (200, 100),
        (Proportionality.FIXED, Thoroughness.VERY): (1000, 500),
        (Proportionality.FIXED, Thoroughness.MOST): (2000, 1000),
        (Proportionality.PROBLEM_PARAMETERS, Thoroughness.LEAST): (problem_coefficient * 2, problem_coefficient),
        (Proportionality.PROBLEM_PARAMETERS, Thoroughness.KINDA): (problem_coefficient * 10, problem_coefficient * 5),
        (Proportionality.PROBLEM_PARAMETERS, Thoroughness.AVERAGE): (
            problem_coefficient * 20, problem_coefficient * 10),
        (Proportionality.PROBLEM_PARAMETERS, Thoroughness.VERY): (problem_coefficient * 100, problem_coefficient * 50),
        (Proportionality.PROBLEM_PARAMETERS, Thoroughness.MOST): (problem_coefficient * 200, problem_coefficient * 100),
        (Proportionality.ENTIRE_SEARCH_SPACE, Thoroughness.LEAST): (total * 0.005, total * 0.001),
        (Proportionality.ENTIRE_SEARCH_SPACE, Thoroughness.KINDA): (total * 0.010, total * 0.005),
        (Proportionality.ENTIRE_SEARCH_SPACE, Thoroughness.AVERAGE): (total * 0.020, total * 0.010),
        (Proportionality.ENTIRE_SEARCH_SPACE, Thoroughness.VERY): (total * 0.100, total * 0.050),
        (Proportionality.ENTIRE_SEARCH_SPACE, Thoroughness.MOST): (total * 0.200, total * 0.1)
    }

    if search_method == SearchMethod.TOTAL_SEARCH:
        generated_feature_amount, kept_feature_amount = total, total
    else:
        generated_feature_amount, kept_feature_amount = generated_and_kept_map[(proportionality, thoroughness)]

    # to prevent oversearching
    generated_feature_amount = min(generated_feature_amount, total)
    kept_feature_amount = min(generated_feature_amount, total)

    mixing_iterator = None
    if search_method == SearchMethod.TOTAL_SEARCH:
        mixing_iterator = IterationParameters.TOTAL_SEARCH
    elif search_method == SearchMethod.HEURISTIC_SEARCH:
        mixing_iterator = IterationParameters.HEURISTIC_SEARCH
    elif search_method == SearchMethod.STOCHASTIC_SEARCH:
        mixing_iterator = IterationParameters.STOCHASTIC_SEARCH

    only_explainability = only_explainability_criterion(criteria_and_weights)
    effective_criteria_and_weights = None
    if criteria_start == CriteriaStart.ALWAYS:
        effective_criteria_and_weights = criteria_and_weights
    elif criteria_start == CriteriaStart.FROM_MIDPOINT:
        if weight <= explored_depth // 2:
            effective_criteria_and_weights = only_explainability
        else:
            effective_criteria_and_weights = criteria_and_weights
    elif criteria_start == CriteriaStart.ONLY_LAST:
        if weight < explored_depth:
            effective_criteria_and_weights = only_explainability
        else:
            effective_criteria_and_weights = criteria_and_weights

    return IterationParameters(generated_feature_amount,
                               kept_feature_amount,
                               mixing_iterator,
                               criteria_and_weights)
