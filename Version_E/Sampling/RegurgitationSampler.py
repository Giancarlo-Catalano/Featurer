from typing import Iterable, Callable

import SearchSpace
from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.MeasurableCriterion.CriterionUtilities import Balance, Extreme, All
from Version_E.MeasurableCriterion.Explainability import Explainability
from Version_E.MeasurableCriterion.ForSampling import Completeness, ExpectedFitness
from Version_E.MeasurableCriterion.MeasurableCriterion import MeasurableCriterion
from Version_E.PrecomputedFeatureInformation import PrecomputedFeatureInformation
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import Miners


def get_reference_features_for_regurgitation_sampling(fitness_criterion: MeasurableCriterion,
                                                      problem: CombinatorialProblem,
                                                      termination_predicate: Callable,
                                                      ppi: PrecomputedPopulationInformation,
                                                      reference_miner_parameters: dict,
                                                      amount_to_return: int,
                                                      importance_of_explainability: float) -> list[Feature]:
    search_criterion = Balance([
        Explainability(problem),
        Extreme(fitness_criterion)],   # note the Extreme
        weights=[importance_of_explainability, 1 - importance_of_explainability])

    selector = FeatureSelector(ppi, search_criterion)

    miner = Miners.decode_miner(reference_miner_parameters,
                                selector=selector,
                                termination_predicate=termination_predicate)

    mined_features = miner.get_meaningful_features(amount_to_return)

    return mined_features

def regurgitation_sample(reference_features: Iterable[Feature],
                          fitness_criterion: MeasurableCriterion,
                          termination_predicate: Callable,
                          original_ppi: PrecomputedPopulationInformation,
                          sampling_miner_parameters: dict,
                          amount_to_return: int) -> list[SearchSpace.Candidate]:
    reference_feature_pfi = PrecomputedFeatureInformation(original_ppi, reference_features)
    generation_criterion = All([Completeness(),
                                ExpectedFitness(criterion=fitness_criterion, pfi=reference_feature_pfi)])

    selector = FeatureSelector(original_ppi, generation_criterion)
    sampling_miner = Miners.decode_miner(sampling_miner_parameters,
                                         selector=selector,
                                         termination_predicate=termination_predicate)
    sampled_features = sampling_miner.get_meaningful_features(amount_to_return)

    sampled_candidates = [feature.to_candidate() for feature in sampled_features
                          if feature.is_convertible_to_candidate()]

    return sampled_candidates
