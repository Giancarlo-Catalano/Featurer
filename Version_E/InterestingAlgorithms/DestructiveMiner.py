from typing import Callable

from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.ArchiveMiner import ArchiveMiner
from Version_E.InterestingAlgorithms.Miner import FeatureSelector


class DestructiveMiner(ArchiveMiner):
    stochastic: bool

    Population = list[Feature]
    EvaluatedPopulation = list[(Feature, float)]

    def __init__(self, selector: FeatureSelector,
                 population_size: int,
                 stochastic: bool,
                 termination_criteria_met: Callable):
        super().__init__(selector, population_size, termination_criteria_met)
        self.stochastic = stochastic

    def __repr__(self):
        return (f"Destructive(population = {self.population_size}, "
                f"stochastic = {self.stochastic}, "
                f"pop_size = {self.population_size})")

    def get_initial_population(self) -> Population:
        return super().get_complex_feature_population(self.population_size)

    def get_children(self, parents: EvaluatedPopulation) -> list[Feature]:
        return super().get_simplified_children(parents)

    def select(self, population: EvaluatedPopulation) -> EvaluatedPopulation:
        amount_to_select = self.population_size // 3
        if self.stochastic:
            return super().tournament_selection(population, amount_to_select)
        else:
            return super().truncation_selection(population, amount_to_select)
