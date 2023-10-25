from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.ArchiveMiner import ArchiveMiner
from Version_E.InterestingAlgorithms.Miner import LayeredFeatureMiner, FeatureSelector
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation


class ConstructiveMiner(ArchiveMiner):
    stochastic: bool

    Population = list[Feature]
    EvaluatedPopulation = list[(Feature, float)]

    def __init__(self, selector: FeatureSelector, population_size: int, generations: int, stochastic: bool):
        super().__init__(selector, population_size, generations)
        self.stochastic = stochastic

    def __repr__(self):
        return (f"Constructive(population = {self.population_size}, "
                f"stochastic = {self.stochastic}, "
                f"pop_size = {self.population_size},"
                f"generations = {self.generations})")

    def get_initial_population(self) -> Population:
        return super().get_empty_feature_population()

    def get_children(self, parents: EvaluatedPopulation) -> list[Feature]:
        return super().get_complected_children(parents)


    def select(self, population: EvaluatedPopulation) -> EvaluatedPopulation:
        amount_to_select = self.population_size // 3
        if self.stochastic:
            return super().tournament_selection(population, amount_to_select)
        else:
            return super().truncation_selection(population, amount_to_select)