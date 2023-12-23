from typing import Callable

import utils
from Version_E.Sampling.EDAs.UnivariateUtils import UnivariateModel
from SearchSpace import SearchSpace, Candidate
from Version_E.Sampling.FullSolutionSampler import FullSolutionSampler, EvaluatedPopulation, Population, Evaluator


class EDASampler(FullSolutionSampler):
    population_size: int
    selection_proportion = 0.3

    def __init__(self,
                 search_space: SearchSpace,
                 fitness_function: Callable,
                 population_size: int,
                 termination_criteria: Callable):
        super().__init__(search_space, fitness_function, termination_criteria)
        self.population_size = population_size

    def combine_new_model(self, current_model: UnivariateModel,
                          selected_population_model: UnivariateModel) -> UnivariateModel:
        raise Exception("In an implementation of EDASampler, self.update is not implemented")

    def sample(self, how_many_to_return: int) -> EvaluatedPopulation:
        model = UnivariateModel.get_uniform(self.search_space)

        population = self.evaluator.with_fitnesses(model.sample_many(self.population_size))

        iteration_count = 0

        def should_continue():
            return self.termination_criteria(iteration=iteration_count, used_budget=self.evaluator.used_evaluations)

        while should_continue():
            amount_of_population_to_keep = int(len(population) * self.selection_proportion)
            selected = self.evaluator.select(population, amount_of_population_to_keep)
            model_from_selected = UnivariateModel.get_from_selected_population(self.evaluator.without_fitnesses(selected),
                                                                               self.search_space)
            model = self.combine_new_model(model, model_from_selected)
            children = model.sample_many(self.population_size - len(selected))
            evaluated_children = self.evaluator.with_fitnesses(children)
            population = selected + evaluated_children

            iteration_count += 1

        population.sort(key=utils.second, reverse=True)
        return population[:how_many_to_return]


class UMDA(EDASampler, FullSolutionSampler):

    def __init__(self,
                 search_space: SearchSpace,
                 fitness_function: Callable,
                 population_size: int,
                 termination_predicate):
        super().__init__(search_space, fitness_function, population_size, termination_predicate)

    def __repr__(self):
        return "UMDA"

    def combine_new_model(self, current_model: UnivariateModel,
                          selected_population_model: UnivariateModel) -> UnivariateModel:
        return selected_population_model


class PBIL(EDASampler):
    rho: float

    def __init__(self,
                 search_space: SearchSpace,
                 fitness_function: Callable,
                 population_size: int,
                 termination_predicate: Callable,
                 rho: float):
        super().__init__(search_space, fitness_function, population_size, termination_predicate)
        self.rho = rho

    def __repr__(self):
        return "UMDA"

    def combine_new_model(self, current_model: UnivariateModel,
                          selected_population_model: UnivariateModel) -> UnivariateModel:
        return UnivariateModel.weighted_sum(current_model, selected_population_model, (1 - self.rho), self.rho)
