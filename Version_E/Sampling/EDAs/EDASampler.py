from typing import Callable

import utils
from UnivariateUtils import Evaluator, Population, EvaluatedPopulation, UnivariateModel
from SearchSpace import SearchSpace, Candidate


class EDASampler:
    population_size: int
    selection_proportion = 0.3

    def __init__(self, population_size: int):
        self.population_size = population_size

    def combine_new_model(self, current_model: UnivariateModel,
                          selected_population_model: UnivariateModel) -> UnivariateModel:
        raise Exception("In an implementation of EDASampler, self.update is not implemented")

    def run(self, search_space: SearchSpace, fitness_function: Callable, budget: int) -> Candidate:
        evaluator = Evaluator(fitness_function)
        model = UnivariateModel.get_uniform(search_space)

        population = evaluator.with_fitnesses(model.sample_many(self.population_size))

        while evaluator.used_evaluations < budget:
            amount_of_population_to_keep = int(len(population) * self.selection_proportion)
            selected = evaluator.select(population, amount_of_population_to_keep)
            model_from_selected = UnivariateModel.get_from_selected_population(evaluator.without_fitnesses(selected),
                                                                               search_space)
            model = self.combine_new_model(model, model_from_selected)
            children = model.sample_many(self.population_size - len(selected))
            evaluated_children = evaluator.with_fitnesses(children)
            population = selected + evaluated_children


        return population.sort(key=utils.second, reverse=True)[0]


class UMDA(EDASampler):

    def __init__(self):
        pass

    def __repr__(self):
        return "UMDA"

    def combine_new_model(self, current_model: UnivariateModel,
                          selected_population_model: UnivariateModel) -> UnivariateModel:
        return selected_population_model


class PBIL(EDASampler):
    rho: float

    def __init__(self, rho: float):
        self.rho = rho

    def __repr__(self):
        return "UMDA"

    def combine_new_model(self, current_model: UnivariateModel,
                          selected_population_model: UnivariateModel) -> UnivariateModel:
        return UnivariateModel.weighted_sum(current_model, selected_population_model, (1 - self.rho), self.rho)
