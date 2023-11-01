import json
from typing import Callable

from Version_E.BaselineAlgorithms import RandomSearch, HillClimber
from Version_E.BaselineAlgorithms.GA import GAMiner
from Version_E.BaselineAlgorithms.HillClimber import HillClimber
from Version_E.BaselineAlgorithms.RandomSearch import RandomSearch
from Version_E.InterestingAlgorithms.BiDirectionalMiner import BiDirectionalMiner
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner
from Version_E.InterestingAlgorithms.Miner import FeatureMiner
from Version_E.InterestingAlgorithms.Miner import FeatureSelector

TerminationPredicate = Callable


def decode_miner(properties: dict, selector: FeatureSelector,
                 termination_predicate: TerminationPredicate) -> FeatureMiner:
    """ converts a json string into an instance of a FeatureMiner object"""

    kind = properties["which"]

    if kind in "constructive":
        return ConstructiveMiner(selector,
                                 stochastic=properties["stochastic"],
                                 population_size=properties["population_size"],
                                 uses_archive=properties["uses_archive"],
                                 termination_criteria_met=termination_predicate)
    elif kind == "destructive":
        return DestructiveMiner(selector,
                                stochastic=properties["stochastic"],
                                population_size=properties["population_size"],
                                uses_archive=properties["uses_archive"],
                                termination_criteria_met=termination_predicate)
    elif kind == "bidirectional":
        return BiDirectionalMiner(selector,
                                  stochastic=properties["stochastic"],
                                  population_size=properties["population_size"],
                                  uses_archive=properties["uses_archive"],
                                  termination_criteria_met=termination_predicate)
    elif kind == "ga":
        return GAMiner(selector,
                       population_size=properties["population_size"],
                       termination_criteria_met=termination_predicate)
    elif kind == "hill_climber":
        return HillClimber(selector,
                           termination_criteria_met=termination_predicate)
    elif kind == "random":
        return RandomSearch(selector,
                            termination_criteria_met=termination_predicate)


def generate_miner_name(miner_json: dict) -> str:
    miner_name = miner_json["which"]
    if miner_name in {"constructive", "destructive", "bidirectional"}:
        stochastic = 'S' if miner_json["stochastic"] else 'H'
        pop_size = miner_json["population_size"]
        uses_archive = "A" if miner_json["uses_archive"] else "nA"
        return f"{miner_name}_{stochastic}_{pop_size}_{uses_archive}"
    elif miner_name == "ga":
        pop_size = miner_json["population_size"]
        return f"{miner_name}_{pop_size}"
    elif miner_name in {"hill_climber", "random"}:
        return f"{miner_name}"


