from Version_E.BaselineAlgorithms import RandomSearch, HillClimber
from Version_E.BaselineAlgorithms.GA import GAMiner
from Version_E.BaselineAlgorithms.HillClimber import HillClimber
from Version_E.BaselineAlgorithms.RandomSearch import RandomSearch
from Version_E.Feature import Feature
from Version_E.InterestingAlgorithms.ConstructiveMiner import ConstructiveMiner
from Version_E.InterestingAlgorithms.DestructiveMiner import DestructiveMiner
from Version_E.InterestingAlgorithms.Miner import FeatureMiner
from Version_E.InterestingAlgorithms.Miner import FeatureSelector

def decode_miner(properties: dict, selector: FeatureSelector) -> FeatureMiner:
    """ converts a json string into an instance of a FeatureMiner object"""

    kind = properties["which"]

    if kind in "constructive":
        return ConstructiveMiner(selector,
                                 stochastic=properties["stochastic"],
                                 at_most_parameters=properties["at_most"],
                                 amount_to_keep_in_each_layer=properties["population_size"])
    elif kind == "destructive":
        return DestructiveMiner(selector,
                                stochastic=properties["stochastic"],
                                at_least_parameters=properties["at_least"],
                                amount_to_keep_in_each_layer=properties["population_size"])
    elif kind == "ga":
        return GAMiner(selector,
                       population_size=properties["population_size"],
                       iterations=properties["iterations"])
    elif kind == "hill_climber":
        return HillClimber(selector,
                           amount_to_generate=properties["population_size"])
    elif kind == "random":
        return RandomSearch(selector,
                            amount_to_generate=properties["population_size"])