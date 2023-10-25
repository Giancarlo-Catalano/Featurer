import json

import utils
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
                                 population_size=properties["population_size"],
                                 generations = properties["generations"])
    elif kind == "destructive":
        return DestructiveMiner(selector,
                                stochastic=properties["stochastic"],
                                population_size=properties["population_size"],
                                generations = properties["generations"])
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

def generate_miner_name(miner_json: dict) -> str:
    miner_name = miner_json["which"]
    if miner_name in {"constructive", "destructive"}:
        stochastic = miner_json["stochastic"]
        pop_size = miner_json["population_size"]
        return f"{miner_name}_{'S' if stochastic else 'H'}_{pop_size}"
    elif miner_name in {"hill_climber", "random"}:
        pop_size = miner_json["population_size"]
        return f"{miner_name}_{pop_size}"
    elif miner_name == "ga":
        pop_size = miner_json["population_size"]
        iterations = miner_json["iterations"]
        return f"{miner_name}_{iterations}_{pop_size}"


def aggregate_algorithm_jsons_into_csv(json_file_list: list[str], output_file_name: str, for_time=False):

    def get_miner_result_dict_from_file(single_file_name: str) -> dict:
        with open(single_file_name, "r") as file:
            data = json.load(file)
            miner_runs: list[dict] = data["result"]["test_results"]
            as_dict = {generate_miner_name(item['miner']): [result_item["time" if for_time else "found"]
                                                            for result_item in item['result']]
                       for item in miner_runs}
            return as_dict

    def merge_results_by_keys(results: list[dict]) -> dict:
        def get_aggregated_by_key(key) -> list[int]:
            return [item for single_result in results for item in single_result.setdefault(key, [])]

        all_keys = {key for single_result in results for key in single_result.keys()}

        return {key: get_aggregated_by_key(key) for key in all_keys}

    miner_result_dicts : list[dict] = [get_miner_result_dict_from_file(input_file) for input_file in json_file_list]
    aggregated = merge_results_by_keys(miner_result_dicts)

    with open(output_file_name, "w") as output_file:
        for category, results in aggregated.items():
            output_file.write(f"{category},")
            output_file.write(",".join([f"{item:.3f}" for item in results]))
            output_file.write("\n")
