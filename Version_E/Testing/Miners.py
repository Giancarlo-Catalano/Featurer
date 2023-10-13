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


def aggregate_jsons_into_csv(json_file_list: list[str], output_file_name: str):
    def get_runs_from_file(single_file_name) -> list[dict]:
        with open(single_file_name, "r") as file:
            data = json.load(file)
            return data["result"]["test_results"]

    def get_results_from_run(single_run: dict) -> dict[str, list[int]]:
        miners_and_results = [(item["miner"], item["result"]) for item in single_run]

        def convert_pairs(pairs_list) -> list[int]:
            return [item["found"] for item in pairs_list]

        return {f"{miner}": convert_pairs(result) for miner, result in miners_and_results}

    def merge_results(results: list[dict]) -> dict:
        def get_aggregated_by_key(key) -> list[int]:
            return [item for single_result in results for item in single_result.setdefault(key, [])]

        all_keys = {key for single_result in results for key in single_result.keys()}

        return {key: get_aggregated_by_key(key) for key in all_keys}

    all_runs: list[dict] = [run for file_name in json_file_list
                            for run in get_runs_from_file(file_name)]

    aggregated_results= merge_results([get_results_from_run(run) for run in all_runs])
    aggregated_results = sorted(aggregated_results.items(), key=lambda p: int(p[0]))

    with open(output_file_name, "w") as output_file:
        for category, results in aggregated_results:
            output_file.write(f"{category}, ")
            output_file.write(", ".join([f"{item}" for item in results]))
            output_file.write("\n")



