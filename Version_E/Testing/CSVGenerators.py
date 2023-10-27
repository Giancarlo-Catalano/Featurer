import json

import utils
from Version_E.Testing.Miners import generate_miner_name


def make_csv_for_successes(input_name, output_name: str):
    with open(input_name, "r") as input_file:
        data = json.load(input_file)
        miners_and_results = [(item["miner"], item["result"]) for item in data["result"]["test_results"]]

        def convert_pairs(pairs_list):
            return [item["found"] for item in pairs_list]

        miners_and_results = [(miner, convert_pairs(result)) for miner, result in miners_and_results]

        with open(output_name, "w") as output_file:
            for miner, results in miners_and_results:
                output_file.write(f"{miner}\t" + '\t'.join(f"{value}" for value in results))


def make_csv_for_connectedness(input_name, output_name: str):
    def merge_dicts(dicts: list[dict]):
        def items_for_key(key):
            recovered_items = [single_dict.setdefault(key, []) for single_dict in dicts]
            return utils.concat_lists(recovered_items)

        all_keys = {key for single_dict in dicts
                    for key in single_dict.keys()}
        return {key: items_for_key(key) for key in all_keys}

    # with open(input_name, "r") as input_file:
    #     data = json.load(input_file)
    #     trials = [item["edge_counts"] for item in data["result"]["test_results"]]
    #
    #     # aggregate different trial runs
    #     merged_trials = merge_dicts(trials)
    #     def get_samples_for_amount_of_nodes(amount_of_nodes) -> list[int]:
    #         possible_connections = amount_of_nodes*(amount_of_nodes-1)//2
    #         chance_of_connection = data["parameters"]["problem"]["chance_of_connection"]
    #         sample_size = 1000
    #         return sample_from_binomial_distribution(possible_connections, chance_of_connection, sample_size)
    #     merged_baselines = {amount_of_nodes: get_samples_for_amount_of_nodes(int(amount_of_nodes))
    #                         for amount_of_nodes in merged_trials.keys()}
    #
    #     # sort by amount of nodes
    #     merged_trials = dict(sorted(merged_trials.items(), key=utils.first))
    #     merged_baselines = dict(sorted(merged_baselines.items(), key=utils.first))
    #
    #
    #     with open(output_name, "w") as output_file:
    #         for (observed_key, observed_values), (expected_key, expected_values) in zip(merged_trials.items(),
    #                                                                                     merged_baselines.items()):
    #             output_file.write(f"Observed_{observed_key}")
    #             output_file.write("".join([f"\t{item}" for item in observed_values]))
    #             output_file.write("\n")
    #             output_file.write(f"Expected_{expected_key}")
    #             output_file.write("".join([f"\t{item}" for item in expected_values]))
    #             output_file.write("\n")



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

    miner_result_dicts: list[dict] = [get_miner_result_dict_from_file(input_file) for input_file in json_file_list]
    aggregated = merge_results_by_keys(miner_result_dicts)

    with open(output_file_name, "w") as output_file:
        for category, results in aggregated.items():
            output_file.write(f"{category},")
            output_file.write(",".join([f"{item:.3f}" for item in results]))
            output_file.write("\n")
