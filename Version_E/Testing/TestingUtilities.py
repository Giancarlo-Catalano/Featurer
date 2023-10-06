import datetime
import json
from json import JSONDecodeError

from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from SearchSpace import Candidate
from Version_E.InterestingAlgorithms.Miner import FeatureSelector
from Version_E.PrecomputedPopulationInformation import PrecomputedPopulationInformation
from Version_E.Testing import Miners, Criteria, Problems, Tests

JSON = dict


def to_json_object(input) -> JSON:
    return json.loads(input)


def get_random_candidates_and_fitnesses(problem: CombinatorialProblem,
                                        sample_size) -> (list[Candidate], list[float]):
    random_candidates = [problem.get_random_candidate_solution() for _ in range(sample_size)]

    scores = [problem.score_of_candidate(c) for c in random_candidates]
    return random_candidates, scores

def error_result(description) -> JSON:
    return {"error": description}


def run_test(arguments: dict):
    print(f"Received the arguments {arguments}")

    problem = Problems.decode_problem(arguments["problem"])
    criterion = Criteria.decode_criterion(arguments["criterion"], problem)
    sample_size = arguments["sample_size"]
    training_ppi = PrecomputedPopulationInformation.from_problem(problem, sample_size)

    selector = FeatureSelector(training_ppi, criterion)
    miner = Miners.decode_miner(arguments["miner"], selector)

    test_dict = arguments["test"]
    result_json = Tests.apply_test(test_dict, problem, miner, arguments)  # important part

    test_type = arguments["test"]["which"]
    problem = Problems.decode_problem(arguments["problem"])
    now = datetime.datetime.now()
    output_name = f"{test_type}~{problem}~{miner}_({now.day}-{now.month})_[{now.hour}_{now.minute}].json"

    output_json = {"parameters": arguments, "result": result_json}

    print(f"The result is {output_json}")

    with open(output_name, 'w') as json_file:
        json.dump(output_json, json_file)



