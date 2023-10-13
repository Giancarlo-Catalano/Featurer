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
    #print(f"Received the arguments {arguments}")

    result_json = Tests.apply_test(arguments)  # important part

    test_type = arguments["test"]["which"]
    problem_kind = arguments["problem"]["which"]
    miner_kind = arguments["miner"]["which"]
    now = datetime.datetime.now()
    output_name = f"{test_type}~{problem_kind}~{miner_kind}_({now.day}-{now.month})_[{now.hour}_{now.minute}].json"

    output_json = {"parameters": arguments, "result": result_json}

    print(json.dumps(output_json))



