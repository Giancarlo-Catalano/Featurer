import random

import utils
from BenchmarkProblems.ArtificialProblem import ArtificialProblem
from BenchmarkProblems.BinVal import BinValProblem
from BenchmarkProblems.CheckerBoard import CheckerBoardProblem
from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem, TestableCombinatorialProblem
from BenchmarkProblems.GraphColouring import GraphColouringProblem, InsularGraphColouringProblem
from BenchmarkProblems.Knapsack import KnapsackProblem
from BenchmarkProblems.OneMax import OneMaxProblem
from BenchmarkProblems.PlateauProblem import PlateauProblem
from BenchmarkProblems.Shuffle import ShuffleProblem
from BenchmarkProblems.TrapK import TrapK


def decode_problem(properties: dict) -> CombinatorialProblem:
    problem_string = properties["which"]

    if problem_string == "binval":
        return BinValProblem(amount_of_bits=properties["size"],
                             base=properties["base"])
    elif problem_string == "onemax":
        return OneMaxProblem(amount_of_bits=properties["size"])
    elif problem_string == "trapk":
        return TrapK(amount_of_groups=properties["amount_of_groups"],
                     k=properties["k"])
    elif problem_string == "checkerboard":
        return CheckerBoardProblem(rows=properties["rows"],
                                   cols=properties["cols"])
    elif problem_string == "artificial":
        return ArtificialProblem(amount_of_bits=properties["size"],
                                 size_of_partials=properties["size_of_partials"],
                                 amount_of_features=properties["amount_of_features"],
                                 allow_overlaps=properties["allow_overlaps"])
    elif problem_string == "knapsack":
        return KnapsackProblem(expected_price=properties["expected_price"],
                               expected_weight=properties["expected_weight"],
                               expected_volume=properties["expected_volume"])
    elif problem_string == "graph":
        return GraphColouringProblem(amount_of_colours=properties["amount_of_colours"],
                                     amount_of_nodes=random.randrange(10, 30),
                                     chance_of_connection=properties["chance_of_connection"])
    elif problem_string == "insular":
        return InsularGraphColouringProblem(amount_of_islets = properties["amount_of_islets"])
    elif problem_string == "plateau":
        return PlateauProblem(amount_of_groups = properties["amount_of_groups"])
    elif problem_string == "shuffle":
        return ShuffleProblem(decode_problem(properties["problem"]))
    else:
        raise Exception("The problem was not recognised")

