import utils
from BenchmarkProblems.ArtificialProblem import ArtificialProblem
from BenchmarkProblems.BinVal import BinValProblem
from BenchmarkProblems.CheckerBoard import CheckerBoardProblem
from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem, TestableCombinatorialProblem
from BenchmarkProblems.GraphColouring import GraphColouringProblem
from BenchmarkProblems.Knapsack import KnapsackProblem
from BenchmarkProblems.OneMax import OneMaxProblem
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
                                     amount_of_nodes=properties["amount_of_nodes"],
                                     chance_of_connection=properties["chance_of_connection"])
    else:
        raise Exception("The problem was not recognised")

def make_problem(which, size: str):
    if size == "small":
        amount_of_bits = 5
    elif size == "medium":
        amount_of_bits = 25
    elif size == "large":
        amount_of_bits = 100
    else:
        amount_of_bits = 20


    if which == "binval":
        return {"which": which,
                "size": amount_of_bits,
                "base": 2}
    elif which == "onemax":
        return {"which": which,
                "size": amount_of_bits}
    elif which == "trapk":
        return {"which": which,
                "k": 5,
                "amount_of_groups": (amount_of_bits // 5)}
    elif which == "checkerboard":
        if size == "small":
            rows = 4
        elif size == "medium":
            rows = 8
        else:
            rows = 16
        return {"which": which,
                "rows": rows,
                "cols": rows}
    elif which == "artificial":
        return {"which": which,
                "size": amount_of_bits,
                "size_of_partials": 5,
                "amount_of_features": 5,
                "allow_overlaps": False}
    elif which == "knapsack":
        return {"which": which,
                "expected_price": 50,
                "expected_weight": 1000,
                "expected_volume": 15}
    elif which == "graph":
        return {"which": which,
                "amount_of_colours": 4,
                "amount_of_nodes": amount_of_bits,
                "chance_of_connection": 0.3}
    else:
        raise Exception("Problem string was not recognised")



