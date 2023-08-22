from enum import Enum
from typing import Optional

from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from BenchmarkProblems.CombinatorialConstrainedProblem import CombinatorialConstrainedProblem
import SearchSpace


class Item:
    name: str
    price: float
    weight: int
    volume: int  # a value from 1 to 10, where 1 is a pen and 10 is a laptop

    def __init__(self, name, price, weight, volume):
        self.name = name
        self.price = price
        self.weight = weight  # in grams
        self.volume = volume

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self.name.__hash__()


water_bottle = Item("water bottle", 1.00, 250, 3)
pen = Item("pen", 0.20, 5, 1)
bananas = Item("bananas", 1.25, 300, 4)
oranges = Item("oranges", 1.25, 400, 4)
hat = Item("hat", 4.00, 100, 2)
socks = Item("socks", 1.00, 200, 2)
laptop = Item("laptop", 500.00, 200, 10)
playing_cards = Item("cards", 1.00, 50, 2)
cash = Item("cash", 0.00, 100, 3)
credit_card = Item("credit card", 0.00, 10, 1)
phone_charger = Item("phone charger", 1.00, 15, 2)
tissues = Item("tissues", 1.00, 5, 1)
book = Item("book", 2.00, 250, 4)
candy = Item("candy", 2.00, 100, 3)
energy_drink = Item("energy drink", 1.50, 150, 3)
lock = Item("lock", 4.00, 50, 1)
local_map = Item("local map", 3.00, 100, 2)
jacket = Item("jacket", 15.00, 500, 8)
knitting = Item("knitting", 5.00, 300, 6)
shampoo = Item("shampoo", 3.00, 250, 4)
cutlery = Item("cutlery", 2.00, 50, 2)
headphones = Item("headphones", 10.00, 150, 5)
earphones = Item("earphones", 4.00, 15, 3)
sunglasses = Item("sunglasses", 2.00, 20, 4)
travel_towel = Item("travel towel", 2.00, 30, 7)
thermos = Item("thermos", 2.00, 50, 6)
crosswords = Item("crosswords", 1.50, 25, 3)
swimming_trunks = Item("swimming_trunks", 3.00, 150, 4)
bread = Item("bread", 0.75, 120, 5)
coins = Item("coins", 0.00, 200, 3)
sunscreen = Item("sunscreen", 2.00, 100, 3)

all_items = [water_bottle, pen, bananas, oranges, hat, socks, laptop, playing_cards, cash, credit_card, phone_charger,
             tissues,
             book, candy, energy_drink, lock, local_map, jacket, knitting, shampoo, cutlery, headphones, earphones,
             sunglasses,
             travel_towel, thermos, crosswords, swimming_trunks, bread, coins, sunscreen]


class KnapsackProblem(CombinatorialProblem):
    expected_price: float
    expected_weight: int
    expected_weight: int

    def __init__(self, expected_price, expected_weight, expected_volume):
        search_space = SearchSpace.SearchSpace([2] * len(all_items))
        self.expected_price = expected_price
        self.expected_weight = expected_weight
        self.expected_volume = expected_volume
        super().__init__(search_space)

    def __repr__(self):
        return f"Knapsack({self.expected_price}, {self.expected_weight}, {self.expected_volume})"

    def feature_as_bools(self, feature: SearchSpace.Feature) -> list[Optional[bool]]:
        return [None if value is None else bool(value) for value in super().get_positional_values(feature)]

    def feature_repr(self, feature: SearchSpace.Feature):
        present_items = [all_items[var] for var, val in feature.var_vals if val == 1]
        absent_items = [all_items[var] for var, val in feature.var_vals if val == 0]

        result = ""
        if len(present_items) > 0:
            result += f"Bring:{present_items}"
        if len(absent_items) > 0:
            if result != "":
                result += "\n"
            result += f"DO NOT Bring:{absent_items}"

        price, weight, volume = self.get_properties_of_candidate(self.search_space.feature_to_candidate(feature))
        # result += f"\n{price = }, {weight = }, {volume = }"

        return result

    def get_items_brought_in_candidate(self, candidate: SearchSpace.Candidate):
        return [item for item, bring in zip(all_items, candidate.values) if bring == 1]

    def get_properties_of_candidate(self, candidate: SearchSpace.Candidate) -> (float, int, int):
        brought_items = self.get_items_brought_in_candidate(candidate)
        price = sum(item.price for item in brought_items)
        weight = sum(item.weight for item in brought_items)
        volume = sum(item.volume for item in brought_items)

        return price, weight, volume

    def score_of_candidate(self, candidate: SearchSpace.Candidate) -> float:
        price, weight, volume = self.get_properties_of_candidate(candidate)

        def score_for_property(observed, expected):
            return abs(observed - expected) / expected

        return sum(score_for_property(total_property, expected)
                   for total_property, expected in zip([price, weight, volume],
                                                       [self.expected_price, self.expected_weight,
                                                        self.expected_volume]))

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        return super().amount_of_set_values_in_feature(feature)


class KnapsackConstraint(Enum):
    DRINK = 1
    FOOD = 2
    PAYMENT = 3
    BEACH = 4

    def __repr__(self):
        return ["drink", "food", "payment", "beach"][self.value - 1]

    def __str__(self):
        return self.__repr__()


class ConstrainedKnapsackProblem(CombinatorialConstrainedProblem):
    needs: list[KnapsackConstraint]
    original_problem: KnapsackProblem

    def __repr__(self):
        price = self.original_problem.expected_price
        weight = self.original_problem.expected_weight
        volume = self.original_problem.expected_volume
        return f"ConstrainedKnapSack(price = {price}, weight = {weight}, volume = {volume}, {self.needs})"

    def __init__(self, unconstrained_problem: KnapsackProblem, needs: list[KnapsackConstraint]):
        self.needs = needs
        constraint_space = SearchSpace.SearchSpace(tuple(2 for need in self.needs))
        self.original_problem = unconstrained_problem
        super().__init__(unconstrained_problem, constraint_space)

    def candidate_contains_any(self, candidate: SearchSpace.Candidate, any_of_these):
        present_items = self.original_problem.get_items_brought_in_candidate(candidate)
        return any(wanted_item in present_items for wanted_item in any_of_these)

    def candidate_contains_all(self, candidate: SearchSpace.Candidate, all_of_these):
        present_items = self.original_problem.get_items_brought_in_candidate(candidate)
        return all(wanted_item in present_items for wanted_item in all_of_these)

    def can_drink(self, candidate: SearchSpace.Candidate) -> bool:
        drinkable = [water_bottle, oranges, energy_drink]
        return self.candidate_contains_any(candidate, drinkable)

    def can_eat(self, candidate: SearchSpace.Candidate) -> bool:
        edible = [bananas, oranges, candy, bread]
        return self.candidate_contains_any(candidate, edible)

    def can_pay(self, candidate: SearchSpace.Candidate) -> bool:
        payment_methods = [cash, credit_card, coins]
        return self.candidate_contains_any(candidate, payment_methods)

    def can_go_to_the_beach(self, candidate: SearchSpace.Candidate) -> bool:
        for_the_beach = [hat, swimming_trunks, sunscreen]
        return self.candidate_contains_all(candidate, for_the_beach)

    def satisfies_constraint(self, candidate: SearchSpace.Candidate, constraint: KnapsackConstraint):
        if constraint == KnapsackConstraint.DRINK:
            return self.can_drink(candidate)
        elif constraint == KnapsackConstraint.FOOD:
            return self.can_eat(candidate)
        elif constraint == KnapsackConstraint.PAYMENT:
            return self.can_pay(candidate)
        elif constraint == KnapsackConstraint.BEACH:
            return self.can_go_to_the_beach(candidate)
        else:
            raise Exception("Constraint was not recognised")

    def get_predicates(self, candidate: SearchSpace.Candidate) -> SearchSpace.Candidate:
        def value_for_need(need):
            return int(self.satisfies_constraint(candidate, need))

        return SearchSpace.Candidate(tuple(value_for_need(need) for need in self.needs))

    def predicate_feature_repr(self, constraint: SearchSpace.Feature) -> str:

        yes = "✓"
        no = "⤬"

        return ", ".join(f"{self.needs[need_index]}({yes if satisfied else no})"
                         for (need_index, satisfied) in constraint.var_vals)

    def get_complexity_of_feature(self, feature: SearchSpace.Feature) -> float:
        unconstrained_feature, predicates = super().split_feature(feature)
        return self.unconstrained_problem.get_complexity_of_feature(unconstrained_feature)
