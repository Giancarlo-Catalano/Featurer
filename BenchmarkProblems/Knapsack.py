from typing import Optional

from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
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

all_items = [water_bottle, pen, bananas, oranges, hat, socks, laptop, playing_cards, cash, credit_card, phone_charger,
             tissues,
             book, candy, energy_drink, lock, local_map, jacket, knitting, shampoo, cutlery, headphones, earphones,
             sunglasses,
             travel_towel, thermos, crosswords, swimming_trunks]


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
            return observed / expected

        return sum(score_for_property(total_property, expected)
                   for total_property, expected in zip([price, weight, volume],
                                                       [self.expected_price, self.expected_weight,
                                                        self.expected_volume]))

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        return super().amount_of_set_values_in_feature(feature)
