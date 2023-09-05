import random
from typing import Optional

import SearchSpace
import utils
from enum import Enum, auto
from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
from BenchmarkProblems.CombinatorialConstrainedProblem import CombinatorialConstrainedProblem
import numpy as np


class Weekday(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

    @classmethod
    def random(cls):
        return cls(random.randrange(7))

    def __repr__(self):
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][self.value]

    def __lt__(self, other):
        return self.value < other.value


weekdays = [Weekday.MONDAY, Weekday.TUESDAY, Weekday.WEDNESDAY, Weekday.THURSDAY, Weekday.FRIDAY, Weekday.SATURDAY,
            Weekday.SUNDAY]


class WeeklySchedule:
    working_days: list

    def __init__(self, working_days: list):
        self.working_days = working_days

    @classmethod
    def random(cls, how_many_days_to_work):
        which_days = random.sample(weekdays, how_many_days_to_work)
        which_days.sort()
        return cls(which_days)

    def __repr__(self):
        return self.working_days.__repr__()


worker_first_names = ["Amy", "Bob", "Chris", "Darcy", "Elly", "Frank", "Gian", "Hugh", "Igo", "Joe", "Kyle", "Lola",
                      "Moira",
                      "Naomi", "Otto", "Pascal", "Quinn", "Ruth", "Seth", "Ted", "Ugo", "Vicky", "Walter", "Xenia",
                      "Yves",
                      "Zeno"]

worker_surnames = ["Anderson", "Bokmal", "Catalano", "Devon", "Elsinor", "Fourier", "Gomez", "Hayashi", "Immaut",
                   "Jackson", "Kingstone", "Ling", "Morris", "Noivern", "Olowasamelori", "Pulitser", "Quasimodo",
                   "Rossi", "Stradivarius", "Turner", "Umm Summa",
                   "Vladivosov", "Wieux", "Xerox", "Ypritte", "Zeppelin"]


def get_worker_names(amount_of_workers) -> list[str]:
    if len(worker_first_names) >= amount_of_workers:
        return worker_first_names[:amount_of_workers]

    def add_surnames(surname_permutation):
        return [first_name + " " + surname for first_name, surname in zip(worker_first_names, surname_permutation)]

    resulting_names = []

    while len(resulting_names) < amount_of_workers:
        resulting_names.extend(add_surnames(random.choices(worker_surnames, k=len(worker_first_names))))
        if len(resulting_names) >= amount_of_workers:
            break

    return resulting_names[:amount_of_workers]


class WorkerRota:
    week_size: int
    pattern: list[bool]

    def __init__(self, week_size, pattern):
        self.week_size = week_size
        self.pattern = pattern

    @property
    def total_days(self):
        return len(self.pattern)

    def __repr__(self):
        week_starts = [day_index for day_index in range(len(self.pattern)) if day_index % self.week_size == 0]
        weeks = [self.pattern[week_start:(week_start + self.week_size)] for week_start in week_starts]

        def repr_week(week):
            return "[" + "".join(["*" if works else "_" for works in week]) + "]"

        return " ".join(repr_week(week) for week in weeks)

    @classmethod
    def get_random(cls, week_size, total_days):
        def random_day():
            return random.choice([True, False])

        return cls(week_size, [random_day() for _ in range(total_days)])

    def get_rotated(self, starting_day):
        wrap_around = self.pattern[:starting_day]
        after_start = self.pattern[starting_day:]
        new_pattern = after_start + wrap_around
        return WorkerRota(self.week_size, new_pattern)


class Worker:
    name: str
    options: list[WorkerRota]
    days_in_pattern: int

    def __init__(self, name, options):
        self.name = name
        self.options = options

    @property
    def amount_of_rota_choices(self) -> int:
        return len(self.options)

    @property
    def days_in_pattern(self) -> int:
        # we assume that there's always at least one option
        return len(self.options[0].pattern)

    @property
    def week_size(self) -> int:
        return self.options[0].week_size

    @classmethod
    def get_random(cls, name, amount_of_options):
        week_size = random.randrange(4, 8)
        amount_of_weeks = random.randrange(1, 6)
        days_in_pattern = week_size * amount_of_weeks
        options = [WorkerRota.get_random(week_size, days_in_pattern) for _ in range(amount_of_options)]
        return cls(name, options)

    def __repr__(self):
        return f"{self.name}, with rotas:\n\t" + (
            "\n\t".join(f"{rota}" for rota in self.options)
        )

    def get_search_space_for_worker(self):
        return SearchSpace.SearchSpace([self.amount_of_rota_choices, self.days_in_pattern])

    def get_effective_rota_from_indices(self, rota_option, starting_day) -> WorkerRota:
        return self.options[rota_option].get_rotated(starting_day)


class BTProblem(CombinatorialProblem):
    total_workers: int
    amount_of_choices_per_worker: int
    workers: list[Worker]
    total_roster_length: int

    def get_worker_names(self):
        if self.total_workers < len(worker_first_names):
            return worker_first_names[:self.total_workers]

    def __init__(self, total_workers, amount_of_choices_per_worker, total_rota_length):
        self.total_workers = total_workers
        self.amount_of_choices_per_worker = amount_of_choices_per_worker
        self.workers = [Worker.get_random(name, amount_of_choices_per_worker)
                        for name in get_worker_names(self.total_workers)]
        self.total_roster_length = total_rota_length

        search_space = SearchSpace.merge_many_spaces([worker.get_search_space_for_worker() for worker in self.workers])
        super().__init__(search_space)

    def __repr__(self):
        return f"BTProblem({self.total_workers}, {self.amount_of_choices_per_worker}, {self.total_roster_length})"

    def long_repr(self):
        return f"The workers and their options are:\n\t" + "\n\t".join([f"{worker}" for worker in self.workers])

    def get_rotas_in_feature(self, feature: SearchSpace.Feature) -> list[WorkerRota]:
        return [self.workers[worker_index].options[which_rota]
                for worker_index, which_rota in feature.var_vals]

    def break_candidate_by_worker(self, candidate: SearchSpace.Candidate) -> list[(int, int)]:
        return [(candidate.values[2 * i], candidate.values[2 * i + 1]) for i, _ in enumerate(self.workers)]

    def get_rotas_in_candidate(self, candidate: SearchSpace.Candidate) -> list[WorkerRota]:
        rota_choices_and_starts = self.break_candidate_by_worker(candidate)
        return [worker.get_effective_rota_from_indices(which_rota, starting_day)
                for (worker, (which_rota, starting_day)) in zip(self.workers, rota_choices_and_starts)]

    def break_feature_by_worker(self, feature: SearchSpace.Feature) -> list[(Optional[int], Optional[int])]:
        result: list[[Optional[int], Optional[int]]] = [[None, None] for _ in self.workers]

        # the items are lists of 2 values rather than tuples, because they will be mutated
        def update_result(var, val):
            worker_index, is_starting_day_var = divmod(var, 2)
            if is_starting_day_var:
                result[worker_index][is_starting_day_var] = val

        for var, val in feature.var_vals:
            update_result(var, val)

        # now we can convert them into tuples
        return [tuple(params) for params in result]

    def feature_repr(self, feature: SearchSpace.Feature) -> str:

        def repr_worker_parameters(worker: Worker, rota_index: Optional[int], starting_day: Optional[int]) -> str:
            result = f"{worker.name} "
            if rota_index is not None:
                result += f" rota #{rota_index},"
            if starting_day is not None:
                result += f" starting from day #{starting_day}"
            return result

        def is_worth_showing(rota_index, starting_day) -> bool:
            return (rota_index is not None) or (starting_day is not None)

        worker_parameters = self.break_feature_by_worker(feature)

        return "\n".join(repr_worker_parameters(worker, rota_index, starting_day)
                         for (worker, (rota_index, starting_day)) in zip(self.workers, worker_parameters)
                         if is_worth_showing(rota_index, starting_day))

    def extend_rota_to_total_roster(self, rota: WorkerRota) -> list[bool]:
        result = []
        while len(result) < self.total_roster_length:
            result.extend(rota.pattern)

        return result[: self.total_roster_length]

    def get_counts_for_overlapped_rotas(self, rotas: list[WorkerRota]) -> np.ndarray:
        extended_rotas = [self.extend_rota_to_total_roster(rota) for rota in rotas]
        as_int_grid = np.array(extended_rotas, dtype=int)
        return np.sum(as_int_grid, axis=0)

    def get_min_and_max_for_each_work_day(self, candidate: SearchSpace.Candidate) -> list[(int, int)]:
        rotas = self.get_rotas_in_candidate(candidate)
        counts_for_each_day = self.get_counts_for_overlapped_rotas(rotas)
        counts_for_each_day = counts_for_each_day.reshape((-1, 7))
        minimums = np.min(counts_for_each_day, axis=0)
        maximums = np.max(counts_for_each_day, axis=0)

        return list(zip(list(minimums), list(maximums)))

    def get_amount_of_first_choices(self, candidate: SearchSpace.Candidate) -> int:
        return len([value for value in candidate.values if value == 0])

    def get_range_scores_for_each_day(self, candidate: SearchSpace.Candidate) -> list[float]:
        def range_score(min_value, max_value):
            if max_value == 0:
                return 1000
            else:
                return (max_value - min_value) / max_value

        mins_and_maxs = self.get_min_and_max_for_each_work_day(candidate)
        return [range_score(max_val, min_val) for max_val, min_val in mins_and_maxs]

    def get_range_score_of_candidate(self, candidate: SearchSpace.Candidate) -> float:
        range_scores = self.get_range_scores_for_each_day(candidate)
        weights_for_days = [1, 1, 1, 1, 1, 10, 10]
        return sum((range_val ** 2) * weight for range_val, weight in zip(range_scores, weights_for_days))

    def score_of_candidate(self, candidate: SearchSpace.Candidate) -> float:
        range_score = self.get_range_score_of_candidate(candidate)
        preference_score = self.get_amount_of_first_choices(candidate)
        weight_of_preference = 0.001

        return range_score - weight_of_preference * preference_score

    def get_complexity_of_feature(self, feature: SearchSpace.Feature) -> float:
        # order of preference:
        # nothing at all: bestest
        # worker with a rota and starting day: best
        # worker with a rota: almost worst
        # worker with just a starting day: worst

        def complexity_for_values(rota_index: Optional[int], starting_day: Optional[int]):
            if (rota_index is None) and (starting_day is None):
                return 0
            elif (rota_index is not None) and (starting_day is None):
                return 50
            elif (rota_index is None) and (starting_day is not None):
                return 100
            elif (rota_index is not None) and (starting_day is not None):
                return 1

        worker_params = self.break_feature_by_worker(feature)
        amount_of_workers = len(
            [1 for rota_index, starting_day in worker_params if rota_index is not None or starting_day is not None])
        worker_amount_malus = abs(amount_of_workers - 3) * 5
        return sum(complexity_for_values(*params) for params in worker_params) # + worker_amount_malus


class BTPredicate(Enum):
    BAD_MONDAY = auto()
    BAD_TUESDAY = auto()
    BAD_WEDNESDAY = auto()
    BAD_THURSDAY = auto()
    BAD_FRIDAY = auto()
    BAD_SATURDAY = auto()
    BAD_SUNDAY = auto()

    EXCEEDS_WEEKLY_HOURS = auto()
    CONSECUTIVE_WEEKENDS = auto()

    def __repr__(self):
        return ["Stable Monday", "Stable Tuesday", "Stable Wednesday",
                "Stable Thursday", "Stable Friday", "Stable Saturday",
                "Stable Sunday", "Exceeds weekly working hours", "Has consecutive weekends"][self.value - 1]

    def __str__(self):
        return self.__repr__()

    def to_week_day(self):
        return self.value - 1  # THIS MEANS YOU CAN'T ADD MORE PREDICATES ABOVE MONDAY


class ExpandedBTProblem(CombinatorialConstrainedProblem):
    original_problem: BTProblem
    predicates: list[BTPredicate]

    def __init__(self, original_problem, predicates: list[BTPredicate]):
        self.original_problem = original_problem
        self.predicates = predicates
        constraint_space = SearchSpace.SearchSpace([2 for _ in self.predicates])
        super().__init__(original_problem, constraint_space)

    def __repr__(self):
        return f"ConstrainedBTProblem({self.original_problem}, {self.predicates})"

    def long_repr(self):
        return self.original_problem.long_repr()

    def rota_exceeds_weekly_working_hours(self, worker_rota: WorkerRota) -> bool:
        max_allowed_weekly_hours = 48
        extended_rota = self.original_problem.extend_rota_to_total_roster(worker_rota)
        as_matrix = np.array(extended_rota, dtype=int)
        as_matrix = as_matrix.reshape((-1, 7))
        hours_per_day = 8
        weekly_hours = np.sum(as_matrix, axis=1) * hours_per_day
        return any(week_hours > max_allowed_weekly_hours for week_hours in weekly_hours)

    def any_rotas_exceed_weekly_working_hours(self, candidate: SearchSpace.Candidate):
        rotas = self.original_problem.get_rotas_in_candidate(candidate)
        return any(self.rota_exceeds_weekly_working_hours(rota) for rota in rotas)

    def amount_of_consecutive_weekends_in_rota(self, worker_rota: WorkerRota) -> int:
        def get_working_weekends(rota):
            extended = self.original_problem.extend_rota_to_total_roster(rota)
            as_matrix = np.array(extended)
            as_matrix.reshape((-1, 7))
            return [week[5] or week[6] for week in as_matrix]

        def count_consecutives(working_weekends):
            return len(list(utils.adjacent_pairs(working_weekends)))

        return count_consecutives(get_working_weekends(worker_rota))

    def any_rotas_have_consecutive_weekends(self, candidate: SearchSpace.Candidate) -> bool:
        amounts_of_consecutive_weekends = [self.amount_of_consecutive_weekends_in_rota(rota)
                                           for rota in self.original_problem.get_rotas_in_candidate(candidate)]
        return any(amount > 1 for amount in amounts_of_consecutive_weekends)
        # in the future this might be returning a percentage of how many workers have to work consecutive weekdays

    def get_bad_weekdays(self, candidate: SearchSpace.Candidate) -> list[int]:
        """Note: the week days are 0 indexed!!"""
        ranges = self.original_problem.get_range_scores_for_each_day(candidate)
        weekdays_and_scores = list(enumerate(ranges))
        weekdays_and_scores.sort(key=utils.second, reverse=True)
        bad_days = utils.unzip(weekdays_and_scores[:3])[0]
        return bad_days

    def get_predicates(self, candidate: SearchSpace.Candidate):
        """the predicates are TRUE when the constraint is VIOLATED"""
        bad_weekdays = self.get_bad_weekdays(candidate)

        def result_of_predicate(predicate: BTPredicate):
            if predicate == BTPredicate.EXCEEDS_WEEKLY_HOURS:
                return self.any_rotas_exceed_weekly_working_hours(candidate)
            elif predicate == BTPredicate.CONSECUTIVE_WEEKENDS:
                return self.any_rotas_have_consecutive_weekends(candidate)
            else:  # weekday check
                weekday = predicate.to_week_day()
                return weekday in bad_weekdays

        return SearchSpace.Candidate([int(result_of_predicate(predicate)) for predicate in self.predicates])

    def predicate_feature_repr(self, predicates: SearchSpace.Feature) -> str:

        def repr_predicate(predicate: BTPredicate, value):
            if predicate == BTPredicate.EXCEEDS_WEEKLY_HOURS:
                return "Exceeds weekly hours" if value else "Within weekly hours"
            elif predicate == BTPredicate.CONSECUTIVE_WEEKENDS:
                return "Contains consecutive working weekends" if value \
                    else "Does not have consecutive working weekends"
            else:
                weekday = predicate.to_week_day()
                weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]
                return f"{weekday_name} is UNstable" if value else f"{weekday_name} is Stable"

        return ", ".join(repr_predicate(self.predicates[var], bool(val)) for var, val in predicates.var_vals)

    def score_of_candidate(self, candidate: SearchSpace.Candidate) -> float:
        original_candidate, predicates = self.split_candidate(candidate)
        normal_score = self.original_problem.score_of_candidate(candidate)

        if (BTPredicate.EXCEEDS_WEEKLY_HOURS in self.predicates
                and self.any_rotas_exceed_weekly_working_hours(original_candidate)):
            return 1000.0  # this is a minimisation task, so we return a big value when the constraint is broken
        else:
            return normal_score

    def get_complexity_of_predicates(self, predicates: SearchSpace.Feature):
        def complexity_of_predicate(predicate, value):
            if predicate == BTPredicate.EXCEEDS_WEEKLY_HOURS:
                return 2 if value else 20
            elif predicate == BTPredicate.CONSECUTIVE_WEEKENDS:
                return 5 if value else 2
            else:
                return 5 if value else 1

        if predicates.var_vals:
            return sum(complexity_of_predicate(self.predicates[index], bool(val))
                       for index, val in predicates.var_vals)

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        partial_solution_parameters, descriptors_partial_solution = super().split_feature(feature)

        amount_of_workers = super().amount_of_set_values_in_feature(partial_solution_parameters)
        predicates_are_present = super().amount_of_set_values_in_feature(descriptors_partial_solution) > 0

        if predicates_are_present:
            return amount_of_workers + \
                self.get_complexity_of_predicates(descriptors_partial_solution)
        else:
            return 1000
