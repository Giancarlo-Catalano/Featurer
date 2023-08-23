import random

import SearchSpace
import utils
from enum import Enum
from BenchmarkProblems.CombinatorialProblem import CombinatorialProblem
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


worker_names = ["Amy", "Bob", "Chris", "Darcy", "Elly", "Frank", "Gian", "Hugh", "Igo", "Joe", "Kyle", "Lola", "Moira",
                "Naomi", "Otto", "Pascal", "Quinn", "Ruth", "Seth", "Ted", "Ugo", "Vicky", "Walter", "Xenia", "Yves",
                "Zeno"]


class SimpleWorker:
    name: str
    available_schedules: list

    def __init__(self, name, available_schedules):
        self.name = name
        self.available_schedules = available_schedules

    def __repr__(self):
        return self.name + " has options " + \
            "; ".join([schedule.__repr__() for schedule in self.available_schedules])

    @classmethod
    def random(cls, how_many_options):
        random_name = random.choice(worker_names)
        how_many_days_to_work = random.choice(range(1, 4))
        options = [WeeklySchedule.random(how_many_days_to_work) for _ in range(how_many_options)]
        return cls(random_name, options)


class SimplifiedBTProblem(CombinatorialProblem):
    total_workers: int
    amount_of_choices: int
    workers: list

    def __init__(self, amount_of_workers, amount_of_choices):
        self.total_workers = amount_of_workers
        self.amount_of_choices = amount_of_choices
        self.workers = [SimpleWorker.random(amount_of_choices) for _ in range(amount_of_workers)]
        super().__init__(SearchSpace.SearchSpace([self.amount_of_choices] * self.total_workers))

    def __repr__(self):
        # return f"BTProblem:(" \
        #        f"\n\t workers: " + \
        #     "\n\t\t".join([worker.__repr__() for worker in self.workers])
        return "BTProblem"

    def get_complexity_of_feature(self, feature: SearchSpace.Feature):
        amount_of_workers = super().amount_of_set_values_in_feature(feature)
        return amount_of_workers

    def get_resulting_roster(self, candidate: SearchSpace.Candidate):
        chosen_schedules = [worker.available_schedules[choice]
                            for (worker, choice) in zip(self.workers, candidate.values)]
        counts_for_each_day = [0] * 7
        for schedule in chosen_schedules:
            for day in schedule.working_days:
                counts_for_each_day[day.value] += 1

        return counts_for_each_day

    def get_homogeneity_of_roster(self, roster):
        (min_amount, max_amount) = utils.min_max(roster)
        if max_amount == 0:
            return 0

        return (max_amount - min_amount) / max_amount

    def get_preference_score_of_candidate(self, candidate: SearchSpace.Candidate):
        how_many_top_choices = len([chosen_schedule_index for chosen_schedule_index
                                    in candidate.values if chosen_schedule_index == 0])
        return how_many_top_choices / self.total_workers

    def score_of_candidate(self, candidate: SearchSpace.Candidate):
        roster = self.get_resulting_roster(candidate)
        return self.get_homogeneity_of_roster(roster)

    def feature_repr(self, feature):
        def repr_of_var_val(var, val):
            worker = self.workers[var]
            chosen_schedule = worker.available_schedules[val]
            return f"{worker.name} with rota #{val}:{chosen_schedule.__repr__()}"

        return "\n".join([repr_of_var_val(var, val) for (var, val) in feature.var_vals])


class WorkerRota:
    week_size: int
    pattern: list[bool]

    def __init__(self, week_size, pattern):
        self.week_size = week_size
        self.pattern = pattern


    def __repr__(self):
        week_starts = [day_index for day_index in range(len(self.pattern)) if day_index % self.week_size == 0]
        weeks = [self.pattern[week_start:(week_start+self.week_size)] for week_start in week_starts]
        def repr_week(week):
            return "["+"".join(["*" if works else "_" for works in week])+"]"

        return " ".join(repr_week(week) for week in weeks)

    @classmethod
    def get_random(cls, week_size):
        def random_day():
            return random.choice([True, False])

        how_many_weeks = random.randrange(1, 5)
        amount_of_days = how_many_weeks * week_size
        return cls(week_size, [random_day() for _ in range(amount_of_days)])



class Worker:
    name: str
    options: list[WorkerRota]


    def __init__(self, name, options):
        self.name = name
        self.options = options


    @classmethod
    def get_random(cls, name, amount_of_options):
        week_size = random.randrange(4, 8)
        options = [WorkerRota.get_random(week_size) for _ in range(amount_of_options)]
        return cls(name, options)


    def __repr__(self):
        return f"{self.name}, with rotas:"+ (
            "\n\t".join(f"{rota}" for rota in self.options)
        )



class BTProblem(CombinatorialProblem):
    total_workers: int
    amount_of_choices_per_worker: int
    workers: list[Worker]
    total_roster_length: int


    def __init__(self, total_workers, amount_of_choices_per_worker, total_rota_length):
        self.total_workers = total_workers
        self.amount_of_choices_per_worker = amount_of_choices_per_worker
        self.workers = [Worker.get_random(name, amount_of_choices_per_worker) for name in worker_names[:total_workers]]
        self.total_roster_length = total_rota_length

        super().__init__(SearchSpace.SearchSpace(self.amount_of_choices_per_worker for worker in self.workers))


    def __repr__(self):
        return f"BTProblem({self.total_workers}, {self.amount_of_choices_per_worker}, {self.total_roster_length})"

    def get_rotas_in_feature(self, feature: SearchSpace.Feature) -> list[WorkerRota]:
        return [self.workers[worker_index].options[which_rota]
                for worker_index, which_rota in feature.var_vals]


    def get_rotas_in_candidate(self, candidate: SearchSpace.Candidate) -> list[WorkerRota]:
        return [worker.options[which_rota] for worker, which_rota in zip(self.workers, candidate.values)]

    def feature_repr(self, feature: SearchSpace.Feature) -> str:
        def repr_worker_and_rota(worker_index, rota_index):
            worker = self.workers[worker_index]
            rota = worker.options[rota_index]
            return f"{worker.name} on #{rota_index} = \t\t{rota}"

        return "\n".join(repr_worker_and_rota(worker_index, rota_index)
                         for worker_index, rota_index in feature.var_vals)


    def extend_rota_to_total_roster(self, rota: WorkerRota) -> list[bool]:
        result = []
        while len(result) < self.total_roster_length:
            result.extend(rota.pattern)

        return result[: self.total_roster_length]

    def get_counts_for_overlapped_rotas(self, rotas: list[WorkerRota]) -> np.ndarray:
        extended_rotas = [self.extend_rota_to_total_roster(rota) for rota in rotas]
        as_int_grid = np.array(extended_rotas, dtype=int)
        return (np.sum(as_int_grid, axis=0))


    def get_min_and_max_for_each_work_day(self, candidate: SearchSpace.Candidate) -> list[(int, int)]:
        rotas = self.get_rotas_in_candidate(candidate)
        counts_for_each_day = self.get_counts_for_overlapped_rotas(rotas)
        counts_for_each_day = counts_for_each_day.reshape((-1, 7))
        minimums = np.min(counts_for_each_day, axis=0)
        maximums = np.max(counts_for_each_day, axis=0)

        return list(zip(list(minimums), list(maximums)))

    def get_amount_of_first_choices(self, candidate:SearchSpace.Candidate) -> int:
        return len([value for value in candidate.values if value == 0])


    def get_range_score_of_candidate(self, candidate: SearchSpace.Candidate) -> float:
        def range_score(min_value, max_value):
            if max_value == 0:
                return 1000
            else:
                return (max_value-min_value)/max_value
        mins_and_maxs = self.get_min_and_max_for_each_work_day(candidate)
        range_scores = [range_score(max_val, min_val) for max_val, min_val in mins_and_maxs]
        weights_for_days = [1, 1, 1, 1, 1, 10, 10]
        return sum((range_val**2) * weight for range_val, weight in zip(range_scores, weights_for_days))

    def score_of_candidate(self, candidate: SearchSpace.Candidate) -> float:
        range_score = self.get_range_score_of_candidate(candidate)
        preference_score = self.get_amount_of_first_choices(candidate)
        weight_of_preference = 0.001

        return range_score - weight_of_preference * preference_score

    def get_complexity_of_feature(self, feature: SearchSpace.Feature) -> float:
        return super().amount_of_set_values_in_feature(feature)










