import random

import SearchSpace
import utils
from enum import Enum
import CombinatorialProblem


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


class Worker:
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


class BTProblem(CombinatorialProblem.CombinatorialProblem):
    total_workers: int
    amount_of_choices: int
    workers: list

    def __init__(self, amount_of_workers, amount_of_choices):
        super().__init__(SearchSpace.SearchSpace([self.amount_of_choices] * self.total_workers))
        self.total_workers = amount_of_workers
        self.amount_of_choices = amount_of_choices
        self.workers = [Worker.random(amount_of_choices) for _ in range(amount_of_workers)]

    def __repr__(self):
        return f"BTProblem:(" \
               f"\n\t workers: " + \
            "\n\t\t".join([worker.__repr__() for worker in self.workers])

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

    def pretty_print_feature(self, feature):
        def repr_of_var_val(var, val):
            worker = self.workers[var]
            chosen_schedule = worker.available_schedules[val]
            return f"{worker.name} with rota #{val}:{chosen_schedule.__repr__()}"

        print("\n".join([repr_of_var_val(var, val) for (var, val) in feature.var_vals]))
