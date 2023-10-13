import utils

constructive_miners = [{"which": "constructive",
                        "stochastic": stochastic_item,
                        "at_most": 5,
                        "population_size": population_item}
                       for stochastic_item in [True, False]
                       for population_item in [36, 72, 144]]

destructive_miners = [{"which": "destructive",
                        "stochastic": stochastic_item,
                        "at_least": 1,
                        "population_size": population_item}
                       for stochastic_item in [True, False]
                       for population_item in [36, 72, 144]]

ga_miners = [{"which": "ga",
             "iterations": iteration_item,
             "population_size": population_item}
            for iteration_item in [5, 10, 20]
            for population_item in [36, 72, 144]]


random_miners = [{"which": "random",
                  "population_size": population_item}
                 for population_item in [36, 72, 144]]

hill_climber = [{"which": "hill_climber",
                 "population_size": population_item}
                for population_item in [36, 72, 144]]

many_miners = utils.concat_lists([constructive_miners, destructive_miners, ga_miners, random_miners, hill_climber])