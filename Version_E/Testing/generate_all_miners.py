import json

stochastic_options = [True, False]
uses_archive_options = [True, False]
population_size_options = [50, 100, 150]
gc_miner_options = ["constructive", "destructive", "bidirectional"]

gc_miners = [{"which": which,
              "population_size": pop_size,
              "uses_archive": uses_archive,
              "stochastic": stochastic}
             for which in gc_miner_options
             for pop_size in population_size_options
             for uses_archive in uses_archive_options
             for stochastic in stochastic_options]


ga_miners = [{"which": "ga",
              "population_size": pop_size}
             for pop_size in population_size_options]


other_miners = [{"which": "hill_climber"}, {"which": "random"}]


all_miners = gc_miners + ga_miners + other_miners


def print_all_miners():
    print(json.dumps(all_miners))


print_all_miners()