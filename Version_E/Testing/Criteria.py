from Version_E.MeasurableCriterion.CriterionUtilities import All, Any, Not, Balance
from Version_E.MeasurableCriterion.Explainability import Explainability
from Version_E.MeasurableCriterion.GoodFitness import HighFitness, ConsistentFitness
from Version_E.MeasurableCriterion.Popularity import Overrepresentation, Commonality
from Version_E.MeasurableCriterion.Robustness import Robustness

consistently_high_fitness = Balance([HighFitness(), ConsistentFitness()], weights=[2, 1])
consistently_low_fitness = Balance([Not(HighFitness()), ConsistentFitness()], weights=[2, 1])

relevant_to_fitness = ConsistentFitness()
robust = Robustness(0, 2)