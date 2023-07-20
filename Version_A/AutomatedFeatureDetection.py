from Version_A import SimpleVariableExplainer, SurrogateScoreModel
import utils
import SearchSpace
import ProgressiveFeatures


def induce_features(search_space: SearchSpace.SearchSpace,
                    objective_function,
                    feature_complexity_evaluator,
                    importance_of_explainability=0.5,
                    merging_power=2):
    amount_of_random_candidates = 1000
    remove_duplicate_candidates = False

    candidate_solutions = [search_space.get_random_candidate() for _ in range(amount_of_random_candidates)]
    if remove_duplicate_candidates:
        candidate_solutions = utils.remove_duplicates(candidate_solutions)
    scores = [objective_function(candidate) for candidate in candidate_solutions]
    # experimental
    # candidate_solutions = utils.sort_using_scores(candidate_solutions, scores)  # from lowest to highest
    # scores = [position for (position, _) in enumerate(candidate_solutions)] #the scores are now just the position in the list
    # end of experimental
    print(f"the candidate solutions are \n{list(zip(candidate_solutions, scores))}")
    feature_inducer = ProgressiveFeatures.ProgressiveFeatures(candidate_solutions,
                                                              scores,
                                                              search_space,
                                                              feature_complexity_evaluator,
                                                              importance_of_explainability=importance_of_explainability,
                                                              merging_power=merging_power)

    feature_inducer.build()
    features_with_scores = feature_inducer.get_most_relevant_features()
    features_with_scores.sort(key=utils.second, reverse=True)

    return features_with_scores


def induce_features_to_generate_solutions(search_space: SearchSpace.SearchSpace,
                                          objective_function,
                                          feature_complexity_evaluator,
                                          pretty_print_feature,
                                          importance_of_explainability=0.5,
                                          merging_power=2):
    amount_of_random_candidates = 1000
    remove_duplicate_candidates = False
    candidate_solutions = [search_space.get_random_candidate() for _ in range(amount_of_random_candidates)]
    if remove_duplicate_candidates:
        candidate_solutions = utils.remove_duplicates(candidate_solutions)
    scores = [objective_function(candidate) for candidate in candidate_solutions]

    feature_detection_model = ProgressiveFeatures.ProgressiveFeatures(candidate_solutions,
                                                    scores,
                                                    search_space,
                                                    feature_complexity_evaluator,
                                                    importance_of_explainability=importance_of_explainability,
                                                    merging_power=merging_power)

    # NOTE: this needs to be done before calling .build, but this is an implementation detail that will be fixed later
    simple_variable_explainer = SimpleVariableExplainer.SimpleVariableExplainer(feature_detection_model)

    feature_detection_model.build()
    features_with_scores = feature_detection_model.get_most_relevant_features()
    features_with_scores.sort(key=utils.second, reverse=True)


    print("-"*30)
    print("This is the feature analysis section:")
    simple_variable_explainer.print_correlation_report()

    importance_candidate = simple_variable_explainer.get_importance_candidate()
    print(f"the variance of each variable is")
    pretty_print_feature(importance_candidate)


    print("-" * 30)
    print("The features obtained, with scores, are:")
    for (feature, score) in features_with_scores:
        pretty_print_feature(feature)
        print(f"-> has score {score:.2f}\n")

    selection_of_random_candidates = [search_space.get_random_candidate() for _ in range(6)]
    selection_of_lucky_candidates = [feature_detection_model.MCMC() for _ in range(6)]

    def print_list_of_candidates(cands):
        for candidate in cands:
            pretty_print_feature(candidate)
            print()

    surrogate_score_model = SurrogateScoreModel.SurrogateScoreModel(feature_detection_model)


    print("*"*30)
    print("This is the Gibbs sampling and surrogate scoring section")
    print("The scores are:")
    for candidate in selection_of_lucky_candidates:
        surrogate_score = surrogate_score_model.get_surrogate_score(candidate)
        actual_score = objective_function(candidate)
        pretty_print_feature(candidate)
        print(f"pseudo-score = {surrogate_score:.3f}, actual score = {actual_score:.3f}")
