# DeepFM
final_implicit.py file is our final code for this approach

implicit_fullrank.py file use a different type of evaluation. In the first file in the function evaluate_model() there is num_negatives = min(99, len(unseen_items)) and these means its not a Full-ranking evaluation, i.e. for each user the held-out item is ranked against ALL unseen items, but at most 99.

Note about the parameters used for the quick run of this two file: The parameters like lr,epoch,patience,... are the result of the gridsearch on implicit_fullrank.py.

example of quick run: python final_implicit.py
example of grid_search run: python final_implicit.py --grid_search 1
example of quick run with plot: python final_implicit.py --plot 1
