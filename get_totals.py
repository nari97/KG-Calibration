from TripleManager import TripleManager
from Evaluator import Evaluator
import time
import sys
import statistics

if __name__ == '__main__':
    #folder = sys.argv[1]
    #corruption_mode = "LCWA"

    folder = ""
    corruption_mode = "LCWA"

    start = time.perf_counter()

    for dataset in range(8):
        dataset_name = ""
        if dataset == 0:
            dataset_name = "FB13"
        if dataset == 1:
            dataset_name = "FB15K"
        if dataset == 2:
            dataset_name = "FB15K237"
        if dataset == 3:
            dataset_name = "NELL-995"
        if dataset == 4:
            dataset_name = "WN11"
        if dataset == 5:
            dataset_name = "WN18"
        if dataset == 6:
            dataset_name = "WN18RR"
        if dataset == 7:
            dataset_name = "YAGO3-10"

        path = folder + "Datasets/" + dataset_name + "/"

        manager = TripleManager(path, splits=["new_valid", "new_train"], corruption_mode=corruption_mode)
        rel_anomaly_max = .75
        evaluator = Evaluator(manager, rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=0)
        totals = evaluator.get_totals()

        print('Dataset: ', dataset_name, '; Validation; Avg. totals (filtered): ', statistics.mean(totals),
              '; Std. dev. totals: ', statistics.stdev(totals))

        manager = TripleManager(path, splits=["new_test", "new_valid", "new_train"], corruption_mode=corruption_mode)
        rel_anomaly_max = 1
        evaluator = Evaluator(manager, rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=0)
        totals = evaluator.get_totals()

        print('Dataset: ', dataset_name, '; Test; Avg. totals (filtered): ', statistics.mean(totals),
              '; Std. dev. totals: ', statistics.stdev(totals))

    end = time.perf_counter()
    print("Time elapsed:", str(end - start))