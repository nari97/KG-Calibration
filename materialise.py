from TripleManager import TripleManager
from MaterialEvaluator import MaterialEvaluator
from ModelUtils import ModelUtils
import time
import sys
import glob

if __name__ == '__main__':
    folder = sys.argv[1]
    model_name = sys.argv[2]
    dataset = int(sys.argv[3])
    corruption_mode = "LCWA"

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

    start = time.perf_counter()

    manager = TripleManager(path, splits=["new_test", "new_valid", "new_train"], corruption_mode=corruption_mode)

    pending = 0
    for model_file in glob.glob(folder + "Model/" + str(dataset) + "/" + model_name + "*.model"):
        pending = pending + 1

    # We are assuming that minimum and maximum scores and the other stuff is already in the database.
    for model_file in glob.glob(folder + "Model/" + str(dataset) + "/" + model_name + "*.model"):
        util = ModelUtils(model_name, ModelUtils.get_params(model_file))
        model = util.get_model(manager.entityTotal, manager.relationTotal, 0)
        model.model.load_checkpoint(model_file)

        evaluator = MaterialEvaluator(manager, rel_anomaly_max=1, rel_anomaly_min=0)
        rc = evaluator.evaluate(model.model, name=model_file)

    end = time.perf_counter()
    print("Time elapsed to materialize:", str(end - start))
