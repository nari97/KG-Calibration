from TripleManager import TripleManager
from Evaluator import Evaluator
from ModelUtils import ModelUtils
import sys
import glob

if __name__ == '__main__':
    #folder = sys.argv[1]
    #model_name = sys.argv[2]
    #dataset = int(sys.argv[3])  # 0--6

    # TODO: Remove!
    folder = ''
    model_name = 'simple'
    dataset = 5

    type = 'test'  # Always test for evaluation
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

    evaluators = {}
    manager = TripleManager(path, splits=["new_test", "new_valid", "new_train"], corruption_mode=corruption_mode)
    evaluators["MR_Global"] = {'metric_str': 'mr', 'rel_anomaly_max': 1, 'rel_anomaly_min': 0}
    evaluators["MR_Q1"] = {'metric_str': 'mr', 'rel_anomaly_max': 1, 'rel_anomaly_min': .75}
    evaluators["MR_Q2"] = {'metric_str': 'mr', 'rel_anomaly_max': .7499, 'rel_anomaly_min': .5}
    evaluators["MR_Q3"] = {'metric_str': 'mr', 'rel_anomaly_max': .4999, 'rel_anomaly_min': .25}
    evaluators["MR_Q4"] = {'metric_str': 'mr', 'rel_anomaly_max': .2499, 'rel_anomaly_min': 0}
    evaluator = Evaluator(manager, rel_anomaly_max=1, rel_anomaly_min=0)

    models = []
    collectors = []
    pending = 0
    for model_file in glob.glob(folder + "Model/" + str(dataset) + "/" + model_name + "*.model"):
        pending = pending + 1

    for model_file in glob.glob(folder + "Model/" + str(dataset) + "/" + model_name + "*.model"):
        print('Pending:', pending)
        pending = pending - 1

        # Load model.
        util = ModelUtils(model_name, ModelUtils.get_params(model_file))
        model = util.get_model(manager.entityTotal, manager.relationTotal, 0)
        model.model.load_checkpoint(model_file)

        rc = evaluator.evaluate(model.model, False)

        models.append(model_file)
        collectors.append(rc)

    rank_values = {}
    for k in evaluators.keys():
        ev = evaluators[k]

        for i in range(len(models)):
            rc = collectors[i].prune(ev['rel_anomaly_max'], ev['rel_anomaly_min'])
            if len(rc.all_ranks) == 0:
                continue

            if models[i] not in rank_values.keys():
                rank_values[models[i]] = {'amr':'NULL', 'amr_q1':'NULL', 'amr_q2':'NULL', 'amr_q3':'NULL',
                                          'amr_q4':'NULL', 'ties':'NULL', 'below':'NULL'}

            amr = 1.0 - (rc.get_metric(metric_str=ev['metric_str']).get() /
                         rc.get_expected(metric_str=ev['metric_str']).get())
            if k == "MR_Global":
                rank_values[models[i]]['amr'] = amr
                rank_values[models[i]]['ties'] = rc.all_ties.count(True) / len(rc.all_ties)
                rank_values[models[i]]['below'] = rc.get_ranks_below_expected().count(True) / len(rc.all_ranks)
            elif k == "MR_Q1":
                rank_values[models[i]]['amr_q1'] = amr
            elif k == "MR_Q2":
                rank_values[models[i]]['amr_q2'] = amr
            elif k == "MR_Q3":
                rank_values[models[i]]['amr_q3'] = amr
            elif k == "MR_Q4":
                rank_values[models[i]]['amr_q4'] = amr

    for i in range(len(models)):
        print('Model:', models[i])
        print('Metrics:', rank_values[models[i]])
