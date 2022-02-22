from TripleManager import TripleManager
from Evaluator import Evaluator
from ModelUtils import ModelUtils
from plattscaling import ModelCalibration
import sys
import glob
import sqlite3

def get_scn(scn):
    i = 7 * 2 * 2  # datasets, losses, calibtype
    for dataset in [0, 1, 2, 3, 4, 5, 6]:
        for losstype in ['norank', 'rank']:
            for calibtype in [False, True]:
                i = i - 1
                if scn == i:
                    return dataset, losstype, calibtype


if __name__ == '__main__':
    folder = sys.argv[1]
    model_name = sys.argv[2]
    dbname = sys.argv[3]  # database name
    scn = int(sys.argv[4])  # 0--27

    dataset, losstype, calibtype = get_scn(scn)

    type = 'test'  # Always test for evaluation
    corruption_mode = "LCWA"
    # Let's fix the scaling (min-max) and always using weights.
    normtype, weighttype = 'norm', True

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

    # db connection
    con = sqlite3.connect(folder + 'Model/'+dbname)
    cur = con.cursor()
    cur.executescript('CREATE TABLE IF NOT EXISTS ranks(model, calibtype, normtype, losstype, weighttype, '
                      'amr, ties, below, qone, qtwo, qthree, qfour,'
                      'PRIMARY KEY(model, calibtype, normtype, losstype, weighttype));')
    cur.close()
    con.commit()

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

        tempname = model_file.split("Model")[1]

        # Check whether the model is already present.
        checkquery = "SELECT * FROM calibration WHERE model='" + tempname + "' AND " \
                                        "calibtype='" + str(calibtype) + "' AND " \
                                        "normtype='" + str(normtype) + "' AND " \
                                        "losstype='" + str(losstype) + "' AND " \
                                        "weighttype='" + str(weighttype) + "';"
        cur = con.cursor()
        cur.execute(checkquery)
        check = cur.fetchone()
        cur.close()

        # if check != None:
        #    # results already saved
        #    print("Results already in db")
        #    continue

        cur = con.cursor()
        cur.execute('SELECT * FROM minmax WHERE model="' + tempname + '";')
        minmax = cur.fetchone()
        cur.close()
        calcweights = minmax[3:]

        cur = con.cursor()
        cur.execute("SELECT weight, bias FROM calibration WHERE model='" + tempname + "' AND " \
                                        "calibtype='" + str(calibtype) + "' AND " \
                                        "normtype='" + str(normtype) + "' AND " \
                                        "losstype='" + str(losstype) + "' AND " \
                                        "weighttype='" + str(weighttype) + "';")
        temp = cur.fetchone()
        cur.close()
        weight, bias = temp[0], temp[1]

        calibratedmodel = ModelCalibration(model.model, minmax[1], minmax[2], weighttype, A=weight, B=bias,
                                 losstype=losstype, scaletype=normtype, opencalib=calibtype, customweights=calcweights)
        rc = evaluator.evaluate(calibratedmodel, False)

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
        tempname = models[i].split("Model")[1]
        values = rank_values[models[i]]
        update = "REPLACE INTO ranks(model, calibtype, normtype, losstype, weighttype, "+\
                      "amr, ties, below, qone, qtwo, qthree, qfour) VALUES('" + tempname + "', "+\
                      "'"+str(calibtype)+"','"+str(normtype)+"','"+str(losstype)+"','"+str(weighttype)+"', "+\
                      "'"+str(values['amr'])+"', '"+str(values['ties'])+"', '"+str(values['below'])+"', "+\
                      "'"+str(values['amr_q1'])+"', '"+str(values['amr_q2'])+"', "+\
                      "'"+str(values['amr_q3'])+"', '"+str(values['amr_q4'])+"');"
        #print('Update query:',update)
        cur = con.cursor()
        cur.execute(update)
        cur.close()
        con.commit()
    con.close()
