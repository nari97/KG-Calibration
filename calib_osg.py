from Evaluator import Evaluator
from plattscaling import ModelCalibration
from ModelUtils import ModelUtils
from TripleManager import TripleManager
import sys
import glob
import itertools
import sqlite3
import time
import math

def get_scn(scn):
    i = 5 * 7 * 25 # (losses, normtype, target schemes)
    target_schemes = list(itertools.permutations([.375, .250, .125, .0], 3))
    target_schemes.append((.0,))
    for losstype in ['bce', 'se', 'huber', 'l1', 'l2']:
        for normtype in ['sigmoid', 'norm1', 'norm5', 'norm10', 'norm20', 'mean', 'zscr']:
            for target_scheme in target_schemes:
                i=i-1
                if scn==i:
                    return losstype, normtype, list(target_scheme)

if __name__ == '__main__':
    model_name = sys.argv[1]
    dataset = int(sys.argv[2])
    dbname = sys.argv[3]  # database name
    scn = int(sys.argv[4])  # 0--874

    losstype, normtype, target_scheme = get_scn(scn)
    
    type = 'valid'  # Always validation for training calibration.
    corruption_mode = 'LCWA'

    print('Dataset: ', dataset, '; Loss: ', losstype, '; Targets scheme: ', str(target_scheme), '; Rescaling: ', normtype)

    # db connection
    con = sqlite3.connect('Model/' + dbname)

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

    path = "Datasets/" + dataset_name + "/"

    start = time.perf_counter()

    if type == "valid":
        manager = TripleManager(path, splits=["new_valid", "new_train"], corruption_mode=corruption_mode)
    elif type == "test":
        manager = TripleManager(path, splits=["new_test", "new_valid", "new_train"], corruption_mode=corruption_mode)

    pending = 0
    for model_file in glob.glob("Model/" + str(dataset) + "/" + model_name + "*.model"):
        pending = pending + 1

    # We are assuming that minimum and maximum scores and the other stuff is already in the database.
    for model_file in glob.glob("Model/" + str(dataset) + "/" + model_name + "*.model"):
        print('Pending:', pending, '; Model:', model_file)
        pending = pending - 1

        # Load model.
        util = ModelUtils(model_name, ModelUtils.get_params(model_file))
        model = util.get_model(manager.entityTotal, manager.relationTotal, 0)
        model.model.load_checkpoint(model_file)

        tempname = model_file.split("Model")[1]
        tempname = tempname.replace('\\','/')

        # Check whether the model is already present.
        checkquery = "SELECT * FROM calibration WHERE model='" + tempname + "' AND " \
                                                    "targets='"+str(target_scheme)+"' AND " \
                                                    "normtype='"+str(normtype)+"' AND " \
                                                    "losstype='"+str(losstype)+"';"
        cur = con.cursor()
        cur.execute(checkquery)
        check = cur.fetchone()
        cur.close()

        if check != None:
            # results already saved
            print("Results already in db")
            continue

        cur = con.cursor()
        cur.execute('SELECT minvalue, maxvalue, mean, std, pos, tclcwa, lcwa, global '
                    'FROM minmax WHERE model="' + tempname + '";')
        minmax = cur.fetchone()
        cur.close()
        calcweights = minmax[4:]
        tempmodel = ModelCalibration(model.model, minmax[0], minmax[1], minmax[2], minmax[3], scaletype=normtype,
                                     losstype=losstype, totals=calcweights, targets=target_scheme)
        weight, bias = tempmodel.train_calib(manager)

        # This is only for informative purposes and can be eliminated.
        evaluator = Evaluator(manager, rel_anomaly_max=1, rel_anomaly_min=0)
        rc, tp, tn, fp, fn = evaluator.evaluate(tempmodel, False, accuracy=True)

        print('MR: ', rc.get_metric(metric_str='mr').get(), '; Adjusted MR: ', rc.get_metric(metric_str='mrh').get(),
              '; TP: ', tp, '; TN: ', tn, '; FP: ', fp, '; FN: ', fn)

        print("REPLACE INTO calibration(model, targets, normtype, losstype, weight, bias) "
                    "VALUES ('" + tempname + "','"+str(target_scheme)+"','"+str(normtype)+"', '"+str(losstype)+"',"
                        + (str(weight) if not math.isnan(weight) else 'NULL') + ","
                        + (str(bias) if not math.isnan(bias) else 'NULL') + ");")

    con.close()
