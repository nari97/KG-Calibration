from TripleManager import TripleManager
from ModelUtils import ModelUtils
import time
from plattscaling import ModelCalibration
from Evaluator import Evaluator
import itertools
import sys
import glob
import sqlite3
from os.path import exists

def get_scn(scn):
    i = 5 * 7 * 25 * 2  # (losses, normtype, target schemes, brier/accuracy)
    target_schemes = list(itertools.permutations([.375, .250, .125, .0], 3))
    target_schemes.append((.0,))
    for losstype in ['bce', 'se', 'huber', 'l1', 'l2']:
        for normtype in ['sigmoid', 'norm1', 'norm5', 'norm10', 'norm20', 'mean', 'zscr']:
            for tocompute in ['brier', 'accuracy']:
                for target_scheme in target_schemes:
                    i = i - 1
                    if scn == i:
                        return losstype, normtype, list(target_scheme), tocompute

if __name__ == '__main__':
    model_name = sys.argv[1]
    dataset = int(sys.argv[2])
    dbname = sys.argv[3]  # database name
    scn = int(sys.argv[4])  # 0--1749

    losstype, normtype, target_scheme, tocompute = get_scn(scn)

    type = 'test'  # Always test for evaluation
    corruption_mode = "LCWA"

    compute_brier, compute_divergence, compute_accuracy = \
        tocompute == 'brier', tocompute == 'divergence', tocompute == 'accuracy'

    print('Dataset: ', dataset, '; Loss: ', losstype, '; Targets scheme: ', str(target_scheme), '; Rescaling: ',
          normtype, 'Computing: ', tocompute)

    # db connection
    print('Database file exists: ', exists('Model/' + dbname))
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
        tempname = tempname.replace('\\', '/')

        if compute_brier or compute_accuracy:
            cur = con.cursor()
            # We are just checking after, assuming that before was already computed if after was there.
            cur.execute("SELECT * FROM "+("brier_after" if compute_brier else "accuracy")+" WHERE " \
                            "model='" + tempname + "' AND " \
                            "targets='" + str(target_scheme) + "' AND " \
                            "normtype='" + str(normtype) + "' AND " \
                            "losstype='" + str(losstype) + "';")
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

        cur = con.cursor()
        cur.execute("SELECT weight, bias FROM calibration WHERE model='" + tempname + "' AND " \
                                                    "targets='"+str(target_scheme)+"' AND " \
                                                    "normtype='"+str(normtype)+"' AND " \
                                                    "losstype='"+str(losstype)+"';")
        temp = cur.fetchone()
        cur.close()
        weight, bias = temp[0], temp[1]

        if compute_brier:
            brier_results = {}
            # We will compute Brier before only once; just selected a specific configuration to do it.
            if normtype=='sigmoid' and losstype=='bce' and len(target_scheme)==1:
                for mode in ['before_sigmoid', 'before_norm1', 'before_norm5', 'before_norm10', 'before_norm20']:
                    uncalibdirect = ModelCalibration(model.model, minmax[0], minmax[1], minmax[2], minmax[3],
                                            A=1, B=0, scaletype=mode.replace('before_', ''), losstype=losstype,
                                                     totals=calcweights, targets=target_scheme)
                    print('Brier ', mode)
                    brier_results[mode] = uncalibdirect.evaluate_brier(manager)

        calibmodel = ModelCalibration(model.model, minmax[0], minmax[1], minmax[2], minmax[3], A=weight, B=bias,
                                 losstype=losstype, scaletype=normtype, totals=calcweights, targets=target_scheme)
        if compute_brier:
            print('Brier after')
            brier_results['after'] = calibmodel.evaluate_brier(manager)

            # Get and store all results.
            def get_brier(brier_results, mode):
                ret = ""
                for pos_neg in ['pos', 'global', 'pos_neg']:
                    to_add = "NULL"
                    if pos_neg in brier_results[mode]['cwa']['binary']:
                        to_add = "'" + str(brier_results[mode]['cwa']['binary'][pos_neg]) + "'"
                    ret += to_add + ","
                # Remove last char.
                return ret[:-1]

            for mode in brier_results.keys():
                sql = "REPLACE INTO brier_" + mode + "(model, " +\
                    ("" if mode.startswith('before') else "targets, normtype, losstype, ")+\
                    "pos, global, pos_neg) VALUES('" + tempname + "', " +\
                    ("" if mode.startswith('before') else
                        "'"+str(target_scheme)+"','"+str(normtype)+"','"+str(losstype)+"',")+\
                    get_brier(brier_results, mode) + ");"
                print(sql)

        if compute_divergence:
            # TODO This is not ready!
            print('Divergence')
            div_results = calibmodel.evaluate_divergence(manager)

            def get_div(div_results, mode):
                ret = ""
                for pos_neg in ['pos', 'global', 'local', 'tclcwa', 'lcwa', 'all_neg', 'pos_neg']:
                    to_add = "NULL"
                    if pos_neg in div_results[mode].keys():
                        to_add = str(div_results[mode][pos_neg])
                    ret += "'"+to_add+"',"
                # Remove last char.
                return ret[:-1]

            for mode in ['direct', 'minmax']:
                sql = "REPLACE INTO divergence_" + mode + "(model, calibtype, normtype, losstype, "+\
                      " pos, global, local, tclcwa, lcwa, all_neg, pos_neg) VALUES('" + tempname +\
                      "','"+str(target_scheme)+"','"+str(normtype)+"','"+str(losstype)+"',"+\
                       get_div(div_results, mode)+");"
                print(sql)

        if compute_accuracy:
            evaluator = Evaluator(manager, rel_anomaly_max=1, rel_anomaly_min=0)
            rc, tp, tn, fp, fn = evaluator.evaluate(calibmodel, False, accuracy=True)

            sql = "REPLACE INTO accuracy(model, targets, normtype, losstype, "+\
                  " tp, tn, fp, fn) VALUES('" + tempname +\
                  "','" + str(target_scheme) + "','" + str(normtype) + "','" + str(losstype) +\
                  "','" + str(tp) + "','" + str(tn) + "','" + str(fp) + "','" + str(fn) + "');"
            print(sql)

    con.close()
