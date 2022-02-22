from TripleManager import TripleManager
from ModelUtils import ModelUtils
import time
from plattscaling import ModelCalibration
from Evaluator import Evaluator
import itertools
import sys
import glob
import sqlite3

def get_scn(scn):
    i = 5 * 7 * 2 * 25  # (losses, normtype, brier/accuracy)
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
    folder = sys.argv[1]
    model_name = sys.argv[2]
    dataset = int(sys.argv[3])
    dbname = sys.argv[4]  # database name
    scn = int(sys.argv[5])  # 0--1749

    losstype, normtype, target_scheme, tocompute = get_scn(scn)

    type = 'test'  # Always test for evaluation
    corruption_mode = "LCWA"

    compute_brier, compute_divergence, compute_accuracy = \
        tocompute == 'brier', tocompute == 'divergence', tocompute == 'accuracy'

    print('Dataset: ', dataset, '; Loss: ', losstype, '; Targets scheme: ', str(target_scheme), '; Rescaling: ',
          normtype, 'Computing: ', tocompute)

    # db connection
    con = sqlite3.connect(folder + 'Model/' + dbname)

    # This is to avoid database locked due to many jobs accessing the file database at the same time.
    def create_table(con, table_name, sql):
        # Check if table exists first.
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='" + table_name + "';")
        check = cur.fetchone()
        cur.close()

        if not check:
            cur = con.cursor()
            cur.executescript(sql)
            cur.close()
            con.commit()

    for mode in ['before_sigmoid', 'before_norm1', 'before_norm5', 'before_norm10', 'before_norm20', 'after']:
        otherattribs = '' if mode.startswith('before') else ', targets, normtype, losstype'
        create_table(con, 'brier_'+mode, 'CREATE TABLE IF NOT EXISTS brier_'+mode+'(model ' +
            otherattribs + ', pos, global, pos_neg, PRIMARY KEY(model'+otherattribs+'));')
    for mode in ['direct', 'minmax']:
        create_table(con, 'divergence_'+mode, 'CREATE TABLE IF NOT EXISTS divergence_'+mode+'(model, targets, normtype, losstype, '
            ' pos, global, pos_neg, PRIMARY KEY (model, targets, normtype, losstype));')
    create_table(con, 'accuracy', 'CREATE TABLE IF NOT EXISTS accuracy(model, targets, normtype, losstype, '
            ' tp, tn, fp, fn, PRIMARY KEY (model, targets, normtype, losstype));')

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

    if type == "valid":
        manager = TripleManager(path, splits=["new_valid", "new_train"], corruption_mode=corruption_mode)
    elif type == "test":
        manager = TripleManager(path, splits=["new_test", "new_valid", "new_train"], corruption_mode=corruption_mode)

    pending = 0
    for model_file in glob.glob(folder + "Model/" + str(dataset) + "/" + model_name + "*.model"):
        pending = pending + 1

    # We are assuming that minimum and maximum scores and the other stuff is already in the database.
    for model_file in glob.glob(folder + "Model/" + str(dataset) + "/" + model_name + "*.model"):
        print('Pending:', pending, '; Model:', model_file)
        pending = pending - 1

        # Load model.
        util = ModelUtils(model_name, ModelUtils.get_params(model_file))
        model = util.get_model(manager.entityTotal, manager.relationTotal, 0)
        model.model.load_checkpoint(model_file)

        tempname = model_file.split("Model")[1]
        tempname = tempname.replace('\\', '/')

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
                cur = con.cursor()
                cur.execute(sql)
                cur.close()
                con.commit()

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
                cur = con.cursor()
                cur.execute("REPLACE INTO divergence_" + mode + "(model, calibtype, normtype, losstype, "
                            " pos, global, local, tclcwa, lcwa, all_neg, pos_neg) VALUES('" + tempname +
                                    "','"+str(target_scheme)+"','"+str(normtype)+"','"+str(losstype)+"',"
                            +get_div(div_results, mode)+");")
                cur.close()
                con.commit()

        if compute_accuracy:
            evaluator = Evaluator(manager, rel_anomaly_max=1, rel_anomaly_min=0)
            rc, tp, tn, fp, fn = evaluator.evaluate(calibmodel, False, accuracy=True)

            cur = con.cursor()
            cur.execute("REPLACE INTO accuracy(model, targets, normtype, losstype, "
                        " tp, tn, fp, fn) VALUES('" + tempname +
                        "','" + str(target_scheme) + "','" + str(normtype) + "','" + str(losstype) +
                        "','" + str(tp) + "','" + str(tn) + "','" + str(fp) + "','" + str(fn) + "');")
            cur.close()
            con.commit()

    con.close()
