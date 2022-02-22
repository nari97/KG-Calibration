from Evaluator import Evaluator
from plattscaling import ModelCalibration
from ModelUtils import ModelUtils
from TripleManager import TripleManager
import sys
import sqlite3
import time

def get_scn(scn, folder, pendingfile):
    mid, tid, eid = None, None, None
    with open(folder + pendingfile, 'r') as f:
        linecount = 0
        for line in f:
            linecount += 1
            # First line contains header, we want to start in the second line
            if scn + 2 == linecount:
                (mid, tid, eid) = line.split(",")[0:3]
                break
    return mid, tid, eid

def to_db_or_print(con, query, params):
    # This is for RC.
    if len(sys.argv) == 5:
        cur = con.cursor()
        cur.execute(query, params)
        cur.close()
        con.commit()
    # This is for OSG.
    else:
        found = 0
        while True:
            pos = query.find('?')
            if pos < 0:
                break
            else:
                quote = ""
                if isinstance(params[found], str):
                    quote = "'"
                query = query[:pos] + quote + str(params[found]) + quote + query[pos + 1:]
                found += 1
        print(query)

if __name__ == '__main__':
    folder = sys.argv[1]
    dbname = sys.argv[2]  # database name
    # This is for RC.
    if len(sys.argv) == 5:
        pendingfile = sys.argv[3]  # file with pending (ModelId,TrialId,ExperimentId)
        scn = int(sys.argv[4])  # 0--6124
        mid, tid, eid = get_scn(scn, folder, pendingfile)
    # This is for OSG.
    else:
        mid, tid, eid = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    type = 'valid'  # Always validation for training calibration.
    corruption_mode = 'LCWA'

    print('Model:', mid, 'Trial:', tid, 'Experiment:', eid)

    # db connection
    con = sqlite3.connect(folder + 'Model/' + dbname)

    cur = con.cursor()
    cur.execute("SELECT status, loss, norm, global, tclcwa, lcwa, pos "
                "FROM Experiment_Trial WHERE mid=? AND tid=? AND eid=?;", (mid, tid, eid))
    (status, losstype, normtype, glbl, tclcwa, lcwa, pos) = cur.fetchone()
    glbl, tclcwa, lcwa, pos = float(glbl), float(tclcwa), float(lcwa), float(pos)
    cur.close()

    if status == 'Completed':
        print('Marked as completed')
    else:
        target_scheme = {'global': glbl, 'tclcwa': tclcwa, 'lcwa': lcwa, 'pos': pos}

        cur = con.cursor()
        cur.execute("SELECT dataset, modelfile, algorithm, minvalue, maxvalue, mean, std, pos, tclcwa, lcwa, global "
                    "FROM Model WHERE id=?;", (mid,))
        (dataset, model_file, model_name, min, max, mean, std, totpos, tottclcwa, totlcwa, totglobal) = cur.fetchone()
        min, max, mean, std, totpos, tottclcwa, totlcwa, totglobal = \
            float(min), float(max), float(mean), float(std), \
            float(totpos), float(tottclcwa), float(totlcwa), float(totglobal)
        cur.close()
        # global, lcwa and tclcwa are strings!
        totals_scheme = {'pos': totpos, 'global': totglobal, 'lcwa': totlcwa, 'tclcwa': tottclcwa}

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
            manager = TripleManager(path, splits=["new_test", "new_valid", "new_train"],
                                    corruption_mode=corruption_mode)

        model_file = folder + "Model/" + model_file

        print(losstype, normtype, target_scheme, model_file, model_name)

        # Load model.
        util = ModelUtils(model_name, ModelUtils.get_params(model_file))
        model = util.get_model(manager.entityTotal, manager.relationTotal, 0)
        model.model.load_checkpoint(model_file)

        cur = con.cursor()
        cur.execute("SELECT a, b FROM Experiment_Trial WHERE mid=? AND tid=? AND eid=?;", (mid, tid, eid))
        (a, b) = cur.fetchone()
        cur.close()
        con.commit()

        if a is None or b is None:
            tempmodel = ModelCalibration(model.model, min, max, mean, std, scaletype=normtype, losstype=losstype,
                                         totals=totals_scheme, targets=target_scheme)
            a, b = tempmodel.train_calib(manager)

            query = "UPDATE Experiment_Trial SET a=?, b=? " \
                     "WHERE mid=? AND tid=? AND eid=?;"
            params = (str(a), str(b), mid, tid, eid)
            to_db_or_print(con, query, params)
            # Checkpoint
            #sys.exit(0)
        else:
            tempmodel = ModelCalibration(model.model, min, max, mean, std, A=float(a), B=float(b), scaletype=normtype,
                                         losstype=losstype, totals=totals_scheme, targets=target_scheme)

        evaluator = Evaluator(manager, rel_anomaly_max=1, rel_anomaly_min=0)
        rc, tp, tn, fp, fn = evaluator.evaluate(tempmodel, False, accuracy=True)
        acc = ModelCalibration.get_accuracy(tp, tn, fp, fn)

        print('MR: ', rc.get_metric(metric_str='mr').get(), '; Adjusted MR: ', rc.get_metric(metric_str='mrh').get(),
              '; TP: ', tp, '; TN: ', tn, '; FP: ', fp, '; FN: ', fn, '; Acc: ', acc)

        query = "UPDATE Experiment_Trial SET status='Completed', accvalid=? "\
                        "WHERE mid=? AND tid=? AND eid=?;"
        params = (str(acc), mid, tid, eid)
        to_db_or_print(con, query, params)

    con.close()