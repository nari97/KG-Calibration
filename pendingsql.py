import sqlite3
import itertools

if __name__ == '__main__':
    folder = './'
    dbname = 'calib_4.db'  # database name

    index_to_setting = {}
    i = 5 * 7 * 2 * 25  # (losses, normtype, brier/accuracy)
    target_schemes = list(itertools.permutations([.375, .250, .125, .0], 3))
    target_schemes.append((.0,))
    for losstype in ['bce', 'se', 'huber', 'l1', 'l2']:
        for normtype in ['sigmoid', 'norm1', 'norm5', 'norm10', 'norm20', 'mean', 'zscr']:
            for tocompute in ['brier', 'accuracy']:
                for target_scheme in target_schemes:
                    i = i - 1
                    index_to_setting[i] = (losstype, normtype, list(target_scheme), tocompute)

    def find_setting(losstype, normtype, target_scheme, tocompute):
        for key in index_to_setting.keys():
            (losstype_i, normtype_i, target_scheme_i, tocompute_i) = index_to_setting[key]
            if losstype_i == losstype and normtype_i == normtype and \
                target_scheme == target_scheme_i and tocompute == tocompute_i:
                return key

    # db connection
    con = sqlite3.connect(folder + 'Model/'+dbname)

    target_schemes = list(itertools.permutations([.375, .250, .125, .0], 3))
    target_schemes.append((.0,))
    # , 'rotate'
    for model in ['analogy', 'complex', 'distmult', 'hole', 'simple', 'transd', 'transe', 'transh']:
        total_missing = 0
        model_missing = {}
        for dataset in [0, 1, 2, 3, 4, 5, 6]:
            count_acc, count_brier_before, count_brier_after = 0, 0, 0
            checkquery = "SELECT DISTINCT model FROM calibration " \
                         "WHERE model LIKE '/" + str(dataset) + "/" + model + "%.model';"

            pending_idxs = set()

            model_files = []
            cur = con.cursor()
            cur.execute(checkquery)
            for row in cur:
                model_files.append(row[0])
            cur.close()

            for model_file in model_files:
                checkquery = "SELECT COUNT(*) FROM accuracy WHERE model = '" + str(model_file) + "';"
                cur = con.cursor()
                cur.execute(checkquery)
                cnt = cur.fetchone()[0]
                cur.close()
                #print(model_file, cnt)

                stop = False
                for mode in ['before_sigmoid', 'before_norm1', 'before_norm5', 'before_norm10', 'before_norm20']:
                    checkquery = "SELECT model FROM brier_" + mode + " WHERE model = '" + str(model_file) + "';"
                    cur = con.cursor()
                    cur.execute(checkquery)
                    check = cur.fetchone()
                    if not check:
                        count_brier_before += 1
                        pending_idxs.add(find_setting('bce', 'sigmoid', [0.0], 'brier'))
                        stop = True
                    cur.close()

                    if stop:
                        break

            print(model, dataset, 'Missing brier before: ', count_brier_before)

            for losstype in ['bce', 'se', 'huber', 'l1', 'l2']:
                for normtype in ['sigmoid', 'norm1', 'norm5', 'norm10', 'norm20', 'mean', 'zscr']:
                    for target_scheme in target_schemes:
                        target_scheme = list(target_scheme)
                        stop = False
                        for model_file in model_files:
                            checkquery = "SELECT model FROM accuracy WHERE model = '" + str(model_file) + "' AND " \
                                            "targets='" + str(target_scheme) + "' AND " \
                                            "normtype='" + str(normtype) + "' AND " \
                                            "losstype='" + str(losstype) + "';"
                            cur = con.cursor()
                            cur.execute(checkquery)
                            check = cur.fetchone()
                            if not check:
                                count_acc += 1
                                pending_idxs.add(find_setting(losstype, normtype, target_scheme, 'accuracy'))
                                stop = True
                            cur.close()

                            if stop:
                                break

                        stop = False
                        for model_file in model_files:
                            checkquery = "SELECT model FROM brier_after WHERE model = '" + str(model_file) + "' AND " \
                                                    "targets='" + str(target_scheme) + "' AND " \
                                                    "normtype='" + str(normtype) + "' AND " \
                                                    "losstype='" + str(losstype) + "';"
                            cur = con.cursor()
                            cur.execute(checkquery)
                            check = cur.fetchone()
                            if not check:
                                count_brier_after += 1
                                pending_idxs.add(find_setting(losstype, normtype, target_scheme, 'brier'))
                                stop = True
                            cur.close()

                            if stop:
                                break

            print(model, dataset, 'Missing accuracy: ', count_acc)
            print(model, dataset, 'Missing brier after: ', count_brier_after)
            if len(pending_idxs) > 0:
                model_missing[dataset] = pending_idxs
                total_missing += len(pending_idxs)

        if total_missing > 0:
            with open(model+'_pending.txt', 'w') as the_file:
                for dataset in model_missing.keys():
                    for idx in model_missing[dataset]:
                        the_file.write(model+','+str(dataset)+','+str(idx)+'\n')

    con.close()
