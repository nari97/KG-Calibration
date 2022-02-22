import sqlite3
import itertools
import glob
import sys

if __name__ == '__main__':
    folder = ''
    dbname = 'finalw_calib.db'  # database name

    # db connection
    con = sqlite3.connect(folder + 'Model/' + dbname)

    # cur = con.cursor()
    # cur.execute("SELECT id FROM Model WHERE algorithm='distmult';")
    # for row in cur:
    #     print('Distmult model:', str(row[0]))
    # cur.close()
    #
    cur = con.cursor()
    cur.execute("SELECT mid, tid, eid FROM Experiment_Trial WHERE status <> 'Completed';")
    for row in cur:
        # The last should be the dataset id but is only
        print(str(row[0])+','+str(row[1])+","+str(row[2])+",0")
    cur.close()

    # TODO Get best models by accvalid (and other stats, e.g., targets).
    # SELECT status, loss, norm, global, tclcwa, lcwa, pos FROM Experiment_Trial

    # eid=1: binary targets; eid=2: non-binary targets; eid=3: non-binary targets with pos=1.0.
    eid = 3
    cur = con.cursor()
    cur.execute("SELECT dataset, algorithm, MAX(accvalid) FROM Model JOIN Experiment_Trial ON id=mid "
                "WHERE eid=? GROUP BY dataset, algorithm;", (eid,))
    for (dataset, algorithm, accvalid) in cur:
        # TODO Remove!!!
        if algorithm != 'distmult':
            continue
        print('Dataset: ', dataset, '; Algorithm: ', algorithm, '; Accuracy (valid): ', accvalid)

        innercur = con.cursor()
        innercur.execute("SELECT loss, norm, mid, Experiment_Trial.global, Experiment_Trial.tclcwa, Experiment_Trial.lcwa, "
                         "Experiment_Trial.pos FROM Model JOIN Experiment_Trial ON id=mid "
                         "WHERE eid=? AND dataset=? AND algorithm=? AND accvalid=?;", (eid, dataset, algorithm, accvalid))
        (loss, norm, mid, glbl, tclcwa, lcwa, pos) = innercur.fetchone()
        print('\t', 'Loss:', loss, '; Norm:' , norm , '; Mid:', mid, '; Global:', glbl, '; TCLCWA:', tclcwa,
              '; LCWA:', lcwa, '; Pos:', pos)
        innercur.close()
    cur.close()


    con.close()
