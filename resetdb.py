import sys
import sqlite3

# /home/crrvcs/ActivePython-3.7/bin/python3 -u ./Code/resetdb.py /home/crrvcs/OpenKE/ finalw_calib.db
if __name__ == '__main__':
    #folder = sys.argv[1]
    #dbname = sys.argv[2]  # database name

    folder = './'
    dbname =  'finalw_calib.db'

    # db connection
    con = sqlite3.connect(folder + 'Model/'+dbname)
    cur = con.cursor()
    #cur.executescript('DROP TABLE IF EXISTS Experiment;')
    #cur.executescript('DROP TABLE IF EXISTS Experiment_Trial;')

    for row in cur.execute('SELECT * FROM Model WHERE dataset=6'):
        print('Row: ', row)
    print(cur.description)

    #for mode in ['before_direct', 'before_minmax', 'after']:
    #    for type_brier in ['owa', 'cwa']:
    #        for score in ['binary', 'original']:
    #            cur.executescript('DROP TABLE IF EXISTS brier_'+mode+'_'+type_brier+'_'+score+';')
    #for mode in ['direct', 'minmax']:
    #    cur.executescript('DROP TABLE IF EXISTS divergence_'+mode+';')
    cur.close()
    con.commit()
    con.close()
