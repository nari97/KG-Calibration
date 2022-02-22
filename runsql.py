import sys
import sqlite3
from datetime import datetime
import struct

if __name__ == '__main__':
    folder = './'
    dbname = 'final_calib.db'  # database name

    # db connection
    con = sqlite3.connect(folder + 'Model/'+dbname)
    cur = con.cursor()

    file1 = open('OSG/updates.txt', 'r')
    count = 0

    while True:
        count += 1

        # Get next line from file
        line = file1.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break

        if count % 100 == 0:
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), ' -- ', 'Done: ', count)
            con.commit()

        cur.executescript(line)

    file1.close()

    cur.close()
    con.commit()
    con.close()
