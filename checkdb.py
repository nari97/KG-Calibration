import sqlite3

if __name__ == '__main__':
    folder=''
    dbname = 'calib.db'
    print('DB:',dbname)

    # db connection
    con = sqlite3.connect(folder + 'Model/'+dbname)
    cur = con.cursor()

    cur.execute('SELECT * FROM minmax;')
    for row in cur.fetchall():
        print('\t',row)
    con.close()
