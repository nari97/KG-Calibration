import sys
import sqlite3

# /home/crrvcs/ActivePython-3.7/bin/python3 -u ./Code/gzip_models.py /home/crrvcs/OpenKE/ final_calib.db
def run():
    folder = sys.argv[1]
    dbname = sys.argv[2]  # database name

    # Let's create all tables first.
    con = sqlite3.connect(folder + 'Model/' + dbname)

    cur = con.cursor()

    cur.execute("SELECT id, modelfile FROM Model;")
    for row in cur:
        (id, file) = row
        print("tar -czvf "+str(id)+".tar.gz Model"+file)
    cur.close()
    con.close()

if __name__ == '__main__':
    run()
