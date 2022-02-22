from TripleManager import TripleManager
from minmaxeval import MinMaxEval
from ModelUtils import ModelUtils
import sys
import glob
import sqlite3

def get_scn(scn):
    i = 7 * 9 # datasets
    for dataset in [0, 1, 2, 3, 4, 5, 6]:
        for model_name in ["transe", "transh", "transd", "distmult", "complex", "hole", "simple", "analogy", "rotate"]:
            i=i-1
            if scn==i:
                return dataset, model_name

if __name__ == '__main__':
    folder = sys.argv[1]
    dbname = sys.argv[2]  # database name
    dataset, model_name = get_scn(int(sys.argv[3])) # 0--62
    corruption_mode = "LCWA"

    # db connection
    con = sqlite3.connect(folder + 'Model/' + dbname)

    # This is to avoid database locked due to many jobs accessing the file database at the same time.
    # Check if table exists first.
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Model';")
    check = cur.fetchone()
    cur.close()

    if not check:
        cur = con.cursor()
        # These are all the models we have trained.
        cur.executescript("CREATE TABLE IF NOT EXISTS Model(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                          "dataset INTEGER, algorithm TEXT, modelfile TEXT, minvalue TEXT, maxvalue TEXT, "
                          "mean TEXT, std TEXT, pos TEXT, tclcwa TEXT, lcwa TEXT, global TEXT);")
        con.commit()
        cur.close()
    con.close()

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

    manager = TripleManager(path, splits=["new_valid", "new_train"], corruption_mode=corruption_mode)

    # Find min-max values.
    print("Collecting min-max values for each model")
    # using all data for min-max
    minmaxevaluator = MinMaxEval(manager, use_gpu=False, rel_anomaly_max=1, rel_anomaly_min=0)

    # Let's load all models!
    for model_file in glob.glob(folder + "Model/" + str(dataset) + "/" + model_name + "*.model"):
        print(model_file.split("Model")[1])

        filename = model_file.split("Model")[1]
        filename = filename.replace('\\', '/')

        # Ask if it is there.
        con = sqlite3.connect(folder + 'Model/' + dbname)
        cur = con.cursor()
        cur.execute("SELECT id FROM Model WHERE dataset=? AND algorithm=? AND modelfile=?;",
                    (dataset, model_name, filename))
        check = cur.fetchone()
        cur.close()
        con.close()

        util = ModelUtils(model_name, ModelUtils.get_params(model_file))
        model = util.get_model(manager.entityTotal, manager.relationTotal, 0)
        model.model.load_checkpoint(model_file)
        maxvalue, minvalue, mean, std, pos, tclcwa, lcwa, glbl = minmaxevaluator.evaluate(model.model)

        print('Min:', minvalue, 'Max:', maxvalue, 'Mean: ', mean, 'Std: ', std)

        # Saving min-max values.
        con = sqlite3.connect(folder + 'Model/' + dbname)
        cur = con.cursor()
        if check is None:
            cur.execute("INSERT INTO Model(dataset, algorithm, modelfile, minvalue, maxvalue, "
                      "mean, std, pos, tclcwa, lcwa, global) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    (dataset, model_name, filename, str(minvalue), str(maxvalue), str(mean), str(std), str(pos),
                     str(tclcwa), str(lcwa), str(glbl)))
        else:
            cur.execute("UPDATE Model SET minvalue=?, maxvalue=?, mean=?, std=?, pos=?, "
                        "tclcwa=?, lcwa=?, global=? WHERE id=?;",
                    (str(minvalue), str(maxvalue), str(mean), str(std), str(pos),
                     str(tclcwa), str(lcwa), str(glbl), check[0]))
        cur.close()
        con.commit()
        con.close()
