import sys
from ax.service.ax_client import AxClient
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements
from ax.service.utils.best_point import get_best_raw_objective_point
import sqlite3

def get_params():
    parameters = {}
    parameters['global'] = {"name": "global", "type": "range", "bounds": [1e-10, .5 - 1e-10], "value_type": "float"}
    parameters['tclcwa'] = {"name": "tclcwa", "type": "range", "bounds": [1e-10, .5 - 1e-10], "value_type": "float"}
    parameters['lcwa'] = {"name": "lcwa", "type": "range", "bounds": [1e-10, .5 - 1e-10], "value_type": "float"}
    parameters['pos'] = {"name": "pos", "type": "range", "bounds": [.5 + 1e-10, 1.0], "value_type": "float"}

    parameters['loss'] = {"name": "loss", "type": "range", "bounds": [0, 4], "value_type": "int"}
    parameters['norm'] = {"name": "norm", "type": "range", "bounds": [0, 6], "value_type": "int"}
    return parameters

# /home/crrvcs/ActivePython-3.7/bin/python3 -u ./Code/exp_calib.py /home/crrvcs/OpenKE/ final_calib.db pending_1.txt
def run():
    folder = sys.argv[1]
    dbname = sys.argv[2]  # database name
    outname = sys.argv[3]  # name of the file to write

    # Let's create all tables first.
    con = sqlite3.connect(folder + 'Model/' + dbname)

    cur = con.cursor()
    # These are all the models we have trained. Run mm.py to create the Model table!
    # This is to record an experiment, that is, a number of trials we would like to evaluate.
    cur.executescript("CREATE TABLE IF NOT EXISTS Experiment(mid INTEGER, eid INTEGER, axfile TEXT, complete TEXT, "
                      "description TEXT, PRIMARY KEY(mid, eid), FOREIGN KEY (mid) REFERENCES Model(id));")
    # This is to record trials associated with an experiment; tid may come from Ax.
    cur.executescript("CREATE TABLE IF NOT EXISTS Experiment_Trial(mid INTEGER, eid INTEGER, tid INTEGER, status TEXT,"
                      "loss TEXT, norm TEXT, global TEXT, tclcwa TEXT, lcwa TEXT, pos TEXT, "
                      "a TEXT, b TEXT, delta TEXT, accvalid TEXT, acctest TEXT, brier TEXT, "
                      "PRIMARY KEY(mid, eid, tid), FOREIGN KEY (mid, eid) REFERENCES Experiment(mid, eid));")
    cur.close()
    con.commit()

    losstypes = ['bce', 'se', 'huber', 'l1', 'l2']
    # sigmoid does nothing, no scaling.
    normtypes = ['sigmoid', 'norm1', 'norm5', 'norm10', 'norm20', 'mean', 'zscr']

    def next_trials(ax_client, con, mid, eid, n=None):
        if n is None:
            num_trials = trials_in_parallel(ax_client)
        else:
            num_trials = n

        for i in range(num_trials):
            trial_params, trial_index = ax_client.get_next_trial()
            pos = 1.0
            if eid==2:
                pos = trial_params['pos']
            innercur = con.cursor()
            innercur.execute("INSERT INTO Experiment_Trial(mid, eid, tid, status, loss, norm, "
                             "global, tclcwa, lcwa, pos) VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (mid, eid, trial_index, 'Pending', losstypes[trial_params['loss']],
                              normtypes[trial_params['norm']], str(trial_params['global']),
                              str(trial_params['tclcwa']), str(trial_params['lcwa']), str(pos)))
            innercur.close()
            con.commit()

    def trials_in_parallel(ax_client):
        (num_trials, max_setting) = ax_client.get_max_parallelism()[0]
        if max_setting == -1:
            return num_trials
        else:
            return max_setting

    def get_parameters(parameters, eid):
        params = ['global', 'tclcwa', 'lcwa', 'loss', 'norm']
        if eid==2:
            params.append('pos')
        ret = []
        for p in params:
            ret.append(parameters[p])
        return ret

    parameters = get_params()

    # We are going to define experiments: 1 is for binary targets, which we are going to try all loss and norm
    #   options; 2 is for non-binary targets, which we are going to use Ax; 3 is for non-binary targets but pos=1.0.

    # Experiment 1.
    cur = con.cursor()
    cur.execute("SELECT id FROM Model;")
    for row in cur:
        innercur = con.cursor()
        innercur.execute("SELECT mid FROM Experiment WHERE mid=? AND eid=1;", (row[0],))
        check = innercur.fetchone()
        innercur.close()

        # This experiment does not exist.
        if not check:
            innercur = con.cursor()
            innercur.execute("INSERT INTO Experiment(mid, eid, complete, description) "
                             "VALUES (?,1,'N','Binary targets')", (row[0],))
            innercur.close()
            con.commit()

            # Create all trials.
            tid = 0
            innercur = con.cursor()
            for loss in losstypes:
                for norm in normtypes:
                    innercur.execute("INSERT INTO Experiment_Trial(mid, eid, tid, status, "
                                     "loss, norm, global, tclcwa, lcwa, pos) "
                                     "VALUES (?,1,?,?,?,?,'.0','.0','.0','1.0')", (row[0], tid, 'Pending', loss, norm))
                    tid += 1
            innercur.close()
            con.commit()
    cur.close()

    # Experiments 2 and 3.
    for eid in [2, 3]:
        cur = con.cursor()
        cur.execute("SELECT id FROM Model;")
        for row in cur:
            innercur = con.cursor()
            innercur.execute("SELECT mid FROM Experiment WHERE mid=? AND eid=?;", (row[0],eid))
            check = innercur.fetchone()
            innercur.close()

            # This experiment does not exist.
            if not check:
                # Create Ax file.
                ax_file = folder + "Ax_Calib/" + str(row[0]) + "_" + str(eid) + ".ax"

                # Create client
                ax_client = AxClient(enforce_sequential_optimization=False, verbose_logging=False)
                ax_client.create_experiment(
                    name=row[0],
                    parameters=get_parameters(parameters, eid),
                    objective_name="accvalid",
                    minimize=False,
                )

                innercur = con.cursor()
                innercur.execute("INSERT INTO Experiment(mid, eid, axfile, complete, description) "
                                 "VALUES (?,?,?,'N','Non-binary targets')", (row[0], eid, ax_file))
                innercur.close()
                con.commit()

                # Get the next trials.
                next_trials(ax_client, con, row[0], eid)

                ax_client.save_to_json_file(ax_file)
        cur.close()

        # Only for experiments 2/3: Register completed trials in Ax and check if it suggests more experiments.
        cur = con.cursor()
        cur.execute("SELECT id FROM Model;")
        for row in cur:
            innercur = con.cursor()
            innercur.execute("SELECT axfile FROM Experiment WHERE mid=? AND eid=? AND complete='N';", (row[0], eid))
            ax_file = innercur.fetchone()
            if ax_file is None:
                # we are done!
                continue
            ax_file = ax_file[0]
            innercur.close()

            if ax_file is not None:
                # Load client.
                ax_client = AxClient.load_from_json_file(ax_file, verbose_logging=False)

                # Get trials (completed and not completed).
                pending = 0

                innercur = con.cursor()
                innercur.execute("SELECT tid, status, accvalid FROM Experiment_Trial WHERE mid=? AND eid=?;", (row[0], eid))
                for trial in innercur:
                    # Get trial in file.
                    trial_in_file = ax_client.experiment.trials[trial[0]]

                    # Check w.r.t. to the file
                    if trial[1] == 'Completed' and not trial_in_file.status.is_completed:
                        ax_client.complete_trial(trial[0], raw_data={'accvalid': float(trial[2])})
                    elif trial[1] != 'Completed' and trial_in_file.status.is_completed:
                        print('Ax file: ', ax_file, ' has trial ', trial[0],
                              ' marked as completed but not completed in DB')
                    elif trial[1] != 'Completed':
                        pending += 1
                innercur.close()

                # Save client.
                ax_client.save_to_json_file(ax_file)

                # Check if it suggests more experiments.
                if pending == 0:
                    compute_next = False

                    try:
                        predictions = ax_client.get_model_predictions()
                    except:
                        predictions = None

                    if predictions is None or not predictions.keys():
                        compute_next = True
                        print('No model predictions')
                    else:
                        # Get current, real mean.
                        current_best_parameters, values = get_best_raw_objective_point(ax_client.experiment)
                        current_acc, current_sem = values['accvalid']

                        # Get new, expected mean.
                        new_best_parameters, values = ax_client.get_best_parameters()
                        means, covariances = values
                        new_acc = means['accvalid']

                        if current_acc >= new_acc:
                            print('Experiment is over! Best model: ', current_best_parameters, '; Acc:', current_acc)
                            plot = ax_client.get_optimization_trace(objective_optimum=1.0)
                            with open(folder + "Ax_Calib/" + str(row[0]) + '_' + str(eid) + '.html', 'w') as outfile:
                                outfile.write(render_report_elements(
                                    row[0], html_elements=[plot_config_to_html(plot)], header=False,
                                ))

                            innercur = con.cursor()
                            innercur.execute("UPDATE Experiment SET complete='Y' WHERE mid=? AND eid=?;", (row[0], eid))
                            con.commit()
                            innercur.close()
                        else:
                            print('Current Acc:', current_acc, '; Expected Acc:', new_acc)
                            # Request next trial, one by one now.
                            next_trials(ax_client, con, row[0], eid, n=1)

                    if compute_next:
                        next_trials(ax_client, con, row[0], eid)

                # Save client
                ax_client.save_to_json_file(ax_file)
        cur.close()

    with open(folder + outname, 'w') as f:
        f.write('ModelId,TrialId,ExperimentId,DatasetId\n')

        # Pending trials!
        cur = con.cursor()
        cur.execute("SELECT id, dataset FROM Model;")
        for row in cur:
            innercur = con.cursor()
            innercur.execute("SELECT tid, eid FROM Experiment_Trial WHERE mid=? AND status='Pending';", (row[0],))
            for trial in innercur:
                f.write(str(row[0]))
                f.write(',')
                f.write(str(trial[0]))
                f.write(',')
                f.write(str(trial[1]))
                f.write(',')
                f.write(str(row[1]))
                f.write('\n')
            innercur.close()

            innercur = con.cursor()
            innercur.execute("UPDATE Experiment_Trial SET status='Running' WHERE mid=? AND status='Pending';", (row[0],))
            con.commit()
            innercur.close()
        cur.close()

    con.close()


if __name__ == '__main__':
    run()
