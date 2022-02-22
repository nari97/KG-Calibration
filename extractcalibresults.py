import sys
import sqlite3

if __name__ == '__main__':
    #folder = sys.argv[1]
    #dbname = sys.argv[2]  # database name

    folder = ''
    # 2 is regular targets and 3 is targets flipped!
    dbname = 'calib_2.db'

    models = ['analogy', 'complex', 'distmult', 'hole', 'rotate', 'simple', 'transd', 'transe', 'transh'
              ]
    model_map = {'analogy':'Analogy', 'complex': 'ComplEx', 'distmult':'DistMult', 'hole':'HolE', 'simple':'SimplE',
                 'transe':'TransE', 'transd':'TransD', 'transh':'TransH', 'rotate':'RotatE'
                 }
    datasets = ['0', '1', '2', '4', '5', '6', '3']
    assumptions = ['cwa', 'owa']

    # db connection
    con = sqlite3.connect(folder + 'Model/' + dbname)
    cur = con.cursor()

    print('Before calibration')
    for attrib_name in ['pos', 'all_neg']:
        print('Attribute:', attrib_name)
        for assumption in assumptions:
            if (attrib_name=='pos' or attrib_name=='tclcwa') and assumption=='cwa':
                continue

            for type in ['before_direct', 'before_minmax']:
                if type == 'before_direct':
                    print('Using sigmoid only')
                elif type == 'before_minmax':
                    print('Using minmax norm')

                best_by_model_dataset = {}
                for model_name in models:
                    if model_name not in best_by_model_dataset.keys():
                        best_by_model_dataset[model_name] = {}
                    cur.execute("SELECT model, "+attrib_name+" FROM brier_"+type+"_"+assumption+"_binary "
                                    "WHERE model LIKE '%" + model_name + "%';")

                    best_by_dataset = {}
                    for model, brier in cur.fetchall():
                        brier = float(brier)
                        model_split = model.split('_')
                        # Get dataset
                        dataset = model_split[0].split('/')[1]

                        if dataset not in best_by_dataset.keys():
                            best_by_dataset[dataset] = (brier, model)
                        else:
                            (existing_brier, existing_model) = best_by_dataset[dataset]
                            if existing_brier > brier:
                                best_by_dataset[dataset] = (brier, model)

                    for dataset in best_by_dataset.keys():
                        (brier, model) = best_by_dataset[dataset]
                        if dataset not in best_by_model_dataset[model_name].keys():
                            best_by_model_dataset[model_name][dataset] = (brier, type, model)
                        else:
                            (other_brier, other_type, other_model) = best_by_model_dataset[model_name][dataset]
                            if other_brier > brier:
                                best_by_model_dataset[model_name][dataset] = (brier, type, model)

                # Print results.
                print('Assumption:', assumption)
                for model in best_by_model_dataset.keys():
                    print(model_map[model], end='')
                    for dataset in datasets:
                        print('&', end='')

                        if dataset in best_by_model_dataset[model].keys():
                            (brier, type, model_name) = best_by_model_dataset[model][dataset]
                            print('{:.3f}'.format(float(brier)), end='')
                        else:
                            print('null', end='')
                    print('\\\\ \hline', end='')
                    print()

    # Those with largest Brier score of positives only
    selected_models = {}
    assumption, attrib_name, type = 'cwa', 'pos', 'before_minmax'
    for model_name in models:
        if model_name not in selected_models.keys():
            selected_models[model_name] = {}
        cur.execute("SELECT model, "+attrib_name+" FROM brier_"+type+"_"+assumption+"_binary "
                        "WHERE model LIKE '%" + model_name + "%';")

        best_by_dataset = {}
        for model, brier in cur.fetchall():
            brier = float(brier)
            model_split = model.split('_')
            # Get dataset
            dataset = model_split[0].split('/')[1]

            if dataset not in best_by_dataset.keys():
                best_by_dataset[dataset] = (brier, model)
            else:
                (existing_brier, existing_model) = best_by_dataset[dataset]
                if existing_brier < brier:
                    best_by_dataset[dataset] = (brier, model)

        for dataset in best_by_dataset.keys():
            (brier, model) = best_by_dataset[dataset]
            if dataset not in selected_models[model_name].keys():
                selected_models[model_name][dataset] = (brier, model)
            else:
                (other_brier, other_type, other_model) = selected_models[model_name][dataset]
                if other_brier < brier:
                    selected_models[model_name][dataset] = (brier, model)

    print('Min-max values')
    mm_by_model_dataset = {}
    for model_name in selected_models.keys():
        if model_name not in mm_by_model_dataset.keys():
            mm_by_model_dataset[model_name] = {}
        for dataset in selected_models[model_name].keys():
            (existing_brier, model_file) = selected_models[model_name][dataset]

            cur.execute("SELECT minvalue, maxvalue FROM minmax "
                        " WHERE model = '" + model_file + "';")
            count = 0
            for row in cur.fetchall():
                count += 1
            if count > 1:
                print('This cannot happen!')
                exit()
            elif count == 0:
                continue

            min, max = float(row[0]), float(row[1])
            mm_by_model_dataset[model_name][dataset] = (min, max)

    for model in mm_by_model_dataset.keys():
        print(model_map[model], end='')
        for dataset in datasets:
            print('&', end='')
            if dataset in mm_by_model_dataset[model].keys():
                (min, max) = mm_by_model_dataset[model][dataset]
                print('{:.3f}'.format(float(min)), end='')
                print('--', end='')
                print('{:.3f}'.format(float(max)), end='')
            else:
                print('null', end='')
        print('\\\\ \hline', end='')
        print()

    print('After calibration')
    type = 'after'

    for losstype in ['norank', 'rank']:
        print('Loss type:', losstype)
        for attrib_name in ['amr']:
            print('Attribute:', attrib_name)

            for semantic in [False, True]:
                if not semantic:
                    print('Closed-world calibration')
                else:
                    print('Open-world calibration')

                rank_by_model_dataset = {}
                for weights in [True]:

                    for model_name in selected_models.keys():
                        if model_name not in rank_by_model_dataset.keys():
                            rank_by_model_dataset[model_name] = {}
                        for dataset in selected_models[model_name].keys():
                            (existing_brier, model_file) = selected_models[model_name][dataset]

                            cur.execute("SELECT amr, ties, below FROM ranks "
                                        " WHERE model = '" + model_file + "' AND losstype='"+losstype+"' AND "
                                        "calibtype='"+str(semantic)+"';")
                            count = 0
                            for row in cur.fetchall():
                                count+=1
                            if count > 1:
                                print('This cannot happen!')
                                exit()
                            elif count == 0:
                                continue

                            amr, ties, below = float(row[0]), float(row[1]), float(row[2])

                            cur.execute("SELECT weight, bias FROM calibration "
                                        " WHERE model='" + model_file + "' AND losstype='"+losstype+"' AND "
                                        "calibtype='"+str(semantic)+"';")
                            count = 0
                            for row in cur.fetchall():
                                count+=1
                            if count > 1:
                                print('This cannot happen!')
                                exit()

                            A, B = row[0], row[1]

                            if dataset not in rank_by_model_dataset[model_name].keys():
                                rank_by_model_dataset[model_name][dataset] = (amr, ties, below, A, B)
                            else:
                                print('This cannot happen!')
                                exit()

                # Print results.
                print('Ranks:')
                for model in rank_by_model_dataset.keys():
                    print(model_map[model], end='')
                    for dataset in datasets:
                        print('&', end='')
                        if dataset in rank_by_model_dataset[model].keys():
                            (amr, ties, below, A, B) = rank_by_model_dataset[model][dataset]
                            print('{:.3f}'.format(float(amr)), end='')
                        else:
                            print('null', end='')
                    print('\\\\ \hline', end='')
                    print()
                print('A:')
                for model in rank_by_model_dataset.keys():
                    print(model_map[model], end='')
                    for dataset in datasets:
                        print('&', end='')
                        if dataset in rank_by_model_dataset[model].keys():
                            (amr, ties, below, A, B) = rank_by_model_dataset[model][dataset]
                            print('{:.3f}'.format(float(A)), end='')
                        else:
                            print('null', end='')
                    print('\\\\ \hline', end='')
                    print()
                print('Ties:')
                for model in rank_by_model_dataset.keys():
                    print(model_map[model], end='')
                    for dataset in datasets:
                        print('&', end='')
                        if dataset in rank_by_model_dataset[model].keys():
                            (amr, ties, below, A, B) = rank_by_model_dataset[model][dataset]
                            print('{:.3f}'.format(float(ties)), end='')
                        else:
                            print('null', end='')
                    print('\\\\ \hline', end='')
                    print()
                print('Below expected:')
                for model in rank_by_model_dataset.keys():
                    print(model_map[model], end='')
                    for dataset in datasets:
                        print('&', end='')
                        if dataset in rank_by_model_dataset[model].keys():
                            (amr, ties, below, A, B) = rank_by_model_dataset[model][dataset]
                            print('{:.3f}'.format(float(below)), end='')
                        else:
                            print('null', end='')
                    print('\\\\ \hline', end='')
                    print()

    print('Brier')
    for losstype in ['norank', 'rank']:
        print('Loss type:', losstype)
        for attrib_name in ['pos', 'all_neg',
                            'tclcwa', #'pos_neg',
                            'weighted'
                            ]:
            print('Attribute:', attrib_name)

            for semantic in [False, True]:
                assumptions_to_check = ['owa']
                if not semantic:
                    print('Closed-world calibration')
                    # In this case, we will check both closed-world and open-world Brier scores.
                    if attrib_name != 'pos':
                        assumptions_to_check.append('cwa')
                else:
                    print('Open-world calibration')

                for assumption in assumptions_to_check:
                    if attrib_name == 'tclcwa' and assumption == 'cwa':
                        continue

                    for weights in [True]:
                        brier_by_model_dataset = {}
                        for model_name in selected_models.keys():
                            if model_name not in brier_by_model_dataset.keys():
                                brier_by_model_dataset[model_name] = {}
                            for dataset in selected_models[model_name].keys():
                                (existing_brier, model_file) = selected_models[model_name][dataset]
                                cur.execute("SELECT "+attrib_name+" FROM brier_"+type+"_"+assumption+"_binary "
                                                "WHERE model = '" + model_file + "' AND losstype='"+losstype+"' AND "
                                                "calibtype='"+str(semantic)+"';")
                                count = 0
                                for row in cur.fetchall():
                                    count += 1
                                if count > 1:
                                    print('This cannot happen!')
                                    exit()
                                brier = float(row[0])

                                if dataset not in brier_by_model_dataset[model_name].keys():
                                    brier_by_model_dataset[model_name][dataset] = brier
                                else:
                                    print('This cannot happen!')
                                    exit()

                    # Print results.
                    print('Brier score:', assumption)
                    for model in brier_by_model_dataset.keys():
                        print(model_map[model], end='')
                        for dataset in datasets:
                            print('&', end='')
                            if dataset in brier_by_model_dataset[model].keys():
                                brier = brier_by_model_dataset[model][dataset]
                                print('{:.3f}'.format(float(brier)), end='')
                            else:
                                print('null', end='')
                        print('\\\\ \hline', end='')
                        print()

    # TODO Continue here!

    print('Divergence')
    losstype = 'rank'
    for div_type in ['minmax']:
        for attrib_name in ['pos', 'all_neg', 'pos_neg']:
            print('Attribute:', attrib_name)

            for semantic in [False, True]:
                if not semantic:
                    print('Closed-world calibration')
                else:
                    print('Open-world calibration')

                div_by_model_dataset = {}
                for model_name in selected_models.keys():
                    if model_name not in div_by_model_dataset.keys():
                        div_by_model_dataset[model_name] = {}
                    for dataset in selected_models[model_name].keys():
                        (existing_brier, model_file) = selected_models[model_name][dataset]
                        cur.execute(
                            "SELECT " + attrib_name + " FROM divergence_" + div_type + " "
                                "WHERE model = '" + model_file + "' AND losstype='" + losstype + "' AND "
                                    "calibtype='" + str(semantic) + "';")
                        count = 0
                        for row in cur.fetchall():
                            count += 1
                        if count > 1:
                            print('This cannot happen!')
                            exit()
                        div = float(row[0])

                        if dataset not in div_by_model_dataset[model_name].keys():
                            div_by_model_dataset[model_name][dataset] = div
                        else:
                            print('This cannot happen!')
                            exit()

                # Print results.
                for model in div_by_model_dataset.keys():
                    print(model_map[model], end='')
                    for dataset in datasets:
                        print('&', end='')
                        if dataset in div_by_model_dataset[model].keys():
                            div = div_by_model_dataset[model][dataset]
                            print('{:.3f}'.format(float(div)), end='')
                        else:
                            print('null', end='')
                    print('\\\\ \hline', end='')
                    print()

    print('Accuracy')
    losstype = 'norank'
    for semantic in [False, True]:
        if not semantic:
            print('Closed-world calibration')
        else:
            print('Open-world calibration')

        acc_by_model_dataset = {}
        for model_name in selected_models.keys():
            if model_name not in acc_by_model_dataset.keys():
                acc_by_model_dataset[model_name] = {}
            for dataset in selected_models[model_name].keys():
                (existing_brier, model_file) = selected_models[model_name][dataset]
                cur.execute(
                    "SELECT tp, tn, fp, fn FROM accuracy "
                    "WHERE model = '" + model_file + "' AND losstype='" + losstype + "' AND "
                    "calibtype='" + str(semantic) + "';")
                count = 0
                for row in cur.fetchall():
                    count += 1
                if count > 1:
                    print('This cannot happen!')
                    exit()
                tp, tn, fp, fn = float(row[0]), float(row[1]), float(row[2]), float(row[3])

                if dataset not in acc_by_model_dataset[model_name].keys():
                    acc_by_model_dataset[model_name][dataset] = (tp, tn, fp, fn)
                else:
                    print('This cannot happen!')
                    exit()

        # Print results.
        for model in acc_by_model_dataset.keys():
            print(model_map[model], end='')
            for dataset in datasets:
                print('&', end='')
                if dataset in acc_by_model_dataset[model].keys():
                    (tp, tn, fp, fn) = acc_by_model_dataset[model][dataset]
                    if  tp>0 or fp>0:
                        precision, recall = tp/(tp+fp), tp/(tp+fn)
                        f1 = 2.0*precision*recall/(precision+recall)
                    else:
                        f1 = .0
                    print('{:.3f}'.format(float(f1)), end='')
                else:
                    print('null', end='')
            print('\\\\ \hline', end='')
            print()

    con.close()
