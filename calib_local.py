from Evaluator import Evaluator
from plattscaling import ModelCalibration
from ModelUtils import ModelUtils
from TripleManager import TripleManager
import sys
import sqlite3
import time


if __name__ == '__main__':
    folder = './'
    type = 'valid'  # Always validation for training calibration.
    corruption_mode = 'LCWA'

    glbl, tclcwa, lcwa, pos = .95, .951, .952, 0.05

    target_scheme = {'global': glbl, 'tclcwa': tclcwa, 'lcwa': lcwa, 'pos': pos}

    dataset = 6
    model_file = '/6/simple_lr_0.4169427064143164_nr_6_nbatches_127_wd_2.873718479268297e-08_m_0.7979868658570197_dim_229_bern_True_trial_index_12.model'
    model_name = 'simple'
    losstype = 'bce'
    normtype = 'norm10'

    (min, max, mean, std, totpos, tottclcwa, totlcwa, totglobal) = \
        (-1.7752957344055176, 5.70140266418457, 1.0456207232900932e-06, 0.001237435597695769,
         2044, 45969801, 104042224, 17310636)

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

    calibmodel = ModelCalibration(model.model, min, max, mean, std, scaletype=normtype, losstype=losstype,
                                  totals=totals_scheme, targets=target_scheme)
    calibmodel.train_calib(manager)
    a, b = calibmodel.A.item(), calibmodel.B.item()

    print('A:',a,'; B:',b)

    evaluator = Evaluator(manager, rel_anomaly_max=1, rel_anomaly_min=0)
    rc, tp, tn, fp, fn = evaluator.evaluate(calibmodel, False, accuracy=True)
    acc = ModelCalibration.get_accuracy(tp, tn, fp, fn)

    print('MR: ', rc.get_metric(metric_str='mr').get(), '; Adjusted MR: ', rc.get_metric(metric_str='mrh').get(),
          '; TP: ', tp, '; TN: ', tn, '; FP: ', fp, '; FN: ', fn, '; Acc: ', acc)
