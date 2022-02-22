import torch
from torch import nn, optim
import numpy as np
from torch.autograd import Variable
from datetime import datetime

# Calibration: take input model, manager, min, max, weights and return A and B.
class ModelCalibration(nn.Module):
    # Targets here. Positive is always 1.0; negatives should always be less than .5.
    @staticmethod
    def get_ground_truth_targets(targets=None):
        ground_truth_targets = {}
        ground_truth_targets['pos'] = 1.0
        ground_truth_targets['tclcwa'] = .0
        ground_truth_targets['lcwa'] = .0
        ground_truth_targets['global'] = .0

        if targets is not None:
            for key in ['pos', 'tclcwa', 'lcwa', 'global']:
                if key in targets.keys():
                    ground_truth_targets[key] = targets[key]
        return ground_truth_targets

    @staticmethod
    def get_accuracy(tp, tn, fp, fn):
        recall, tnr = tp/(tp + fn), tn/(tn+fp)
        return 2.0*recall*tnr/(recall+tnr)

    def __init__(self, model, min, max, mean, std, scaletype="norm", losstype="rank",
                 A=-.5, B=-.5, totals=None, targets=None):
        super(ModelCalibration, self).__init__()
        # self.cuda()
        self.model = model
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std
        self.scale = scaletype
        self.loss = losstype
        self.A = nn.Parameter(torch.tensor([A], dtype=torch.float64))
        self.B = nn.Parameter(torch.tensor([B], dtype=torch.float64))
        # https://stats.stackexchange.com/questions/465937/how-to-choose-delta-parameter-in-huber-loss-function
        self.delta = nn.Parameter(torch.tensor([1.35], dtype=torch.float64))

        self.totals = totals
        self.ground_truth_targets = ModelCalibration.get_ground_truth_targets(targets)

    def rescaling(self, logits, a=-20, b=20):
        return (b-a)*((logits-self.min)/(self.max-self.min))+a

    def meannorm(self, logits):
        return (logits-self.mean)/(self.max-self.min)

    def zscorenorm(self, logits):
        return (logits-self.mean)/self.std

    def forward(self, input, usesigmoid=True):
        logits = torch.tensor(self.model.predict(input)).float()

        if self.scale.startswith('norm'):
            a = float(self.scale.replace('norm', ''))
            logits = self.rescaling(logits, a=-a, b=a)
        if self.scale == 'mean':
            logits = self.meannorm(logits)
        if self.scale == 'zscr':
            logits = self.zscorenorm(logits)

        logits = torch.add(torch.mul(logits, self.A), self.B)
        if usesigmoid:
            logits = torch.sigmoid(logits)
        return logits

    def predict(self, input):
        # Positives are 1.0 and negatives are below .5; ranking is expected to be lowest is best, so leading minus.
        return -self.forward(input)

    def triples_by_relation(self, manager=None, rel_anomaly_max=1, rel_anomaly_min=0):
        # We will split the evaluation by relation
        relations = {}
        for t in manager.get_triples():
            # We will not consider these anomalies.
            if manager.relation_anomaly[t.r] < rel_anomaly_min or \
                    manager.relation_anomaly[t.r] > rel_anomaly_max:
                continue
            if t.r not in relations.keys():
                relations[t.r] = []
            relations[t.r].append(t)
        return relations

    def prepare_batch(self, t, manager=None):
        corruptedHeads, headscores = manager.get_corrupted_scores(t.h, t.r, t.t,
                                            targets=self.ground_truth_targets, type="head")
        corruptedTails, tailscores = manager.get_corrupted_scores(t.h, t.r, t.t,
                                            targets=self.ground_truth_targets, type="tail")
        groundTruth = np.concatenate([np.array([self.ground_truth_targets['pos']]), headscores, tailscores])

        totalTriples = 1 + len(corruptedHeads) + len(corruptedTails)
        arrH = np.zeros(totalTriples, dtype=np.int64)
        arrR = np.zeros(totalTriples, dtype=np.int64)
        arrT = np.zeros(totalTriples, dtype=np.int64)

        arrH[0], arrR[0], arrT[0] = t.h, t.r, t.t

        arrH[1:1 + len(corruptedHeads)] = list(corruptedHeads)
        arrR[1:1 + len(corruptedHeads)] = t.r
        arrT[1:1 + len(corruptedHeads)] = t.t

        arrH[1 + len(corruptedHeads):] = t.h
        arrR[1 + len(corruptedHeads):] = t.r
        arrT[1 + len(corruptedHeads):] = list(corruptedTails)

        batch = {
            'batch_h': Variable(torch.from_numpy(np.array(arrH))),
            'batch_r': Variable(torch.from_numpy(np.array(arrR))),
            'batch_t': Variable(torch.from_numpy(np.array(arrT))),
            'mode': 'normal'
        }
        return batch, groundTruth, 1 + len(corruptedHeads)

    def train_calib(self, manager=None, rel_anomaly_max=1, rel_anomaly_min=0):
        # We will split the evaluation by relation
        relations = self.triples_by_relation(manager, rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min)

        optimizer = optim.Adam(self.parameters(), lr=.0001)
        #optimizer = optim.SGD(params, lr=.0001)

        nplus, nminus = self.totals['pos'], self.totals['global'] + self.totals['lcwa'] + self.totals['tclcwa']

        usesigmoid = True
        if self.loss == 'bce':
            usesigmoid = False

        for r in relations.keys():
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), ' -- ', 'Relation: ', r)
            for i, t in enumerate(relations[r]):
                batch, ground_truth, object_corrupt_pos = self.prepare_batch(t, manager=manager)

                # Weights to account for imbalanced data.
                weights = np.zeros(len(ground_truth), dtype=np.float64)
                weights[np.where(ground_truth == self.ground_truth_targets['pos'])[0]] = \
                    self.totals['pos'] / (nplus + nminus)
                weights[np.where(ground_truth == self.ground_truth_targets['global'])[0]] = \
                    self.totals['global'] / (nplus + nminus)
                weights[np.where(ground_truth == self.ground_truth_targets['lcwa'])[0]] = \
                    self.totals['lcwa'] / (nplus + nminus)
                weights[np.where(ground_truth == self.ground_truth_targets['tclcwa'])[0]] = \
                    self.totals['tclcwa'] / (nplus + nminus)
                weights = torch.tensor(weights)

                # Let's reduce overfitting.
                ground_truth[np.where(ground_truth == self.ground_truth_targets['pos'])[0]] = \
                    self.ground_truth_targets['pos'] - (1 / (nplus + 2))
                ground_truth[np.where(ground_truth == self.ground_truth_targets['global'])[0]] = \
                    self.ground_truth_targets['global'] + (1 / (nminus + 2))
                ground_truth[np.where(ground_truth == self.ground_truth_targets['lcwa'])[0]] = \
                    self.ground_truth_targets['lcwa'] + (1 / (nminus + 2))
                ground_truth[np.where(ground_truth == self.ground_truth_targets['tclcwa'])[0]] = \
                    self.ground_truth_targets['tclcwa'] + (1 / (nminus + 2))
                ground_truth = torch.tensor(ground_truth)

                logits = self(batch, object_corrupt_pos=object_corrupt_pos, usesigmoid=usesigmoid)

                # Optimize
                if self.loss == 'bce':
                    criterion = nn.BCEWithLogitsLoss(weight=weights)
                    loss = criterion(logits, ground_truth)
                if self.loss == 'se':
                    def squared_error_loss(weights, logits, targets):
                        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.184.5203&rep=rep1&type=pdf
                        return torch.mean(weights * (targets * (1 - logits) ** 2 + (1 - targets) * logits ** 2))
                    loss = squared_error_loss(weights, logits, ground_truth)
                if self.loss == 'huber':
                    def huber_loss(weights, logits, targets):
                        # https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
                        return torch.mean(
                            weights * (self.delta ** 2 * (torch.sqrt(1 + ((targets - logits) / self.delta) ** 2) - 1)))
                    loss = huber_loss(weights, logits, ground_truth)
                if self.loss == 'l1':
                    def l1_loss(weights, logits, targets):
                        # http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/
                        return torch.mean(weights * torch.abs(logits - targets))
                    loss = l1_loss(weights, logits, ground_truth)
                if self.loss == 'l2':
                    def l2_loss(weights, logits, targets):
                        # http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/
                        return torch.mean(weights * (logits - targets) ** 2)
                    loss = l2_loss(weights, logits, ground_truth)

                # This gives a NaN error!
                # if self.loss == 'boosting':
                #    def boosting_loss(logits, targets):
                #        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.184.5203&rep=rep1&type=pdf
                #        return torch.mean(targets*torch.sqrt((1-logits)/logits) +
                #                          (1-targets)*torch.sqrt(logits/(1-logits)))
                #    siglogits = torch.sigmoid(logits)
                #    loss = boosting_loss(siglogits, torch.tensor(groundTruth))

                # Do the magic!
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #print('A:', self.A, '; B:', self.B)
                #print('Loss:', loss.item())

    def init_dict_types(self):
        dict = {}
        dict['pos'] = 0
        dict['global'] = 0
        return dict

    def update_dict_types(self, dict, gt, a):
        dict['pos'] += torch.sum(a[gt==self.ground_truth_targets['pos']]).item()
        dict['global'] += torch.sum(a[gt == .0]).item()

    def sum_scores(self, dict):
        dict['pos_neg'] = dict['global'] + dict['pos']

    # Compute sum of squared errors for each type and all types
    def evaluate_brier(self, manager=None, rel_anomaly_max=1, rel_anomaly_min=0):
        # TODO Only CWA and only binary!

        # We will split the evaluation by relation
        relations = self.triples_by_relation(manager, rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min)

        brier_results = {}
        type  = 'cwa'
        #brier_results['owa'] = (self.init_dict_types(True), self.init_dict_types(True), self.init_dict_types(True))
        brier_results['cwa'] = (self.init_dict_types(), self.init_dict_types(), self.init_dict_types())

        for r in relations.keys():
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), ' -- ', 'Relation: ', r)
            for i, t in enumerate(relations[r]):
                batch, groundTruth = self.prepare_batch(t, manager=manager)
                predictions = self.forward(batch)

                # Change ground truth
                groundTruth[np.where(groundTruth < .5)[0]] = .0
                (sum_sq_errors, sum_sq_errors_all, totals) = brier_results[type]

                # Get squared errors and sum.
                sq_errors = torch.pow(predictions - torch.tensor(groundTruth), 2)
                self.update_dict_types(sum_sq_errors, groundTruth, sq_errors)

                # Create tensor of ones and count totals.
                self.update_dict_types(totals, groundTruth,
                        torch.tensor([self.ground_truth_targets['pos']] * groundTruth.size))

                # Get squared errors for each type and sum.
                sq_errors = torch.pow(predictions -
                        torch.tensor([self.ground_truth_targets['pos']] * groundTruth.size), 2)
                self.update_dict_types(sum_sq_errors_all, groundTruth, sq_errors)
                sq_errors = torch.pow(predictions - torch.tensor([.0] * groundTruth.size), 2)
                self.update_dict_types(sum_sq_errors_all, groundTruth, sq_errors)

        brier_return = {}
        brier_return[type] = {}
        brier_return[type]['binary'] = {}

        (sum_sq_errors, sum_sq_errors_all, totals) = brier_results[type]
        # Create all negatives and all.
        self.sum_scores(sum_sq_errors)
        self.sum_scores(sum_sq_errors_all)
        self.sum_scores(totals)

        for key in sum_sq_errors.keys():
            brier_return[type]['binary'][key] = sum_sq_errors[key]/totals[key]
            print('\tBrier ('+key+'):', brier_return[type]['binary'][key])
        return brier_return

    # Compute divergence: we compute p and q; p is before and q is after. For p, there is the option of using
    #   only sigmoid(x), a.k.a. direct, or sigmoid(min-max(x)), a.k.a. minmax. For both p and q, each element
    #   is the score at hand divided by the sum of all scores, so we need to compute such sum first.
    def evaluate_divergence(self, manager=None, rel_anomaly_max=1, rel_anomaly_min=0):
        # TODO Are we using this?

        # We will split the evaluation by relation
        relations = self.triples_by_relation(manager, rel_anomaly_max=rel_anomaly_max, rel_anomaly_min=rel_anomaly_min)

        # Compute the sum of all scores.
        sum_scores_before_direct, sum_scores_before_minmax, sum_scores_after = \
            self.init_dict_types(), self.init_dict_types(), self.init_dict_types()

        for r in relations.keys():
            for i, t in enumerate(relations[r]):
                batch, groundTruth = self.prepare_batch(t, manager=manager)

                # Compute predictions.
                model_predictions = self.model.forward(batch)
                sigmoid_predictions = torch.sigmoid(model_predictions)
                self.update_dict_types(sum_scores_before_direct, groundTruth, sigmoid_predictions)
                normalized_predictions = torch.sigmoid(self.rescaling(model_predictions))
                self.update_dict_types(sum_scores_before_minmax, groundTruth, normalized_predictions)
                calib_predictions = self.forward(batch)
                self.update_dict_types(sum_scores_after, groundTruth, calib_predictions)

        # Compute divergence using m = 1/2(p+q) (Jensen-Shannon divergence).
        # Include all negatives and all.
        self.sum_scores(sum_scores_before_direct)
        self.sum_scores(sum_scores_before_minmax)
        self.sum_scores(sum_scores_after)

        # Divergence scores.
        divergence_direct, divergence_minmax = self.init_dict_types(), self.init_dict_types()
        # We will also compute all negatives and all, so add them to the dictionaries.
        self.sum_scores(divergence_direct)
        self.sum_scores(divergence_minmax)

        def update_divergence(p, totalp, q, totalq, gt, div, type):
            def update(condition, key):
                # Do nothing if skip.
                if self.skip(type, key):
                    return
                newp = p[condition] / totalp[key]
                newq = q[condition] / totalq[key]
                newm = .5 * (newp + newq)

                # Avoiding zeros.
                newp[newp!=0] = newp[newp!=0] * torch.log(newp[newp!=0]/newm[newp!=0])
                newq[newq!=0] = newq[newq!=0] * torch.log(newq[newq!=0]/newm[newq!=0])

                div[key] += torch.sum(newp).item() + torch.sum(newq).item()

            update(gt == self.ground_truth_targets['pos'], 'pos')
            update(gt == self.ground_truth_targets['global'], 'global')
            update(gt == self.ground_truth_targets['tclcwa'], 'tclcwa')
            update(gt == self.ground_truth_targets['lcwa'], 'lcwa')
            update(gt < 0.5, 'all_neg')
            update(gt <= 1.0, 'pos_neg')

        type = 'owa' if len(self.ground_truth_targets)>2 else 'cwa'
        for r in relations.keys():
            for i, t in enumerate(relations[r]):
                batch, groundTruth = self.prepare_batch(t, manager=manager)

                # Compute predictions.
                model_predictions = self.model.forward(batch)
                sigmoid_predictions = torch.sigmoid(model_predictions)
                normalized_predictions = torch.sigmoid(self.rescaling(model_predictions))
                calib_predictions = self.forward(batch)

                update_divergence(sigmoid_predictions, sum_scores_before_direct,
                                  calib_predictions, sum_scores_after, groundTruth, divergence_direct, type)
                update_divergence(normalized_predictions, sum_scores_before_minmax,
                                  calib_predictions, sum_scores_after, groundTruth, divergence_minmax, type)

        div_results = {}
        # The actual divergence is .5*(D(p||m) + D(q||m)).
        for i, divergence in enumerate([divergence_direct, divergence_minmax]):
            type = 'direct' if i==0 else 'minmax'
            div_results[type] = {}
            for key in divergence.keys():
                div_results[type][key] = .5*divergence[key]
                print('Divergence '+type+' ('+key+'): ', div_results[type][key])
        return div_results
