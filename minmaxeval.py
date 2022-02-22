import torch
from torch.autograd import Variable
import numpy as np
from plattscaling import ModelCalibration

class MinMaxEval(object):

    """ This can be either the validator or tester depending on the manager used. E"""
    def __init__(self, manager=None, use_gpu=False, rel_anomaly_max=.75, rel_anomaly_min=0):
        self.manager = manager
        self.use_gpu = use_gpu

        self.rel_anomaly_max = rel_anomaly_max
        self.rel_anomaly_min = rel_anomaly_min

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    # Compute mean and variance online (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm).
    def update_aggregations(self, count, mean, M2, new):
        count += 1
        delta = new - mean
        mean += delta / count
        delta2 = new - mean
        M2 += delta * delta2
        return count, mean, M2

    def finalize_aggregations(self, count, mean, M2):
        if count < 2:
            return float("nan")
        else:
            mean, variance, sampleVariance = mean, M2 / count, M2 / (count - 1)
            return mean, variance, sampleVariance

    def evaluate(self, model):
        if self.use_gpu:
            model.cuda()
        # We will split the evaluation by relation
        relations = {}
        for t in self.manager.get_triples():
            # We will not consider these anomalies.
            if self.manager.relation_anomaly[t.r] < self.rel_anomaly_min or \
                    self.manager.relation_anomaly[t.r] > self.rel_anomaly_max:
                continue
            if t.r not in relations.keys():
                relations[t.r] = []
            relations[t.r].append(t)

        # To compute mean and standard deviation online.
        (count, mean, M2) = (0, 0, 0)
        # Let's get the ground truth targets. These values must not be repeated but they do not really matter
        #       since we are just matching them and counting.
        gttargets = ModelCalibration.get_ground_truth_targets({'tclcwa':.375, 'lcwa':.250, 'global':.125})

        minvalue, maxvalue = None, None
        ng, nt, nl, npos = 0, 0, 0, 0
        for r in relations.keys():
            for t in relations[r]:
                npos+=1 #every triple is a positive
                corruptedHeads, headscores = self.manager.get_corrupted_scores(t.h, t.r, t.t, gttargets, type="head")
                corruptedTails, tailscores = self.manager.get_corrupted_scores(t.h, t.r, t.t, gttargets, type="tail")

                for scores in [headscores, tailscores]:
                    num, counts = np.unique(scores, return_counts=True)
                    mapping = dict(zip(num,counts))

                    ng+=mapping.get(gttargets['global'],0) #global
                    nt+=mapping.get(gttargets['tclcwa'],0) #type
                    nl+=mapping.get(gttargets['lcwa'],0) #lcwa

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

                scores = self.predict(arrH, arrR, arrT, model)
                currentmin, currentmax = min(scores), max(scores)
                if minvalue is None or currentmin < minvalue:
                    minvalue = currentmin
                if maxvalue is None or currentmax > maxvalue:
                    maxvalue = currentmax

                # TODO Remove!
                print('Currentmin:',currentmin,'Minvalue:',minvalue,'Currentmax:',currentmax,'Maxvalue:',maxvalue)

                for s in scores:
                    count, mean, M2 = self.update_aggregations(count, mean, M2, s)

        mean, variance, sampleVariance = self.finalize_aggregations(count, mean, M2)
        return maxvalue, minvalue, mean, sampleVariance**.5, npos, nt, nl, ng

    def predict(self, arrH, arrR, arrT, model):
        return model.predict({
            'batch_h': self.to_var(arrH, self.use_gpu),
            'batch_r': self.to_var(arrR, self.use_gpu),
            'batch_t': self.to_var(arrT, self.use_gpu),
            'mode': 'normal'
        })