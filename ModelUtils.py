from Amie import Amie
from Analogy import Analogy
from ComplEx import ComplEx
from DistMult import DistMult
from HolE import HolE
from RESCAL import RESCAL
from RotatE import RotatE
from SimplE import SimplE
from TransD import TransD
from TransE import TransE
from TransH import TransH

from NegativeSampling import NegativeSampling
from MarginLoss import MarginLoss
from SigmoidLoss import SigmoidLoss
from SoftplusLoss import SoftplusLoss

import os
import ast

class ModelUtils:
    def __init__(self, model_name, params):
        self.model_name = model_name
        self.params = params

    # https://stackoverflow.com/questions/2859674/converting-python-list-of-strings-to-their-type
    @staticmethod
    def tryeval(val):
        try:
            val = ast.literal_eval(val)
        except ValueError:
            pass
        return val

    @staticmethod
    def get_params(model_file):
        filename, ext = os.path.splitext(model_file)
        # No extension!
        s = filename.split("_")

        paramaters = {}
        for i in range(1, len(s), 2):
            p = s[i]
            value = s[i + 1]

            if p=='trial':
                break
            paramaters[p] = ModelUtils.tryeval(value)
        return paramaters

    def get_name(self):
        s = self.model_name
        for p in self.params:
            s = s + "_" + p + "_" + str(self.params[p])
        return s

    def get_model(self, ent_total, rel_total, batch_size):
        if self.model_name == "transe":
            m = TransE(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"],
                p_norm=self.params["pnorm"],
                norm_flag=self.params["norm"])
        elif self.model_name == "transh":
            m = TransH(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"],
                p_norm=self.params["pnorm"],
                norm_flag=self.params["norm"])
        elif self.model_name == "transd":
            m = TransD(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim_e=self.params["dime"],
                dim_r=self.params["dimr"],
                p_norm=self.params["pnorm"],
                norm_flag=self.params["norm"])
        elif self.model_name == "rescal":
            m = RESCAL(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"])
        elif self.model_name == "distmult":
            m = DistMult(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"])
        elif self.model_name == "complex":
            m = ComplEx(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"])
        elif self.model_name == "hole":
            m = HolE(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"])
        elif self.model_name == "simple":
            m = SimplE(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"])
        elif self.model_name == "analogy":
            m = Analogy(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"])
        elif self.model_name == "rotate":
            m = RotatE(
                ent_tot=ent_total,
                rel_tot=rel_total,
                dim=self.params["dim"])
        elif self.model_name == "amie":
            m = Amie()

        if self.model_name == "transe" or self.model_name == "transh" or self.model_name == "transd" or \
                self.model_name == "rescal":
            loss=MarginLoss(margin=self.params["gamma"])
        elif self.model_name == "rotate":
            loss=SigmoidLoss()
        else:
            loss=SoftplusLoss()
        return NegativeSampling(
                    model=m,
                    loss=loss,
                    batch_size=batch_size)