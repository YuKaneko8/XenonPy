from typing import Union
from xenonpy.descriptor.base import BaseFeaturizer, BaseDescriptor

from xenonpy.model.training import Trainer
from xenonpy.datatools import Scaler
import torch, numpy




class NNPropDescriptor(BaseFeaturizer):
    def __init__(self,FP,checker,trainer,colnames):
        super().__init__(n_jobs=0, on_errors="nan", return_type="any")
        self.FP = FP
        self.trainer = trainer
        self.checker = checker
        self.colnames = colnames

    def featurize(self, x):
        tmp = self.FP.transform(x)
        output = self.trainer.predict(x_in=torch.tensor(tmp.values, dtype=torch.float)).detach().numpy()
        return pd.DataFrame(self.checker.inverse_transform(output), index=tmp.index, columns=self.colnames)

    @property
    def feature_labels(self):
        return self.colnames

