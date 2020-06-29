"""
HE2RNA: Computation of correlations
Copyright (C) 2020  Owkin Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
import pickle as pkl
from joblib import Parallel, delayed

def corr(pred, label, i):
    return np.corrcoef(
        label[:, i],
        pred[:, i])[0, 1]

def compute_metrics(label, pred):
    res = Parallel(n_jobs=16)(
        delayed(corr)(pred, label, i) for i in range(label.shape[1])
    )
    return res
