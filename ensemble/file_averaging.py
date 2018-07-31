# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This is an obsolete script from https://github.com/antoine77340/Youtube-8M-WILLOW/blob/master/file_averaging.py
For large inference files or for many files, this script is likely to fail because it loads all the inferences at the same time.
'''

import os
from collections import defaultdict, Counter
import pickle
import pandas as pd

SUBMIT_PATH = '/media/TB2/chuck/__MODEL_VAULT__'
SIGFIGS = 6

def read_models(model_weights, blend=None):
    if not blend:
        blend = defaultdict(Counter)
    for m, w in model_weights.items():
        print(m, w)
        with open(os.path.join(SUBMIT_PATH, m + '.csv'), 'r') as f:
            f.readline()
            for l in f:
                id, r = l.split(',')
                r = r.split(' ')
                n = len(r) // 2
                for i in range(0, n * 2, 2):
                    k = int(r[i])
                    v = int(10**(SIGFIGS - 1) * float(r[i+1]))
                    blend[id][k] += w * v
    return blend


def write_models(blend, file_name, total_weight):
    with open(os.path.join(SUBMIT_PATH, file_name + '.csv'), 'w') as f:
        f.write('VideoID,LabelConfidencePairs\n')
        for id, v in blend.items():
            l = ' '.join(['{} {:{}f}'.format(t[0]
                                            , float(t[1]) / 10 ** (SIGFIGS - 1) / total_weight
                                            , SIGFIGS) for t in v.most_common(20)])
            f.write(','.join([id, l + '\n']))
    return None


model_pred = {
        'inference-on-train-and-val0-4+6-9-model03-02-cp473348-top50': 1,
        'inference-on-train-and-val0-4+6-9-model05-02-cp800020-top50': 1,
        'inference-on-train-and-val0-4+6-9-model06-02-cp173998-top50': 1,
        'inference-on-train-and-val0-4+6-9-model07-01-cp380325-top50': 1,
        'inference-on-train-and-val0-4+6-9-model07-02-cp243673-top50': 1,
        }

avg = read_models(model_pred)
write_models(avg, 'inference-ens5-0302-0502-0602-0701-0702', sum(model_pred.values()))
