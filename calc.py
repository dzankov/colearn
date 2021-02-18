import numpy as np
import pandas as pd
from CGRtools import RDFRead
from utils import ModelBuider

from CGRtools import SDFRead, SDFWrite
from CGRtools import RDFRead, RDFWrite
from CIMtools.preprocessing.conditions_container import DictToConditions, ConditionsToDataFrame
from CIMtools.preprocessing import Fragmentor, CGR, EquationTransformer, SolventVectorizer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from os import environ
fragmentor_path='/home/zankov/dev'
environ['PATH'] += ':{}'.format(fragmentor_path)

import sys
sys.path.append('/home/zankov/dev')


reacts = RDFRead('arrhenius_dataset.rdf', remap=False).read()
for reaction in reacts:
    reaction.standardize()
    reaction.kekule()
    reaction.implicify_hydrogens()
    reaction.thiele()
    reaction.clean2d()

#
def extract_meta(x):
    return [y[0].meta for y in x]

features = ColumnTransformer([('solv', SolventVectorizer(), ['solvent.1']),
                              ('amount', 'passthrough', ['solvent_amount.1']),])

conditions = Pipeline([('meta', FunctionTransformer(extract_meta)),
                       ('cond', DictToConditions(solvents=('additive.1',), amounts=('amount.1',))),
                       ('desc', ConditionsToDataFrame()),
                       ('final', features)])

graph = Pipeline([('CGR', CGR()), ('frg', Fragmentor(fragment_type=3, max_length=3, useformalcharge=True, version='2017.x'))])

frag = ColumnTransformer([('graph', graph, [0]),
                          ('cond', conditions, [0])])

#
frag.fit([[i] for i in reacts])

XK = frag.transform([[i] for i in reacts])

YK = np.array([float(i.meta['logK']) for i in reacts])
YA = np.array([float(i.meta.get('lgA', np.nan)) for i in reacts])
YE = np.array([float(i.meta.get('activationenergy', np.nan)) for i in reacts])

T = 1000 / (np.array([float(i.meta['temperature']) for i in reacts]) * 2.303 * 8.314)

import shutil
del frag
frg_files = [i for i in os.listdir() if i.startswith('frg')]
for file in frg_files:
    if os.path.isfile(file):
        os.remove(file)
    else:
        shutil.rmtree(file)

X = pd.read_csv('data/CA_fragments_1549.csv', sep=';').values
YK = pd.read_csv('data/lgK_train.csv', sep=';', header=None).values.flatten()
YA = pd.read_csv('data/lgA_train.csv', sep=';', header=None).values.flatten()
YE = pd.read_csv('data/E_train.csv', sep=';', header=None).values.flatten()
T = pd.read_csv('data/T_train.csv', sep=';', header=None).values.flatten()
reacts = RDFRead('TRAIN_1549.rdf', remap=False).read()

hparams = {'a': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
           'lmbA': [10 ** i for i in range(-3, 5)],
           'lmbE': [10 ** i for i in range(-3, 5)]}

model_builder = ModelBuider(reacts=reacts, local_dir='models_exp', hparams=hparams, init_cuda=True, random_state=42)
model_builder.train_test_split(X, YK, YA, YE, T)
#
model_builder.train_individual_model(task='YK')
model_builder.train_individual_model(task='YA')
model_builder.train_individual_model(task='YE')
model_builder.train_individual_arrhenius_model()
model_builder.train_arrhenius_based_model()
model_builder.train_arrhenius_conjugated_model()
