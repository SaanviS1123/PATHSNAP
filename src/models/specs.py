"""Frozen baseline model configurations and model-family tuning spaces."""
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED=42
BASELINES=['logistic_regression','random_forest','hist_gradient_boosting']


def make_estimator(name,params=None,seed=SEED):
    if name=='logistic_regression':
        config=dict(C=1.,max_iter=1000,class_weight='balanced',solver='lbfgs',random_state=seed)
        config.update(params or {})
        return make_pipeline(StandardScaler(),LogisticRegression(**config))
    if name=='random_forest':
        config=dict(n_estimators=80,max_depth=14,min_samples_leaf=10,max_features='sqrt',
                    max_samples=.5,class_weight='balanced_subsample',n_jobs=2,random_state=seed)
        config.update(params or {})
        return RandomForestClassifier(**config)
    if name=='hist_gradient_boosting':
        config=dict(max_iter=100,learning_rate=.1,max_leaf_nodes=31,l2_regularization=1.,
                    early_stopping=False,class_weight='balanced',random_state=seed)
        config.update(params or {})
        return HistGradientBoostingClassifier(**config)
    if name=='knn':
        config=dict(n_neighbors=15,weights='distance',algorithm='kd_tree',leaf_size=40,n_jobs=2)
        config.update(params or {})
        return make_pipeline(StandardScaler(),KNeighborsClassifier(**config))
    if name=='mlp':
        config=dict(hidden_layer_sizes=(32,16),activation='relu',solver='adam',alpha=.001,
                    batch_size=512,learning_rate_init=.001,max_iter=40,early_stopping=False,
                    n_iter_no_change=8,random_state=seed)
        config.update(params or {})
        return make_pipeline(StandardScaler(),MLPClassifier(**config))
    raise ValueError(f'Unknown family: {name}')


def suggest_params(trial,name):
    if name=='logistic_regression':
        return dict(C=trial.suggest_float('C',.001,10.,log=True))
    if name=='random_forest':
        return dict(n_estimators=trial.suggest_categorical('n_estimators',[64,80,128]),
                    max_depth=trial.suggest_categorical('max_depth',[10,14,18]),
                    min_samples_leaf=trial.suggest_categorical('min_samples_leaf',[5,10,20]),
                    max_features=trial.suggest_categorical('max_features',['sqrt',.5,1.]))
    if name=='hist_gradient_boosting':
        return dict(max_iter=trial.suggest_categorical('max_iter',[60,100,160]),
                    learning_rate=trial.suggest_float('learning_rate',.03,.15,log=True),
                    max_leaf_nodes=trial.suggest_categorical('max_leaf_nodes',[15,31,63]),
                    min_samples_leaf=trial.suggest_categorical('min_samples_leaf',[20,50,100]),
                    l2_regularization=trial.suggest_float('l2_regularization',.01,10.,log=True))
    if name=='knn':
        return dict(n_neighbors=trial.suggest_int('n_neighbors',5,31,step=2),p=trial.suggest_categorical('p',[1,2]))
    if name=='mlp':
        layers=trial.suggest_categorical('layers',['32,16','64,32'])
        return dict(hidden_layer_sizes=tuple(map(int,layers.split(','))),alpha=trial.suggest_float('alpha',1e-5,.01,log=True),
                    learning_rate_init=trial.suggest_float('learning_rate_init',.0003,.003,log=True))
    raise ValueError(name)


def decode_trial_params(name,params):
    result=dict(params)
    if name=='mlp' and 'layers' in result:
        result['hidden_layer_sizes']=tuple(map(int,result.pop('layers').split(',')))
    return result
