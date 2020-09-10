from colearn.metrics import rmse
from sklearn.model_selection import RepeatedKFold
from CIMtools.model_selection.transformation_out import TransformationOut
from sklearn.metrics import r2_score
from statistics import mean
import numpy as np



def cross_val(estimator, X, y, n_splits=10, n_repeats=5, random_state=888):
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    y_pred = []
    y_true = []
    
    y_rmse = []
    y_r2 = []
    
    n_fold = 0
    
    for train, test in rkf.split(X):
    
        estimator.fit(X[train], y[train])
        
        y_pred.extend(estimator.predict(X[test]))
        y_true.extend(y[test])
       
        n_fold += 1
        
        if not n_fold % n_splits:
            
            y_rmse.append(rmse(y_true, y_pred))
            y_r2.append(r2_score(y_true, y_pred))
            
            y_pred.clear()
            y_true.clear()
            
            
    stats = {   'rmseMax': max(y_rmse),
                'rmseMin': min(y_rmse),
                'rmseAvg': mean(y_rmse),
                'r2Max': max(y_r2),
                'r2Min': min(y_r2),
                'r2Avg': mean(y_r2)   
            }

    return stats
    
    
def cross_val_trans_out(estimator, reacts, groups, X, y, n_splits=10, n_repeats=1, random_state=888):
    
    rkf = TransformationOut(n_splits=n_splits, n_repeats=n_repeats, shuffle=True, random_state=random_state)

    y_pred = []
    y_true = []
    
    y_rmse = []
    y_r2 = []
    
    n_fold = 0
    
    for train, test in rkf.split(reacts, groups=groups):
    
        estimator.fit(X[train], y[train])
        
        y_pred.extend(estimator.predict(X[test]))
        y_true.extend(y[test])
       
        n_fold += 1
        
        if not n_fold % n_splits:
            
            y_rmse.append(rmse(y_true, y_pred))
            y_r2.append(r2_score(y_true, y_pred))
            
            y_pred.clear()
            y_true.clear()
            
            
    stats = {   'rmseMax': max(y_rmse),
                'rmseMin': min(y_rmse),
                'rmseAvg': mean(y_rmse),
                'r2Max': max(y_r2),
                'r2Min': min(y_r2),
                'r2Avg': mean(y_r2)   
            }

    return stats    

    
def tau_cross_val(estimator, X1, X2, X, yT, yK, n_splits=10, n_repeats=5, random_state=5):
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    yT_pred, yT_true = [], []
    yK_pred, yK_true = [], []
    
    yT_rmse, yK_rmse = [], []
    yT_r2, yK_r2 = [], []

    n_fold = 0
    
    for (train_const, test_const), (train_acid, test_acid) in \
        zip(rkf.split(X1), rkf.split(X)):
        
        estimator.fit(X1[train_const], X2[train_const],
                      X[train_acid], yT[train_const], yK[train_acid])
        
        yT_pred.extend(estimator.predict_const(X1[test_const], X2[test_const]))
        yT_true.extend(yT[test_const])
        
        yK_pred.extend(estimator.predict(X[test_acid]))
        yK_true.extend(yK[test_acid])
        
        n_fold += 1
        
        if not n_fold % n_splits:
            
            yT_r2.append(r2_score(yT_true, yT_pred))
            yT_rmse.append(rmse(yT_true, yT_pred))
           
            yK_r2.append(r2_score(yK_true, yK_pred))
            
            yK_rmse.append(rmse(yK_true, yK_pred))
            
            
            yT_pred, yT_true = [], []
            yK_pred, yK_true = [], []
            
    stats = {'lmb': estimator.lmb,
             'alpha': estimator.alpha,
             'yT_RMSE': mean(yT_rmse),
             'yK_RMSE': mean(yK_rmse),
             'yT_R2': mean(yT_r2),
             'yK_R2': mean(yK_r2)}
        
    return stats


    
def tau_cross_val_trans_out(estimator, reacts, groups,
                            X1, X2, X, yT, yK, n_splits=5, n_repeats=5, random_state=5):
    
    rkf_tau = TransformationOut(n_splits=n_splits, n_repeats=n_repeats, shuffle=True, random_state=random_state)
    rkf_acidity = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    yT_pred, yT_true = [], []
    yK_pred, yK_true = [], []
    
    yT_rmse, yK_rmse = [], []
    yT_r2, yK_r2 = [], []

    n_fold = 0
    
    for (train_const, test_const), (train_acid, test_acid) in \
        zip(rkf_tau.split(reacts, groups=groups), rkf_acidity.split(X)):
        
        estimator.fit(X1[train_const], X2[train_const],
                      X[train_acid], yT[train_const], yK[train_acid])
        
        yT_pred.extend(estimator.predict_const(X1[test_const], X2[test_const]))
        yT_true.extend(yT[test_const])
        
        yK_pred.extend(estimator.predict(X[test_acid]))
        yK_true.extend(yK[test_acid])
        
        n_fold += 1
        
        if not n_fold % n_splits:
            
            yT_r2.append(r2_score(yT_true, yT_pred))
            yT_rmse.append(rmse(yT_true, yT_pred))
           
            yK_r2.append(r2_score(yK_true, yK_pred))
            
            yK_rmse.append(rmse(yK_true, yK_pred))
            
            
            yT_pred, yT_true = [], []
            yK_pred, yK_true = [], []
            
    stats = {'lmb': estimator.lmb,
             'alpha': estimator.alpha,
             'yT_RMSE': mean(yT_rmse),
             'yK_RMSE': mean(yK_rmse),
             'yT_R2': mean(yT_r2),
             'yK_R2': mean(yK_r2)}
        
    return stats
    
    
def arrhenius_cross_val(estimator, X, yA, yE, yK, T, n_splits=10, n_repeats=1, model_type='coop', random_state=888):
                        
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    yA_pred = []
    yE_pred = []
    yK_pred = []
    
    yA_true = []
    yE_true = []
    yK_true = []
    
    yA_rmse = []
    yE_rmse = []
    yK_rmse = []
    
    yA_r2   = []
    yE_r2   = []
    yK_r2   = []
    
    n_fold = 0
    
    for train, test in rkf.split(X):

        trainA = train[np.isfinite(yA[train])]
        trainE = train[np.isfinite(yE[train])]
    
        testA = test[np.isfinite(yA[test])]
        testE = test[np.isfinite(yE[test])]
        
        if model_type == 'coop':
        
            estimator.fit(X[trainA], X[trainE], X[train], yA[trainA], yE[trainE], yK[train], T[train])
            
        else:
            
            estimator.fit(X[train], yK[train], T[train])
        
        yA_pred.extend(estimator.predict_lgA(X[testA]))
        yE_pred.extend(estimator.predict_E(X[testE]))
        yK_pred.extend(estimator.predict_lgK(X[test], T[test]))
        
        yA_true.extend(yA[testA])
        yE_true.extend(yE[testE])
        yK_true.extend(yK[test])        
        
        n_fold += 1
        
        if not n_fold % n_splits:
            
            yA_r2.append(r2_score(yA_true, yA_pred))
            yE_r2.append(r2_score(yE_true, yE_pred))
            yK_r2.append(r2_score(yK_true, yK_pred))
            
            yA_rmse.append(rmse(yA_true, yA_pred))
            yE_rmse.append(rmse(yE_true, yE_pred))
            yK_rmse.append(rmse(yK_true, yK_pred))
            
            yA_pred = []
            yE_pred = []
            yK_pred = []
    
            yA_true = []
            yE_true = []
            yK_true = []

    
    stats = {   
                'lmbA': estimator.lmbA,
                'lmbE': estimator.lmbE,
                'yA_RMSE': mean(yA_rmse),
                'yE_RMSE': mean(yE_rmse),
                'yK_RMSE': mean(yK_rmse),
                'yA_R2': mean(yA_r2),
                'yE_R2': mean(yE_r2),
                'yK_R2': mean(yK_r2)   
            }
    
    return stats
    


'''def arrhenius_cross_val_trans_out(estimator, reacts, groups, X, yA, yE, yK, T, n_splits=10, n_repeats=1, 
                                  model_type='coop', random_state=888):
                        
    rkf = TransformationOut(n_splits=n_splits, n_repeats=n_repeats, shuffle=True, random_state=random_state)
    
    yA_pred = []
    yE_pred = []
    yK_pred = []
    
    yA_true = []
    yE_true = []
    yK_true = []
    
    yA_rmse = []
    yE_rmse = []
    yK_rmse = []
    
    yA_r2  = []
    yE_r2  = []
    yK_r2  = []
    
    n_fold = 0
    
    for train, test in rkf.split(reacts, groups=groups):

        trainA = train[np.isfinite(yA[train])]
        trainE = train[np.isfinite(yE[train])]
    
        testA = test[np.isfinite(yA[test])]
        testE = test[np.isfinite(yE[test])]
        
        if model_type == 'coop':
        
            estimator.fit(X[trainA], X[trainE], X[train], yA[trainA], yE[trainE], yK[train], T[train])
            
        else:
            
            estimator.fit(X[train], yK[train], T[train])
        
        yA_pred.extend(estimator.predict_lgA(X[testA]))
        yE_pred.extend(estimator.predict_E(X[testE]))
        yK_pred.extend(estimator.predict_lgK(X[test], T[test]))
        
        yA_true.extend(yA[testA])
        yE_true.extend(yE[testE])
        yK_true.extend(yK[test])        
        
        n_fold += 1
        
        if not n_fold % n_splits:
            
            yA_r2.append(r2_score(yA_true, yA_pred))
            yE_r2.append(r2_score(yE_true, yE_pred))
            yK_r2.append(r2_score(yK_true, yK_pred))
            
            yA_rmse.append(rmse(yA_true, yA_pred))
            yE_rmse.append(rmse(yE_true, yE_pred))
            yK_rmse.append(rmse(yK_true, yK_pred))
            
            yA_pred = []
            yE_pred = []
            yK_pred = []
    
            yA_true = []
            yE_true = []
            yK_true = []

    
    stats = {   
                'lmbA': estimator.lmbA,
                'lmbE': estimator.lmbE,
                'yA_RMSE': mean(yA_rmse),
                'yE_RMSE': mean(yE_rmse),
                'yK_RMSE': mean(yK_rmse),
                'yA_R2': mean(yA_r2),
                'yE_R2': mean(yE_r2),
                'yK_R2': mean(yK_r2)   
            }
    
    return stats'''
    
def arrhenius_cross_val_trans_out(estimator, reacts, groups, X, yA, yE, yK, T, n_splits=5, n_repeats=1, random_state=5):

    rkf = TransformationOut(n_splits=n_splits, n_repeats=n_repeats, shuffle=True, random_state=random_state)
    
    yA_pred = []
    yE_pred = []
    yK_pred = []
    
    yA_true = []
    yE_true = []
    yK_true = []
    
    yA_rmse = []
    yE_rmse = []
    yK_rmse = []
    
    yA_r2  = []
    yE_r2  = []
    yK_r2  = []
    
    n_fold = 0
    
    for train, test in rkf.split(reacts, groups=groups):

        trainA = train[np.isfinite(yA[train])]
        trainE = train[np.isfinite(yE[train])]
    
        testA = test[np.isfinite(yA[test])]
        testE = test[np.isfinite(yE[test])]

        estimator.fit(X[trainE], X[train], yE[trainE], yK[train], T[train])

        yA_pred.extend(estimator.predict_lgA(X[testA]))
        yE_pred.extend(estimator.predict_E(X[testE]))
        yK_pred.extend(estimator.predict_lgK(X[test], T[test]))
        
        yA_true.extend(yA[testA])
        yE_true.extend(yE[testE])
        yK_true.extend(yK[test])        
        
        n_fold += 1
        
        if not n_fold % n_splits:
            
            yA_r2.append(r2_score(yA_true, yA_pred))
            yE_r2.append(r2_score(yE_true, yE_pred))
            yK_r2.append(r2_score(yK_true, yK_pred))
            
            yA_rmse.append(rmse(yA_true, yA_pred))
            yE_rmse.append(rmse(yE_true, yE_pred))
            yK_rmse.append(rmse(yK_true, yK_pred))
            
            yA_pred = []
            yE_pred = []
            yK_pred = []
    
            yA_true = []
            yE_true = []
            yK_true = []

    
    stats = {   
                'lmbA': estimator.lmbA,
                'lmbE': estimator.lmbE,
                'a':estimator.a,
                'yA_RMSE': mean(yA_rmse),
                'yE_RMSE': mean(yE_rmse),
                'yK_RMSE': mean(yK_rmse),
                'yA_R2': mean(yA_r2),
                'yE_R2': mean(yE_r2),
                'yK_R2': mean(yK_r2)   
            }
    
    return stats
    