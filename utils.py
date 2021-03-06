import random
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error
from CIMtools.model_selection.transformation_out import TransformationOut
from sklearn.preprocessing import MinMaxScaler

random.seed(42)
np.random.seed(42)


def tau_cross_val(estimator, X1, X2, X, YT, YA, n_splits=10, n_repeats=5, random_state=5):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    YT_pred, YT_true = [], []
    YA_pred, YA_true = [], []

    YT_rmse, YA_rmse = [], []
    YT_r2, YA_r2 = [], []

    n_fold = 0
    for (T_train, T_test), (A_train, A_test) in zip(rkf.split(X1), rkf.split(X)):

        estimator.fit(X1[T_train], X2[T_train], X[A_train], YT[T_train], YA[A_train])

        YT_pred.extend(estimator.predict_constant(X1[T_test], X2[T_test]))
        YT_true.extend(YT[T_test])

        YA_pred.extend(estimator.predict_acidity(X[A_test]))
        YA_true.extend(YA[A_test])

        n_fold += 1
        if not n_fold % n_splits:
            YT_r2.append(r2_score(YT_true, YT_pred))
            YT_rmse.append(mean_squared_error(YT_true, YT_pred) ** 0.5)

            YA_r2.append(r2_score(YA_true, YA_pred))
            YA_rmse.append(mean_squared_error(YA_true, YA_pred) ** 0.5)

            YT_pred, YT_true = [], []
            YA_pred, YA_true = [], []

    stats = {'YT_RMSE': np.mean(YT_rmse), 'YA_RMSE': np.mean(YA_rmse), 'YT_R2': np.mean(YT_r2), 'YA_R2': np.mean(YA_r2)}

    return stats


def arrhenius_cross_val(estimator, reacts, X, YK, YA, YE, T, n_splits=10, n_repeats=1, task=None, random_state=42):

    #rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rkf = TransformationOut(n_splits=n_splits, n_repeats=n_repeats, shuffle=True, random_state=9)

    solvs = [i.meta['additive.1'] for i in reacts]
    solvs_num = {v: n for n, v in enumerate(set(solvs))}
    groups = [solvs_num[i] for i in solvs]

    YK_pred, YK_true = [], []
    YA_pred, YA_true = [], []
    YE_pred, YE_true = [], []

    YK_rmse, YA_rmse, YE_rmse = [], [], []
    YK_r2, YA_r2, YE_r2 = [], [], []

    n_fold = 0
    train_idx = []
    test_idx = []
    T_true = []
    for train, test in rkf.split(reacts, groups=groups):

        train_idx.extend(train)
        test_idx.extend(test)

        trainK = train[np.isfinite(YK[train])]
        trainA = train[np.isfinite(YA[train])]
        trainE = train[np.isfinite(YE[train])]

        testK = test[np.isfinite(YK[test])]
        testA = test[np.isfinite(YA[test])]
        testE = test[np.isfinite(YE[test])]

        T_true.extend(T[test])

        if estimator.__class__.__name__ == 'ArrheniusConjugatedNet':

            estimator.fit(X[train], X[train], X[train], YK[train], YA[train], YE[train], T[train],
                          reacts=list(np.array(reacts)[train]))

            YK_pred.extend(estimator.predict_YK(X[testK], T[testK]))
            YA_pred.extend(estimator.predict_YA(X[testA]))
            YE_pred.extend(estimator.predict_YE(X[testE]))

        if estimator.__class__.__name__ == 'IndividualNet':

            if task == 'YK':
                estimator.fit(X[trainK], YK[trainK], reacts=list(np.array(reacts)[trainK]))
            elif task == 'YA':
                estimator.fit(X[trainA], YA[trainA], reacts=list(np.array(reacts)[trainA]))
            elif task == 'YE':
                estimator.fit(X[trainE], YE[trainE], reacts=list(np.array(reacts)[trainE]))

            YK_pred.extend(estimator.predict(X[testK]))
            YA_pred.extend(estimator.predict(X[testA]))
            YE_pred.extend(estimator.predict(X[testE]))

        elif estimator.__class__.__name__ == 'IndividualRidge':
            if task == 'YK':
                estimator.fit(X[trainK], YK[trainK])
            elif task == 'YA':
                estimator.fit(X[trainA], YA[trainA])
            elif task == 'YE':
                estimator.fit(X[trainE], YE[trainE])

            YK_pred.extend(estimator.predict(X[testK]))
            YA_pred.extend(estimator.predict(X[testA]))
            YE_pred.extend(estimator.predict(X[testE]))

        elif estimator.__class__.__name__ == 'ArrheniusIndividualRidge':
            estimator.fit(X[trainA], X[trainE], YA[trainA], YE[trainE])

            YK_pred.extend(estimator.predict_YK(X[testK], T[testK]))
            YA_pred.extend(estimator.predict_YA(X[testA]))
            YE_pred.extend(estimator.predict_YE(X[testE]))


        elif estimator.__class__.__name__ == 'ArrheniusConjugatedRidge':
            estimator.fit(X[trainK], X[trainA], X[trainE],
                          YK[trainK], YA[trainA], YE[trainE], T[trainK])

            YK_pred.extend(estimator.predict_YK(X[testK], T[testK]))
            YA_pred.extend(estimator.predict_YA(X[testA]))
            YE_pred.extend(estimator.predict_YE(X[testE]))

        YK_true.extend(YK[testK])
        YA_true.extend(YA[testA])
        YE_true.extend(YE[testE])

        n_fold += 1
        if not n_fold % n_splits:
            YK_r2.append(r2_score(YK_true, YK_pred))
            YA_r2.append(r2_score(YA_true, YA_pred))
            YE_r2.append(r2_score(YE_true, YE_pred))

            YK_rmse.append(mean_squared_error(YK_true, YK_pred) ** 0.5)
            YA_rmse.append(mean_squared_error(YA_true, YA_pred) ** 0.5)
            YE_rmse.append(mean_squared_error(YE_true, YE_pred) ** 0.5)

            predictions = {'test_idx': test_idx,
                           'YK_pred': np.array(YK_pred).flatten(),
                           'YA_pred': np.array(YA_pred).flatten(),
                           'YE_pred': np.array(YE_pred).flatten(),
                           'YK_true': np.array(YK_true).flatten(),
                           'YA_true': np.array(YA_true).flatten(),
                           'YE_true': np.array(YE_true).flatten(),
                           'T_true':np.array(T_true).flatten()}

            YK_pred, YA_pred, YE_pred = [], [], []
            YK_true, YA_true, YE_true = [], [], []


    stats = {'YK_RMSE': np.mean(YK_rmse), 'YA_RMSE': np.mean(YA_rmse), 'YE_RMSE': np.mean(YE_rmse),
             'YK_R2': np.mean(YK_r2), 'YA_R2': np.mean(YA_r2), 'YE_R2': np.mean(YE_r2)}

    return predictions, stats