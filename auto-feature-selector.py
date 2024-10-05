
def get_mic(i):
    import numpy as np
    import pandas as pd
    from minepy import MINE
    mine = MINE()
    mine.compute_score(X.values[:,i], y.values)
    return mine.mic()


# 4min  for 2.5k features with 3k instances, single process
# 30sec for 2.5k features with 3k instances, joblib 16 processes
def feature_selection_univariate_mic(X, y, nr_features_selected=-1): 
    # lib
    import numpy as np
    import pandas as pd
    try:
        from minepy import MINE
    except ImportError, e:
        msg = "lib minepy not installed, non-root users can install by: pip2 install minepy --user"
        print(msg)
        return msg
    
    # param config
    if isinstance(X, pd.DataFrame):
        X_is_df = True
        row_nr = X.shape[0]
        column_nr = X.shape[1]
        if isinstance(X.columns, pd.core.index.MultiIndex):
            df_columns_is_multi_index = True
        else:
            df_columns_is_multi_index = False
    else: # non-DF
        X_is_df = False
        row_nr = np.shape(X)[0]
        column_nr = np.shape(X)[1]
    if nr_features_selected == -1:
        nr_features_selected = np.ceil(np.sqrt(column_nr))
    
    # selecting
    from joblib import Parallel, delayed
    mic_scores = Parallel(n_jobs=parallel_job_nr)(delayed(get_mic)(i) for i in range(column_nr))
    indices = np.argsort(mic_scores)[::-1]
    column_idxs_selected = indices[:nr_features_selected]
    column_mask_selected = np.zeros(len(indices), dtype=int)
    column_mask_selected[column_idxs_selected] = 1

    if X_is_df and not df_columns_is_multi_index:
        column_names_selected = X.columns[column_idxs_selected]
    else:
        column_names_selected = None

    return({'column_idxs_selected':column_idxs_selected, 'column_mask_selected':column_mask_selected, 'column_names_selected':column_names_selected, 'selector':mic_scores})


def feature_selection_importance(X, y, nr_features_selected = -1, algorithm = 'xgb'):
    # lib
    import pandas as pd
    import numpy as np
    
    # param config
    if isinstance(X, pd.DataFrame):
        X_is_df = True
        row_nr = X.shape[0]
        column_nr = X.shape[1]
        if isinstance(X.columns, pd.core.index.MultiIndex):
            df_columns_is_multi_index = True
        else:
            df_columns_is_multi_index = False
    else: # non-DF
        X_is_df = False
        row_nr = np.shape(X)[0]
        column_nr = np.shape(X)[1]
    if nr_features_selected == -1:
        nr_features_selected = np.ceil(np.sqrt(column_nr))
    if algorithm == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier()
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    
    # selecting
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    column_idxs_selected = indices[:nr_features_selected]
    column_mask_selected = np.zeros(len(indices), dtype=int)
    column_mask_selected[column_idxs_selected] = 1
    
    if X_is_df and not df_columns_is_multi_index:
        column_names_selected = X.columns[column_idxs_selected]
    else:
        column_names_selected = None
    
    return({'column_idxs_selected':column_idxs_selected, 'column_mask_selected':column_mask_selected, 'column_names_selected':column_names_selected, 'selector':model})


def feature_selection_rfe(X, y, nr_features_selected=-1, algorithm='xgb', scoring='f1'):
    # lib
    from sklearn.feature_selection import RFECV
    import pandas as pd
    import numpy as np
    
    # param config
    if isinstance(X, pd.DataFrame):
        X_is_df = True
        row_nr = X.shape[0]
        column_nr = X.shape[1]
        if isinstance(X.columns, pd.core.index.MultiIndex):
            df_columns_is_multi_index = True
        else:
            df_columns_is_multi_index = False
    else: # non-DF
        X_is_df = False
        row_nr = np.shape(X)[0]
        column_nr = np.shape(X)[1]
    if nr_features_selected == -1:
        nr_features_selected = np.ceil(np.sqrt(column_nr))
    if algorithm == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier()
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    
    # selecting
    selector = RFECV(estimator=model, step=min(column_nr/10, nr_features_selected), cv=5, scoring=scoring)
    selector = rfecv.fit(X, y)
    indices = np.argsort(selector.ranking_)
    column_idxs_selected = indices[:nr_features_selected]
    
    column_mask_selected = np.zeros(column_nr, dtype=int)
    column_mask_selected[column_idxs_selected] = 1
    
    if X_is_df and not df_columns_is_multi_index:
        column_names_selected = X.columns[column_idxs_selected]
    else:
        column_names_selected = None
    
    return({'column_idxs_selected':column_idxs_selected, 'column_mask_selected':column_mask_selected, 'column_names_selected':column_names_selected, 'selector':selector})


def feature_selection_classification_univariate(X, y, nr_features_selected = -1, algorithm='chi2'):
    # lib
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest # URLKDvQT6

    # param config
    if isinstance(X, pd.DataFrame):
        X_is_df = True
        row_nr = X.shape[0]
        column_nr = X.shape[1]
        if isinstance(X.columns, pd.core.index.MultiIndex):
            df_columns_is_multi_index = True
        else:
            df_columns_is_multi_index = False
    else: # non-DF
        X_is_df = False
        row_nr = np.shape(X)[0]
        column_nr = np.shape(X)[1]
    if nr_features_selected == -1:
        nr_features_selected = np.ceil(np.sqrt(column_nr))
    if algorithm == 'chi2':
        if sum((X < 0).sum(axis=0)) > 0:
            X = (X - X.min()) / (X.max() - X.min())
        if (y < 0).sum() > 0:
            y = (y - y.min()) / (y.max() - y.min())
        from sklearn.feature_selection import chi2
        selector = SelectKBest(score_func=chi2, k=nr_features_selected)
    elif algorithm == 'f_classif':
        from sklearn.feature_selection import f_classif
        selector = SelectKBest(score_func=f_classif, k=nr_features_selected)
    else:
        from sklearn.feature_selection import mutual_info_classif
        selector = SelectKBest(score_func=mutual_info_classif, k=nr_features_selected)
    selector_fitted = selector.fit(X, y) # URL8EM5cY
    
    column_idxs_selected = selector.get_support(indices=True)
    column_mask_selected = selector.get_support()
    if X_is_df and not df_columns_is_multi_index:
        column_names_selected = X.columns[idxs_selected]
    else:
        column_names_selected = None
    
    fitted_scores = selector_fitted.scores_
    fitted_pvalues = selector_fitted.pvalues_
    
    return({'column_idxs_selected':column_idxs_selected, 'column_mask_selected':column_mask_selected, 'column_names_selected':column_names_selected, 'selector':selector_fitted})


def feature_selection_regression_univariate(X, y, nr_features_selected = -1, algorithm='f_regression'):
    # lib
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest # URLKDvQT6

    # param config
    if isinstance(X, pd.DataFrame):
        X_is_df = True
        row_nr = X.shape[0]
        column_nr = X.shape[1]
        if isinstance(X.columns, pd.core.index.MultiIndex):
            df_columns_is_multi_index = True
        else:
            df_columns_is_multi_index = False
    else: # non-DF
        X_is_df = False
        row_nr = np.shape(X)[0]
        column_nr = np.shape(X)[1]
    if nr_features_selected == -1:
        nr_features_selected = np.ceil(np.sqrt(column_nr))
    if algorithm == 'f_regression':
        from sklearn.feature_selection import f_regression
        selector = SelectKBest(score_func=f_regression, k=nr_features_selected)
    else:
        from sklearn.feature_selection import mutual_info_regression
        selector = SelectKBest(score_func=mutual_info_regression, k=nr_features_selected)
    selector_fitted = selector.fit(X, y) # URL8EM5cY
    
    column_idxs_selected = selector.get_support(indices=True)
    column_mask_selected = selector.get_support()
    if X_is_df and not df_columns_is_multi_index:
        column_names_selected = X.columns[idxs_selected]
    else:
        column_names_selected = None
    
    fitted_scores = selector_fitted.scores_
    fitted_pvalues = selector_fitted.pvalues_
    
    return({'column_idxs_selected':column_idxs_selected, 'column_mask_selected':column_mask_selected, 'column_names_selected':column_names_selected, 'selector':selector_fitted})
