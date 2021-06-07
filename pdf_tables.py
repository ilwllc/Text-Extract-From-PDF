import itertools
import numpy as np
import pandas as pd
import re
from typing import Union

from nltk import ngrams
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

import fitz


#   Related to
#   Step 1: Get text and positional INFORMATION
def get_page_elements(doc: fitz.fitz.Document, page_num: int) -> Union[pd.DataFrame, None]:
    page = doc.loadPage(page_num)
    page_dict = page.getText('dict')

    blocks = [page_dict['blocks'][i] for i in np.arange(len(page_dict['blocks']))]
    lines = [blocks[line]['lines'] for line in range(len(blocks)) if blocks[line].get('lines')]
    spans = [s for s in itertools.chain(*[s for s in [lines[l] for l in np.arange(len(lines))]])]
    _data = pd.DataFrame(data=itertools.chain.from_iterable([[[s_i['bbox'][0], s_i['bbox'][1]
                                                                  , s_i['bbox'][2], s_i['bbox'][3]
                                                                  , s_i['text']] for s_i in s['spans']]
                                                             for s in spans])
                         , columns=['x0', 'y0', 'x1', 'y1', 'line']
                         ).sort_values(['y0', 'x0'])
    _data['line'] = _data['line'].str.strip()
    _data['x_avg'] = _data[['x0', 'x1']].apply(np.mean, axis=1)
    _data['y_avg'] = _data[['y0', 'y1']].apply(np.mean, axis=1)

    return _data


#   Used in
#   Step 2: REMOVE unrelated text
#   Step 3: Build table by GROUPING on X and Y axis
def get_table_via_clustering(data, cluster_data_columns: list
                             , *
                             , remove_outliers: bool = False
                             , outlierprops: dict = None
                             , copy: bool = True
                             , cluster_label: tuple = ('cluster_id_x', 'cluster_id_y')
                             , clusterprops: tuple = ({'eps': 0.5, 'min_samples': 0, 'metric': 'manhattan'}
                                                      , {'eps': 0.5, 'min_samples': 0, 'metric': 'manhattan'})
                             ) -> (pd.DataFrame, Union[pd.DataFrame, None]):
    """
    Use clustering to determine how table elements should be grouped together

    Keyword arguments:
    data: A pandas DataFrame that contains the table
    cluster_data_columns : List of numeric columns to use as features for clustering
    copy: Boolean flag to make a copy of the original or update the original (default=True)
    cluster_label: column name for cluster labels (defaul='cluster_id')
    clusterprops: dictionary of parameters to pass to the clustering algorithm
            Keys are
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
            metric: The metric to use when calculating distance between instances in a feature array. Examples include
            cityblock, cosine, euclidean, l1, l2, and manhattan. See sklearn.metrics.pairwise_distances for other options

    Returns:
    Dictionary of clustering results
    Original data with cluster label column appended
    """

    if copy:
        _data = data.copy()
    else:
        _data = data

    def get_1d_clusters(data, cluster_data_columns: list, cluster_label: str,
                        clusterprops: dict = None):

        _data = data

        if clusterprops:
            _eps = clusterprops.get('eps', 0.5)
            _min_samples = clusterprops.get('min_samples', 5)
            _metric = clusterprops.get('metric', 'euclidean')
            _n_clusters = clusterprops.get('n_clusters', 2)
        else:
            _eps = 0.5
            _min_samples = 5
            _metric = 'euclidean'
            _n_clusters = 2

        cluster_model = DBSCAN(eps=_eps, min_samples=_min_samples, metric=_metric)
        _ = cluster_model.fit(_data[cluster_data_columns].values)
        _data[cluster_label] = cluster_model.labels_

        n_clusters = len(set(cluster_model.labels_)) - (1 if -1 in cluster_model.labels_ else 0)

        if (cluster_model.labels_.shape[0] >= 2) & \
                (cluster_model.labels_.shape[0] <= _data[cluster_data_columns].values.shape[0] - 1):
            _silhouette_score = silhouette_score(_data[cluster_data_columns].values, cluster_model.labels_)
        else:
            _silhouette_score = np.nan

        return {'n_clusters': n_clusters, 'silhouette_score': _silhouette_score}, _data

    results, _ = get_1d_clusters(_data, cluster_data_columns[0]
                                 , cluster_label=cluster_label[0]
                                 , clusterprops=clusterprops[0]
                                 )

    results, _ = get_1d_clusters(_data, cluster_data_columns[1]
                                 , cluster_label=cluster_label[1]
                                 , clusterprops=clusterprops[1]
                                 )

    clustered_data = _data.copy()

    if outlierprops:
        rng = outlierprops.get('outlier_range', 1.5)
    else:
        rng = 1.5

    if remove_outliers:
        mid_x, mid_y = clustered_data[['x0', 'y0']].median(axis=0)
        sd_x, sd_y = clustered_data[['x0', 'y0']].std(axis=0)

        clustered_data['sep_x'] = abs((clustered_data['x0'] - mid_x) / sd_x)
        clustered_data['sep_y'] = abs((clustered_data['y0'] - mid_y) / sd_y)

        rng_low = ((clustered_data['x0'] < mid_x - sd_x * rng) | (clustered_data['y0'] < mid_y - sd_y * rng))
        rng_high = ((clustered_data['x0'] > mid_x + sd_x * rng) | (clustered_data['y0'] > mid_y + sd_y * rng))

        outlier_idx = clustered_data.loc[rng_low | rng_high,].index

        clustered_data.drop(index=outlier_idx, inplace=True)

    x = clustered_data[['cluster_id_x', 'x0']].sort_values(['x0']).groupby(['cluster_id_x']).first().sort_values(
        ['x0']).reset_index()
    x['x_seq'] = np.arange(x.shape[0])
    clustered_data = clustered_data.join(x.set_index(['cluster_id_x', 'x0']), how='left', on=['cluster_id_x', 'x0'])
    clustered_data[['x0', 'x_seq']] = clustered_data[['x0', 'x_seq']].sort_values(['x0', 'x_seq']).fillna(
        method='ffill')

    y = clustered_data[['cluster_id_y', 'y0']].sort_values(['y0']).groupby(['cluster_id_y']).first().sort_values(
        ['y0']).reset_index()
    y['y_seq'] = np.arange(y.shape[0])
    clustered_data = clustered_data.join(y.set_index(['cluster_id_y', 'y0']), how='left', on=['cluster_id_y', 'y0'])
    clustered_data[['y0', 'y_seq']] = clustered_data[['y0', 'y_seq']].sort_values(['y0', 'y_seq']).fillna(
        method='ffill')

    clustered_data = clustered_data.astype({'x_seq': 'int32', 'y_seq': 'int32'})

    table = clustered_data.loc[:, ['x_seq', 'y_seq', 'cluster_id_x', 'x0', 'y0', 'line']].sort_values(['y_seq', 'x0']) \
        .pivot(index='y_seq', columns='x_seq', values='line').reset_index(drop=True)

    return table, clustered_data


#   Used in
#   Step 4: Find page HEADERS
def get_page_headers(data, header_row_detector: str
                     , *
                     , copy: bool = False
                     ):
    if copy:
        _data = data.copy()
    else:
        _data = data

    _data.loc[-1, :] = _data.columns.values
    _data = _data.sort_index()
    idx_all = {}

    all_grams = []
    for i in range(1, len(header_row_detector.split()) + 1):
        ngram_values = ngrams(header_row_detector.split(), i)
        all_grams = all_grams + [' '.join(val) for val in ngram_values]

    for col in _data.columns:
        # check only against string values .isin and join will both fail otherwise
        str_idx = _data[_data[col].apply(lambda val: isinstance(val, str))].index.values
        d = _data.loc[str_idx, col].isin(all_grams)
        if d.any():
            idx = d[d == True].index.values.tolist()

            if ' '.join(_data.loc[idx, col]) == header_row_detector:
                idx_all[col] = idx

    s = set(i for i in itertools.chain(*idx_all.values()))
    if len(s) > 0:
        col = idx_all.keys()

        col_header_idx = np.arange(min(s), max(s) + 1).tolist()

        new_col_name = _data.loc[col_header_idx, col].fillna('').apply(lambda r: ' '.join([str(r[i]) for i in
                                                                                           r.index if len(
                str(r[i])) > 0]).strip()).values[0]
        _data.rename(columns={list(col)[0]: new_col_name}, inplace=True)

        new_col_names = _data.loc[col_header_idx, :] \
            .fillna('') \
            .replace(['^Unnamed.*'], [''], regex=True) \
            .apply(lambda r: ' '.join([str(r[i]) for i in r.index if len(str(r[i])) > 0]))

        new_column_names = [x2 if len(x2) > 0 else x1 for x1, x2 in zip(_data.columns, new_col_names)]

        _data.rename(columns={old_colname: new_colname for old_colname, new_colname in
                              zip(_data.columns, new_column_names)} \
                     , inplace=True
                     )
        max_col_header_idx = max(col_header_idx)
        min_idx = min(_data.index)

        drop_row_idx = _data.loc[min_idx:max_col_header_idx].index
        _data.drop(index=drop_row_idx, inplace=True, errors='ignore')
    else:
        raise ValueError('No combination of {} found in columns'.format(header_row_detector))

    return _data


#   Used in
#   Step 5: CONSOLIDATE rows
def group_rows(data, row_grouper_columns: list
               , *
               , fillna: bool = True
               , copy: bool = False
               ):
    def collapse(vals):
        ret = re.sub(' +', ' ', ' '.join(['' if val == np.nan else str(val) for val in vals.values])).strip()
        return ret

    if copy:
        _data = data.copy()
    else:
        _data = data
    original_column_seq = _data.columns.values.tolist()

    # Add a sequence column to keep the order in case there are duplicates in the row_grouper_columns
    idx = _data.loc[:, row_grouper_columns].dropna().index
    _data.loc[idx, 'SEQ'] = idx

    # fill the grouper and SEQ columns down, so we have something to group on
    for col in ['SEQ'] + row_grouper_columns:
        _data[col].fillna(method='ffill', inplace=True)

    if fillna:
        _data.fillna('', inplace=True)

    cols_to_collapse = list(set(_data.columns.values).difference(['SEQ'] + row_grouper_columns))
    _data = _data.groupby(['SEQ'] + row_grouper_columns, sort=False) \
        .agg({col: lambda r: collapse(r) for col in cols_to_collapse}, axis=1)
    _data = _data.reset_index(
        level=list(np.arange(1, _data.index.nlevels)))  # reset index leaving just the SEQ column as the index

    _data = _data[original_column_seq].reset_index(drop=True)

    # the row_grouper_columns are key columns. If they are blank, drop the rows
    d_idx = np.where(_data[row_grouper_columns].applymap(lambda x: x == ''))[0].tolist()
    _data.drop(index=d_idx, inplace=True)

    return _data

