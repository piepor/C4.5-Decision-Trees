""" Functions related to te filtering of a dataset """

import pandas as pd
import numpy as np

def filter_dataset_cat(data: pd.DataFrame, attr_name: str, attr_value: str) -> pd.DataFrame:
    """ create the dataset corresponding to the split of a categorical value """
    data_knw = data[data[attr_name] != '?'].copy(deep=True)
    split_data_knw = data_knw[data_knw[attr_name] == attr_value].copy(deep=True)
    data_unknw = data[data[attr_name] == '?'].copy(deep=True)
    return create_weight_ds(data_knw, data_unknw, split_data_knw)

def filter_dataset_low(data: pd.DataFrame, attr_name: str, threshold: float) -> pd.DataFrame:
    """ create the dataset corresponding to the low split of a continuous value """
    data_knw = data[data[attr_name] != '?'].copy(deep=True)
    data_unknw = data[data[attr_name] == '?'].copy(deep=True)
    # lower than the threshold
    split_data_knw = data_knw[data_knw[attr_name] <= threshold].copy(deep=True)
    return create_weight_ds(data_knw, data_unknw, split_data_knw)

def filter_dataset_high(data: pd.DataFrame, attr_name: str, threshold: float) -> pd.DataFrame:
    """ create the dataset corresponding to the high split of a continuous value """
    data_knw = data[data[attr_name] != '?'].copy(deep=True)
    data_unknw = data[data[attr_name] == '?'].copy(deep=True)
    # higher than the threshold
    split_data_knw = data_knw[data_knw[attr_name] > threshold].copy(deep=True)
    return create_weight_ds(data_knw, data_unknw, split_data_knw)

def create_weight_ds(
        data_knw: pd.DataFrame,
        data_unknw: pd.DataFrame,
        split_data_knw: pd.DataFrame) -> pd.DataFrame:
    """ create the weight for the unkonwn part of data """
    weight_unknw = len(split_data_knw) / len(data_knw)
    weight_unknw = np.array([weight_unknw] * len(data_unknw))
    weight = (weight_unknw * np.array(data_unknw['weight'].copy(deep=True))).tolist()
    data_unknw = data_unknw
    data_unknw.loc[:, ['weight']] = weight
    return pd.concat([split_data_knw, data_unknw], ignore_index=True)
