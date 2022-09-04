"""
Basic pipline where request logs -> item events -> CTR model -> batch inference output
"""
from typing import List

import pandas as pd


def get_click_position(impressions: List[str], click: str) -> int:
    try:
        return impressions.index(click) + 1
    except ValueError:
        return -1


def get_click_item_and_pos(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    _df['position'] = _df.apply(lambda x: get_click_position(x['impressions'], x['event_item']), axis=1)
    _df = _df.rename(columns={'event_item': 'item'})
    return _df[['request_id', 'item', 'position', 'event_type']]


def get_updated_impressions(impressions, impressions_visible):
    if isinstance(impressions_visible, list) and len(impressions) >= len(impressions_visible):
        return impressions_visible
    return impressions


def update_impression_col(df):
    _df = df.copy()
    _df['impressions'] = _df.apply(lambda x: get_updated_impressions(x['impressions'],
                                                                     x['impressions_visible']), axis=1)
    _df = _df.drop(columns=['impressions_visible'])
    return _df


def get_impress_positions(df: pd.DataFrame) -> pd.Series:
    positions = df.groupby('request_id').cumcount() + 1
    positions = positions.reset_index(drop=True)

    return positions


def get_impress_item_and_pos(df):
    _df = df.explode('impressions')
    _df['position'] = get_impress_positions(_df)
    _df = _df.rename(columns={'impressions': 'item'})
    _df['event_type'] = 'impress'
    return _df[['request_id', 'item', 'position', 'event_type']]


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    events_agg = events.pivot_table(index=['item'], columns=['event_type'], values=['request_id'],
                                    aggfunc='count', fill_value=0)
    events_agg.columns = events_agg.columns.droplevel()
    events_agg = events_agg.reset_index()

    # Clear index name
    events_agg = events_agg.rename_axis(None, axis=1)

    return events_agg


class NaiveCTR:
    def __init__(self):
        self.item_dict = None

    def fit(self, events: pd.DataFrame) -> 'NaiveCTR':
        events_agg = aggregate_events(events)
        events_agg['ctr'] = events_agg['click'] / events_agg['impress']

        self.item_dict = events_agg.set_index('item')['ctr'].to_dict()

        return self

    def predict(self, item_id: str) -> int:
        return self.item_dict.get(item_id, -1)

    def batch_predict(self, item_df: pd.DataFrame) -> pd.DataFrame:
        result = item_df.copy()
        result['expected_ctr'] = result['item_id'].apply(lambda x: self.predict(x))

        return result
