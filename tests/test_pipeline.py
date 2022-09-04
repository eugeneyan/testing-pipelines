import numpy as np
import pandas as pd
import pytest

from src.pipeline import get_click_position, get_impress_positions, aggregate_events, get_impress_item_and_pos, \
    get_click_item_and_pos, NaiveCTR


@pytest.fixture
def logs():
    arr = (('r1', ['i1', 'i2', 'i3', 'i4'], None, 'impress'),
           ('r2', ['i2', 'i3', 'i4', 'i1'], 'i2', 'click'),
           ('r3', ['i3', 'i4', 'i1', 'i2'], None, 'impress'),
           ('r4', ['i4', 'i1', 'i2', 'i3'], 'i3', 'click'),
           ('r4', ['i1', 'i2', 'i3', 'i4'], None, 'impress'))
    cols = ('request_id', 'impressions', 'event', 'event_type')
    df = pd.DataFrame(arr, columns=cols)
    return df


@pytest.fixture
def events(logs):
    impress_logs = logs[logs['event_type'] == 'impress']
    click_logs = logs[logs['event_type'] == 'click']

    impress_events = get_impress_item_and_pos(impress_logs)
    click_events = get_click_item_and_pos(click_logs)

    events = pd.concat([impress_events, click_events])
    return events


@pytest.fixture
def items():
    return pd.Series(['i1', 'i2', 'i3', 'i4', 'i5']).to_frame(name='item_id')


@pytest.fixture
def model():
    ctr = NaiveCTR()
    return ctr


# Unit test: Row level
@pytest.mark.parametrize('impressions,click,expected',
                         [(['i1', 'i2', 'i3', 'i4'], 'i1', 1),
                          (['i1', 'i2', 'i3', 'i4'], 'i3', 3),
                          (['i1', 'i2', 'i3', 'i4'], None, -1),
                          (['i1', 'i2', 'i3', 'i4'], 'NA', -1)])
def test_get_click_position(impressions, click, expected):
    assert get_click_position(impressions, click) == expected


# Unit test: Column level
def test_get_impress_position(logs):
    impress_logs = logs[logs['event_type'] == 'impress']
    impress_events = impress_logs.explode('impressions')
    impress_positions = get_impress_positions(impress_events)

    # Since each impress log has 4 items, expect impression positions to be three sets of 1-4
    pd.testing.assert_series_equal(impress_positions, pd.Series([1, 2, 3, 4] * 3))


# Unit test: Dataframe level
def test_aggregate_events(events):
    result = aggregate_events(events)

    arr = [['i1', 0, 3],
           ['i2', 1, 3],
           ['i3', 1, 3],
           ['i4', 0, 3]]
    cols = ['item', 'click', 'impress']
    expected = pd.DataFrame(arr, columns=cols)

    pd.testing.assert_frame_equal(result, expected)


# Schema check: Columns and datatypes
def test_events_schema(events):
    expected_cols = {'request_id': np.object,
                     'item': np.object,
                     'position': np.int64,
                     'event_type': np.object}

    # Check all expected columns are present
    assert len(set(expected_cols.keys()) - set(events.columns)) == 0, \
        f'{set(expected_cols.keys()) - set(events.columns)} columns missing!'

    # Check all column data types are correct
    for col, dtype in expected_cols.items():
        assert events.dtypes[col] == dtype, \
            f'Expected column {col} to be of {dtype} type but found {events.dtypes[col]} type!'


# Integration test: Input logs to aggregated events
def test_feature_pipeline(logs):
    impress_logs = logs[logs['event_type'] == 'impress']
    click_logs = logs[logs['event_type'] == 'click']

    impress_events = get_impress_item_and_pos(impress_logs)
    click_events = get_click_item_and_pos(click_logs)

    sample_events = pd.concat([impress_events, click_events])
    result = aggregate_events(sample_events)

    arr = [['i1', 0, 3],
           ['i2', 1, 3],
           ['i3', 1, 3],
           ['i4', 0, 3]]
    cols = ['item', 'click', 'impress']
    expected = pd.DataFrame(arr, columns=cols)

    pd.testing.assert_frame_equal(result, expected)


# Integration test: Aggregated events to batch inference
def test_model_pipeline(events, model, items):
    model = model.fit(events)
    result = model.batch_predict(items)

    arr = [['i1', 0.0],
           ['i2', 0.3333333333],
           ['i3', 0.3333333333],
           ['i4', 0.0],
           ['i5', -1.0]]
    cols = ['item_id', 'expected_ctr']
    expected = pd.DataFrame(arr, columns=cols)

    pd.testing.assert_frame_equal(result, expected)
