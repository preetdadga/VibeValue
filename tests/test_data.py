import pandas as pd
from vibevalue.data import load_phrasebank, prepare_splits_from_df, validate_df


def make_df():
    return pd.DataFrame({
        "text": [f"sentence {i}" for i in range(10)],
        # labels: 0,1,2 repeated (not perfectly balanced but enough for stratify)
        "label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 1],
    })


def test_validate_and_split():
    df = make_df()
    dup = validate_df(df)
    assert dup == 0
    train, val, test = prepare_splits_from_df(df, seed=0)
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1


def test_load_phrasebank_csv_fallback():
    df = load_phrasebank()
    assert df.shape == (4840, 2)
    counts = df["label"].value_counts().sort_index().to_dict()
    assert counts == {0: 604, 1: 2873, 2: 1363}
    assert df["text"].notna().all()
    assert (df["text"].astype(str).str.strip() != "").all()
