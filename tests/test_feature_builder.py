import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from feature_builder import (
    DraftValidationError,
    build_feature_row,
    calculate_role_counts,
    get_hero_choices,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "artifacts" / "dota2_draft_model_bundle_compressed.joblib"


@pytest.fixture(scope="module")
def bundle():
    if not BUNDLE_PATH.exists():
        pytest.skip("Model bundle is not present in artifacts/.")
    return joblib.load(BUNDLE_PATH)


def make_valid_payload(bundle):
    _, pick_options, ban_options = get_hero_choices(bundle)
    payload = {}
    chosen_picks = set()
    for side in ("radiant", "dire"):
        for slot in range(1, 6):
            field = f"{side}_hero_{slot}"
            available = [h["id"] for h in pick_options[field] if h["id"] not in chosen_picks]
            if not available:
                raise AssertionError(f"No unused supported hero found for {field}")
            payload[field] = str(available[0])
            chosen_picks.add(available[0])
    chosen_bans = set()
    for side in ("radiant", "dire"):
        for slot in range(1, 8):
            field = f"{side}_ban_{slot}"
            available = [h["id"] for h in ban_options[field]
                         if h["id"] not in chosen_picks and h["id"] not in chosen_bans]
            # Missing ban slots are allowed; some valid test drafts have fewer bans.
            payload[field] = str(available[0]) if available else ""
            if available:
                chosen_bans.add(available[0])
    return payload


def test_bundle_schema_consistency(bundle):
    columns = bundle["feature_columns"]
    assert len(columns) == 2996
    assert len(set(columns)) == 2996
    assert list(bundle["decision_tree"].feature_names_in_) == columns
    assert list(bundle["random_forest"].feature_names_in_) == columns
    assert bundle["decision_tree"].n_features_in_ == 2996
    assert bundle["random_forest"].n_features_in_ == 2996


def test_build_feature_row_has_exact_schema_and_predictions_run(bundle):
    payload = make_valid_payload(bundle)
    row, notes = build_feature_row(
        payload, bundle["feature_columns"], bundle["hero_constants"], bundle["hero_role_map"]
    )
    assert row.shape == (1, 2996)
    assert row.columns.tolist() == bundle["feature_columns"]
    assert bundle["decision_tree"].predict(row).shape == (1,)
    assert bundle["random_forest"].predict(row).shape == (1,)
    assert isinstance(notes, list)


def test_duplicate_picks_are_rejected(bundle):
    payload = make_valid_payload(bundle)
    payload["dire_hero_1"] = payload["radiant_hero_1"]
    with pytest.raises(DraftValidationError, match="cannot be picked more than once"):
        build_feature_row(payload, bundle["feature_columns"], bundle["hero_constants"], bundle["hero_role_map"])


def test_picked_and_banned_overlap_is_rejected(bundle):
    payload = make_valid_payload(bundle)
    payload["radiant_ban_1"] = payload["radiant_hero_1"]
    with pytest.raises(DraftValidationError, match="also be banned"):
        build_feature_row(payload, bundle["feature_columns"], bundle["hero_constants"], bundle["hero_role_map"])


def test_missing_pick_is_rejected(bundle):
    payload = make_valid_payload(bundle)
    payload["radiant_hero_1"] = ""
    with pytest.raises(DraftValidationError, match="is required"):
        build_feature_row(payload, bundle["feature_columns"], bundle["hero_constants"], bundle["hero_role_map"])


def test_role_counts_follow_role_map(bundle):
    role_map = bundle["hero_role_map"]
    heroes = bundle["hero_constants"]
    antimage_ids = heroes.loc[heroes["localized_name"].eq("Anti-Mage"), "id"]
    assert len(antimage_ids) == 1
    antimage = int(antimage_ids.iloc[0])
    counts = calculate_role_counts({"radiant": [antimage]*5, "dire": [antimage]*5}, role_map)
    expected = role_map[antimage]
    assert counts["radiant_disabler_count"] == 5 * ("Disabler" in expected)
    assert counts["radiant_initiator_count"] == 5 * ("Initiator" in expected)
    assert counts["radiant_pusher_count"] == 5 * ("Pusher" in expected)
