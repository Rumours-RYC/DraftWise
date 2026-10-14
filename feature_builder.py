"""Faithful prediction-row builder for the exported Dota 2 draft models."""
from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

PICK_COLS = [f"{side}_hero_{i}" for side in ("radiant", "dire") for i in range(1, 6)]
BAN_COLS = [f"{side}_ban_{i}" for side in ("radiant", "dire") for i in range(1, 8)]
ROLE_COLS = [
    "radiant_disabler_count", "dire_disabler_count",
    "radiant_initiator_count", "dire_initiator_count",
    "radiant_pusher_count", "dire_pusher_count",
]

PICK_FEATURE_PATTERN = re.compile(r"^(radiant|dire)_hero_([1-5])_(\d+)$")
BAN_FEATURE_PATTERN = re.compile(r"^(radiant|dire)_ban_([1-7])_(.+)$")


class DraftValidationError(ValueError):
    """Raised when a draft cannot validly be passed to the research models."""


def _as_int(value: Any, label: str) -> int:
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            raise DraftValidationError(f"{label} is required.")
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, DraftValidationError):
            raise
        raise DraftValidationError(f"{label} must be a valid hero ID.") from exc


def get_supported_ids_by_slot(feature_columns: list[str], group: str) -> dict[str, set[int]]:
    """Return hero IDs represented by the model schema for each exact input slot."""
    result: dict[str, set[int]] = {}
    if group == "pick":
        pattern = PICK_FEATURE_PATTERN
    elif group == "ban":
        pattern = BAN_FEATURE_PATTERN
    else:
        raise ValueError("group must be 'pick' or 'ban'")
    for column in feature_columns:
        match = pattern.match(column)
        if not match:
            continue
        side, slot = match.group(1), match.group(2)
        value = match.group(3)
        if group == "pick" or value != "<NA>":
            try:
                hero_id = int(value)
            except ValueError:
                continue
            key = f"{side}_{'hero' if group == 'pick' else 'ban'}_{slot}"
            result.setdefault(key, set()).add(hero_id)
    return result


def _normalize_role_map(role_map: Mapping[Any, Any]) -> dict[int, list[str]]:
    normalized: dict[int, list[str]] = {}
    for key, roles in role_map.items():
        try:
            hero_id = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(roles, (list, tuple, set)):
            normalized[hero_id] = [str(role) for role in roles]
        else:
            normalized[hero_id] = []
    return normalized


def calculate_role_counts(picks_by_side: Mapping[str, list[int]], role_map: Mapping[Any, Any]) -> dict[str, int]:
    """Match notebook count_role behavior for Disabler, Initiator and Pusher."""
    roles = _normalize_role_map(role_map)
    counts: dict[str, int] = {}
    for side in ("radiant", "dire"):
        heroes = picks_by_side[side]
        for role in ("disabler", "initiator", "pusher"):
            target = role.title()
            counts[f"{side}_{role}_count"] = sum(
                1 for hero_id in heroes if target in roles.get(int(hero_id), [])
            )
    return counts


def validate_draft(payload: Mapping[str, Any], heroes: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, list[int | None]]]:
    """Validate unique picks/bans and return normalized IDs by team."""
    known_ids = {int(v) for v in heroes["id"].dropna().tolist()}
    picks: dict[str, list[int]] = {}
    bans: dict[str, list[int | None]] = {}

    for side in ("radiant", "dire"):
        team_picks = []
        for slot in range(1, 6):
            key = f"{side}_hero_{slot}"
            hero_id = _as_int(payload.get(key), f"{side.title()} pick {slot}")
            if hero_id not in known_ids:
                raise DraftValidationError(f"Unknown hero ID in {key}.")
            team_picks.append(hero_id)
        picks[side] = team_picks

        team_bans: list[int | None] = []
        for slot in range(1, 8):
            key = f"{side}_ban_{slot}"
            raw = payload.get(key)
            if raw is None or str(raw).strip() == "":
                team_bans.append(None)
                continue
            hero_id = _as_int(raw, f"{side.title()} ban {slot}")
            if hero_id not in known_ids:
                raise DraftValidationError(f"Unknown hero ID in {key}.")
            team_bans.append(hero_id)
        bans[side] = team_bans

    all_picks = picks["radiant"] + picks["dire"]
    if len(set(all_picks)) != 10:
        raise DraftValidationError("A hero cannot be picked more than once across both teams.")

    all_bans = [hero_id for values in bans.values() for hero_id in values if hero_id is not None]
    if len(set(all_bans)) != len(all_bans):
        raise DraftValidationError("A hero cannot be banned more than once across both teams.")
    if set(all_picks).intersection(all_bans):
        raise DraftValidationError("A picked hero cannot also be banned in the same draft.")

    return picks, bans


def build_feature_row(
    payload: Mapping[str, Any],
    feature_columns: list[str],
    heroes: pd.DataFrame,
    role_map: Mapping[Any, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Build a one-row DataFrame aligned exactly to the original X.columns schema.

    Categorical values follow the notebook convention: nullable integer IDs are
    converted to strings before one-hot encoding; missing ban slots use '<NA>'.
    """
    if len(feature_columns) != 2996:
        raise ValueError(f"Expected 2,996 model features, got {len(feature_columns)}.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature schema contains duplicate column names.")

    picks, bans = validate_draft(payload, heroes)
    column_set = set(feature_columns)
    # Values are numeric from the start. Feature columns not represented by this
    # draft remain zero, equivalent to reindexing one-hot columns to training X.
    row = pd.DataFrame([[0] * len(feature_columns)], columns=feature_columns, dtype=np.int64)
    warnings: list[str] = []

    for side in ("radiant", "dire"):
        for slot, hero_id in enumerate(picks[side], start=1):
            field = f"{side}_hero_{slot}"
            dummy = f"{field}_{hero_id}"
            if dummy in column_set:
                row.at[0, dummy] = 1
            else:
                warnings.append(
                    f"{field}: hero ID {hero_id} was not observed in this pick position during training; "
                    "this slot's one-hot identity is all zeros."
                )
        for slot, hero_id in enumerate(bans[side], start=1):
            field = f"{side}_ban_{slot}"
            category = "<NA>" if hero_id is None else str(hero_id)
            dummy = f"{field}_{category}"
            if dummy in column_set:
                row.at[0, dummy] = 1
            elif hero_id is not None:
                warnings.append(
                    f"{field}: hero ID {hero_id} was not observed in this ban position during training; "
                    "this slot's one-hot identity is all zeros."
                )
            else:
                warnings.append(
                    f"{field}: the missing-ban category is absent from the saved schema; this slot is all zeros."
                )

    role_counts = calculate_role_counts(picks, role_map)
    for column in ROLE_COLS:
        if column not in column_set:
            raise ValueError(f"Required role feature '{column}' is absent from schema.")
        row.at[0, column] = role_counts[column]

    if row.shape != (1, 2996):
        raise ValueError(f"Constructed feature row has unexpected shape {row.shape}.")
    if list(row.columns) != feature_columns:
        raise ValueError("Constructed feature row order differs from saved schema.")
    return row, warnings


def get_hero_choices(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Return metadata heroes and per-slot hero choices representable by the schema."""
    heroes = bundle["hero_constants"].copy()
    if "id" not in heroes.columns or "localized_name" not in heroes.columns:
        raise ValueError("hero_constants must contain 'id' and 'localized_name'.")
    heroes = heroes.dropna(subset=["id", "localized_name"]).copy()
    heroes["id"] = heroes["id"].astype(int)
    heroes["localized_name"] = heroes["localized_name"].astype(str)
    heroes = heroes.drop_duplicates(subset=["id"]).sort_values("localized_name")
    cols = bundle["feature_columns"]
    pick_support = get_supported_ids_by_slot(cols, "pick")
    ban_support = get_supported_ids_by_slot(cols, "ban")

    options = [{"id": int(row.id), "name": str(row.localized_name)} for row in heroes.itertuples()]
    by_slot_pick: dict[str, list[dict[str, Any]]] = {}
    by_slot_ban: dict[str, list[dict[str, Any]]] = {}
    lookup = {item["id"]: item for item in options}
    for side in ("radiant", "dire"):
        for slot in range(1, 6):
            key = f"{side}_hero_{slot}"
            by_slot_pick[key] = [lookup[i] for i in sorted(pick_support.get(key, set())) if i in lookup]
        for slot in range(1, 8):
            key = f"{side}_ban_{slot}"
            by_slot_ban[key] = [lookup[i] for i in sorted(ban_support.get(key, set())) if i in lookup]
    return options, by_slot_pick, by_slot_ban
