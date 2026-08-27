"""Filename and metadata parsing helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


BEARING_CLASSES = {"healthy", "bearing_inner", "bearing_outer", "bearing_ball"}

GEARBOX_2025_TEST_MAP = {
    1: ("healthy", "variable_speed"),
    2: ("healthy", "variable_load"),
    3: ("healthy", "stationary"),
    4: ("healthy", "variable_speed_load"),
    5: ("broken_gear_tooth", "variable_speed"),
    6: ("broken_gear_tooth", "variable_load"),
    7: ("broken_gear_tooth", "variable_speed_load"),
    8: ("broken_gear_tooth", "stationary"),
    10: ("combined_gear_bearing", "variable_load"),
    11: ("combined_gear_bearing", "variable_speed_load"),
    12: ("combined_gear_bearing", "stationary"),
    14: ("bearing_outer", "variable_load"),
    15: ("bearing_outer", "stationary"),
    16: ("bearing_outer", "variable_speed_load"),
}


def _number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_mcc5_filename(filename: str) -> dict[str, Any]:
    """Parse MCC5 filename tokens into label and condition metadata."""
    name = Path(str(filename)).name
    stem = Path(name).stem
    text = stem.lower()

    condition_type = "unknown"
    if "speed_circulation" in text:
        condition_type = "speed_circulation"
    elif "torque_circulation" in text:
        condition_type = "torque_circulation"
    elif "speed" in text:
        condition_type = "speed_related"
    elif "torque" in text or "load" in text:
        condition_type = "torque_related"

    is_compound = "_and_" in text or "compound" in text or text.count("bearing_") > 1
    severity = None
    sev_match = re.search(r"(?:^|_)([hlm])(?:_|$)", text)
    if sev_match:
        severity = sev_match.group(1).upper()

    if "normal" in text or "healthy" in text or re.search(r"(?:^|_)health(?:_|$)", text):
        label_group = "healthy"
        fault_family = "healthy"
    elif "bearing_inner" in text or "inner" in text:
        label_group = "bearing_inner" if not is_compound else "compound"
        fault_family = "bearing"
    elif "bearing_outer" in text or "outer" in text:
        label_group = "bearing_outer" if not is_compound else "compound"
        fault_family = "bearing"
    elif "bearing_ball" in text or "ball" in text:
        label_group = "bearing_ball" if not is_compound else "compound"
        fault_family = "bearing"
    elif "unbalance" in text:
        label_group = "unbalance"
        fault_family = "mechanical"
    elif "eccentricity" in text:
        label_group = "eccentricity"
        fault_family = "mechanical"
    elif "winding" in text or "stator" in text or "short" in text:
        label_group = "stator_winding"
        fault_family = "electrical"
    elif "broken_bar" in text or "bar" in text:
        label_group = "broken_bar"
        fault_family = "electrical"
    elif "bend" in text:
        label_group = "bend"
        fault_family = "mechanical"
    else:
        label_group = "unknown"
        fault_family = "unknown"

    return {
        "label_raw": stem,
        "label_group": label_group,
        "fault_family": fault_family,
        "condition_type": condition_type,
        "rpm_nominal": _number(r"(\d+(?:\.\d+)?)\s*rpm", text),
        "load_nm": _number(r"(\d+(?:\.\d+)?)\s*nm", text),
        "severity": severity,
        "is_bearing_fault": label_group in {"bearing_inner", "bearing_outer", "bearing_ball"},
        "is_compound_fault": is_compound or label_group == "compound",
    }


def parse_gearbox_filename(filename: str) -> dict[str, Any]:
    """Parse gearbox fault and condition names from a path."""
    text = str(filename).lower()
    file_match = re.search(r"nextmon_gps_(\d+)", text)
    if file_match:
        test_id = int(file_match.group(1))
        if test_id in GEARBOX_2025_TEST_MAP:
            label, condition = GEARBOX_2025_TEST_MAP[test_id]
            return {
                "label_group": label,
                "condition_type": condition,
                "test_id": test_id,
                "label_source": "mendeley_test_distribution",
            }
    if "broken" in text and ("bearing" in text or "outer" in text):
        label = "combined_gear_bearing"
    elif "broken" in text or "tooth" in text:
        label = "broken_gear_tooth"
    elif "bearing" in text or "outer" in text:
        label = "bearing_outer"
    elif "healthy" in text or "normal" in text:
        label = "healthy"
    else:
        label = "unknown"
    if "variable" in text and "load" in text and "speed" in text:
        condition = "variable_speed_load"
    elif "variable" in text and "speed" in text:
        condition = "variable_speed"
    elif "variable" in text and "load" in text:
        condition = "variable_load"
    else:
        condition = "stationary"
    return {"label_group": label, "condition_type": condition, "test_id": None, "label_source": "filename_tokens"}


def parse_vat_filename(filename: str) -> dict[str, Any]:
    """Parse VAT file names into signal type and bearing label."""
    text = Path(str(filename)).stem.lower()
    if "inner" in text:
        label = "bearing_inner"
    elif "outer" in text:
        label = "bearing_outer"
    elif "ball" in text:
        label = "bearing_ball"
    elif "normal" in text or "healthy" in text:
        label = "healthy"
    else:
        label = "unknown"
    if "current" in text:
        signal = "current"
    elif "rpm" in text or "speed" in text:
        signal = "rpm"
    elif "vibration" in text or "vib" in text:
        signal = "vibration"
    else:
        signal = "unknown"
    return {"label_group": label, "signal_type": signal}


def parse_lenze_metadata_row(row) -> dict[str, Any]:
    """Infer a Lenze label from a metadata row without assuming exact columns."""
    raw_condition = str(row.to_dict().get("Condition", "")).strip()
    joined = " ".join(str(v).lower() for v in row.to_dict().values())
    condition_text = raw_condition.lower()
    if "healthy" in joined or "normal" in joined:
        label = "healthy"
        fault_family = "healthy"
    elif "pitting" in condition_text or "scientific_fault" in condition_text or "fault" in condition_text:
        label = "bearing_fault"
        fault_family = "bearing"
    elif "inner" in joined:
        label = "bearing_inner"
        fault_family = "bearing"
    elif "outer" in joined:
        label = "bearing_outer"
        fault_family = "bearing"
    elif "ball" in joined:
        label = "bearing_ball"
        fault_family = "bearing"
    else:
        label = "unknown"
        fault_family = "unknown"
    severity = None
    severity_match = re.search(r"pitting[_\s-]*([ivx]+|\d+)", condition_text)
    if severity_match:
        severity = severity_match.group(1).upper()
    elif "scientific_fault" in condition_text:
        severity = "scientific_fault"
    return {
        "label_group": label,
        "label_raw": raw_condition or joined[:200],
        "fault_family": fault_family,
        "severity": severity,
    }
