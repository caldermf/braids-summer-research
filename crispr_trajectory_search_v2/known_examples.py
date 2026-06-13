"""Known trajectories used only for calibration and repair experiments."""

KNOWN_P5_LENGTH54_FACTOR_IDS = (
    7,
    7,
    10,
    13,
    4,
    13,
    4,
    2,
    13,
    20,
    13,
    20,
    13,
    10,
    2,
    13,
    4,
    13,
    4,
    13,
    7,
    21,
    20,
    13,
    20,
    13,
    10,
    16,
    16,
    2,
    13,
    4,
    13,
    4,
    13,
    21,
    20,
    13,
    20,
    13,
    21,
    10,
    13,
    4,
    13,
    4,
    2,
    16,
    13,
    11,
    13,
    11,
    13,
    21,
)

KNOWN_EXAMPLES = {
    "p5_length54": KNOWN_P5_LENGTH54_FACTOR_IDS,
}


def known_example(name: str) -> tuple[int, ...]:
    try:
        return KNOWN_EXAMPLES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(KNOWN_EXAMPLES))
        raise ValueError(f"unknown calibration example {name!r}; choose from {choices}") from exc
