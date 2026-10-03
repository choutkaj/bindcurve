"""Arithmetic response aggregation shared by data preparation and plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_responses(
    table: pd.DataFrame, *, by: list[str], count_name: str
) -> pd.DataFrame:
    """Return group means, sample SD, counts, and SEM in sorted group order.

    Grand means must receive experiment means, never pooled raw replicates.
    Known observation sigma is propagated separately by ``fit_observations``.
    """
    aggregated = table.groupby(by, as_index=False)["response"].agg(
        **{"response": "mean", "response_sd": "std", count_name: "count"}
    )
    aggregated["response_sem"] = aggregated["response_sd"] / np.sqrt(
        aggregated[count_name]
    )
    return aggregated.sort_values(by)
