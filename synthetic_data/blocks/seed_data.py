"""
SeedDataBlock: Creates the initial DataFrame with N rows for generation.

Each row will be populated by downstream sampler blocks with the
specific field values for that synthetic example.
"""

from __future__ import annotations

import pandas as pd
from sdg_hub import BaseBlock, BlockRegistry


@BlockRegistry.register(
    "SeedDataBlock", "transform",
    "Create initial DataFrame with N empty rows for synthetic generation",
)
class SeedDataBlock(BaseBlock):
    num_samples: int = 150

    def generate(self, samples: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({"row_id": range(self.num_samples)})
