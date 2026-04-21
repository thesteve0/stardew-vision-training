# SDG Hub custom blocks for Stardew Vision synthetic data generation.
# Import all blocks so BlockRegistry auto-discovers them.

from synthetic_data.blocks.seed_data import SeedDataBlock  # noqa: F401
from synthetic_data.blocks.caught_fish_sampler import CaughtFishSamplerBlock  # noqa: F401
from synthetic_data.blocks.pierre_shop_sampler import PierreShopSamplerBlock  # noqa: F401
from synthetic_data.blocks.tv_dialog_sampler import TVDialogSamplerBlock  # noqa: F401
from synthetic_data.blocks.render_screenshot import RenderScreenshotBlock  # noqa: F401
from synthetic_data.blocks.build_chatml import BuildChatMLBlock  # noqa: F401
