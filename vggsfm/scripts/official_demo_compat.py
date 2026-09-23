#!/usr/bin/env python3
"""Run pinned official demo with one non-1024 invalid-mask compatibility fix.

LightGlue extractors preprocess query images to a fixed 1024-pixel long side.
The pinned ``Extractor.extract`` passes ``invalid_mask`` through at the caller's
resolution, so VGGSfM ``img_size != 1024`` fails when the extractor indexes its
1024 score map with (for example) a 640 mask. This adapter reproduces the
upstream method and nearest-resizes only that boolean invalid mask to the
preprocessed image shape. Keypoints are still mapped back with the unchanged
upstream scale calculation. No model, camera, track or point code is changed.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


def install_invalid_mask_compatibility() -> None:
    from lightglue.utils import Extractor, ImagePreprocessor

    @torch.no_grad()
    def extract(self, img: torch.Tensor, invalid_mask=None, **conf) -> dict:
        if img.dim() == 3:
            img = img[None]
        if img.dim() != 4 or img.shape[0] != 1:
            raise ValueError(f"Expected one BCHW image; received {tuple(img.shape)}")
        shape = img.shape[-2:][::-1]
        processed, scales = ImagePreprocessor(
            **{**self.preprocess_conf, **conf}
        )(img)
        if invalid_mask is not None:
            mask = invalid_mask
            if mask.dim() == 3:
                mask = mask[:, None]
            if mask.dim() != 4 or mask.shape[0] != 1 or mask.shape[1] != 1:
                raise ValueError(f"Expected one B1HW invalid mask; received {tuple(mask.shape)}")
            if mask.shape[-2:] != processed.shape[-2:]:
                mask = torch.nn.functional.interpolate(
                    mask.float(), size=processed.shape[-2:], mode="nearest"
                )
            invalid_mask = mask[:, 0].bool()
        feats = self.forward({"image": processed}, invalid_mask=invalid_mask)
        feats["image_size"] = torch.tensor(shape)[None].to(processed).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        return feats

    Extractor.extract = extract


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: official_demo_compat.py OFFICIAL_DEMO [Hydra overrides ...]")
    official_demo = Path(sys.argv[1]).resolve(strict=True)
    expected_tail = Path("downloads/source/vggsfm/demo.py")
    if not str(official_demo).endswith(str(expected_tail)):
        raise SystemExit(f"refusing unexpected official demo path: {official_demo}")
    install_invalid_mask_compatibility()
    sys.argv = [str(official_demo), *sys.argv[2:]]
    runpy.run_path(str(official_demo), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
