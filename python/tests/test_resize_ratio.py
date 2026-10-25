"""Resize ratios across both size limits (RapidOCR main #761, issue #759)."""

import numpy as np

from rapidocr.utils.process_img import map_boxes_to_original, resize_image_within_bounds


def test_resize_ratios_map_boxes_after_both_size_limits():
    original = np.zeros((100, 1000, 3), dtype=np.uint8)
    resized, ratio_h, ratio_w = resize_image_within_bounds(
        original, min_side_len=128, max_side_len=512
    )

    assert resized.shape[:2] == (128, 1024)
    assert ratio_h == original.shape[0] / resized.shape[0]
    assert ratio_w == original.shape[1] / resized.shape[1]

    boxes = np.array([[[128, 32], [256, 32], [256, 64], [128, 64]]], dtype=np.float32)
    ori_h, ori_w = original.shape[:2]
    mapped = map_boxes_to_original(boxes, ratio_w, ratio_h, 0, ori_w, ori_h)
    np.testing.assert_allclose(mapped, [[[125, 25], [250, 25], [250, 50], [125, 50]]])


def test_resize_ratios_single_limit_unchanged():
    img = np.zeros((3000, 1500, 3), dtype=np.uint8)
    resized, ratio_h, ratio_w = resize_image_within_bounds(img, 30, 2000)
    assert ratio_h == 3000 / resized.shape[0]
    assert ratio_w == 1500 / resized.shape[1]

    small = np.zeros((200, 300, 3), dtype=np.uint8)
    same, ratio_h, ratio_w = resize_image_within_bounds(small, 30, 2000)
    assert same is small and ratio_h == 1.0 and ratio_w == 1.0
