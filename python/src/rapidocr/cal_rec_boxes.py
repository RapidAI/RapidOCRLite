"""Word and character boxes (rapidocr main ``return_word_box``).

A line-for-line port of rapidocr main's ``CTCLabelDecode.get_word_info``
(ch_ppocr_rec/utils.py) and ``CalRecBoxes`` (cal_rec_boxes/main.py) at
RapidAI/RapidOCR 8ac45f4e. The native engine supplies what main reads from
the recognizer output (the time step and probability of every decoded
character, the batch's time-step count and max_wh_ratio, and the crop size);
everything after that is the same Python. The two OpenCV calls main makes,
``cv2.getPerspectiveTransform`` and ``cv2.invert`` (DECOMP_LU), are ported
below with OpenCV's arithmetic so the integer word boxes match bit for bit
without an OpenCV dependency.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Sequence, Tuple

import numpy as np


class WordType(Enum):
    CN = "cn"
    EN = "en"
    NUM = "num"
    EN_NUM = "en&num"


@dataclass
class WordInfo:
    words: List[List[str]] = field(default_factory=list)
    word_cols: List[List[int]] = field(default_factory=list)
    word_types: List[WordType] = field(default_factory=list)
    line_txt_len: float = 0.0
    confs: List[float] = field(default_factory=list)


class Direction(Enum):
    HORIZONTAL = "horizontal_direct"
    VERTICAL = "vertical_direct"


def is_chinese_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f" or "\uff00" <= ch <= "\uffef"


def has_chinese_char(text: str) -> bool:
    return any(is_chinese_char(ch) for ch in text)


def quads_to_rect_bbox(bbox: np.ndarray) -> Tuple[float, float, float, float]:
    if bbox.ndim != 3:
        raise ValueError("bbox shape must be 3")
    if bbox.shape[1] != 4 and bbox.shape[2] != 2:
        raise ValueError("bbox shape must be (N, 4, 2)")
    all_x, all_y = (bbox[:, :, 0].flatten(), bbox[:, :, 1].flatten())
    x_min, y_min = np.min(all_x), np.min(all_y)
    x_max, y_max = np.max(all_x), np.max(all_y)
    return float(x_min), float(y_min), float(x_max), float(y_max)


def build_word_info(
    text: str,
    cols: Sequence[int],
    probs: Sequence[float],
    steps: int,
    wh_ratio: float,
    max_wh_ratio: float,
) -> WordInfo:
    """main's CTCLabelDecode.decode(return_word_box=True) for one line."""

    conf_list = [round(float(conf), 5) for conf in probs]
    if len(conf_list) == 0:
        conf_list = [0]
    selection = np.zeros(int(steps), dtype=bool)
    if len(cols):
        selection[np.asarray(cols, dtype=np.int64)] = True
    info = get_word_info(text, selection)
    info.line_txt_len = int(steps) * wh_ratio / max_wh_ratio
    info.confs = conf_list
    return info


def get_word_info(text: str, selection: np.ndarray) -> WordInfo:
    word_list = []
    word_col_list = []
    state_list = []
    word_content = []
    word_col_content = []

    valid_col = np.where(selection)[0]
    if len(valid_col) <= 0:
        return WordInfo()

    col_width = np.zeros(valid_col.shape)
    col_width[1:] = valid_col[1:] - valid_col[:-1]
    col_width[0] = min(3 if has_chinese_char(text[0]) else 2, int(valid_col[0]))

    state = None
    for c_i, char in enumerate(text):
        if char.isspace():
            if word_content:
                word_list.append(word_content)
                word_col_list.append(word_col_content)
                state_list.append(state)
                word_content = []
                word_col_content = []
            continue

        c_state = WordType.CN if has_chinese_char(char) else WordType.EN_NUM
        if state is None:
            state = c_state

        if state != c_state or col_width[c_i] > 5:
            if len(word_content) != 0:
                word_list.append(word_content)
                word_col_list.append(word_col_content)
                state_list.append(state)
                word_content = []
                word_col_content = []
            state = c_state

        word_content.append(char)
        word_col_content.append(int(valid_col[c_i]))

    if len(word_content) != 0:
        word_list.append(word_content)
        word_col_list.append(word_col_content)
        state_list.append(state)

    return WordInfo(words=word_list, word_cols=word_col_list, word_types=state_list)


def _lu_solve(a: List[List[float]], b: List[float]) -> List[float]:
    """OpenCV LUImpl<double> (DECOMP_LU) for one right-hand side."""

    m = len(a)
    eps = np.finfo(np.float64).eps * 100
    for i in range(m):
        k = i
        for j in range(i + 1, m):
            if abs(a[j][i]) > abs(a[k][i]):
                k = j
        if abs(a[k][i]) < eps:
            raise np.linalg.LinAlgError("singular perspective system")
        if k != i:
            a[i], a[k] = a[k], a[i]
            b[i], b[k] = b[k], b[i]
        d = -1.0 / a[i][i]
        for j in range(i + 1, m):
            alpha = a[j][i] * d
            for c in range(i + 1, m):
                a[j][c] += alpha * a[i][c]
            b[j] += alpha * b[i]
    for i in range(m - 1, -1, -1):
        s = b[i]
        for c in range(i + 1, m):
            s -= a[i][c] * b[c]
        b[i] = s / a[i][i]
    return b


def get_perspective_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """cv2.getPerspectiveTransform(src, dst) (float32 points, DECOMP_LU)."""

    src = np.asarray(src, dtype=np.float32).reshape(4, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(4, 2)
    a = [[0.0] * 8 for _ in range(8)]
    b = [0.0] * 8
    for i in range(4):
        sx, sy = src[i, 0], src[i, 1]
        dx, dy = dst[i, 0], dst[i, 1]
        a[i][0] = a[i + 4][3] = float(sx)
        a[i][1] = a[i + 4][4] = float(sy)
        a[i][2] = a[i + 4][5] = 1.0
        # Point2f products are float32 in OpenCV.
        a[i][6] = float(np.float32(-sx) * dx)
        a[i][7] = float(np.float32(-sy) * dx)
        a[i + 4][6] = float(np.float32(-sx) * dy)
        a[i + 4][7] = float(np.float32(-sy) * dy)
        b[i] = float(dx)
        b[i + 4] = float(dy)
    x = _lu_solve(a, b)
    return np.array(x + [1.0], dtype=np.float64).reshape(3, 3)


def invert3x3(m: np.ndarray) -> np.ndarray:
    """cv2.invert(m) for a 3x3 CV_64F matrix (OpenCV's closed form)."""

    s = [[float(m[r, c]) for c in range(3)] for r in range(3)]
    d = (s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1])
         - s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0])
         + s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0]))
    out = np.zeros((3, 3), dtype=np.float64)
    if d == 0.0:
        return out
    d = 1.0 / d
    out[0, 0] = (s[1][1] * s[2][2] - s[1][2] * s[2][1]) * d
    out[0, 1] = (s[0][2] * s[2][1] - s[0][1] * s[2][2]) * d
    out[0, 2] = (s[0][1] * s[1][2] - s[0][2] * s[1][1]) * d
    out[1, 0] = (s[1][2] * s[2][0] - s[1][0] * s[2][2]) * d
    out[1, 1] = (s[0][0] * s[2][2] - s[0][2] * s[2][0]) * d
    out[1, 2] = (s[0][2] * s[1][0] - s[0][0] * s[1][2]) * d
    out[2, 0] = (s[1][0] * s[2][1] - s[1][1] * s[2][0]) * d
    out[2, 1] = (s[0][1] * s[2][0] - s[0][0] * s[2][1]) * d
    out[2, 2] = (s[0][0] * s[1][1] - s[0][1] * s[1][0]) * d
    return out


class CalRecBoxes:
    """main's CalRecBoxes; ``crop_sizes`` stand in for the crop images (only
    their (h, w) is read)."""

    def __call__(self, crop_sizes, dt_boxes, txts, word_infos, return_single_char_box=False):
        word_results = []
        for idx, ((h, w), box) in enumerate(zip(crop_sizes, dt_boxes)):
            if txts is None or h * w == 0 or word_infos is None:
                continue
            img_box = np.array([[0, 0], [w, 0], [w, h], [0, h]])
            word_box_content_list, word_box_list, conf_list = self.cal_ocr_word_box(
                txts[idx], img_box, word_infos[idx], return_single_char_box
            )
            word_box_list = self.adjust_box_overlap(copy.deepcopy(word_box_list))
            direction = self.get_box_direction(box)
            word_box_list = self.reverse_rotate_crop_image(copy.deepcopy(box), word_box_list, direction)
            word_results.append(list(zip(word_box_content_list, conf_list, word_box_list)))
        return tuple(word_results)

    @staticmethod
    def get_box_direction(box: np.ndarray) -> Direction:
        edge_lengths = [
            float(np.linalg.norm(box[0] - box[1])),
            float(np.linalg.norm(box[1] - box[2])),
            float(np.linalg.norm(box[2] - box[3])),
            float(np.linalg.norm(box[3] - box[0])),
        ]
        width = max(edge_lengths[0], edge_lengths[2])
        height = max(edge_lengths[1], edge_lengths[3])
        if width < 1e-6:
            return Direction.VERTICAL
        aspect_ratio = round(height / width, 2)
        return Direction.VERTICAL if aspect_ratio >= 1.5 else Direction.HORIZONTAL

    def cal_ocr_word_box(self, rec_txt, bbox, word_info, return_single_char_box=False):
        if not rec_txt or word_info.line_txt_len == 0:
            return [], [], []

        bbox_points = quads_to_rect_bbox(bbox[None, ...])
        avg_col_width = (bbox_points[2] - bbox_points[0]) / word_info.line_txt_len

        is_all_en_num = all(v is WordType.EN_NUM for v in word_info.word_types)

        line_cols, char_widths, word_contents = [], [], []
        for word, word_col in zip(word_info.words, word_info.word_cols):
            if is_all_en_num and not return_single_char_box:
                line_cols.append(word_col)
                word_contents.append("".join(word))
            else:
                line_cols.extend(word_col)
                word_contents.extend(word)

            if len(word_col) == 1:
                continue

            avg_width = self.calc_avg_char_width(word_col, avg_col_width)
            char_widths.append(avg_width)

        avg_char_width = self.calc_all_char_avg_width(
            char_widths, bbox_points[0], bbox_points[2], len(rec_txt)
        )

        if is_all_en_num and not return_single_char_box:
            word_boxes = self.calc_en_num_box(line_cols, avg_char_width, avg_col_width, bbox_points)
        else:
            word_boxes = self.calc_box(line_cols, avg_char_width, avg_col_width, bbox_points)
        return word_contents, word_boxes, word_info.confs

    def calc_en_num_box(self, line_cols, avg_char_width, avg_col_width, bbox_points):
        results = []
        for one_col in line_cols:
            cur_word_cell = self.calc_box(one_col, avg_char_width, avg_col_width, bbox_points)
            x0, y0, x1, y1 = quads_to_rect_bbox(np.array(cur_word_cell))
            results.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return results

    @staticmethod
    def calc_box(line_cols, avg_char_width, avg_col_width, bbox_points):
        x0, y0, x1, y1 = bbox_points
        results = []
        for col_idx in line_cols:
            center_x = (col_idx + 0.5) * avg_col_width
            char_x0 = max(int(center_x - avg_char_width / 2), 0) + x0
            char_x1 = min(int(center_x + avg_char_width / 2), x1 - x0) + x0
            cell = [[char_x0, y0], [char_x1, y0], [char_x1, y1], [char_x0, y1]]
            results.append(cell)
        return sorted(results, key=lambda x: x[0][0])

    @staticmethod
    def calc_avg_char_width(word_col, each_col_width):
        char_total_length = (word_col[-1] - word_col[0]) * each_col_width
        return char_total_length / (len(word_col) - 1)

    @staticmethod
    def calc_all_char_avg_width(width_list, bbox_x0, bbox_x1, txt_len):
        if txt_len == 0:
            return 0.0
        if len(width_list) > 0:
            return sum(width_list) / len(width_list)
        return (bbox_x1 - bbox_x0) / txt_len

    @staticmethod
    def adjust_box_overlap(word_box_list):
        for i in range(len(word_box_list) - 1):
            cur, nxt = word_box_list[i], word_box_list[i + 1]
            if cur[1][0] > nxt[0][0]:
                distance = abs(cur[1][0] - nxt[0][0])
                cur[1][0] -= distance / 2
                cur[2][0] -= distance / 2
                nxt[0][0] += distance - distance / 2
                nxt[3][0] += distance - distance / 2
        return word_box_list

    def reverse_rotate_crop_image(self, bbox_points, word_points_list, direction):
        left = int(np.min(bbox_points[:, 0]))
        top = int(np.min(bbox_points[:, 1]))
        bbox_points[:, 0] = bbox_points[:, 0] - left
        bbox_points[:, 1] = bbox_points[:, 1] - top

        img_crop_width = int(np.linalg.norm(bbox_points[0] - bbox_points[1]))
        img_crop_height = int(np.linalg.norm(bbox_points[0] - bbox_points[3]))

        pts_std = np.array(
            [[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]]
        ).astype(np.float32)
        M = get_perspective_transform(bbox_points, pts_std)
        IM = invert3x3(M)

        new_word_points_list = []
        for word_points in word_points_list:
            new_word_points = []
            for point in word_points:
                new_point = point
                if direction == Direction.VERTICAL:
                    new_point = self.s_rotate(math.radians(-90), new_point[0], new_point[1], 0, 0)
                    new_point[0] = new_point[0] + img_crop_width

                p = np.array(new_point + [1])
                x, y, z = np.dot(IM, p)
                new_point = [x / z, y / z]

                new_point = [int(new_point[0] + left), int(new_point[1] + top)]
                new_word_points.append(new_point)
            new_word_points = self.order_points(new_word_points)
            new_word_points_list.append(new_word_points)
        return new_word_points_list

    @staticmethod
    def s_rotate(angle, valuex, valuey, pointx, pointy):
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
        sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
        return [sRotatex, sRotatey]

    @staticmethod
    def order_points(ori_box):
        def convert_to_1x2(p):
            if p.shape == (2,):
                return p.reshape((1, 2))
            if p.shape == (1, 2):
                return p
            return p[:1, :]

        box = np.array(ori_box).reshape((-1, 2))
        center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
        if np.any(box[:, 0] == center_x) and np.any(box[:, 1] == center_y):
            p1 = box[np.where(box[:, 0] == np.min(box[:, 0]))]
            p2 = box[np.where(box[:, 1] == np.min(box[:, 1]))]
            p3 = box[np.where(box[:, 0] == np.max(box[:, 0]))]
            p4 = box[np.where(box[:, 1] == np.max(box[:, 1]))]
        elif np.all(box[:, 0] == center_x):
            y_sort = np.argsort(box[:, 1])
            p1 = box[y_sort[0]]
            p2 = box[y_sort[1]]
            p3 = box[y_sort[2]]
            p4 = box[y_sort[3]]
        elif np.any(box[:, 0] == center_x) and np.all(box[:, 1] != center_y):
            p12, p34 = (box[np.where(box[:, 1] < center_y)], box[np.where(box[:, 1] > center_y)])
            p1, p2 = (
                p12[np.where(p12[:, 0] == np.min(p12[:, 0]))],
                p12[np.where(p12[:, 0] == np.max(p12[:, 0]))],
            )
            p3, p4 = (
                p34[np.where(p34[:, 0] == np.max(p34[:, 0]))],
                p34[np.where(p34[:, 0] == np.min(p34[:, 0]))],
            )
        else:
            p14, p23 = (box[np.where(box[:, 0] < center_x)], box[np.where(box[:, 0] > center_x)])
            p1, p4 = (
                p14[np.where(p14[:, 1] == np.min(p14[:, 1]))],
                p14[np.where(p14[:, 1] == np.max(p14[:, 1]))],
            )
            p2, p3 = (
                p23[np.where(p23[:, 1] == np.min(p23[:, 1]))],
                p23[np.where(p23[:, 1] == np.max(p23[:, 1]))],
            )

        p1 = convert_to_1x2(p1)
        p2 = convert_to_1x2(p2)
        p3 = convert_to_1x2(p3)
        p4 = convert_to_1x2(p4)
        return np.array([p1, p2, p3, p4]).reshape((-1, 2)).tolist()
