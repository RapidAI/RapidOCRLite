# lite_examples 图片来源与许可

| 文件 | 来源 | 许可 |
|---|---|---|
| ch_en_num.jpg | RapidAI/RapidOCR `python/tests/test_files/ch_en_num.jpg` | 随 RapidOCR 仓库以 Apache-2.0 分发 |
| check_return_word_len.jpeg | RapidAI/RapidOCR `python/tests/test_files/check_return_word_len.jpeg` | 同上 |
| return_word_debug.jpg | RapidAI/RapidOCR `python/tests/test_files/return_word_debug.jpg` | 同上 |
| text_vertical_words.jpg | RapidAI/RapidOCR `python/tests/test_files/text_vertical_words.png`（转为 JPEG q80） | 同上 |
| synthetic_scan_report.jpg | 本目录 `render_synthetic.py` 生成（内容为虚构示例数据） | 与本仓库相同（Apache-2.0）；所用字体 Noto CJK 为 SIL OFL-1.1 |
| synthetic_dense_mixed.png | 同上 | 同上 |

说明：前四张图来自 RapidOCR 测试集，原始出处（截图/拍照）由 RapidOCR 仓库收录，这里沿用其仓库许可。
`images/*_ppocr_boxes.jpg`、`images/*_lite_boxes.jpg` 和 `results/*.json` 由 `run_compare.py` 生成。
lite 的识别结果属于 lite 模型的"输出"，按 `models/lite/LICENSE` 第 8 条适用该许可证（OpenRAIL-M）。

颜色约定：`run_compare.py` 与 CI 检查（`python/tests/ci_lite_examples_check.py`）传入的是**真 RGB** 的
ndarray；RapidOCR 3.x 的 Python 入口按 OpenCV 惯例把 ndarray 视为 BGR，会在内部交换 R/B 通道
（`LoadImage.convert_img`）。因此 `results/*.json` 记录的是**引擎实际看到的 R/B 交换图像**的识别结果。
重新生成基线时务必保持同样的输入约定（直接传真 RGB、不要预先交换通道），否则全部基线都会变化。
详见 issue #1。
