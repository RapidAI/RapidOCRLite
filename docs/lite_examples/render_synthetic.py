from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, io, textwrap
SANS='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; SERIF='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
SB='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
def f(p,s): return ImageFont.truetype(p,s,index=2)  # SC
def wrap(d,text,font,width):
    lines=[];cur=''
    for ch in text:
        if d.textlength(cur+ch,font=font)>width: lines.append(cur);cur=ch
        else: cur+=ch
    if cur: lines.append(cur)
    return lines
# --- sample A: two-column report + table, scanned look
W,H=1240,1600; im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im)
d.text((W//2,70),'城市公共交通年度运行简报（2025）',font=f(SB,40),fill='black',anchor='mm')
d.text((W//2,125),'编制：示例数据组　　发布日期：2026年3月15日',font=f(SANS,22),fill=(60,60,60),anchor='mm')
col_l="一、总体情况。全年公共交通客运总量为18.6亿人次，同比增长4.2%。其中轨道交通客运量占比首次超过55%，常规公交客运量同比下降1.8%。早高峰平均出行时间为38分钟，较上年缩短2分钟。二、线网建设。年内新开通轨道交通线路2条，新增运营里程41.5公里；优化调整公交线路63条，新辟社区微循环线路17条，撤销重复线路9条。"
col_r="三、服务质量。乘客满意度调查共回收有效问卷12,480份，综合满意度为87.3分。投诉主要集中在发车间隔不稳定（占32%）和换乘指引不清晰（占21%）两方面。四、绿色出行。新能源公交车占比达到96.4%，全年减少二氧化碳排放约21万吨。下一步将推进站点无障碍改造，并试点需求响应式公交服务。"
fb=f(SERIF,26); cw=540
for x0,t in ((60,col_l),(640,col_r)):
    y=180
    for ln in wrap(d,t,fb,cw): d.text((x0,y),ln,font=fb,fill='black'); y+=42
d.text((60,640),'表1　主要运营指标',font=f(SB,26),fill='black')
rows=[['指标','2024年','2025年','变化'],['客运总量（亿人次）','17.85','18.60','+4.2%'],['轨道交通里程（km）','512.3','553.8','+41.5'],['公交线路数（条）','1,208','1,262','+54'],['平均候车时间（min）','6.9','6.1','-0.8'],['准点率','91.2%','93.5%','+2.3pt']]
x=[60,500,740,960,1180]; y0=690; rh=56; ft=f(SANS,24)
for i,r in enumerate(rows):
    for j,c in enumerate(r): d.text((x[j]+14,y0+i*rh+rh//2),c,font=ft,fill='black',anchor='lm')
for i in range(len(rows)+1): d.line((x[0],y0+i*rh,x[-1],y0+i*rh),fill='black',width=2)
for xx in x: d.line((xx,y0,xx,y0+len(rows)*rh),fill='black',width=2)
y=y0+len(rows)*rh+50
for ln in wrap(d,"注：本简报数据为示例数据，仅用于演示 OCR 效果，不代表任何真实城市。准点率按到站时间偏差不超过 3 分钟统计（on-time rate, ±3 min）。",f(SANS,22),1120):
    d.text((60,y),ln,font=f(SANS,22),fill=(40,40,40)); y+=36
im=im.crop((0,0,W,y+40))
im=im.rotate(2.5,resample=Image.BICUBIC,expand=True,fillcolor=(235,233,228))
a=np.asarray(im).astype(np.float32); a=a*np.array([0.97,0.96,0.92])+np.random.default_rng(0).normal(0,9,a.shape)
im=Image.fromarray(a.clip(0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
im.save('synthetic_scan_report.jpg',quality=55)
# --- sample B: dense small mixed zh/en
W=1100; im=Image.new('RGB',(W,1000),'white'); d=ImageDraw.Draw(im)
d.text((40,30),'RapidOCRLite 部署说明 / Deployment Notes v5.0',font=f(SB,26),fill='black')
paras=["1. 安装：执行 pip install rapidocrlite 即可，无需 ONNX Runtime、OpenCV 或 PaddlePaddle；Python 3.9–3.13 均受支持。",
"2. 线程：通过环境变量 PPOCR_BENCH_THREADS=4 设置推理线程数；在 Apple M4 上可设置 PPOCR_INDRA_METAL=2 启用 Metal GPU。",
"3. 模型选择：Det.model_type 与 Rec.model_type 设为 \"lite\" 即使用内置 lite 模型，权重许可证见 models/lite/LICENSE（OpenRAIL-M）。",
"4. C API 示例：rpocr_options o; rpocr_options_init(&o); o.model_type = RPOCR_MODEL_LITE; rpocr_create(dir, NULL, NULL, &o);",
"5. 常见问题（FAQ）：Q: 为什么第一次运行较慢？A: 首次调用会分配线程池和工作缓冲区（约 30–80 ms），之后的调用不再重复分配。",
"6. 性能参考：x86 AVX-512 单页约 0.27 s（4 线程），M4 Metal 约 0.09 s；内存峰值约 0.8–1.2 GB，取决于页面尺寸与文字行数。",
"7. 已知限制：不支持竖排文字与手写体；对旋转超过 ±10° 的页面需先做方向校正（deskew 仅覆盖小角度）。"]
fs=f(SANS,17); y=80
for p in paras:
    for ln in wrap(d,p,fs,W-80): d.text((40,y),ln,font=fs,fill='black'); y+=27
    y+=8
im.crop((0,0,W,y+20)).save('synthetic_dense_mixed.png',optimize=True)
