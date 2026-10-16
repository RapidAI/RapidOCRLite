import json, time, sys, os, statistics
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr import RapidOCR
M=os.environ.get('PPOCR_SMALL_DIR','models')  # PP-OCRv6_det_small.onnx, PP-OCRv6_rec_small.onnx, ppocrv6_dict.txt
engines={
 'ppocr': RapidOCR(params={'Det.model_path':f'{M}/PP-OCRv6_det_small.onnx','Rec.model_path':f'{M}/PP-OCRv6_rec_small.onnx','Rec.rec_keys_path':f'{M}/ppocrv6_dict.txt','Global.log_level':'error','EngineConfig.onnxruntime.intra_op_num_threads':4}),
 'lite': RapidOCR(params={'Det.model_type':'lite','Rec.model_type':'lite','Global.log_level':'error','EngineConfig.onnxruntime.intra_op_num_threads':4}),
}
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',14)
out={}
for fn in sys.argv[1:]:
    name=os.path.splitext(fn)[0]; img=np.asarray(Image.open(fn).convert('RGB'))
    for k,e in engines.items():
        e(img, use_cls=False)
        ts=[]
        for _ in range(3):
            t=time.perf_counter(); r=e(img, use_cls=False); ts.append((time.perf_counter()-t)*1000)
        boxes=[] if r.boxes is None else r.boxes.tolist(); txts=list(r.txts or []); sc=[float(s) for s in (r.scores or [])]
        rec={'image':fn,'model':k,'time_ms_median_of_3':round(statistics.median(ts),1),'threads':int(os.environ.get('PPOCR_BENCH_THREADS','0')),
             'lines':[{'order':i+1,'box':[[round(x,1),round(y,1)] for x,y in b],'text':t,'score':round(s,4)} for i,(b,t,s) in enumerate(zip(boxes,txts,sc))]}
        json.dump(rec,open(f'out/{name}_{k}.json','w'),ensure_ascii=False,indent=1)
        vis=Image.open(fn).convert('RGB'); d=ImageDraw.Draw(vis)
        col=(230,40,40) if k=='lite' else (30,90,230)
        lw=max(2,round(max(vis.size)/600))
        for i,b in enumerate(boxes):
            d.polygon([tuple(p) for p in b],outline=col,width=lw)
            x,y=min(p[0] for p in b),min(p[1] for p in b)
            s=str(i+1); tw=d.textlength(s,font=FONT)
            d.rectangle((x,y-16,x+tw+4,y),fill=col); d.text((x+2,y-16),s,font=FONT,fill='white')
        vis.save(f'out/{name}_{k}_boxes.jpg',quality=80)
        out[(name,k)]=rec; print(name,k,len(boxes),rec['time_ms_median_of_3'],flush=True)
