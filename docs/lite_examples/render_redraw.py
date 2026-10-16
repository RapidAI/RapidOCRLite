"""白描图: redraw recognized text in black on a white canvas at each box position.
Usage: python render_redraw.py   (reads results/*.json + images/, writes images/*_{model}_redraw.jpg)"""
import json, glob, math, os
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.environ.get('REDRAW_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
def font(sz): return ImageFont.truetype(FONT, max(6, int(sz)), index=2)  # index 2 = SC
def text_img(text, h, w):
    """Render text into an RGBA strip of height h, scaled to width w."""
    f = font(h * 0.85); bb = f.getbbox(text)
    tw, th = max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
    im = Image.new('L', (tw + 2, int(h)), 0)
    ImageDraw.Draw(im).text((1 - bb[0], (h - th) / 2 - bb[1]), text, font=f, fill=255)
    return im.resize((max(1, int(w)), max(1, int(h))), Image.LANCZOS)
def vertical_img(text, h, w):
    chars = [c for c in text if not c.isspace()] or [' ']
    cs = min(w, h / len(chars)); f = font(cs * 0.9)
    im = Image.new('L', (max(1, int(w)), max(1, int(h))), 0); d = ImageDraw.Draw(im)
    step = h / len(chars)
    for i, c in enumerate(chars): d.text((w / 2, i * step + step / 2), c, font=f, fill=255, anchor='mm')
    return im
def redraw(img_path, res_path, out_path):
    W, H = Image.open(img_path).size
    canvas = Image.new('L', (W, H), 255)
    for l in json.load(open(res_path))['lines']:
        t = l['text'].strip()
        if not t: continue
        p = l['box']  # [tl, tr, br, bl]
        wl = math.dist(p[0], p[1]); hl = math.dist(p[1], p[2])
        cx = sum(q[0] for q in p) / 4; cy = sum(q[1] for q in p) / 4
        if hl > 1.5 * wl and len(t) > 1:
            m = vertical_img(t, hl, wl); ang = math.degrees(math.atan2(p[3][1] - p[0][1], p[3][0] - p[0][0])) - 90
        else:
            m = text_img(t, hl, wl); ang = math.degrees(math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]))
        if abs(ang) > 0.3: m = m.rotate(-ang, resample=Image.BICUBIC, expand=True)
        canvas.paste(0, (int(round(cx - m.width / 2)), int(round(cy - m.height / 2))), m)
    canvas.convert('RGB').save(out_path, quality=85, optimize=True)
if __name__ == '__main__':
    for rp in sorted(glob.glob(os.path.join(HERE, 'results', '*.json'))):
        r = json.load(open(rp)); base = os.path.basename(rp)[:-5]
        redraw(os.path.join(HERE, 'images', r['image']), rp, os.path.join(HERE, 'images', base + '_redraw.jpg'))
        print(base)
