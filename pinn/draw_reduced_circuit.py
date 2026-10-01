"""Draw the implemented reduced manual-mode circuit; no hardware-netlist claim.

SVG and PNG share the same drawing commands. Values/topology follow
src/ht1b/equations.py, config.py and docs/EQUATIONS.md.
"""
from pathlib import Path
import html
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output'
W,H = 1800,1250
BG='#f7f6f1'; INK='#203431'; MUTED='#5c6a65'; LINE='#cbd3cc'
AUDIO='#22675b'; CONTROL='#326a9d'; GOLD='#916122'; WHITE='#ffffff'
im=Image.new('RGB',(W,H),BG); dr=ImageDraw.Draw(im)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
     '<title id="title">HT-1B 目前程式的簡化等效電路</title>',
     '<desc id="desc">Manual 模式。輸入經 Rs 到節點 a，再經 Ratio 電阻到 b。a 的 Rt 及 b 的 Rg、GRE 均接地。a 的整流訊號控制 C3 電壓 e，進一步產生光學快慢狀態 f、s 及 GRE 導電度。b 經無記憶放大器產生 Wet。虛線為控制關係，並非導線。</desc>',
     f'<rect width="{W}" height="{H}" fill="{BG}"/>']
fonts={}
def font(size,bold=False):
    key=(size,bold)
    if key not in fonts:
        fonts[key]=ImageFont.truetype('C:/Windows/Fonts/msjhbd.ttc' if bold else 'C:/Windows/Fonts/msjh.ttc',size)
    return fonts[key]
def text(x,y,s,size=25,color=INK,bold=False,anchor='left'):
    s=s.replace('−','-')  # Microsoft JhengHei lacks the mathematical minus glyph.
    f=font(size,bold)
    width=dr.textlength(s,font=f)
    left=x-width/2 if anchor=='middle' else x-width if anchor=='right' else x
    assert left>=0 and left+width<=W, (s,left,width)
    dr.text((left,y),s,font=f,fill=color,anchor='lt')
    ascent,_=f.getmetrics()
    # Render text with a top-aligned SVG baseline; preserve editable text.
    svg.append(f'<text x="{x}" y="{y}" dominant-baseline="text-before-edge" text-anchor="'+
        {'left':'start','middle':'middle','right':'end'}[anchor]+f'" font-family="Microsoft JhengHei, Noto Sans CJK TC, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(s)}</text>')
def path(points,color=INK,width=3,dashed=False):
    if dashed:
        for a,b in zip(points,points[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]; length=math.hypot(dx,dy)
            for i in range(0,int(length),15):
                p=i/length; q=min(i+8,length)/length
                dr.line((a[0]+dx*p,a[1]+dy*p,a[0]+dx*q,a[1]+dy*q),fill=color,width=width)
    else: dr.line(points,fill=color,width=width,joint='curve')
    svg.append(f'<polyline points="'+ ' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"'+(' stroke-dasharray="8 7"' if dashed else '')+'/>')
def arrow(points,color=CONTROL,dashed=False,width=3):
    path(points,color,width,dashed)
    x,y=points[-1]; px,py=points[-2]; theta=math.atan2(y-py,x-px)
    tri=[(x,y),(x-13*math.cos(theta-.45),y-13*math.sin(theta-.45)),(x-13*math.cos(theta+.45),y-13*math.sin(theta+.45))]
    dr.polygon(tri,fill=color)
    svg.append(f'<polygon points="'+ ' '.join(f'{a},{b}' for a,b in tri)+f'" fill="{color}"/>')
def box(x,y,w,h,fill=WHITE,stroke=LINE):
    dr.rounded_rectangle((x,y,x+w,y+h),radius=12,fill=fill,outline=stroke,width=2)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
def node(x,y,color=AUDIO):
    dr.ellipse((x-5,y-5,x+5,y+5),fill=color)
    svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
def ground(x,y,color=AUDIO):
    for yy,r in ((y,19),(y+8,12),(y+16,5)):path([(x-r,yy),(x+r,yy)],color,3)
def resistor(x1,y1,x2,y2,color=AUDIO):
    # IEC rectangular resistor, 75 px body centered on the connection.
    if y1==y2:
        mid=(x1+x2)/2
        path([(x1,y1),(mid-38,y1)],color)
        path([(mid+38,y1),(x2,y2)],color)
        coords=(mid-38,y1-13,mid+38,y1+13)
    else:
        mid=(y1+y2)/2
        path([(x1,y1),(x1,mid-38)],color)
        path([(x1,mid+38),(x2,y2)],color)
        coords=(x1-13,mid-38,x1+13,mid+38)
    dr.rectangle(coords,fill=BG,outline=color,width=3)
    a,b,c,d=coords; svg.append(f'<rect x="{a}" y="{b}" width="{c-a}" height="{d-b}" fill="{BG}" stroke="{color}" stroke-width="3"/>')

text(70,45,'HT-1B｜目前模型的簡化等效電路',43,bold=True)
text(72,111,'Manual 模式 · 依現有程式繪製；這是灰箱模型，不是完整 CL-1B 維修電路圖。',25,MUTED)
path([(72,169),(145,169)],AUDIO,4);text(158,153,'音訊／電氣連接',22,AUDIO)
path([(435,169),(510,169)],CONTROL,3,True);text(523,153,'偵測／控制關係（非導線）',22,CONTROL)
text(1295,153,'狀態：e、f、s',23,GOLD,bold=True)
path([(70,204),(1730,204)],LINE,2)
text(72,232,'01  音訊路徑與光控衰減',27,AUDIO,True)

# Passive front network: two nodes, with explicit shunt branches.
path([(115,340),(210,340)],AUDIO,4)
node(115,340);text(115,288,'Dry / u',25,AUDIO,True,anchor='middle')
resistor(210,340,345,340)
text(277,291,'Rs 100 kΩ',23,AUDIO,anchor='middle')
path([(345,340),(550,340)],AUDIO,4)
node(400,340);node(470,340)
text(470,291,'a',28,AUDIO,True,anchor='middle')
resistor(550,340,710,340)
text(630,282,'Rratio',24,AUDIO,True,anchor='middle')
text(630,377,'0–10 kΩ',22,MUTED,anchor='middle')
path([(710,340),(1210,340)],AUDIO,4)
node(815,340);node(1050,340)
text(910,292,'b',28,AUDIO,True,anchor='middle')
resistor(470,340,470,510);ground(470,510)
text(501,417,'Rt',23,AUDIO,True);text(501,453,'100 kΩ',22,MUTED)
resistor(815,340,815,510);ground(815,510)
text(846,417,'Rg',23,AUDIO,True);text(846,453,'100 kΩ',22,MUTED)
resistor(1050,340,1050,510);ground(1050,510)
arrow([(1021,465),(1078,389)],AUDIO,width=2)
text(1100,442,'GRE',24,AUDIO,True)
text(1100,474,'等效可變電阻',20,MUTED)

box(1210,275,330,133,stroke=AUDIO)
text(1375,294,'Make-up＋輸出級',26,AUDIO,True,'middle')
text(1375,335,'增益與 tanh 限幅',23,INK,anchor='middle')
text(1375,369,'無記憶近似',19,MUTED,anchor='middle')
arrow([(1540,340),(1700,340)],AUDIO,width=4)
text(1650,287,'Wet / y',25,AUDIO,True,anchor='middle')

# Detector branches from node a BEFORE Rratio, as in evaluate().
arrow([(400,340),(400,560),(45,560),(45,754),(95,754)],CONTROL,True)
text(72,607,'02  側鏈控制與三個物理狀態',27,CONTROL,True)
box(95,669,350,182,stroke=CONTROL)
text(270,691,'整流／Threshold',27,CONTROL,True,'middle')
text(270,741,'取 |a|，調整偵測靈敏度',23,INK,anchor='middle')
text(270,789,'得到控制目標 d',23,MUTED,anchor='middle')
arrow([(445,754),(555,754)],CONTROL,True)
text(500,711,'d',23,CONTROL,True,'middle')

box(555,669,550,182,stroke=CONTROL)
text(580,690,'Attack／Release 包絡',26,CONTROL,True)
text(580,740,'充電 iA ／放電 iR',23,INK)
text(580,779,'C3 · de/dt = iA − iR',23,GOLD,True)
node(989,731,CONTROL)
text(1017,709,'e',27,GOLD,True)
path([(989,731),(989,765)],CONTROL)
path([(966,765),(1012,765)],CONTROL)
path([(966,777),(1012,777)],CONTROL)
path([(989,777),(989,820)],CONTROL);ground(989,820,CONTROL)
text(1019,767,'C3',21,CONTROL,True);text(1019,797,'10 μF',18,MUTED)
arrow([(1105,754),(1240,754)],CONTROL,True)
text(1172,711,'e → q',22,CONTROL,True,'middle')

box(1240,669,490,182,stroke=CONTROL)
text(1485,691,'光學元件的雙時間尺度模型',25,CONTROL,True,'middle')
text(1485,741,'f：快反應　／　s：慢反應',25,GOLD,True,'middle')
text(1485,789,'合成導電度 G_GRE',23,INK,anchor='middle')
arrow([(1485,669),(1485,560),(1237,560),(1237,425),(1067,425)],CONTROL,True)
text(1380,522,'G_GRE ↑ → 衰減 ↑',23,CONTROL,True,'middle')

text(95,887,'C3 支路保留二極體、Attack 電阻及帶偏壓的 Release 等效回路；圖中以充／放電電流合併表示。',22,MUTED)
path([(70,941),(1730,941)],LINE,2)
text(72,963,'03  PINN 約束：讓預測的 e、f、s 符合這三條動態方程',27,GOLD,True)
for x,w,title,formula in [
    (72,525,'電容電壓 e','C3 · de/dt − iA + iR = 0'),
    (622,525,'快光學狀態 f','τf · df/dt + f − q = 0'),
    (1172,558,'慢光學狀態 s','τs · ds/dt + s − q = 0')]:
    box(x,1013,w,103,fill='#efeee5')
    text(x+22,1029,title,23,GOLD,True)
    text(x+22,1070,formula,24,INK)
text(72,1143,'模型近似：输入／輸出變壓器、真空管內部、噪聲與 C8 的快速動態未展開；GRE 採可辨識的等效模型。'.replace('输入','輸入'),22,MUTED)
text(72,1183,'依據：src/ht1b/equations.py · config.py · physicsnemo_model.py ｜ 電阻值為設定值，非本次學得參數。',21,MUTED)

svg.append('</svg>')
OUT.mkdir(exist_ok=True)
svg_path=OUT/'cl1b-reduced-circuit.svg'; png_path=OUT/'cl1b-reduced-circuit.png'
svg_path.write_text('\n'.join(svg),encoding='utf-8')
im.save(png_path)
print(svg_path);print(png_path)
