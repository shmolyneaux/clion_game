import re
def parse_potrace(fn):
    txt=open(fn).read()
    return [re.findall(r'[MmLlZzCc]|-?\d+\.?\d*', d.replace('\n',' '))
            for d in re.findall(r'<path d="(.*?)"', txt, re.S)]
def transform(toks):
    res=[]; i=0; cx=cy=0; cmd=None
    def emit_abs(): res.append(f'{0.1*cx:.2f},{1024-0.1*cy:.2f}')
    while i<len(toks):
        t=toks[i]
        if t in 'MmLlZzCc':
            cmd=t; i+=1
            if t in 'Zz': res.append('Z')
            elif t=='C': res.append('C')  # absolute cubic
            continue
        if cmd in ('C',):
            # 3 coordinate pairs absolute
            pts=[]
            for k in range(3):
                x=float(toks[i]); y=float(toks[i+1]); i+=2
                pts.append(f'{0.1*x:.2f},{1024-0.1*y:.2f}')
                cx,cy=x,y
            res.append('C'+' '.join(pts))
            continue
        if cmd in ('c',):
            pts=[]
            bx,by=cx,cy
            for k in range(3):
                x=float(toks[i]); y=float(toks[i+1]); i+=2
                ax=bx+x; ay=by+y
                pts.append(f'{0.1*ax:.2f},{1024-0.1*ay:.2f}')
                if k==2: cx,cy=ax,ay
            res.append('C'+' '.join(pts))
            continue
        x=float(toks[i]); y=float(toks[i+1]); i+=2
        if cmd=='M': cx,cy=x,y; res.append('M'); emit_abs(); cmd='L'
        elif cmd=='m': cx+=x;cy+=y; res.append('M'); emit_abs(); cmd='l'
        elif cmd=='L': cx,cy=x,y; res.append('L'); emit_abs()
        elif cmd=='l': cx+=x;cy+=y; res.append('L'); emit_abs()
    return ''.join(res)
def paths_d(fn):
    return ' '.join(transform(p) for p in parse_potrace(fn))

base_d=paths_d('trace_main.svg')
mid_d=paths_d('trace_mid.svg')
hl_d=paths_d('trace_hl.svg')

base_grad='''<linearGradient id="gb" gradientUnits="userSpaceOnUse" x1="512" y1="149" x2="512" y2="810">
<stop offset="0.00" stop-color="#EC85FF"/>
<stop offset="0.08" stop-color="#E377FF"/>
<stop offset="0.30" stop-color="#D761FF"/>
<stop offset="0.53" stop-color="#CC55FF"/>
<stop offset="0.76" stop-color="#A342FF"/>
<stop offset="1.00" stop-color="#8E41FF"/>
</linearGradient>'''
mid_grad='''<linearGradient id="gm" gradientUnits="userSpaceOnUse" x1="512" y1="149" x2="512" y2="810">
<stop offset="0.00" stop-color="#F5A2FF"/>
<stop offset="0.23" stop-color="#F07EFF"/>
<stop offset="0.46" stop-color="#CA75FF"/>
<stop offset="0.72" stop-color="#B266FF"/>
<stop offset="1.00" stop-color="#A25BFF"/>
</linearGradient>'''
hl_grad='''<linearGradient id="gh" gradientUnits="userSpaceOnUse" x1="512" y1="149" x2="512" y2="810">
<stop offset="0.00" stop-color="#F8D2FF"/>
<stop offset="0.17" stop-color="#F7CDFF"/>
<stop offset="0.35" stop-color="#F5C9FF"/>
<stop offset="0.55" stop-color="#E097FF"/>
<stop offset="0.75" stop-color="#C77FFF"/>
<stop offset="1.00" stop-color="#C074FF"/>
</linearGradient>'''

svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
<defs>{base_grad}{mid_grad}{hl_grad}</defs>
<path fill-rule="evenodd" fill="url(#gb)" d="{base_d}"/>
<path fill-rule="evenodd" fill="url(#gm)" d="{mid_d}"/>
<path fill-rule="evenodd" fill="url(#gh)" d="{hl_d}"/>
</svg>'''
open('logo.svg','w').write(svg)
print('wrote', len(svg))
