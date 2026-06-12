import cairosvg, numpy as np, sys
from PIL import Image
def render(svg='logo.svg', out='render.png'):
    cairosvg.svg2png(url=svg, write_to=out, output_width=1024, output_height=1024)
def compose(p):
    # composite RGBA over white for fair comparison
    im=Image.open(p).convert('RGBA')
    bg=Image.new('RGBA',im.size,(255,255,255,255))
    return np.array(Image.alpha_composite(bg,im).convert('RGB')).astype(int)
render()
o=compose('original.png'); r=compose('render.png')
d=np.abs(o-r).sum(2)
print('mean abs diff:', d.mean(), ' max:', d.max(), ' frac px>30:', (d>30).mean())
# save diff heatmap
hm=np.clip(d,0,255).astype('uint8')
Image.fromarray(hm).resize((512,512)).save('diff.png')
# also side render scaled
Image.open('render.png').convert('RGB').resize((512,512)).save('render_512.png')
