import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, grey_closing
o=np.array(Image.open('original.png').convert('RGBA'))
mask=o[:,:,3]>180
G=o[:,:,1].astype(float)
H=1024; baseG=np.full(H,np.nan)
for y in range(H):
    m=mask[y]
    if m.sum()>3: baseG[y]=np.percentile(G[y,m],35)
# fill nan
import numpy as np
idx=np.where(~np.isnan(baseG))[0]
baseG=np.interp(np.arange(H),idx,baseG[idx])
resid=np.clip((G-baseG[:,None]),0,None)*mask
# Clean highlight core: residual high
def clean(m, close=2, op=1):
    m=binary_closing(m,iterations=close)
    m=binary_fill_holes(m)
    if op: m=binary_opening(m,iterations=op)
    return m
strong = clean(mask&(resid>48), close=2, op=1)
broad  = clean(mask&(resid>15), close=3, op=1)
for nm,mm in [('strong',strong),('broad',broad)]:
    Image.fromarray((255-mm*255).astype('uint8')).save(f'mask_{nm}.pbm')
    print(nm,'frac',round(mm.sum()/mask.sum(),3))
