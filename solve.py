import ps_lib as ps
import numpy as np
import scipy.ndimage

# for some reason the code wouldn't run without the numpy & scipy imports here, 
# even though they are imported in ps_lib.py, I don't know why, but it only runs if 
# I add them here so Im just gonna leave it :b

# Q1 Work

# loads in img
im = ps.read_image("../ps3/light-field.png")

# shape
h, w, c = im.shape

# each lenset is 16 x 16
# s & t are coords of the lenset array
s = h // 16
t = w // 16

# u & v are coords on the lenslet’s 16 x 16 sensor block
# rearrange pixels into (u, v, s, t, c)
# & split into 16 x 16 blocks & reorders the axes
lf = np.transpose(im.reshape(s, 16, t, 16, c), (1, 3, 0, 2, 4))

# u & v go from -7 to +8 so shift by +7
# save slice of the array for (u, v) coords (-5, -2)
ui = -5 + 7
vi = -2 + 7
np.save("q1-light-field-slice-a.npy", lf[ui, vi])

# save slice for (u, v) coords (-1, +4)
ui = -1 + 7
vi = 4 + 7
np.save("q1-light-field-slice-b.npy", lf[ui, vi])

# creates all sub aperture views, fixing (u, v) gives L(u, v, s, t, c)
# & arrange them into a 2D mosaic

mos = np.zeros((16 * s, 16 * t, c))

# loop over all u & v views
for u in range(16):
    for v in range(16):
        
        # the sub aperture image at (u, v)
        sub = lf[u, v]
        
        # place each sub image into the correct position in the mosaic
        r = u * s
        col = v * t
        
        # copy into mosaic
        mos[r:r + s, col:col + t] = sub

# save the mosaic image
ps.write_image("q1-sub-aperture-views.png", mos)

# Q2 Work

# just makes th focal stack I(s, t, c, d) for d from -2 to 1
#-2, -1.7, -1.4, -1.1, -0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 1, so 11 depths 
ds = np.linspace(-2, 1, 11)

# just using this to load focal stack images for Q3 & Q4
stk = []

# loops over each depth
for i, d in enumerate(ds):

    # just stores the refocused image for this depth
    out = np.zeros_like(lf[0, 0])

    # sum over all sub aperture views
    for u in range(16):
        for v in range(16):

            # convert index to u, v in range -7- +8
            up = u - 7
            vp = v - 7

            # one light field view L(u, v, s, t, c)
            sub = lf[u, v]

            # shift using s - d * u& t + d * v
            sh_s = d * up
            sh_t = -d * vp

            # shift s & t only, not color
            sh = scipy.ndimage.shift(sub, (sh_s, sh_t, 0), order=1, mode="nearest")

            # add into sum
            out += sh

    # avg over all views
    out = out / (16 * 16)

    # save each depth image
    ps.write_image("q2-depth-" + str(i).zfill(2) + ".png", out)

    # store this refocused image for later use in Q3 & Q4 using the empyty list I made above
    stk.append(out)

# convert list to array to index it 
stk = np.array(stk)


# Q3 Work

# weights w(s, t, d)
wt = np.zeros(stk.shape[:3])

# compute weights from sharpness
for i in range(11):

    # this is just to convert to grayscale
    g = np.mean(stk[i], axis=2)

    # high frequency part, I_HF
    hf = g - (scipy.ndimage.gaussian_filter(g, sigma=2, mode="nearest"))

    # (I_HF)^2 & blur with sigma 8
    wt[i] = scipy.ndimage.gaussian_filter(hf * hf, sigma=8, mode="nearest")

# sum of weights
den = np.sum(wt, axis=0)


# compute all in focus image
top = np.zeros_like(stk[0])

for i in range(11):
    top += stk[i] * wt[i, :, :, None]

# I divide weighted sum by tot weights to get the fin all in focus image
ps.write_image("q3.png", top / den[:, :, None])


# Q4 Work

# just computes depth map using same weights
num = np.zeros_like(den)

for i, d in enumerate(ds):
    num += wt[i] * d

# divides weighted sum of depths by tot weights to get depth at each pxl
dep = num / den

# scale to 0-1 to write the img
dep_im = (dep - dep.min()) / (dep.max() - dep.min())

ps.write_image("q4.png", dep_im)


# Q5 Work

# I set the duration of the capture-video to 9 & it loaded 107 frames onto my computer from the pi
fr = []

for i in range(107):
    # each of the vids
    data = np.load("q5-video/" + str(i).zfill(6) + ".npz")

    # each of the npzs has a frame stored in data
    im = np.float32(data["data"])

    # convert to 0-1 so the image comes out looking right
    if im.max() > 1:
        im = im / 255

    fr.append(im)

fr = np.array(fr)

# saved a couble of frames for the report
ps.write_image("q5-frame-00.png", fr[0])
ps.write_image("q5-frame-20.png", fr[20])
ps.write_image("q5-frame-40.png", fr[40])
ps.write_image("q5-frame-60.png", fr[60])
ps.write_image("q5-frame-80.png", fr[80])
ps.write_image("q5-frame-106.png", fr[106])

# use middle frame of the video sequence
mid = len(fr) // 2
mid_im = fr[mid]

# convert to grayscale 
# it said I & P should be grayscale so I just converted it here
gmid = np.mean(mid_im, axis=2)

# select a small square neighborhood P around the part to be in focus
sz = 93
y0 = 200
x0 = 300

# save template patch P
patch = mid_im[y0:y0+sz, x0:x0+sz]
ps.write_image("q5-template.png", patch)

# grayscale template patch P
P = gmid[y0:y0+sz, x0:x0+sz]

# subtract mean value of P
P_ = P - np.mean(P)

# denom for normalized cross corr
Pd = np.sqrt(np.sum(P_ * P_))

sh = []

rad = 80

# for each frame compute match quality image I_MQ(t, u, v)
for i in range(len(fr)):

    # grayscale frrame
    g = np.mean(fr[i], axis=2)

    best_val = -1
    best = (y0, x0)

    # slide patch over image
    for u in range(max(0, y0-rad), min(g.shape[0]-sz+1, y0+rad)):
        for v in range(max(0, x0-rad), min(g.shape[1]-sz+1, x0+rad)):

            # window same size as P
            W = g[u:u+sz, v:v+sz]

            # subtract mean of window
            W_ = W - np.mean(W)

            # normalized cross correlation
            num = np.sum(P_ * W_)
            # had to add in the 1e-10, bc it wouldnt run
            # bc of 0 in the denom
            den = Pd * np.sqrt(np.sum(W_ * W_)) + 1e-10

            val = num / den

            if val > best_val:
                best_val = val
                best = (u, v)

    # find best match location (argmax of I_MQ)

    # compute shift relative to template location
    sh.append([y0 - best[0], x0 - best[1]])

sh = np.array(sh)

# shift each frame by its shift & avg
acc = np.zeros_like(fr[0])

for i in range(len(fr)):

    # shift so template patch lines up
    al = scipy.ndimage.shift(fr[i], (sh[i,0], sh[i,1], 0), order=1, mode="nearest")

    acc += al

# avg aligned frames
out = acc / len(fr)

# save fin composite image focused around template patch
ps.write_image("q5-composite-image.png", out)