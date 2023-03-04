import colorsys
import numpy as np
import torch
from .layer import *

# checks if hc is in some proximity of hbc
# th = 0.1 will give width of 0.2 out of 1.0
# colors need to be defined in rgb.
def checkColor(hc, bgc, th=0.1, opposite=False):
    h, s, v = colorsys.rgb_to_hsv(*hc)
    if opposite:
        h = (h+0.5)%1.0
    h2, s2, v2 = colorsys.rgb_to_hsv(*bgc)
    
    intervals = [(h+th+i, h-th+i) for i in [-1,0,1]]
    flag = False
    for interval in intervals:
        upper, lower = interval
        if h2 > lower and h2 <upper:
            flag = True
    return flag

def recolor_layer(newlayer, color):
    overlay_layer = torch.ones(newlayer.shape).to(newlayer.device)
    overlay_layer[:,:,:3]  = torch.tensor(color)
    overlay_layer[:,:,3:4] = (newlayer[:,:,-1:]>0)*255
    newlayer = overlay(newlayer, overlay_layer, True)
    return newlayer

def jittercolor(color, h=0.1, s=0, v=0):
    hsv_delta      = np.array([np.random.rand()*h-h/2, np.random.rand()*s-s/2, np.random.rand()*v-v/2])
    temp = np.array(list(colorsys.rgb_to_hsv(*color)))
    temp[0] = (temp[0] + hsv_delta[0])%1.0
    temp[1] = max(0, min(1.0, temp[1] + hsv_delta[1]))
    temp[2] = max(0, min(255, temp[2] + hsv_delta[2]))
    color = colorsys.hsv_to_rgb(*temp)
    return (int(color[0]), int(color[1]), int(color[2]))