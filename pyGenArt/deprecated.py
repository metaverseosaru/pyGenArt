def getOutline1(image, r=1, color=(25,4,3), sigma=0.4, alphath=128):

    # Get the border
    npimg   = np.array(image)
    w,h,d   = npimg.shape
    alpha   = (npimg[:,:,-1] > alphath).astype(int)
    
    # Expand the image
    canvas               = np.zeros((w,h))
    canvas[r:,r:]       += alpha[:-1*r,:-1*r]
    canvas[:-1*r,:-1*r] += alpha[r:,r:]
    canvas[r:,:-1*r]    += alpha[:-1*r,r:]
    canvas[:-1*r,r:]    += alpha[r:,:-1*r]
    expanded             = canvas
    expanded             = (expanded>0).astype(int)
    expanded = expanded - alpha
    expanded = expanded*255
    border_color = np.zeros((w,h,3))
    border_color[:,:] = color
    npoimg = np.concatenate([border_color, np.expand_dims(expanded, axis=-1)],axis=-1)

    # Grab alpha and blur the alpha channel.
    alpha = npoimg[:,:, -1]
    blurred_alpha = signal.convolve2d(alpha, getFilter(5, 0.5, 0.9), boundary='symm', mode='same')
    temp = np.zeros((w,h,3))
    temp[:,:] = color
    # Get the outline only on the alpha channel.
    npoutline = np.concatenate([temp, np.expand_dims(blurred_alpha, axis=-1)], axis=-1)

    # Convert it to image
    outline = Image.fromarray(npoutline.astype(np.uint8))
    return outline

def getOutline2(image, r=1, color=(25,4,3), sigma=0.4, alphath=128):

    # Get the border
    npimg   = np.array(image)
    w,h,d   = npimg.shape
    alpha   = (npimg[:,:,-1] > alphath).astype(int)
    print(np.max(alpha), np.min(alpha))
    
    # Expand the image
    canvas               = np.zeros((w,h))
    canvas[r:,r:]       += alpha[:-1*r,:-1*r]
    canvas[:-1*r,:-1*r] += alpha[r:,r:]
    canvas[r:,:-1*r]    += alpha[:-1*r,r:]
    canvas[:-1*r,r:]    += alpha[r:,:-1*r]
    expanded             = canvas
    expanded             = (expanded>0).astype(int)
    expanded = np.maximum(expanded - alpha, np.zeros(expanded.shape))
    expanded = expanded*255
    border_color = np.zeros((w,h,3))
    border_color[:,:] = color
    npimage = np.concatenate([border_color, np.expand_dims(expanded, axis=-1)],axis=-1)

    # Grab alpha and blur the alpha channel.
    for i in range(4):
        npimage = antialias(npimage)
    
    # Convert it to image
    outline = Image.fromarray(npimage.astype(np.uint8))
    return outline

def getCorners(npimage):
    alpha = npimage[:,:, -1]
    nfilter_v = getFilter2()
    nfilter_h = nfilter_v.T
    vertical   = signal.convolve2d(alpha, nfilter_v, boundary='symm', mode='same')
    horizontal = signal.convolve2d(alpha, nfilter_h, boundary='symm', mode='same')
    corners    = np.multiply(vertical, horizontal)
    corners[alpha>0] = 0 
    corners    = (corners>0)
    return corners

def antialias(npimage):
    corners = getCorners(npimage)
    alpha = npimage[:,:, -1]
    m = 0.2
    filt  = np.array([[0,m,0],[m,0,m],[0,m,0]])
    out = signal.convolve2d(alpha, filt, boundary='symm', mode='same')
    out[corners==False] = 0
    alpha = alpha+out
    return np.concatenate([npimage[:,:, :-1], np.expand_dims(alpha, axis=-1)], axis=-1)

def getFilter(size=5, cov=0.4, bound=1.0):
    offset = size//2
    coords = np.zeros((size,size,2))
    for i in range(-1*offset,-1*offset+size):
        for j in range(-1*offset,-1*offset+size):
            coords[i+offset,j+offset] = i,j
    weights = stats.multivariate_normal.pdf(coords, mean=(0,0), cov=cov)
    return(weights)

def getFilter2(size=3, scale=1.0, bound=1.0):
    offset = size//2
    nfilter = np.zeros((size,1))
    for i in range(-1*offset,-1*offset+size):
        nfilter[i+offset,0] = i
    weights = norm.pdf(nfilter, loc=0, scale=scale)
    weights = np.array([[1,1,1]])
    return(weights)

def getOutline(image, r=1, color=(25,4,3), alphath=128):

    # Get the border
    npimg   = np.array(image)
    w,h,d   = npimg.shape
    alpha   = (npimg[:,:,-1] > alphath).astype(int)
    
    # Expand the image by radius of 1
    # This part needs to be more general.
    canvas               = np.zeros((w,h))
    canvas[r:,r:]       += alpha[:-1*r,:-1*r]
    canvas[:-1*r,:-1*r] += alpha[r:,r:]
    canvas[r:,:-1*r]    += alpha[:-1*r,r:]
    canvas[:-1*r,r:]    += alpha[r:,:-1*r]
    expanded             = canvas
    expanded             = (expanded>0).astype(int)
    expanded             = expanded*255
    print(np.min(expanded), np.max(expanded))
    
    # Apply colors and combine it with the border.
    border_color         = np.zeros((w,h,3))
    border_color[:,:]    = color
    outline              = np.concatenate([border_color, np.expand_dims(expanded, axis=-1)],axis=-1)
    outline              = Image.fromarray(outline.astype(np.uint8))

    return outline

# Grab a random HSV
def chooseHairHSV(srange=0.6, vrange=0.7, sbase=0, vbase=0.3):
    h,s = np.random.rand(), np.random.rand()
    v = np.random.rand()*(1-s*0.7)
    s = s*srange+sbase
    v = v*vrange+vbase
    
    return [h,s,v]# if np.random.rand() < 0.5 else [np.random.rand(), np.random.rand(), np.random.rand()]

def chooseEyeHSV(srange=0.3, vrange=0.3, sbase=0.5, vbase=0.5):
    h,s = np.random.rand(), np.random.rand()
    v = np.random.rand()*(1-s)
    s = s*srange+sbase
    v = v*vrange+vbase
    return [h,s,v]

# Given an image samples hue.
def sampleHue(filename, noise=0.05):
    x = Image.open(filename)
    x = np.array(x)
    
    # Grab a random pixel (in RGB)
    # TODO --> Implement k-means based sampling.
    rgb = x[np.random.randint(x.shape[0]),np.random.randint(x.shape[1]),:3]/255
    
    # Convert it to HSV
    hsv = colorsys.rgb_to_hsv(*rgb.tolist())
    
    # Complementary color activated if necessary
    h = hsv[0]+np.random.randint(2)*0.5
    
    # Add noise to the sampled hue.
    hrandom = np.clip(h+np.random.rand()*2*noise-noise, 0, 1)
    return hrandom


def getOutline(image, r=1, color=(25,4,3), sigma=0.5, alphath=128):

    # Get the border
    npimg = np.array(image)
    w,h,d = npimg.shape
    alpha = npimg[:,:,-1] > alphath
    conved = signal.convolve2d(alpha, np.ones((3,3)), boundary='symm', mode='same')
    outline = np.logical_and(conved>0, conved<9)
    ys,xs = np.where(outline==True)

    # outline with circle 
    oimg = Image.fromarray(np.zeros((w,h,d)).astype(np.uint8))
    draw = ImageDraw.Draw(oimg)
    for y,x in zip(ys, xs):
        box = [(x-r, y-r), (x+r, y+r)]
        draw.ellipse(box, fill=(color[0],color[1],color[2],255))
    npoimg = np.array(oimg)

    # Grab alpha and blur the alpha channel.
    alpha = npoimg[:,:, -1]
    blurred_alpha = signal.convolve2d(alpha, getFilter(7, 0.3, 0.9), boundary='symm', mode='same')
    temp = np.zeros((w,h,3))
    temp[:,:] = color
    # Get the outline only on the alpha channel.
    npoutline = np.concatenate([temp, np.expand_dims(blurred_alpha, axis=-1)], axis=-1)

    # Convert it to image
    outline = Image.fromarray(npoutline.astype(np.uint8))
    
    return outline

def getFilter(size=3, cov=0.4, bound=1.0):
    offset = size//2
    coords = np.zeros((size,size,2))
    for i in range(-1*offset,-1*offset+size):
        for j in range(-1*offset,-1*offset+size):
            coords[i+offset,j+offset] = i,j
    coords.reshape((size**2,2))
    weights = stats.multivariate_normal.pdf(coords, mean=(0,0), cov=cov)
    weights = bound*(weights/np.sum(weights))
    return(weights)

# Change loadAlphaClip c to filename
# May be take in config and evertying is going to be done?
def getClip(self, spec, w=1000, h=1000):

    clip = np.ones((w, h, 1), dtype=bool)
    if spec.strip() == "":
        return clip
    else:    
        # tokenize the command
        parsed = re.split(r'(\ |\)|\()', spec)
        parsed = [p for p in parsed if p.strip()!=""]
        commands = ["("]+parsed+[")"]
        commands = [self.loadAlphaClip(c) if not c in ["and", "or", "not", "(", ")"] else c for c in commands]

        # Parse them
        stack = []
        for p in commands:
            if type(p)!=str or p!=")":
                stack.append(p)
            else:
                popped = ""

                # Get the evaluation chunk
                chunk = []
                while(not(type(popped)==str and popped=="(")):
                    popped = stack.pop()
                    chunk.append(popped)
                chunk = chunk[:-1]
                chunk.reverse()

                # Evaluate the chunk
                # Deal with negation.
                chunk2 = []
                index = 0
                while index < len(chunk):
                    if type(chunk[index])==str and chunk[index] == "not":
                        chunk2.append(np.logical_not(chunk[index+1]))
                        index+=2
                    else:
                        chunk2.append(chunk[index])
                        index+=1

                # Deal with and and or.       
                while len(chunk2) > 1:
                    if chunk2[1] == "or":
                        chunk2 = [np.logical_or(chunk2[0], chunk2[2])] + chunk2[3:]
                    elif chunk2[1] == "and":
                        chunk2 = [np.logical_and(chunk2[0], chunk2[2])] + chunk2[3:]

                # Return it to stack.
                stack.append(chunk2[0])

        assert(len(stack)==1)

        return stack[0]