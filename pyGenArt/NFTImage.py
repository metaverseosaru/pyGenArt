import pandas as pd
import requests
import numpy as np
import json
import os
from os.path import join, isdir, isfile
from PIL import Image
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import ImageFilter
import matplotlib.pyplot as plt
from os import listdir
import scipy.ndimage as ndimage
import scipy.stats as stats
import colorsys
from .layer import *
from .trait import *
from .config import *
from .color import *
from .clipping import *
import re
import scipy.signal as signal
from copy import deepcopy
import pkg_resources

class NFTImage:
    # Constructor. If no attrs are specified, it randomly draws it from pool of traits.
    def __init__(self, 
                 config                ,
                 attrs          = [], 
                 outerid        = 0, 
                 innerid        = 0, 
                 verbose        = False,
                 renderStats    = False):
        
        self.attrs   = {}             # list of tuple holding attributes about this NFT
        self.table   = config.table   # entire trait table
        self.outerid = outerid        # outer (public) ID
        self.innerid = innerid        # inner ID for future use
        self.verbose = verbose        # verbosity
        self.renderStats    = renderStats
        self.config = config        
         
        # assign traits
        if len(attrs) == 0:
            self.roll_traits()
        else:
            assert(len(self.config.traits_to_roll) == len(attrs))
            self.attrs = attrs
            
    # Randomly draws traits.
    def roll_traits(self):
        rolled_ttypes = []
        
        if self.verbose: print("\nROLLING....")
        for t in self.config.traits_to_roll:
            if self.verbose: print("Rolling %s."%t)
            # Draw traits based on prereq
            index, probs, indices = 0, [], []
            for c in self.table[t]:
                if isRollable(self.attrs, c):
                    probs.append(c.weight)
                    indices.append(index)
                index += 1

            # Record the rolled triat.
            probs = probs/np.sum(probs)
            chosen = np.random.choice(indices, p=probs)

            self.attrs[t] = deepcopy(self.table[t][chosen])
            rolled_ttypes.append(t)
                
        if self.verbose: print("DONE ROLLING....")
            
    # Writes metadata to designated outfolder or returns an json object if not specified.
    def writeMetadata(self, 
                      image_url = "replaceThisURL/%i.png",
                      save      = True):
        
        template = {"name": self.config.name%self.outerid,
                    "description": self.config.description,
                    "image": image_url%self.innerid,
                    "attributes":[]}
        
        temp = []
        if self.verbose: print("\nGENERATING METADATA....")
        
        # Write out traits
        for t in self.config.traits_to_display:
            trait = self.attrs[t]
            if self.verbose: print("Processing %s."%trait.name)
            if trait.name != "":
                temp.append({"trait_type": trait.ttype_name, "value": trait.name})
                
        template["attributes"] = temp
  
        if save: 
            outfilename = join(self.config.metadata_outfolder, str(self.innerid))
            outfile = open(outfilename, "w")
            if self.verbose: print("Writing metadata to %s."%outfilename)
            json.dump(template, outfile, indent=4)
            outfile.close()
            
        if self.verbose: print("DONE GENERATING METADATA....\n")
        
        return json.dumps(template, indent=4)
    
    # Renders image out to designated outfolder or returns an IMAGE object if not specified.
    def renderImage(self,
                    save       = True,
                    blur_image = False):
        
        if self.verbose: print("\nRENDERING....")
        if self.config.device == "cuda:0":
            torch.cuda.empty_cache()
            
        # Layers
        layers = LayerHolder()
        
        count = 0
        for tname in self.config.traits_to_render:
            trait = self.attrs[tname]

            if trait.filename!="" and not trait.trait_type.startswith("Background"):
                
                if self.verbose: 
                    print("Rendering %s of %s %s %i"%(trait.name, trait.trait_type, trait.filename, trait.tid))
                newlayer = getImage(trait.filename, cfg)
                
                # Apply colorlink
                if trait.colorlink != None:
                    color = self.attrs[trait.colorlink].color
                    if "Clothes" in trait.trait_type: color = equalizeSV2(color)
                    newlayer = recolor_layer(newlayer, color)

                # Deal with clipping
                if len(trait.clipping) > 0:
                    clip = getClip(self.attrs, trait.clipping, self.config, device=self.config.device)
                    newlayer = clipping(newlayer, clip)
                    
                layers.append(Layer(newlayer, tname, trait.mode))
        
        # Combining layers.
        base = layers.compose().content
        base = Image.fromarray(base.detach().cpu().numpy().astype(np.uint8))
                    
        if self.renderStats:
            base = self.addStatLines(base)
            
        if self.verbose: 
            print("DONE RENDERING....")
            
        if save: 
            if self.verbose: print("Writing image to %s."%outfilename)
            base.save(join(self.config.image_outfolder, "%i.png"%self.innerid))
            
        return base
    
    # Formatting information for dataframe storage
    def toData(self):
        output = dict([("outer_id", self.outerid), ("inner_id", self.innerid)])
        return {**output, **self.attrs}
    
    def addStatLines(self, base):
        if self.verbose: print("RENDERING STATS....")
        temp = []
        for i in self.config.traits_to_roll:
            if "color" in self.attrs[i].trait_type:
                temp.append(self.attrs[i].trait_type+": (%s)"%(self.attrs[i].tid)+str(self.attrs[i].color))
            else:
                temp.append(self.attrs[i].trait_type+": (%s)"%(self.attrs[i].tid)+self.attrs[i].filename)
        stats = "\n".join(temp)+"\n"

        draw = ImageDraw.Draw(base)
        font = ImageFont.truetype(pkg_resources.resource_filename("pyGenArt", "data/moji.ttf"), size=int(18*(self.config.width/1000)))
        draw.text((15*(self.config.width/1000), 15*(self.config.width/1000)), stats, (255,64,64), font=font)
        return base