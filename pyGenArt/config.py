from os.path import join, isdir, isfile
import os
from .util import *
from .trait import *
import torch
from PIL import Image
import numpy as np
from typing import Literal

#####CONFIG#####
class Config:
    
    def __init__(self):
        self.gdrive_id          = ""
        
        # Traits list
        self.traits_to_roll     = []
        self.traits_to_render   = []
        self.traits_to_display  = []
        self.traits_to_diff     = []
        
        self.num_image          = 3000
        self.name               = "GenerativeNFT #%i"
        self.description        = "Placeholder"
        self.width, self.height = 1500, 1500
        
        self.image_infolder     = "./imgsrc"
        self.image_outfolder    = "./img"
        self.metadata_outfolder = "./metadata"
        self.temp_outfolder     = "./temp"
        self.hamming_dist_th    = 3
        self.device             = "cpu"
        self.num_preload        = 100
        
        # Probability table
        self.table = {}
        self.preloaded_images = {}
        
    def initialize(self):
  
        # Make folders if necessary
        if not isdir(self.image_outfolder): os.mkdir(self.image_outfolder)
        if not isdir(self.metadata_outfolder): os.mkdir(self.metadata_outfolder)
        if not isdir(self.temp_outfolder): os.mkdir(self.temp_outfolder)
        if not isdir(self.image_infolder): os.mkdir(self.image_infolder)
            
    def load_traits(self):
        
        assert self.gdrive_id != "", "Google drive Id is not set."
        df = getGDriveAsCSV(self.gdrive_id)
        #df.to_csv(self.temp_outfolder, "triat_data.csv")

        # Should be in the order of combination
        for t in self.traits_to_roll:
            assert t in df["trait_type"].values, "Unknown trait %s in trait_to_roll"%t
        for t in self.traits_to_render:
            assert t in self.traits_to_roll, "Unknown trait %s in trait_to_render"%t
        for t in self.traits_to_display:
            assert t in self.traits_to_roll, "Unknown trait %s in trait_to_display"%t
        for t in self.traits_to_diff:
            assert t in self.traits_to_roll, "Unknown trait %s in trait_to_render"%t

        # Process the raw df in to prob table
        self.table = dict([(t, []) for t in list(set(df["trait_type"].values))])
        count = 0
        num_image = 0
        for i in range(df.shape[0]):
            t = Trait(*df.iloc[i].values.tolist())
            if t.filename != "": 
                assert isfile(join(self.image_infolder, t.filename)), "%s does not exist"%t.filename
                if count < self.num_preload:
                    self.preloaded_images[t.filename] = ImageHolder(join(self.image_infolder, t.filename), device=self.device)
                    count += 1
                num_image += 1
            assert(not t.tid in [i.tid for i in self.table[t.trait_type]])
            self.table[t.trait_type].append(t)
            self.table[t.trait_type].sort(key=lambda x:x.tid)
            
        print("The number of images in imgsrc: %i."%num_image)
        print("The number of images preloaded on GPU: %i."%count)

# A class that holds the original image information in a specified device (cpu or gpu).
class ImageHolder:
    
    def __init__(self, 
                 filename  : str,
                 colors    : list = [],
                 device    : str = "cpu"):
        
        self.filename = filename
        self.image  = torch.tensor(np.array(Image.open(filename)), dtype=torch.uint8).to(device)
        self.colors = colors
        
    def __repr__(self):
        return self.filename
    
        
# Grab Image
def getImage(filename, cfg):
    if filename in cfg.preloaded_images:
        newlayer = cfg.preloaded_images[filename].image
    else:
        newlayer = ImageHolder(join(cfg.image_infolder, filename), device=cfg.device).image
    return newlayer
    