import numpy as np
import os
from os.path import join, isdir, isfile
import re
import torch

# Trait class
# Holds trait information from downloaded from the google Drive.
class Trait:

    def __init__(self,
                 tid,                # Within trate id (consecutive ids of traits).
                 ttype,              # Trait type.
                 ttype_name,         # Display name for trait type (used for metadata display)
                 name,               # Name of trait within the aformentioned trait type.
                 filename,           # Source filename. Not rendered when empty.
                 weight,             # Weight assigned.
                 condition,          # Condition for rolling (always true if left blank)
                 color,              # Color value associated with this trait
                 colorlink,          # Trait to color link to, if exits (mostly haircolor stuff)
                 mode,               # Layer mode (alpha-composite by default)
                 tags,               # Tags for whatever use.
                 clipping,           # Clipping (none by default)
                 comment,            # Comment for the trait
                 ):
    
        self.trait_type      = ttype.strip()
        self.ttype_name      = ttype_name.strip()
        self.tid             = int(tid)
        self.name            = name.strip() #if name.strip()!="" else self.trait_type+":"+str(self.tid)
        self.filename        = filename.strip()
        self.weight          = float(weight)
        self.condition       = condition
        
        if color == "":
            self.color = (-1, -1, -1)
        else:
            self.color = tuple([int(i) for i in color.split(":")])
        self.colorlink       = colorlink if colorlink!= "" else None
            
        assert mode in ["multiply", "overlay", "alpha_composite", ""], "Unknown layer mode %s"%mode
        self.mode            = "alpha_composite" if mode == "" else mode
        self.clipping        = clipping
        self.tags            = tags.split(":") if tags!="" else []
        self.comment         = comment
    
    # Representation
    def __repr__(self):
        output = "%s:%s (@%s, W:%.2f, ID:%s)"%(self.trait_type,
                                               self.name,
                                               self.filename,
                                               self.weight,
                                               self.tid)
        return output
    
# Given an dict of attributes and conditional statments
# this function figures out if the trait should be rolled.
def isRollable(attrs, trait):
    if trait.condition.strip() == "":
        return True
    else:
        parsed = re.split(r'(\ |\)|\()', trait.condition)
        processed = []
        for p in parsed:
            if not p in ["not", " ", "", "and", "or", "(", ")"]:
                ttype, temp = p.split(":")
                tids = []
                for i in temp.split("."):
                    if "-" in i:
                        start, end = i.split("-")
                        for j in range(int(start), int(end)+1):
                            tids.append(j)
                    else:
                        tids.append(int(i))

                processed.append(str(attrs[ttype].tid in tids))
            else:
                processed.append(p)
        return eval("".join(processed))