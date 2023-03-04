from PIL import Image
import numpy as np
import torch
from typing import Literal

# Holds trait information from downloaded from the google Drive.
class Layer:

    def __init__(self,
                 content  :torch.Tensor,     
                 name     :str = "",            
                 mode     :Literal["alpha_composite", "multiply", "overlay"] = "alpha_composite",
                 ):
    
        self.content = content
        self.name    = name
        self.mode    = mode
    
    # Representation
    def __repr__(self):
        output = "%s (%s)"%(self.name, self.mode)
        return output

    # Adding layers
    def combine(self, l2):
        
        # Layer combination 
        if "multiply" == l2.mode:
            out = multiply(self.content, l2.content)
        elif "overlay" == l2.mode:
            out = overlay(self.content, l2.content)
        elif "alpha_composite" == l2.mode:
            # Use PIL default alpha composite
            out = alpha_composite(self.content, l2.content)
            
        return Layer(out, self.name)
    
class LayerHolder:
    
    def __init__(self):
        self.layers = []
        self.namemap = {}
        
    # Appends layers at the end.
    def append(self,
               layer : Layer):
        self.layers.append(layer)
        self.namemap[layer.name] = layer
        
    def insert(self,
               layer : Layer,
               index : int):
        self.layers = self.layers[:index] + [layer] + self.layers[index:]
        self.namemap[layer.name] = layer
        
    def remove(self,
               index : int):
        del self.namemap[self.layers[index].name]
        self.layers = self.layers[:index] + self.layers[index+1:]
        
    # Returns layer object by name ("String") or index (Int)
    def __getitem__(self,
                    key: str):
        if type(key) == str:
            return self.namemap[key]
        elif type(key) == int:
            return self.layers[key]
        else:
            assert False
            return None
        
    def __len__(self):
        return len(self.layers)
    
    # Returns index by name
    def idxByName(self,
                  name: str):
        return [l.name for l in self.layers].index(name)
    
    def compose(self):
        composite = self.layers[0]
        for i in range(1, len(self.layers)):
            composite = composite.combine(self.layers[i])
        base = composite.content
        return Layer(base, composite.name+"_composed")

# Multiplication layer 
def multiply(back  : torch.Tensor,
             front : torch.Tensor):
    # Figure out the colors
    cb = back[:,:,:3]
    cf = front[:,:,:3]
    cr = cb*cf//255
    
    # Get the raw transparency in 0-1 scale
    ab = back[:,:,3:4]/255
    af = front[:,:,3:4]/255
    
    # Calculate new transparency
    a1 = ab*af
    a2 = af*(1.0-ab)
    a3 = ab*(1.0-af)
    
    # Calculate final values
    a_out = a1+a2+a3
    mask  = torch.tile(a_out!=0, (1,1,3))
    c_out = (torch.ones(cb.shape)*255).to(cb.device)
    c_out[mask] = (a1*cr+a2*cf+a3*cb)[mask] / torch.tile(a_out, (1,1,3))[mask]
    
    # Form it into an Image object
    output = torch.concatenate([c_out, a_out*255], axis=2)
    output = torch.clip(output, 0, 255, out=output)
    
    return output

# Overlay layer
def overlay(back  : torch.Tensor,
            front : torch.Tensor,
            flag  : bool = False   # Boolean values whether to consider alpha channel.
           ):
    # Figure out the colors
    cb = back[:,:,:3]
    cf = front[:,:,:3]
    
    cr1 = (cb*cf*2/255).round()
    cr2 = 2*(cb+cf-(cb*cf/255).round())-255
    m1 = cb<128
    m2 = cb>=128
    cr = cr1*m1+cr2*m2
    cr = torch.clip(cr, 0, 255, out=cr)
    
    # Get the raw transparency in 0..1
    ab = back[:,:,3:4]/255
    af = front[:,:,3:4]/255
    
    if flag:
        output = torch.concatenate([cr, ab*255], axis=2)
    else:
        a1 = ab*af         #0.4*1 = 0.4 --> cr
        a2 = af*(1.0-ab)   #1*0.6 = 0.6 --> cf
        a3 = ab*(1.0-af)   #0.4*0 = 0   --> cb

        # Calculate final values
        a_out = a1+a2+a3
        mask  = torch.tile(a_out!=0, (1,1,3))
        c_out = (torch.ones(cb.shape)*255).to(cb.device)
        c_out[mask] = (a1*cr+a2*cf+a3*cb)[mask] / torch.tile(a_out, (1,1,3))[mask]
        
        output = torch.concatenate([c_out, a_out*255], axis=2)
        
    output = torch.clip(output, 0, 255, out=output)
    
    return output

# default alpha_composite layer 
def alpha_composite(back  : torch.Tensor,
                    front : torch.Tensor):
    # Figure out the colors
    cb = back[:,:,:3]
    cf = front[:,:,:3]
    cr = cf
    
    # Get the raw transparency in 0-1 scale
    ab = back[:,:,3:4]/255
    af = front[:,:,3:4]/255
    
    # Calculate new transparency
    a1 = ab*af
    a2 = af*(1.0-ab)
    a3 = ab*(1.0-af)
    
    # Calculate final values
    a_out = a1+a2+a3
    mask  = torch.tile(a_out!=0, (1,1,3))
    c_out = (torch.ones(cb.shape)*255).to(cb.device)
    c_out[mask] = (a1*cr+a2*cf+a3*cb)[mask] / torch.tile(a_out, (1,1,3))[mask]
    
    # Form it into an Image object
    output = torch.concatenate([c_out, a_out*255], axis=2)
    output = torch.clip(output, 0, 255, out=output)
    
    return output