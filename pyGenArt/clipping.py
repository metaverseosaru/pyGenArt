from PIL import Image
import numpy as np
import torch
from .config import *# Clipping
import torch
import re

def clipping(front : torch.Tensor,
             clip  : torch.Tensor):
    front = front*clip
    return front

def loadAlphaClip(attrs,
                  trait   : str,
                  config  : Config,
                  th      : int = 128, 
                  device  : str = "cuda:0",
                  ImageDB : dict = {}):
    #print("Clipping called with", trait, attrs[trait])
    if attrs[trait].filename != "":
        talpha = getImage(attrs[trait].filename, config)[:,:,3:4]
        #print("Actually loaded image")
    else:
        talpha = torch.zeros((config.width,config.height,1)).int().to(device)
    # Figure out how to clip
    return talpha > th

def getClip(attrs,
            spec    : str,
            config  : Config,
            device  : str = "cuda:0"):

    if spec.strip() == "":
        return torch.ones((config.width, config.height, 1), dtype=bool)
    else:    
        # tokenize the command
        parsed = re.split(r'(\ |\)|\()', spec)
        parsed = [p for p in parsed if p.strip()!=""]
        commands = ["("]+parsed+[")"]
        commands = [loadAlphaClip(attrs, c, config, device = device) if not c in ["and", "or", "not", "(", ")"] else c for c in commands]

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
                        chunk2.append(torch.logical_not(chunk[index+1]))
                        index+=2
                    else:
                        chunk2.append(chunk[index])
                        index+=1

                # Deal with and and or.       
                while len(chunk2) > 1:
                    if chunk2[1] == "or":
                        chunk2 = [torch.logical_or(chunk2[0], chunk2[2])] + chunk2[3:]
                    elif chunk2[1] == "and":
                        chunk2 = [torch.logical_and(chunk2[0], chunk2[2])] + chunk2[3:]

                # Return it to stack.
                stack.append(chunk2[0])

        assert(len(stack)==1)

        return stack[0]