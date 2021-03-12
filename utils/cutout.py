import torch
import numpy as np 

class Cutout(object):
    def __init__(self, n_masks, length):
        self.n_masks = n_masks
        self.length = length

    def __call__(self, img):

        H  = img.size(1)
        W = img.size(2)

        mask = np.ones((H, W), np.float32)

        for n in range(self.n_masks):
            x = np.random.randint(W)
            y = np.random.randint(H)

            x1 = np.clip(x - self.length // 2,  0, W)
            x2 = np.clip(x + self.length // 2,  0, W)
            y1 = np.clip(y - self.length // 2,  0, H)
            y2 = np.clip( y + self.length //2,  0, H)
            
            mask[y1 : y2, x1 : x2] = 0 
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask 

        return img 
        




