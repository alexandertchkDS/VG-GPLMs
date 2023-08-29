import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

def model_r3d_18():
    # Load the pre-trained r3d_18 model once
    model = r3d_18(pretrained=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = torch.nn.Sequential(
        *(list(model.children())[:-1]))
    return model


def model_slow_r50():
    # Load the pretrained model
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    
    # Bypass the dropout, proj, and output_pool layers
    model.blocks[5].dropout = nn.Identity()
    model.blocks[5].proj = nn.Identity()
    model.blocks[5].output_pool = nn.Identity()
    
    # Ensure we're only averaging across the spatial and temporal dimensions
    # Changing AvgPool3d kernel size to get the desired output shape
    model.blocks[5].pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    # Send model to appropriate device
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set model to evaluation mode
    
    return model



