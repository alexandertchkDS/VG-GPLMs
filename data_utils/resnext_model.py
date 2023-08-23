import torch
from torchvision.models.video import r3d_18

def model_r3d_18():
    # Load the pre-trained r3d_18 model once
    model = r3d_18(pretrained=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # model = torch.nn.Sequential(
    #     *(list(model.children())[:-1]),
    #     torch.nn.Flatten(),
    #     torch.nn.Linear(512, 2048)
    # ).to("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(
        *(list(model.children())[:-1]))
    return model