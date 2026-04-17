import torch
import torch.nn as nn
import numpy as np
from api.model import MineralCNN

def main():
    print("Initializing model...")
    model = MineralCNN(input_size=30, depth_out=20)
    
    print("Loading weights from mineral_model.pth...")
    model.load_state_dict(torch.load("mineral_model.pth", map_location="cpu"))
    
    # Set to EVAL mode for stable ONNX export
    model.eval() 

    dummy_input = torch.randn(1, 1, 30, 30)

    print("Exporting to mineral_model.onnx (Inference Mode)...")
    torch.onnx.export(
        model,
        dummy_input,
        "mineral_model.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Conversion complete: mineral_model.onnx")

if __name__ == "__main__":
    main()
