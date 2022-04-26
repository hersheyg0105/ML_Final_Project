from sklearn.metrics import mean_absolute_percentage_error
import torch
import torchvision
import argparse
import mobilenet_rm_filt_pt as mb_pt

# from models.vgg11_pt import VGG

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch to ONNX conversion')
parser.add_argument('--model_type', type=str, default='mobileNet', help='model type for conversion')
parser.add_argument('--pytorch_model_path', type=str, default='pruned_model.pt', help='location of the pytorch model')
parser.add_argument('--onnx_model_path', type=str, default='pruned_model.onnx', help='location to store the converted onnx model')
args = parser.parse_args()

model = mb_pt.MobileNetv1()
model.load_state_dict(torch.load(args.pytorch_model_path, map_location=('cpu')))

random_input = torch.randn(1,3,32,32)
torch.onnx.export(model, random_input, args.onnx_model_path, export_params=True, opset_version=10)
