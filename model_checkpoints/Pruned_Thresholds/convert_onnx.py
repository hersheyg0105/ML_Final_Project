from email.policy import strict
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torchvision
import argparse
import mobilenet_rm_filt_pt as mb_pt

# from models.vgg11_pt import VGG

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch to ONNX conversion')
parser.add_argument('--model_type', type=str, default='mobileNet', help='model type for conversion')
parser.add_argument('--pytorch_model_path', type=str, default='pruned_model_0.1.pt', help='location of the pytorch model')
parser.add_argument('--onnx_model_path', type=str, default='pruned_model_0.1.onnx', help='location to store the converted onnx model')
args = parser.parse_args()

model = mb_pt.MobileNetv1()
model.load_state_dict(torch.load(args.pytorch_model_path, map_location=('cpu')), strict=False)

random_input = torch.randn(1,3,32,32)
torch.onnx.export(model, random_input, args.onnx_model_path, export_params=True, opset_version=10)



# from sklearn.metrics import mean_absolute_percentage_error
# import torch
# import torchvision
# import argparse
# import mobilenet_rm_filt_pt as mb_pt

# # from models.vgg11_pt import VGG

# # Argument parser
# parser = argparse.ArgumentParser(description='PyTorch to ONNX conversion')
# parser.add_argument('--model_type', type=str, default='mobileNet', help='model type for conversion')
# parser.add_argument('--pytorch_model_path', type=str, default='pruned_model_0.05.pt', help='location of the pytorch model')
# parser.add_argument('--onnx_model_path', type=str, default='pruned_model.onnx', help='location to store the converted onnx model')
# args = parser.parse_args()



# vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
#                     0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# for prune_thres in vals:
#     model = mb_pt.MobileNetv1()
#     input_model_path = "pruned_model_" + str(prune_thres) + ".pt"
#     print(input_model_path)
#     # model.load_state_dict(torch.load(args.pytorch_model_path, map_location=('cpu')))
#     model.load_state_dict(torch.load(input_model_path, map_location=('cpu')))
#     random_input = torch.randn(1,3,32,32)
#     output_model_path = "pruned_model_" + str(prune_thres) + ".onnx"
#     # torch.onnx.export(model, random_input, args.onnx_model_path, export_params=True, opset_version=10)
#     torch.onnx.export(model, random_input, output_model_path, export_params=True, opset_version=10)