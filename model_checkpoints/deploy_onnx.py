import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time
# from sklearn.metrics import accuracy_score


# TODO: create argument parser object\
parser = argparse.ArgumentParser(description='PyTorch to ONNX conversion')


# TODO: add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('--model_type', type=str, default='MobileNet', help='model type for conversion')

# TODO: Modify the rest of the code to use those arguments correspondingly
args = parser.parse_args()

# TODO: insert ONNX model name, essentially the path to the onnx model
if args.model_type == 'VGG11':
    onnx_model_name = "VGG11_pt.onnx"
elif args.model_type == 'VGG16':
    onnx_model_name = "VGG16_pt.onnx"
else:
    onnx_model_name = "MobileNet_pt.onnx"

# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(onnx_model_name)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation 
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_true = []
label_pred = []

start_time = time.time()
# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("/home/student/HW3_files/test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("/home/student/HW3_files/test_deployment", filename)).resize((32, 32)) as img:
        print("Image shape:", np.float32(img).shape)

        # normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        
        # Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)

        # change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])
        
        print("Input Image shape:", input_image.shape)

        # Run inference and get the prediction for the input image
        pred_onnx = sess.run(None, {input_name: input_image})[0]

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        # TODO: compute test accuracy of the model
        actual_label = filename.split('_')[1]
        actual_label = actual_label.split(".")[0]
        label_true.append(actual_label)
        label_pred.append(pred_class)




end_time = time.time()
total_time = end_time - start_time
# acc = accuracy_score(label_true, label_pred)

total = len(label_true)
count = 0
for i in range(0, len(label_true)):
    if label_true[i] == label_pred[i]:
        count += 1
acc = count/total
file = open('time_deploy_mb_mc1.txt', 'w')
file.write("Inference time: " + str(total_time) + '\n')
file.write("Accuracy: " + str(acc) + '\n')

file.close()








