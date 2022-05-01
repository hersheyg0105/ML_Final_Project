import argparse
from PIL import Image
import numpy as np
import os
import tflite_runtime.interpreter as tflite
from tqdm import tqdm
import time


# TODO: add argument parser
parser = argparse.ArgumentParser(description= "ckpt to tflite conversion")

# TODO: add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument("--model_type", type = str, default = "mobile", help = 'model type for conversion')

# TODO: Modify the rest of the code to use the arguments correspondingly
args = parser.parse_args()

tflite_model_name = "" # TODO: insert TensorFlow Lite model name
if args.model_type == "vgg11":
  tflite_model_name = 'vgg11.tflite'
elif args.model_type == 'vgg16':
  tflite_model_name = 'vgg16.tflite'
else:
  tflite_model_name = 'pruned_model_0.1.tflite'


# Get the interpreter for TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=tflite_model_name)

# Very important: allocate tensor memory
interpreter.allocate_tensors()

# Get the position for inserting the input Tensor
input_details = interpreter.get_input_details()
# Get the position for collecting the output prediction
output_details = interpreter.get_output_details()

# Label names for CIFAR10 Dataset
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

inference_time = 0
validation_correct = 0
validation_total = 0
for filename in tqdm(os.listdir("HW4_files/test_deployment")):
  with Image.open(os.path.join("HW4_files/test_deployment", filename)).resize((32, 32)) as img:

    # input_image = (np.float32(img) / 255.0)
    input_image = np.expand_dims(np.float32(img)/255.0, axis=0)

    # Set the input tensor as the image
    interpreter.set_tensor(input_details[0]['index'], input_image)
    
    start_time = time.time()
    # Run the actual inference
    interpreter.invoke()
    end_time = time.time()

    inference_time += (end_time - start_time)


    # Get the output tensor
    pred_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Find the prediction with the highest probability
    top_prediction = np.argmax(pred_tflite[0])

    # Get the label of the predicted class
    pred_class = label_names[top_prediction]

    validation_total += 1
    gt_class = filename.split('.')[0].split('_')[-1]

    if pred_class == gt_class:
      validation_correct += 1

file_name = "pruned_model_0.1.txt"
file1 = open(file_name,'w')
file1.write("Inference Time: " + str(inference_time) + "\n")
file1.write("Accuracy: " + str(100. * (validation_correct/validation_total)))
file1.close()
# print('Test accuracy: ' + str(100. * (validation_correct/validation_total)))
# print('Inference Time:' + inference_time)


