import subprocess
import threading
from time import sleep
import numpy as np
from threading import Thread, Event
import sys
import numpy as np

for val in np.arange(0, 1, 0.05):
    num = (round(val, 2))
    model_type = "pruned_model_"
    # model_type = "quantized_model_"
    input_name = model_type + str(num) + ".onnx" 
    measurment_csv_output_name = model_type + str(num) + ".csv"

    t1 = subprocess.Popen(['python', 'deploy_onnx.py', input_name], shell=False)
    t2 = subprocess.Popen(['python', 'measuremenet.py', measurment_csv_output_name], shell=False)

    t1.wait()
    t1.terminate()
    t2.terminate()
    sleep(120)

