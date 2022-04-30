{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "import onnxruntime\n",
    "from onnxruntime.quantization import quantize_dynamic, QuantType"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.7.0\n"
     ]
    }
   ],
   "source": [
    "print(onnxruntime.__version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:root:The original model opset version is 10, which does not support node fusions. Please update the model to opset >= 11 for better performance.\n"
     ]
    }
   ],
   "source": [
    "vals = [0.05, .10, .20, .30, .40, .50, .60, .70, .80, .90]\n",
    "# vals = [0.1]\n",
    "\n",
    "for val in vals:\n",
    "        model_input = \"pruned_model_\" + str(val)+  \".onnx\"\n",
    "        model_output = \"quantized_model_\" + str(val) +  \".onnx\"\n",
    "        quant_model = quantize_dynamic(model_input, model_output, weight_type=QuantType.QUInt8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "quantized_model = quantize_dynamic(\"0.1_model_pt.onnx\", \"quantized_test_0.1.onnx\", weight_type=QuantType.QInt8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from onnxruntime.quantization import quantize_dynamic, QuantType\n",
    "\n",
    "tests = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]\n",
    "for prune_thres in tests:\n",
    "    quantized_model = quantize_dynamic(str(prune_thres)+'_model_pt.onnx', str(prune_thres)+'_model_quant_pt.onnx', weight_type = QuantType.QUInt8)"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "49f43478476cfa9c5fb667e34d8f90772758223488791fccb9ae376d503b7c97"
  },
  "kernelspec": {
   "display_name": "Python 3.9.5 ('new_env')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.5"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
