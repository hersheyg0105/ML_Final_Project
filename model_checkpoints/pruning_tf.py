from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import tensorflow_model_optimization as tfmot
from mobilenet_rm_filt_tf import MobileNetv1

def convert_tflite(model, name="", optim=False):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optim:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model
    with open(f"{name}.tflite",'wb') as f:
        f.write(tflite_model)

model = MobileNetv1()
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights("mbnv1_tf.ckpt")
model.save("mbnv1_baseline.h5")
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs
batch_size = 256
epochs = 5
(trainx, trainy), (testx, testy) = cifar10.load_data()
train_images = trainx.astype('float32')
test_images = testx.astype('float32')
train_images /= 255.0
test_images /= 255.0
train_labels = to_categorical(trainy)
test_labels = to_categorical(testy)

_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
convert_tflite(model=model, name="mbnv1_baseline")
convert_tflite(model=model, name="mbnv1_baseline_optim", optim=True)

num_images = train_images.shape[0]
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                             final_sparsity=0.90,
                                                             begin_step=0,
                                                             end_step=end_step)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_for_pruning.summary()

callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

model_for_pruning.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels), callbacks=callbacks)

_, model_for_pruning_accuracy = model_for_pruning.evaluate(test_images,test_labels, verbose=0)

model_for_pruning.save_weights('mbnv1_pruned.ckpt')
model_for_pruning.save('mbnv1_pruned.h5')

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save_weights('mbnv1_strip.ckpt')
model_for_export.save('mbnv1_strip.h5')

def get_zipped_model_size(file, name=""):
    import os
    import zipfile

    zipped_file=f"{name}.zip"
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)/1000000

convert_tflite(model=model_for_export,name="mbnv1_stripped")
convert_tflite(model=model_for_export,name="mbnv1_stripped_optim",optim=True)

from mobilenet_rm_filt_tf import remove_channel
import tensorflow as tf

test_model = MobileNetv1()
test_model.load_weights("mbnv1_strip.ckpt")

removed_filters_model = remove_channel(test_model)
epochs = 5
num_images = train_images.shape[0]
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.90, begin_step=0, end_step=end_step)}
removed_filters_model = prune_low_magnitude(removed_filters_model, **pruning_params)
removed_filters_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
removed_filters_model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels), callbacks=callbacks)
removed_filters_model = tfmot.sparsity.keras.strip_pruning(removed_filters_model)
removed_filters_model.save_weights('mbnv1_removed_filters.ckpt')
removed_filters_model.save('mbnv1_removed_filters.h5')

convert_tflite(model=model_for_export,name="mbnv1_removed_filters")
convert_tflite(model=model_for_export,name="mbnv1_removed_filters_optim",optim=True)

print(f"Baseline test accuracy: {baseline_model_accuracy}")
print(f"Pruned test accuracy: {model_for_pruning_accuracy}")

print("Size of zipped baseline Keras model: %.2f MB" % (get_zipped_model_size("mbnv1_baseline.h5","baseline")))
print("Size of zipped baseline pruned Keras model: %.2f MB" % (get_zipped_model_size("mbnv1_pruned.h5","mbnv1_pruned")))
print("Size of zipped baseline stripped Keras model: %.2f MB" % (get_zipped_model_size("mbnv1_strip.h5","mbnv1_strip")))
print("Size of zipped baseline removed Keras model: %.2f MB" % (get_zipped_model_size("mbnv1_removed_filters.h5","mbnv1_removed_filters")))
print()
print("Size of zipped baseline TFLite model: %.2f MB" % (get_zipped_model_size("mbnv1_baseline.tflite", "baseline")))
print("Size of zipped stripped TFLite model: %.2f MB" % (get_zipped_model_size("mbnv1_stripped.tflite", "baseline")))
print("Size of zipped removed TFLite model: %.2f MB" % (get_zipped_model_size("mbnv1_removed_filters.tflite", "baseline")))
print()
print("Size of zipped baseline TFLite quantized model: %.2f MB" % (get_zipped_model_size("mbnv1_baseline_optim.tflite", "baseline")))
print("Size of zipped stripped TFLite quantized model: %.2f MB" % (get_zipped_model_size("mbnv1_stripped_optim.tflite", "baseline")))
print("Size of zipped removed TFLite quantized model: %.2f MB" % (get_zipped_model_size("mbnv1_removed_filters_optim.tflite", "baseline")))
print()

# Baseline test accuracy: 0.10000000149011612
# Pruned test accuracy: 0.7319999933242798
# Size of zipped baseline Keras model: 12.05 MB
# Size of zipped baseline pruned Keras model: 27.25 MB
# Size of zipped baseline stripped Keras model: 2.77 MB
# Size of zipped baseline removed Keras model: 2.76 MB
#
# Size of zipped baseline TFLite model: 11.93 MB
# Size of zipped stripped TFLite model: 2.64 MB
# Size of zipped removed TFLite model: 2.64 MB
#
# Size of zipped baseline TFLite quantized model: 3.07 MB
# Size of zipped stripped TFLite quantized model: 0.88 MB
# Size of zipped removed TFLite quantized model: 0.88 MB
