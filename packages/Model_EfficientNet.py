import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def get_model(input_shape, output_shape):
    base_model = EfficientNetV2B0(input_shape=(input_shape[0], input_shape[1], 3), include_top=False)
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.tile(inputs, (1, 1, 1, 3))
    # Transfer Learning
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dense
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(output_shape, activation="sigmoid", dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    model = get_model(input_shape=(1024, 1024, 1), output_shape=22)
    print(model.summary())