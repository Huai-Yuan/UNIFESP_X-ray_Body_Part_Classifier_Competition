import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def get_model(input_shape, output_shape):
    inputs = tf.keras.Input(shape=input_shape)
    # CNN
    # x = tf.keras.applications.imagenet_utils.preprocess_input(inputs, mode="tf")
    x = tf.keras.layers.Rescaling(scale=1./255)(inputs)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3))(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3))(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3))(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3))(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3))(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dense
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(output_shape, activation="sigmoid", dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    model = get_model(input_shape=(224, 224, 1), output_shape=22)
    print(model.summary())