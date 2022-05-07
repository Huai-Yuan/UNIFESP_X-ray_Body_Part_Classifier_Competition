import pydicom 
import warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

def split_dataset(X, y, N_SPLITS=4):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for train_index, valid_index in skf.split(X, np.argmax(y, axis=1)):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

    print(f"train size: {y_train.shape[0]}")
    print(f"valid size: {y_valid.shape[0]}")

    ds_splits = {'train':[X_train, y_train], 
                 'valid':[X_valid, y_valid]}
    
    return ds_splits

class tfrecords:
    def __init__(self, IMG_HEIGHT, IMG_WIDTH):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH

    # Write
    def process_dicom_files(self, filename):
        ds = pydicom.dcmread(filename)
        img = ds.pixel_array
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            img = np.invert(img)
        img = img/(2**ds.BitsAllocated - 1)*255
        img = img - img.min()
        img = img / img.max() * 255
        img = img.astype('uint8')
        img = np.expand_dims(img, axis=-1)
        return img

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def image_example(self, image, label):
        label = label.tobytes()

        image = tf.image.resize_with_pad(image, self.IMG_HEIGHT, self.IMG_WIDTH)
        image = tf.cast(image, 'uint8')
        image = tf.io.encode_png(image)

        feature = {
            'label': self._bytes_feature(label),
            'image_raw': self._bytes_feature(image),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def image_example_test(self, image):

        image = tf.image.resize_with_pad(image, self.IMG_HEIGHT, self.IMG_WIDTH)
        image = tf.cast(image, 'uint8')
        image = tf.io.encode_png(image)

        feature = {
            'image_raw': self._bytes_feature(image),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def WriteTFRecords(self, ds_splits, path):
        for ds_split in ds_splits:
            record_file = f"{path}/{ds_split}.tfrecords"
            subset = ds_splits[ds_split]
            if ds_split == "test":
                filenames = subset
                with tf.io.TFRecordWriter(record_file) as writer:
                    for filename in tqdm(filenames):
                        # read dicom files
                        img = self.process_dicom_files(filename)
                        tf_example = self.image_example_test(img)
                        writer.write(tf_example.SerializeToString()) 
            else:
                filenames, labels = subset
                with tf.io.TFRecordWriter(record_file) as writer:
                    for filename, label in tqdm(list(zip(filenames, labels))):
                        # read dicom files
                        img = self.process_dicom_files(filename)
                        tf_example = self.image_example(img, label)
                        writer.write(tf_example.SerializeToString())

    # Read
    def _parse_image_function(self, example_proto):
        image_feature_description = {
            'label': tf.io.FixedLenFeature([], tf.string),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(example_proto, image_feature_description)

        img_raw = example_message['image_raw']
        label = example_message['label']
        
        image = tf.io.decode_png(img_raw)
        image = tf.reshape(image, shape=[self.IMG_HEIGHT, self.IMG_WIDTH, 1])

        label = tf.io.decode_raw(label, tf.int32)
        label = tf.reshape(label, shape=[22])

        return (image, label)

    def _parse_image_function_test(self, example_proto):
        image_feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(example_proto, image_feature_description)

        img_raw = example_message['image_raw']

        image = tf.io.decode_png(img_raw)
        image = tf.reshape(image, shape=[self.IMG_HEIGHT, self.IMG_WIDTH, 1])

        return image

    def get_dataset(self, filename, mode=None):
        options = tf.data.Options()
        options.deterministic = False
        dataset = tf.data.TFRecordDataset(filename)  
        dataset = dataset.with_options(options)  
        dataset = dataset.map(self._parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
    
    def get_test_dataset(self, filename, mode=None):
        options = tf.data.Options()
        options.deterministic = False
        dataset = tf.data.TFRecordDataset(filename)  
        dataset = dataset.with_options(options)  
        dataset = dataset.map(self._parse_image_function_test, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset


    