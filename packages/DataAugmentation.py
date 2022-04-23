import cv2
import tensorflow as tf
import albumentations as album

class dataAugmentation:
    def __init__(self):

        self.transforms = album.Compose([
            album.HorizontalFlip(p=0.5),
            album.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            album.OneOf([
                album.RandomBrightnessContrast(brightness_limit=0.1, 
                                               contrast_limit=0.1, p=0.5),
                album.RandomGamma(p=0.5),
                ], p=0.5),
            album.OneOf([
                album.Blur(p=0.1),
                album.GaussianBlur(p=0.1),
                album.MotionBlur(p=0.1),
                ], p=0.1),
            album.OneOf([
                album.GaussNoise(p=0.1),
                album.GridDropout(ratio=0.5, p=0.2),
                album.CoarseDropout(max_holes=16, max_height=16, max_width=16,
                                    min_holes= 8, min_height= 8, min_width= 8, p=0.2)
                ], p=0.2),
            ])

    def aug_fn(self, image):
        data = {"image":image}
        aug_data = self.transforms(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img, tf.float32)
        return aug_img

    def augment_iamge(self, img, label):
        aug_img = tf.numpy_function(func=self.aug_fn, inp=[img], Tout=tf.float32)
        return aug_img, label