from typing import List
from PIL import Image, ImageOps
import numpy as np
import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import resize


class FormulaImageProcessor:
    """
    Handles image preprocessing: cropping margins, resizing, and padding.
    """
    def __init__(self, image_size: List[int] = [192, 672]):
        self.input_size = image_size 

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        
        # Normalize and threshold
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)
        if coords is None:
            return img
            
        x, y, w, h = cv2.boundingRect(coords)
        return img.crop((x, y, x + w, y + h))

    def prepare_input(self, img: Image.Image, random_padding: bool = False) -> np.ndarray:
        if img is None: return None
        try:
            img = self.crop_margin(img.convert("RGB"))
        except OSError:
            return None
        if img.height == 0 or img.width == 0: return None

        target_h, target_w = self.input_size
        img = resize(img, min(self.input_size)) 
        img.thumbnail((target_w, target_h))
        
        new_w, new_h = img.width, img.height
        delta_width = target_w - new_w
        delta_height = target_h - new_h

        if random_padding:
            pad_left = np.random.randint(low=0, high=delta_width + 1)
            pad_top = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_left = delta_width // 2
            pad_top = delta_height // 2
        padding = (pad_left, pad_top, delta_width - pad_left, delta_height - pad_top)
        return np.array(ImageOps.expand(img, padding, fill="white"))
    

def apply_bitmap(img, **kwargs):
    """Thresholds image to look like a binary bitmap"""
    img = img.copy()
    img[img < 200] = 0
    return img


def apply_erosion(img, **kwargs):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(2, 4, 2)))
    return cv2.erode(img, kernel, iterations=1)


def apply_dilation(img, **kwargs):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(2, 4, 2)))
    return cv2.dilate(img, kernel, iterations=1)


def get_train_transforms(image_size=[192, 672]):
    processor = FormulaImageProcessor(image_size)
    
    aug_pipeline = alb.Compose([
        alb.Compose([
            # 1. Bitmap / Thresholding
            alb.Lambda(name="Bitmap", image=apply_bitmap, p=0.05),

            # 2. Weather Effects (Built-in)
            alb.OneOf([
                alb.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1),
                alb.RandomSnow(brightness_coeff=2.5, snow_point_range=(0.3, 0.5), p=1),
                alb.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.1, p=1),
                alb.RandomShadow(p=1),
            ], p=0.2),

            # 3. Morphology (Erosion/Dilation)
            alb.OneOf([
                alb.Lambda(name="Erosion", image=apply_erosion),
                alb.Lambda(name="Dilation", image=apply_dilation),
            ], p=0.2),

            # 4. Geometry
            alb.Affine(
                translate_percent=0, scale=(0.85, 1.0), rotate=(-1, 1),
                border_mode=0, interpolation=3, fill=(255, 255, 255), p=1
            ),
            alb.GridDistortion(
                distort_limit=0.1, border_mode=0, interpolation=3, 
                fill=(255, 255, 255), p=.5
            )
        ], p=.5),

        # 5. Color & Noise
        alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        alb.GaussNoise(std_range=(0.01, 0.05), p=.2),
        alb.RandomBrightnessContrast(brightness_limit=.05, contrast_limit=(-.2, 0), brightness_by_max=True, p=0.2),
        alb.ImageCompression(quality_range=(95, 100), p=.3),
        
        # 6. Finalize
        alb.ToGray(p=1.0),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ])

    def transform(image):
        img_np = processor.prepare_input(image, random_padding=True)
        if img_np is None:
            img_np = np.full((image_size[0], image_size[1], 3), 255, dtype=np.uint8)
        return aug_pipeline(image=img_np)['image']

    return transform


def get_eval_transforms(image_size=[192, 672]):
    processor = FormulaImageProcessor(image_size)
    
    aug_pipeline = alb.Compose([
        alb.ToGray(p=1.0),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ])
    
    def transform(image):
        img_np = processor.prepare_input(image, random_padding=False)
        if img_np is None:
            img_np = np.full((image_size[0], image_size[1], 3), 255, dtype=np.uint8)
        return aug_pipeline(image=img_np)['image']

    return transform
