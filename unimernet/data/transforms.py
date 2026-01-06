from typing import List, Tuple
from PIL import Image, ImageOps
import numpy as np
import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import random


class FormulaImageProcessor:
    """
    Handles image preprocessing: cropping margins, resizing, and padding.
    """
    def __init__(self, image_size: List[int] = [192, 672]):
        self.input_size = image_size
        self.target_h = image_size[0]
        self.target_w = image_size[1]
        self.min_dim = min(image_size)

    def _compute_resize_dims(self, w: int, h: int) -> Tuple[int, int]:
        if w < h:
            scale = self.min_dim / w
        else:
            scale = self.min_dim / h
        
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w > self.target_w or new_h > self.target_h:
            scale = min(self.target_w / new_w, self.target_h / new_h)
            new_w, new_h = int(new_w * scale), int(new_h * scale)
        
        return max(1, new_w), max(1, new_h)

    def prepare_input(self, img: Image.Image, random_padding: bool = False) -> np.ndarray:
        if img is None:
            return None
        
        try:
            img = img.convert("RGB")
        except OSError:
            return None
            
        img_array = np.array(img)
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        if gray.mean() < 100:  # Dark mode detection
            img_array = 255 - img_array
            gray = 255 - gray
        
        max_val, min_val = gray.max(), gray.min()
        
        if max_val != min_val:
            threshold = int(min_val) + (int(max_val) - int(min_val)) * 200 // 255
            binary = (gray < threshold).astype(np.uint8)
            
            coords = cv2.findNonZero(binary)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                if w > 0 and h > 0:
                    img_array = img_array[y:y+h, x:x+w]
        
        h, w = img_array.shape[:2]
        if h == 0 or w == 0:
            return None
        
        new_w, new_h = self._compute_resize_dims(w, h)
        img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        delta_w = self.target_w - new_w
        delta_h = self.target_h - new_h

        if random_padding:
            pad_left = random.randint(0, delta_w)
            pad_top = random.randint(0, delta_h)
        else:
            pad_left = delta_w // 2
            pad_top = delta_h // 2

        pad_right = delta_w - pad_left
        pad_bottom = delta_h - pad_top
        
        # cv2.copyMakeBorder is faster than np.pad for this use case
        return cv2.copyMakeBorder(
            img_array, 
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)
        )


_EROSION_KERNELS = [
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (i, j))
    for i in range(2, 4) for j in range(2, 4)
]
_DILATION_KERNELS = _EROSION_KERNELS


def apply_bitmap(img, **kwargs):
    result = img.copy()
    result[result < 200] = 0
    return result


def apply_erosion(img, **kwargs):
    kernel = random.choice(_EROSION_KERNELS)
    return cv2.erode(img, kernel, iterations=1)


def apply_dilation(img, **kwargs):
    kernel = random.choice(_DILATION_KERNELS)
    return cv2.dilate(img, kernel, iterations=1)


def get_train_transforms(image_size=[192, 672]):
    processor = FormulaImageProcessor(image_size)
    fallback_image = np.full((image_size[0], image_size[1], 3), 255, dtype=np.uint8)
    
    aug_pipeline = alb.Compose([
        alb.Compose([
            alb.Lambda(name="Bitmap", image=apply_bitmap, p=0.1),

            alb.OneOf([
                alb.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1),
                alb.RandomSnow(brightness_coeff=2.5, snow_point_range=(0.3, 0.5), p=1),
                alb.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.1, p=1),
                alb.RandomShadow(p=1),
            ], p=0.3),

            alb.OneOf([
                alb.Lambda(name="Erosion", image=apply_erosion),
                alb.Lambda(name="Dilation", image=apply_dilation),
            ], p=0.3),

            alb.Affine(
                translate_percent=0, 
                scale=(0.85, 1.15), 
                rotate=(-2, 2),
                border_mode=0, 
                interpolation=3, 
                fill=(255, 255, 255), 
                p=1
            ),

            alb.GridDistortion(
                distort_limit=0.05, 
                border_mode=0, 
                interpolation=3, 
                fill=(255, 255, 255), 
                p=0.5
            )
        ], p=0.65),

        alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        alb.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        alb.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=(-0.2, 0.2), 
            brightness_by_max=True, 
            p=0.3
        ),
        alb.ImageCompression(quality_range=(95, 100), p=0.3),
        
        alb.ToGray(p=1.0),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ])

    def transform(image):
        img_np = processor.prepare_input(image, random_padding=True)
        if img_np is None:
            img_np = fallback_image
        return aug_pipeline(image=img_np)['image']

    return transform


def get_eval_transforms(image_size=[192, 672]):
    processor = FormulaImageProcessor(image_size)
    fallback_image = np.full((image_size[0], image_size[1], 3), 255, dtype=np.uint8)
    
    aug_pipeline = alb.Compose([
        alb.ToGray(p=1.0),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        ToTensorV2(),
    ])
    
    def transform(image):
        img_np = processor.prepare_input(image, random_padding=False)
        if img_np is None:
            img_np = fallback_image
        return aug_pipeline(image=img_np)['image']

    return transform