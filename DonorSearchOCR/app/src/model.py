import timm
import torch
import logging
import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
handler_m = logging.StreamHandler()
formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
handler_m.setFormatter(formatter_m)
m_logger.addHandler(handler_m)

IMG_SIZE = 512
DEVICE = "cpu"
MEAN = [1.3638, 1.5033, 1.6996]
STD = [1.0195, 1.0658, 1.0899]

def prepare_davit(model_name: str, params_path: str, num_classes: int, device: str):
    """Initiates DaViT Visual Transformer as a model
    Loads model weights for the model"""
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(params_path, map_location=torch.device(device))['model_state_dict'])
    m_logger.info(f'model loaded')
    return model

def evaluate_image(image_path, model, mean=MEAN, std=STD):
    def transform_sample(image):
        w, h = image.shape[0], image.shape[1]
        max_wh = np.max([w, h])
        aug = A.Compose([
            A.PadIfNeeded(min_height=max_wh,
                          min_width=max_wh,
                          border_mode=0,
                          value=(255, 255, 255)),
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(
                mean=mean,
                std=std),
            ToTensorV2()
        ])
        return aug(image=image)['image']
    try:
        image = Image.open(image_path)
        img = np.array(image.convert('RGB'))
    except PIL.UnidentifiedImageError:
        m_logger.error(f'something wrong with image')
        status = 'Fail'
        return status, image_path
    angles = {0: 0, 1: 90, 2: 180, 3: 270}
    transformed_img = transform_sample(img).unsqueeze(dim=0)
    transformed_img = transformed_img.to(DEVICE)
    with torch.no_grad():
        model.eval()
        try:
            output = model(transformed_img)
            index = output.data.cpu().numpy().argmax()
            rotated_img = image.rotate(angles[index], expand=True)
            status = 'OK'
            m_logger.info(f'recognition completed')
        except:
            status = 'Fail'
            m_logger.info(f'recognition failed')
            return status
    return rotated_img, index, status



