# python native
import os
import argparse

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A
import ttach as tta

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from src.utils.data_utils import load_yaml_config
from src.plmodules.smp_module import SmpModule
from src.utils.data_utils import load_yaml_config

# 데이터 경로를 입력하세요

IMAGE_ROOT = "/home/taeyoung4060ti/바탕화면/부스트코스/level2-cv-semanticsegmentation-cv-06-lv3/data/test/DCM"

CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}


class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()

        return image, image_name


# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(height, width)


def test(model, data_loader, thr=0.5):
    transform = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Scale(scales=[1, 1.5, 2]),
        ]
    )

    model = model.to("cuda:0").eval()
    tta_model = tta.SegmentationTTAWrapper(model, transform, merge_mode="mean")

    rles = []
    filename_and_class = []
    with torch.no_grad():

        for step, (images, image_names) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images = images.to("cuda:0")
            outputs = tta_model(images)

            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class


def main(config_path, weights):
    print(f"Loading config from {config_path} ...")
    config = load_yaml_config(config_path=config_path)
    print(f"Completed!")

    tf = A.Resize(1024, 1024)

    test_dataset = XRayInferenceDataset(transforms=tf)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    model = SmpModule.load_from_checkpoint(
        model_config_path=config["path"]["model"],
        train_config_path=config_path,
        checkpoint_path=weights,
    )

    model = model.model

    rles, filename_and_class = test(model, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )

    os.makedirs("tta_outputs", exist_ok=True)
    df.to_csv("tta_outputs/TTA_output.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to the weights"
    )
    args = parser.parse_args()
    main(args.config, args.weights)
