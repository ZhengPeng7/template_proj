import os
import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils import data
from torchvision import transforms

from preproc import cv_random_flip, random_crop, random_rotate, color_enhance, random_pepper
from config import Config


Image.MAX_IMAGE_PIXELS = None       # remove DecompressionBombWarning
config = Config()


class MyData(data.Dataset):
    def __init__(self, data_root, image_size, is_train=True):

        self.size_train = image_size
        self.size_test = image_size
        self.data_size = (config.size, config.size)
        self.is_train = is_train
        self.load_all = config.load_all
        if self.load_all:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.transform_label = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform_image = transforms.Compose([
                transforms.Resize(self.data_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.transform_label = transforms.Compose([
                transforms.Resize(self.data_size),
                transforms.ToTensor(),
            ])
        image_root = os.path.join(data_root, 'im')
        self.image_paths = [os.path.join(image_root, p) for p in os.listdir(image_root)]
        self.label_paths = [p.replace('/im/', '/gt/').replace('.jpg', '.png') for p in self.image_paths]
        if self.load_all:
            self.images_loaded, self.labels_loaded = [], []
            for image_path, label_path in tqdm(zip(self.image_paths, self.label_paths), total=len(self.image_paths)):
                self.images_loaded.append(
                    Image.fromarray(
                        cv2.cvtColor(cv2.resize(cv2.imread(image_path), (config.size, config.size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                    ).convert('RGB')
                )
                self.labels_loaded.append(
                    Image.fromarray(
                        cv2.resize(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE), (config.size, config.size), interpolation=cv2.INTER_LINEAR)
                    ).convert('L')
                )


    def __getitem__(self, index):

        if self.load_all:
            image = self.images_loaded[index]
            label = self.labels_loaded[index]
        else:
            image = Image.open(self.image_paths[index]).convert('RGB')
            label = Image.open(self.label_paths[index]).convert('L')

        # loading image and label
        if self.is_train:
            if 'flip' in config.preproc_methods:
                image, label = cv_random_flip(image, label)
            if 'crop' in config.preproc_methods:
                image, label = random_crop(image, label)
            if 'rotate' in config.preproc_methods:
                image, label = random_rotate(image, label)
            if 'enhance' in config.preproc_methods:
                image = color_enhance(image)
            if 'pepper' in config.preproc_methods:
                label = random_pepper(label)

        image, label = self.transform_image(image), self.transform_label(label)

        if self.is_train:
            return image, label
        else:
            return image, label, self.label_paths[index]

    def __len__(self):
        return len(self.image_paths)
