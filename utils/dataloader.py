import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class CycleGanDataset(Dataset):
    def __init__(self, annotation_lines_A, annotation_lines_B, input_shape):
        super(CycleGanDataset, self).__init__()

        self.annotation_lines_A = annotation_lines_A
        self.annotation_lines_B = annotation_lines_B 
        self.length_A           = len(self.annotation_lines_A)
        self.length_B           = len(self.annotation_lines_B)
        
        self.input_shape        = input_shape

    def __len__(self):
        return max(self.length_A, self.length_B)

    def __getitem__(self, index):
        index_A = index % self.length_A
        image_A = Image.open(self.annotation_lines_A[index_A].split(';')[1].split()[0])
        image_A = cvtColor(image_A).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)
        image_A = np.array(image_A, dtype=np.float32)
        image_A = np.transpose(preprocess_input(image_A), (2, 0, 1))
        
        index_B = index % self.length_B
        image_B = Image.open(self.annotation_lines_B[index_B].split(';')[1].split()[0])
        image_B = cvtColor(image_B).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)
        image_B = np.array(image_B, dtype=np.float32)
        image_B = np.transpose(preprocess_input(image_B), (2, 0, 1))
        return image_A, image_B

def CycleGan_dataset_collate(batch):
    images_A = []
    images_B = []
    for image_A, image_B in batch:
        images_A.append(image_A)
        images_B.append(image_B)
    images_A = torch.from_numpy(np.array(images_A, np.float32))
    images_B = torch.from_numpy(np.array(images_B, np.float32))
    return images_A, images_B
