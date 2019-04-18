import chainer
import cv2
import matplotlib.pyplot as plt
import glob
from chainercv.datasets import DirectoryParsingLabelDataset
from chainer.datasets import split_dataset_random

class PreprocessDataset(chainer.dataset.DatasetMixin):
    def __init__(self, pair):
        self.base = pair
        
    def __len__(self):
        return len(self.base)
    
    def get_example(self, i):
        
        image, label = self.base[i]
        image = image /255.0
        
        return (image, label)

def load_dataset(path):
    dataset = DirectoryParsingLabelDataset(path)
    dataset = PreprocessDataset(dataset)
    train_size = int(len(dataset) * 0.9)
    train, test = split_dataset_random(dataset, train_size, seed=0)
    return train, test		
