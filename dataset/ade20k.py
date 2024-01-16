# ------------------------to finetune on ADE20K dataset -----------------------------------
import os

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from .base_dataset import BaseDataset



class ADE20K(BaseDataset):
    def __init__(self, 
                 root, 
                 object_file,
                 mode= "train",
                 num_classes=150,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(ADE20K, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)
        self.dataset_root = root
        self.object_file = object_file 

        self.file_list = list()
        self.mode = mode.lower()

        self.num_classes = num_classes
        self.ignore_label = ignore_label

        self.resize_size = crop_size
        self.multi_scale = multi_scale
        self.flip = flip
        self.upper_limit = 150 # actual classes in ADE20K dataset
        
        if mode == 'train':
            print("Train ::")
            img_dir = os.path.join(self.dataset_root, 'images/training')
            label_dir = os.path.join(self.dataset_root, 'annotations/training')
        elif mode == 'test':
            print("Validation ::")
            img_dir = os.path.join(self.dataset_root, 'images/validation')
            label_dir = os.path.join(self.dataset_root,
                                        'annotations/validation')
            
        self.files = self.read_files(img_dir, label_dir)
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self, img_dir:str, label_dir:str) -> list:
        files = []
        img_files = os.listdir(img_dir)
        print(int(len(img_files)/3)," images")
        label_files = [i.replace('.jpg', '.png') for i in img_files]
        for i in range(int(len(img_files)/3)):
            name, _ = os.path.splitext(img_files[i])
            img_path = os.path.join(img_dir, img_files[i])
            label_path = os.path.join(label_dir, label_files[i])
            files.append({
                "img": img_path,
                "label": label_path,
                "name": name,
            })
        return files
    

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = item["img"]
        label_path = item["label"]
        # Create a transform for resizing
        resize_transform = transforms.Resize(self.resize_size)

        with Image.open(image_path) as io:
            image = resize_transform(io)
            if image.mode in ['L', '1', 'I;16']:
                image = image.convert('RGB')
            image = np.array(image)

        size = image.shape

        if self.mode == 'inference':
            image = self.input_transform(image, city=False)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name
        
        with Image.open(label_path) as io:
            label = resize_transform(io)
            label = np.array(label)
        label = np.where(np.logical_and(label >= 21, label <= self.upper_limit), 255, label) # considering only the first 21 classes

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size, city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
