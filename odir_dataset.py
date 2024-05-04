### CREATING A DATALOADER
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import os

dir = os.listdir('/Users/kamilkon/Desktop/Neuro140FP/ODIR/images')
os.chdir('/Users/kamilkon/Desktop/Neuro140FP/ODIR/images')

class ODIRDataset(Dataset):
    def __init__(self,
                 targ_dir: str,
                 transform=None):
        self.paths = list(os.listdir())
        self.transforms = transform
        self.classes = ['Male', 'Female']
        self.class_to_idx = {'Male': 0, 'Female': 1}
    
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index):
        img = self.load_image(index)
        class_name = self.paths[index].split('_')[0]
        class_idx = self.class_to_idx[class_name]

        if self.transforms:
            return self.transforms(img), class_idx
        else:
            return img, class_idx
        
### TRANSFORMS
import torch
from torchvision.transforms import v2

train_transforms = v2.Compose([
    v2.Resize(size=(300, 300)),
    v2.ToTensor()
    # v2.Normalize(mean=[0.2, 0.2, 0.2], std=[0.229, 0.224, 0.225]),
])

test_transforms = v2.Compose([
    v2.Resize(size=(300, 300)),
    v2.ToTensor()
])

### SPLITTING THE 'IMAGES' FOLDER INTO A TRAINING SET AND A TESTING SET
train_size = int(0.8 * len(dir))
test_size = len(dir) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dir, [train_size, test_size])

