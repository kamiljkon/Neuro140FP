### CREATING A DATALOADER
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import os

def find_classes(directory):
    classes = sorted(set(entry.name.split('_')[0] for entry in os.scandir(directory) if entry.name.endswith('.jpg')))
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}")
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx

try: 
    os.chdir('/Users/kamilkon/Desktop/Neuro140FP')
except FileNotFoundError:
    os.chdir('/home/u_481835/Neuro140FP')

class ODIRDataset(Dataset):
    def __init__(self,
                 targ_dir: str,
                 transform=None):
        self.debug = print("Target directory:", os.path.abspath(targ_dir))
        self.paths = [os.path.join(targ_dir, f) for f in os.listdir(targ_dir) if f.endswith('.jpg')]  # Store full paths
        self.transforms = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
    
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        if os.path.isdir(image_path):
            raise ValueError("Directory found where an image expected: {}".format(image_path))
        return Image.open(image_path)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index):
        img = self.load_image(index)
        class_name = os.path.basename(self.paths[index]).split('_')[0]
        class_idx = self.class_to_idx[class_name]

        if self.transforms:
            return self.transforms(img), class_idx
        else:
            return img, class_idx
        
