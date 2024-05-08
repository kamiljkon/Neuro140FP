### CREATING A DATALOADER
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import os

def find_classes(directory):
  """Finds the class folder names in a target directory."""
  classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
  if not classes:
    raise FileNotFoundError(f"Couldn't find any classes in {directory}")
  class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

  return classes, class_to_idx

try: 
    os.chdir('/Users/kamilkon/Desktop/Neuro140FP')
except FileNotFoundError:
    os.chdir('/home/u_481835/Neuro140FP')

image_path = "COMPOSITE/"

class ODIRDataset(Dataset):
    def __init__(self,
                 targ_dir: str,
                 transform=None):
        self.debug = print("Target directory:", os.path.abspath(targ_dir))
        self.paths = list(Path(targ_dir).glob("*/*.jpg"))
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
        try: 
            img = self.load_image(index)
            class_name = self.paths[index].parent.name
            class_idx = self.class_to_idx[class_name]
        except Exception as e:
            print(f"Error loading data at index {index}: {e}")
            raise

        if self.transforms:
            return self.transforms(img), class_idx
        else:
            return img, class_idx
        
