import os

from torchvision import transforms as T

from data.base_dataset import BaseDataset, __crop
from data.image_folder import make_dataset
from PIL import Image


def _get_transforms():
    return T.Compose([
        T.Lambda(lambda img: __crop(img, (64, 29), 192)),
        T.Resize(64),
        T.ToTensor(),
        T.Normalize(mean=3*(0.5,), std=3*(0.5,))
    ])


class CLEVRDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=3, output_nc=3,
                            crop_size=128,
                            num_slots=11, display_ncols=11)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.transform = _get_transforms()
        p = os.path.join(opt.dataroot, 'images', 'train' if opt.isTrain else 'test')
        self.A_paths = sorted(make_dataset(p, opt.max_dataset_size))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
