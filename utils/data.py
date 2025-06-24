import numpy as np
import os
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

# ImageNet-A is the version defined at https://github.com/zhoudw-zdw/RevisitingCIL from here: 
#   @article{zhou2023revisiting,
#        author = {Zhou, Da-Wei and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
#        title = {Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need},
#        journal = {arXiv preprint arXiv:2303.07338},
#        year = {2023}
#    }

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train, args,isCifar=False):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        if isCifar:
            size = input_size
        else:
            size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    use_path = False

    train_trsf=build_transform(True, None,True)
    test_trsf=build_transform(False, None,True)
    common_trsf = [
        # transforms.ToTensor(),
    ]

    class_order = np.arange(100).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        do_download=True
        if os.path.isfile('/data/cifar-100-python/train'):
            do_download=False
        train_dataset = datasets.cifar.CIFAR100("/data/", train=True, download=do_download)
        test_dataset = datasets.cifar.CIFAR100("/data/", train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNetR(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023), download from https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW
        train_dir = "/data/imagenet_r/train"
        test_dir = "/data/imagenet_r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023), download from  https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL
        train_dir = "/data/imagenet-a/train/"
        test_dir = "/data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class colon(iData):
    '''
    Dataset Name: Colon Dataset
    Task: Classification
    Data Format: Custom `.npz` files for each subfolder group (group1.npz, group2.npz, etc.)
    '''

    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.has_valid = True
        # Define dataset groups and their respective number of classes
        self._dataset_info = [('group1', 2), ('group2', 2), ('group3', 3), ('group4', 2)]  # Example groups
        self.use_path = False
        self.img_size = img_size if img_size is not None else 224  # Default to 224x224 for colon images

        # Define transformations
        self.train_trsf = [
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=self.img_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
        ]
        self.test_trsf = []
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Standard ImageNet values
        ]

        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]
        self.class_order = list(range(sum(self._dataset_inc)))

    def shuffle_order(self, seed):
        random.seed(seed)
        random.shuffle(self._dataset_info)
        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]

    def getdata(self, src_dir):
        train_data = np.array([])
        train_targets = np.array([])
        test_data = np.array([])
        test_targets = np.array([])
        
        known_class = 0  # Keeps track of label offsets across groups

        for data_flag in self._dataset_info:
            # Load the .npz file for the current group
            npz_file = np.load(os.path.join(src_dir, "{}.npz".format(data_flag[0])))

            # Extract all images and labels
            all_images = npz_file['images']
            all_labels = npz_file['labels']

            # Shuffle the data
            indices = np.arange(len(all_images))
            np.random.shuffle(indices)
            all_images = all_images[indices]
            all_labels = all_labels[indices]

            # Determine the split index (7:3 ratio)
            split_index = int(len(all_images) * 0.7)

            # Split into training and testing sets
            train_imgs, test_imgs = all_images[:split_index], all_images[split_index:]
            train_labels, test_labels = all_labels[:split_index], all_labels[split_index:]

            # Adjust the labels to be incremental across groups
            train_labels = train_labels + known_class
            train_imgs = train_imgs.astype(np.uint8)
            train_labels = train_labels.astype(np.uint8)

            test_labels = test_labels + known_class
            test_imgs = test_imgs.astype(np.uint8)
            test_labels = test_labels.astype(np.uint8)

            # Combine data across all groups
            train_data = np.concatenate([train_data, train_imgs]) if len(train_data) != 0 else train_imgs
            train_targets = np.concatenate([train_targets, train_labels]) if len(train_targets) != 0 else train_labels

            test_data = np.concatenate([test_data, test_imgs]) if len(test_data) != 0 else test_imgs
            test_targets = np.concatenate([test_targets, test_labels]) if len(test_targets) != 0 else test_labels

            # Update the known class offset
            known_class = len(np.unique(train_targets))
            print(f"Unique train labels: {np.unique(train_targets)}")

        return train_data, train_targets, test_data, test_targets


    def download_data(self):
        src_dir = "/home/yoyo/project/SAFE/data/colon_processed"  # Path to your `.npz` files
        self.train_data, self.train_targets, self.test_data, self.test_targets = self.getdata(src_dir)
        def remap_labels(labels):
            unique_labels = np.unique(labels)
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            print(f"Label remapping: {label_mapping}")  # Debugging: Show the remapping
            return np.vectorize(label_mapping.get)(labels)

        # Remap the labels to a continuous range
        self.train_targets = remap_labels(self.train_targets)
        self.test_targets = remap_labels(self.test_targets)

        # Debugging outputs
        print(f"Unique train labels after remapping: {np.unique(self.train_targets)}")
        print(f"Unique test labels after remapping: {np.unique(self.test_targets)}")
        
class skin8(iData):
    '''
    Dataset Name:   Skin8 (ISIC_2019_Classification)
    Task:           Skin disease classification
    Data Format:    600x450 color images.
    Data Amount:    3555 for training, 705 for validationg/testing
    Class Num:      8
    Notes:          balanced each sample num of each class

    Reference:      
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(224, (0.8, 1)),
            ]
        
        self.test_trsf = []
        
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.60298395, 0.4887822, 0.46266827], std=[0.25993535, 0.24081337, 0.24418062]),
        ]

        self.class_order = np.arange(8).tolist()
    
    def getdata(self, fn, img_dir):
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                full_path = os.path.join(img_dir, temp[0])
                if os.path.exists(full_path):  # Check if the file actually exists
                    data.append(full_path)
                    targets.append(int(temp[1]))
                else:
                    print(f"Skipping: {full_path} does not exist.")
                #data.append(os.path.join(img_dir, temp[0]))
                #targets.append(int(temp[1]))
        return np.array(data), np.array(targets)

    def download_data(self):
        base_dir = "/home/yoyo/project/SAFE/"
        #base_dir = os.path.join(os.environ["DATA"], "ISIC2019")
        train_dir = os.path.join(base_dir, "Data", "skin8_train_500.txt")
        test_dir = os.path.join(base_dir, "Data", "skin8_test_500.txt")
        
        self.train_data, self.train_targets = self.getdata(train_dir, base_dir)
        self.test_data, self.test_targets = self.getdata(test_dir, base_dir)
        
class medmnist(iData):
    '''
    Dataset Name:   MedMNistv2
    Task:           Diverse classification task (binary/multi-class, ordinal regression and multi-label)
    Data Format:    32x32 color images.
    Data Amount:    Consists of 12 pre-processed 2D datasets and 6 pre-processed 3D datasets with diverse data scales (from 100 to 100,000)
    
    Reference: https://medmnist.com/
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        # 由于 MedMnist 中有些子数据集是多标签或者是3D的，这里没有使用
        # 以下表示中，字符为子数据集名字, 数字为子数据集中包含的类别数
        self.has_valid = True
        self._dataset_info = [('bloodmnist',8), ('organmnist_axial',11), ('pathmnist',9), ('tissuemnist',8)]
        # self._dataset_info = [('bloodmnist',8), ('organamnist',11), ('dermamnist',7), ('pneumoniamnist',2), ('pathmnist',9),
        #                     ('breastmnist',2), ('tissuemnist',8), ('octmnist',4)]
        self.use_path = False
        self.img_size = img_size if img_size != None else 28 # original img size is 28
        self.train_trsf = [
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.24705882352941178),
        ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
        ]
        self.test_trsf = []
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]

        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]
        self.class_order = list(range(sum(self._dataset_inc)))
    
    def shuffle_order(self, seed):
        random.seed(seed)
        random.shuffle(self._dataset_info)
        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]

    def getdata(self, src_dir):
        train_data = np.array([])
        train_targets = np.array([])
        test_data = np.array([])
        test_targets = np.array([])
        
        known_class = 0
        for data_flag in self._dataset_info:
            npz_file = np.load(os.path.join(src_dir, "{}.npz".format(data_flag[0])))

            train_imgs = npz_file['train_images']
            train_labels = npz_file['train_labels']

            test_imgs = npz_file['test_images']
            test_labels = npz_file['test_labels']

            if len(train_imgs.shape) == 3:
                train_imgs = np.expand_dims(train_imgs, axis=3)
                train_imgs = np.repeat(train_imgs, 3, axis=3)

                test_imgs = np.expand_dims(test_imgs, axis=3)
                test_imgs = np.repeat(test_imgs, 3, axis=3)

            train_labels = train_labels + known_class
            train_imgs = train_imgs.astype(np.uint8)
            train_labels = train_labels.astype(np.uint8)

            test_labels = test_labels + known_class
            test_imgs = test_imgs.astype(np.uint8)
            test_labels = test_labels.astype(np.uint8)
            
            train_data = np.concatenate([train_data, train_imgs]) if len(train_data) != 0 else train_imgs
            train_targets = np.concatenate([train_targets, train_labels]) if len(train_targets) != 0 else train_labels
            

            test_data = np.concatenate([test_data, test_imgs]) if len(test_data) != 0 else test_imgs
            test_targets = np.concatenate([test_targets, test_labels]) if len(test_targets) != 0 else test_labels

            known_class = len(np.unique(train_targets))
        
        return train_data, np.squeeze(train_targets,1), test_data, np.squeeze(test_targets,1)


    def download_data(self):
        # 在我们划分的数据集中，后面这一项有几种选项可选
        # mymedmnist_1_in_5: 对不均衡的 PathMNIST,OCTMNIST, TissueMNIST, OrganAMNIST, 将其随机下采样为其原来的1/5, 其余不变
        # mymedmnist_1000_300: 对所有子数据集, 均随机采样成样本数分别为 1000,300 的训练集和测试集
        # origin: MedMNIST 原始数据集
        #src_dir = os.path.join(os.environ["DATA"], "medmnist")
        src_dir = "/home/yoyo/project/SAFE/medmnist"
        self.train_data, self.train_targets, self.test_data, self.test_targets = self.getdata(src_dir)
        # print(self.train_data.shape)
        # print(self.test_data.shape)

        # print(len(np.unique(self.train_targets))) # output: 51
        # print(len(np.unique(self.test_targets))) # output: 51
        
class blood(iData):
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = 224 if img_size is None else img_size
        self.train_trsf = [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomResizedCrop(224, (0.8, 1)),
            ]
        
        self.test_trsf = []
        
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.60298395, 0.4887822, 0.46266827], std=[0.25993535, 0.24081337, 0.24418062]),
        ]

        self.class_order = np.arange(8).tolist()
    
    def getdata(self, fn, img_dir):
        print(fn)
        file = open(fn)
        file_name_list = file.read().split('\n')
        file.close()
        data = []
        targets = []
        for file_name in file_name_list:
            temp = file_name.split(' ')
            if len(temp) == 2:
                relative_path = temp[0].replace("Data/", "")
                full_path = os.path.join(img_dir, relative_path)
                if os.path.exists(full_path):  # Check if the file actually exists
                    data.append(full_path)
                    targets.append(int(temp[1]))
                else:
                    print(f"Skipping: {full_path} does not exist.")
                #data.append(os.path.join(img_dir, temp[0]))
                #targets.append(int(temp[1]))
        return np.array(data), np.array(targets)

    def download_data(self):
        base_dir = "/home/yoyo/project/RanPAC/data/PBC_data"
        train_dir = os.path.join(base_dir, "blood_train.txt")
        test_dir = os.path.join(base_dir, "blood_test.txt")
        
        self.train_data, self.train_targets = self.getdata(train_dir, base_dir)
        self.test_data, self.test_targets = self.getdata(test_dir, base_dir)
        
class covid(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def __init__(self,use_input_norm):
        if use_input_norm:
            self.common_trsf = [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    def download_data(self):
        # as per Zhou et al (2023) download from https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc
        train_dir = "/home/yoyo/project/RanPAC/data/covid/train"
        test_dir = "/home/yoyo/project/RanPAC/data/covid/test"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)