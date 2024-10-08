import torchvision
import numpy as np

import os

from copy import deepcopy
from data.data_utils import subsample_instances
from config import imagenet_gcd_root, imagenet_root, imagenet_200_root

class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

def subsample_dataset(dataset, idxs, absolute=True):
    mask = np.zeros(len(dataset)).astype('bool')
    if absolute==True:
        mask[idxs] = True
    else:
        idxs = set(idxs)
        mask = np.array([i in idxs for i in dataset.uq_idxs])

    dataset.samples = [s for m, s in zip(mask, dataset.samples) if m==True]
    dataset.targets = [t for m, t in zip(mask, dataset.targets) if m==True]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs



def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2

def get_imagenet_datasets(train_transform, test_transform, train_classes=range(50),
                           prop_train_labels=0.8, split_train_val=False, seed=0, use_coarse_label=False):

    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    # subsampled_100_classes = np.random.choice(range(1000), size=(100,), replace=False)
    # subsampled_100_classes = np.sort(subsampled_100_classes)
    # print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    # cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    whole_training_set = ImageNetBase(root=os.path.join(imagenet_root, 'train'), transform=train_transform)

    # Reset dataset
    # whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    # whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    # whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    # whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(imagenet_root, 'val'), transform=test_transform)
    # test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    # test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    # test_dataset.targets = [s[1] for s in test_dataset.samples]
    # test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    # test_dataset.target_transform = None

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


def get_imagenet_200_datasets(train_transform, test_transform, train_classes=range(100),
                          prop_train_labels=0.8, split_train_val=False, seed=0, use_coarse_label=False, args=None):
    np.random.seed(seed)

    whole_training_set = ImageNetBase(root=os.path.join(imagenet_200_root, 'train'), transform=train_transform)

    # Reset dataset
    # whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    # whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    # whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    # whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(imagenet_200_root, 'val'), transform=test_transform)

    # Reset test set
    # test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    # test_dataset.targets = [s[1] for s in test_dataset.samples]
    # test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    # test_dataset.target_transform = None

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


def get_imagenet_100_gcd_datasets(train_transform, test_transform, train_classes=range(50),
                           prop_train_labels=0.5, split_train_val=False, seed=0):

    np.random.seed(seed)

    ### >>>
    subsampled_100_classes = np.arange(100)
    ### <<<
    
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(imagenet_gcd_root, 'train'), transform=train_transform)
    # NOTE: use GCD paper IN split
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None
    

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(imagenet_gcd_root, 'val'), transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets



def get_imagenet_100_gcd_datasets_with_gcdval(train_transform, test_transform, train_classes=range(50), 
                                   prop_train_labels=0.5, split_train_val=True, seed=0, val_split=0.1):

    np.random.seed(seed)

    ### >>>
    subsampled_100_classes = np.arange(100)
    ### <<<
    
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(imagenet_gcd_root, 'train'), transform=train_transform)
    # NOTE: use GCD paper IN split
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)
    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)), absolute=False)
    
    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled, val_split=val_split)
    val_dataset_labelled = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled.transform = test_transform
    train_dataset_labelled = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    
    train_idxs, val_idxs = get_train_val_indices(train_dataset_unlabelled, val_split=val_split)
    val_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset_unlabelled), val_idxs)
    val_dataset_unlabelled.transform = test_transform
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset_unlabelled), train_idxs)
    
    val_dataset_unlabelled.transform = test_transform
    

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(imagenet_gcd_root, 'val'), transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    print(f'total={len(whole_training_set)} train={len(train_dataset_labelled)} {len(train_dataset_unlabelled)} val={len(val_dataset_labelled)} {len(val_dataset_unlabelled)} test={len(test_dataset)}')

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': [val_dataset_labelled, val_dataset_unlabelled],
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_imagenet_100_datasets(None, None, split_train_val=False,
                               train_classes=range(50), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
    
    # imagenet_original_root = r'/home/sheng/dataset/imagenet-100-original'
    # train_dataset_labelled = ImageNetDataset(root=os.path.join(imagenet_original_root, 'train'), anno_file=SPLIT_FILE[0], transform=None)
    # print(list(set(train_dataset_labelled.targets)))
    # train_dataset_unlabelled = ImageNetDataset(root=os.path.join(imagenet_original_root, 'train'), anno_file=SPLIT_FILE[1], transform=None, start_idx=len(train_dataset_labelled))
    # print(list(set(train_dataset_unlabelled.targets)))