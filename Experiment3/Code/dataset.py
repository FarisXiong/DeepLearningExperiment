import os
import torch.utils.data.dataset as dataset
from PIL import Image
from torchvision.transforms import transforms
from transformers import AutoFeatureExtractor

REDUCE_IO_DEBUG = False  # 调试模式，减少 IO 次数


def getDataset(config):
    """
    获取 train + dev + test 的 dataset，以及标签到整数的双向字典

    Args:
        config: 模型配置

    Returns:
        ================================================================================

        当 config.fold_k == 1 时，使用 k 折交叉验证，使用 config 中的 fold_k 作为 k 的值，
        此时返回的数据结构是 (k_fold_dataset, test_dataset), (label_str2int, label_int2str)

        其中 k_fold_dataset 是经过 k 折交叉验证处理后的 train 和 dev 的数据迭代器，结构为
        [
            (train_dataset, dev_dataset),
            (train_dataset, dev_dataset),
            ...
            (train_dataset, dev_dataset)
        ]

        ================================================================================

        当 config.fold_k != 1 时，不用 k 折交叉验证，使用 config 中的 train_dev_frac 比例进行分割，
        此时返回的数据结构是 (train_dev_dataset, test_dataset), (label_str2int, label_int2str)

        其中 train_dev_dataset 是 train 和 dev 的数据迭代器，结构为
        [
            (train_dataset, dev_dataset)
        ]

        相当于 config.fold_k == 1 的特殊情况

        ================================================================================

        train_dataset 和 dev_dataset 中的每一项都是 (tensor, label_int)
        test_dataset 中的每一项都是 (tensor, image_name)
        label_str2int 和 label_int2str 是 str 类型的标签和 int 类型的标签相互转换的字典
    """

    is_transformer = False
    if config.model == "SwinTransformer":
        is_transformer = True

    if config.fold_k == 1:
        # 不用 k 折交叉验证，使用 config 中的 train_dev_frac 比例进行分割

        train_data, dev_data, label_int2str, label_str2int = build_train_dev_data(config)
        train_dataset = MyDataset(train_data, "train_dev", is_transformer)
        dev_dataset = MyDataset(dev_data, "train_dev", is_transformer)

        test_data = build_test_data(config)
        test_dataset = MyDataset(test_data, "test", is_transformer)

        train_dev_dataset = [(train_dataset, dev_dataset)]

        return (train_dev_dataset, test_dataset), (label_str2int, label_int2str)
    else:
        # 用 k 折交叉验证，使用 config 中的 fold_k 作为 k 的值

        k_fold_data, label_int2str, label_str2int = build_train_dev_data_k_fold(config)
        k = config.fold_k
        k_fold_dataset = []
        for i in range(k):
            k_fold_dataset.append([])
        for i in range(k):
            k_fold_dataset[i].append(
                MyDataset(k_fold_data[i][1], "train_dev", is_transformer))  # k_fold_dataset[i][0] = train
            k_fold_dataset[i].append(
                MyDataset(k_fold_data[i][0], "train_dev", is_transformer))  # k_fold_dataset[i][1] = dev

        test_data = build_test_data(config)
        test_dataset = MyDataset(test_data, "test", is_transformer)

        return (k_fold_dataset, test_dataset), (label_str2int, label_int2str)


def build_train_dev_data(config):
    """
    读入数据，并数据增强，生成一个数据例

    Args:
        config: 模型配置

    Returns:
        返回 data_train, data_dev, label_int2str, label_str2int
    """

    # 双向索引 label
    label_int2str = {}  # { 0: "Black-grass", 1: "Charlock", ...}
    label_str2int = {}  # { "Black-grass": 0, "Charlock": 1, ...}

    path_label_train = []
    path_label_dev = []

    path = config.train_path
    path_label_tmp = []
    for label_int, label_dir in enumerate(os.listdir(path)):
        label_int2str[label_int] = label_dir
        label_str2int[label_dir] = label_int
        image_names = os.listdir(path + label_dir)
        for image_name in image_names:
            path_label_tmp.append((path + label_dir + "/" + image_name, label_int))
    for index, item in enumerate(path_label_tmp):
        if index % (config.train_dev_frac + 1) < config.train_dev_frac:
            path_label_train.append(item)
        else:
            path_label_dev.append(item)

    # 根据 config 进行数据增强
    trans_no_aug, trans_no_flp_aug, trans_v_flp_aug, trans_h_flp_aug, trans_all_flp_aug = data_augment(config)

    data_train = []  # data_train 每一项为二元组 (tensor, label)
    data_dev = []  # data_dev 每一项为二元组 (tensor, label)
    if REDUCE_IO_DEBUG:
        for pl in path_label_train:
            data_train.append(pl)  # 本行用于调试，减少 IO 次数
        for pl in path_label_dev:
            data_dev.append(pl)  # 本行用于调试，减少 IO 次数
    else:
        for pl in path_label_train:
            img = Image.open(pl[0]).convert('RGB')
            if config.data_augmentation == 'none':
                data_train.append((trans_no_aug(img), pl[1]))
            else:
                data_train.append((trans_no_flp_aug(img), pl[1]))
                data_train.append((trans_v_flp_aug(img), pl[1]))
                data_train.append((trans_h_flp_aug(img), pl[1]))
                data_train.append((trans_all_flp_aug(img), pl[1]))
        for pl in path_label_dev:
            img = Image.open(pl[0]).convert('RGB')
            if config.data_augmentation == 'none':
                data_dev.append((trans_no_aug(img), pl[1]))
            else:
                data_dev.append((trans_no_flp_aug(img), pl[1]))

    return data_train, data_dev, label_int2str, label_str2int


def build_train_dev_data_k_fold(config):
    """
    读入数据，并数据增强，生成一组 k fold 数据例

    Args:
        config: 模型配置

    Returns:
        返回 k_fold_data, label_int2str, label_str2int
    """

    # 双向索引 label
    label_int2str = {}  # { 0: "Black-grass", 1: "Charlock", ...}
    label_str2int = {}  # { "Black-grass": 0, "Charlock": 1, ...}

    path_label = []

    path = config.train_path
    for label_int, label_dir in enumerate(os.listdir(path)):
        label_int2str[label_int] = label_dir
        label_str2int[label_dir] = label_int
        image_names = os.listdir(path + label_dir)
        for image_name in image_names:
            path_label.append((path + label_dir + "/" + image_name, label_int))

    # 根据 config 进行数据增强
    trans_no_aug, trans_no_flp_aug, trans_v_flp_aug, trans_h_flp_aug, trans_all_flp_aug = data_augment(config)

    data = []  # data 每一项为二元组 (tensor, label)
    if REDUCE_IO_DEBUG:
        for pl in path_label:
            data.append(pl)  # 本行用于调试，减少 IO 次数
    else:
        for pl in path_label:
            img = Image.open(pl[0]).convert('RGB')
            if config.data_augmentation == 'none':
                data.append((trans_no_aug(img), pl[1]))
            else:
                data.append((trans_no_flp_aug(img), pl[1]))
                data.append((trans_v_flp_aug(img), pl[1]))
                data.append((trans_h_flp_aug(img), pl[1]))
                data.append((trans_all_flp_aug(img), pl[1]))
        for pl in path_label:
            img = Image.open(pl[0]).convert('RGB')
            if config.data_augmentation == 'none':
                data.append((trans_no_aug(img), pl[1]))
            else:
                data.append((trans_no_flp_aug(img), pl[1]))

    k = config.fold_k

    # 将所有数据均等分到 k 个桶中
    buckets_k = []
    for i in range(k):
        buckets_k.append([])
    for index, data in enumerate(data):
        for i in range(k):
            if index % k == i:
                buckets_k[i].append(data)

    # 每次将一个桶作为 dev，其他桶合并作为 train
    k_fold_data = []
    for i in range(k):
        k_fold_data.append([])
    for i in range(k):
        k_fold_data[i].append(buckets_k[i])  # k_fold_data[i][0] = dev
        k_fold_data[i].append([])  # k_fold_data[i][1] = train
        for j in range(k):
            if i != j:
                k_fold_data[i][1] += buckets_k[j]

    return k_fold_data, label_int2str, label_str2int


def build_test_data(config):
    """
    读入数据，并数据增强，生成一个数据例

    Args:
        config: 模型配置

    Returns:
        返回 data
    """
    path_name = []
    path = config.test_path
    image_names = os.listdir(path)
    for image_name in image_names:
        path_name.append((path + image_name, image_name))

    trans_no_aug, trans_no_flp_aug, _, _, _ = data_augment(config)

    data = []  # data 每一项为二元组 (tensor, name)
    if REDUCE_IO_DEBUG:
        for pn in path_name:
            data.append(pn)  # 本行用于调试，减少 IO 次数
    else:
        for pn in path_name:
            img = Image.open(pn[0]).convert('RGB')
            if config.data_augmentation == 'none':
                data.append((trans_no_aug(img), pn[1]))
            else:
                data.append((trans_no_flp_aug(img), pn[1]))
    return data


def data_augment(config):
    """
    进行数据增强

    Args:
        config: 模型配置

    Returns:
        返回
        (
            没有增强的 transform.Compose,
            带有全部的颜色增强，但是没有翻转的 transform.Compose,
            带有全部的颜色增强，和垂直翻转的 transform.Compose,
            带有全部的颜色增强，和水平翻转的 transform.Compose,
            带有全部的颜色增强，和全部的翻转的 transform.Compose
        )
    """

    trans_no_aug = []  # 没有增强的 transform
    trans_no_flp_aug = []  # 带有全部的颜色增强，但是没有翻转的 transform
    trans_v_flp_aug = []  # 带有全部的颜色增强，和垂直翻转的 transform
    trans_h_flp_aug = []  # 带有全部的颜色增强，和水平翻转的 transform
    trans_all_flp_aug = []  # 带有全部的颜色增强，和全部的翻转的 transform

    trans_no_aug.append(transforms.Resize((224, 224)))  # 缩放到 224x224
    trans_no_aug.append(transforms.ToTensor())

    trans_no_flp_aug.append(transforms.RandomPosterize(bits=2, p=1))  # 分色
    trans_no_flp_aug.append(transforms.RandomSolarize(threshold=192.0, p=1))  # 曝光
    trans_no_flp_aug.append(transforms.Resize((224, 224)))  # 缩放到 224x224
    trans_no_flp_aug.append(transforms.ToTensor())

    trans_v_flp_aug.append(transforms.RandomVerticalFlip(p=1))  # 垂直翻转
    trans_v_flp_aug.append(transforms.RandomPosterize(bits=2, p=1))  # 分色
    trans_v_flp_aug.append(transforms.RandomSolarize(threshold=192.0, p=1))  # 曝光
    trans_v_flp_aug.append(transforms.Resize((224, 224)))  # 缩放到 224x224
    trans_v_flp_aug.append(transforms.ToTensor())

    trans_h_flp_aug.append(transforms.RandomHorizontalFlip(p=1))  # 水平翻转
    trans_h_flp_aug.append(transforms.RandomPosterize(bits=2, p=1))  # 分色
    trans_h_flp_aug.append(transforms.RandomSolarize(threshold=192.0, p=1))  # 曝光
    trans_h_flp_aug.append(transforms.Resize((224, 224)))  # 缩放到 224x224
    trans_h_flp_aug.append(transforms.ToTensor())

    trans_all_flp_aug.append(transforms.RandomHorizontalFlip(p=1))  # 水平翻转
    trans_all_flp_aug.append(transforms.RandomVerticalFlip(p=1))  # 垂直翻转
    trans_all_flp_aug.append(transforms.RandomPosterize(bits=2, p=1))  # 分色
    trans_all_flp_aug.append(transforms.RandomSolarize(threshold=192.0, p=1))  # 曝光
    trans_all_flp_aug.append(transforms.Resize((224, 224)))  # 缩放到 224x224
    trans_all_flp_aug.append(transforms.ToTensor())

    return transforms.Compose(trans_no_aug), \
           transforms.Compose(trans_no_flp_aug), transforms.Compose(trans_v_flp_aug), \
           transforms.Compose(trans_h_flp_aug), transforms.Compose(trans_all_flp_aug)


class MyDataset(dataset.Dataset):
    """
    对Dataset重写
    """

    def __init__(self, data, choice: str, isTransformer: bool):
        """
        Args:
            data: buildData 的一个实例
            choice: 只能取值为 "train_dev" 或 "test"
            isTransformer: 模型是否是 SwinTransformer
        """

        super(MyDataset, self).__init__()
        self.choice = choice
        self.isTransformer = isTransformer
        self.Data = data
        self.length = len(data)
        if self.isTransformer:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    def __len__(self):
        assert self.length >= 0
        return self.length

    def __getitem__(self, item):
        data = self.Data[item][0]
        if self.isTransformer:
            data = self.feature_extractor(self.Data[item][0], return_tensors="pt").data['pixel_values'].squeeze(0)

        if self.choice == 'train_dev':
            inputs = {
                'data': data,
                'label': self.Data[item][1]
            }
        elif self.choice == 'test':
            inputs = {
                'data': data,
                'name': self.Data[item][1]
            }
        else:
            # 不应该到这个分支
            inputs = {
                'data': data,
                'label': self.Data[item][1]
            }
        return inputs