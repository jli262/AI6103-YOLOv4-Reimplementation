from net.train_pipeline import TrainPipeline
from net.dataset import VOCDataset
from utils.augmentation_utils import YoloAugmentation, ColorAugmentation

anchor = [[[142, 110], [192, 243], [459, 401]], [[36, 75], [76, 55], [72, 146]], [[12, 16], [19, 36], [40, 28]],]

# train config
config = {
    "n_classes": len(VOCDataset.VOC2007_classes),
    "darknet_path": "CSPdarknet53.pth",
    "image_size": 416,
    "freeze_batch_size": 8,
    "freeze": True,
    "freeze_epoch": 50,
    "anchors": anchor,
    "lr": 1e-2,
    "batch_size": 4,
    "max_epoch": 200,
    "start_epoch": 0,
    "num_workers": 4
}

def loadDataset():
    root = 'VOC2007_train'
    datasetDefine(root)


def datasetDefine(root):
    global dataset
    dataset = VOCDataset(
        root,
        'train',
        transformer=YoloAugmentation(config['image_size']),
        colorTransformer=ColorAugmentation(config['image_size']),
        keepDifficult=True,
        ifMosaic=True,
        ifMixup=True,
        imageSize=config["image_size"]
    )


# load dataset
loadDataset()

if __name__ == '__main__':
    train_pipeline = TrainPipeline(dataset=dataset, **config)
    train_pipeline.train()
