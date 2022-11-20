# AI6103-YOLOv4-Reimplementaion

AI6103 Group Project - Reimplementation of YOLOv4 object detection

## Environment setting

1. Create virtual environment:

   ```shell
   conda create -n yolov4 python=3.8
   conda activate yolov4
   pip install -r requirements.txt
   ```

## Train

1. Download VOC2007 dataset from following website and unzip them:

   - http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
   - http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

2. Download pre-trained `CSPDarknet53.pth` model from [Google Drive](https://drive.google.com/file/d/1xqj_yx1Y_jz_UPHzzgNfNAcADtQSbDII/view?usp=share_link).

3. Directory structure is as follows:

   ```txt
   VOC2007_train (VOC2007_test)
   ├───Annotations
   ├───ImageSets
   │   ├───Layout
   │   ├───Main
   │   └───Segmentation
   ├───JPEGImages
   ├───SegmentationClass
   └───SegmentationObject
   ```

4. start training:

   ```shell
   conda activate yolov4
   python train.py
   ```

## Evaluation

Calculate and plot mAP:

```shell
conda activate yolov4
python evals.py
```

### mAP curve

The trained model on VOC2007 dataset and the mAP curse is shown in the figure below (May Modify the `model_path` and `image_path`in evals.py first):

![mAP Curve](mAP_Curve.png)

### best mAP

![mAP](map.png)

## Detection Image

![Detection Image](result sample1.png)
