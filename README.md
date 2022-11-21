# AI6103-YOLOv4-Reimplementaion

AI6103 Group Project - Reimplementation of YOLOv4 object detection

This is the code reimplementation of YOLOv4 object detection for the AI6103 group project of MSAI in Nanyang Technological University.

Group Members: Ge Wenyue, Li Jiayi, Sun Jialiang, Yang Zhuochen, Zhu Qiankun

## Environment setting

1. Create virtual environment:

   ```shell
   conda create -n AI6103-YOLOv4-Reimplementaion python=3.8
   conda activate AI6103-YOLOv4-Reimplementaion
   pip install -r requirements.txt
   ```

## Training Steps

1. Download VOC2007 dataset from following websites and unzip them:

   - http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
   - http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar

2. Download pre-trained `CSPDarknet53.pth` model from our [Google Drive](https://drive.google.com/file/d/1xqj_yx1Y_jz_UPHzzgNfNAcADtQSbDII/view?usp=share_link) and save in the root folder.

3. Directory structure is as follows:

   ```txt
   train.py
   evals.py
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

4. Start training:

   ```shell
   python train.py
   ```

## Evaluation

Calculate and plot mAP:

```shell
python evals.py
```

### mAP curve

The trained model on VOC2007 dataset and the mAP curse is shown in the figure below (May Modify the `model_path` and `image_path`in evals.py first):

![mAP Curve](mAP_Curve.png)

### best mAP

|    Class    |   AP   |  mAP   |
| :---------: | :----: | :----: |
|  aeroplane  | 71.42% | 70.54% |
|   bicycle   | 84.71% |
|    bird     | 74.05% |
|    boat     | 50.76% |
|   bottle    | 59.52% |
|     bus     | 82.31% |
|     car     | 90.21% |
|     cat     | 84.73% |
|    chair    | 59.88% |
|     cow     | 60.25% |
| diningtable | 58.80% |
|     dog     | 81.90% |
|    horse    | 85.75% |
|  motorbike  | 80.61% |
|   person    | 86.49% |
| pottedplant | 41.80% |
|    sheep    | 30.83% |
|    sofa     | 69.42% |
|    train    | 85.11% |
|  tvmonitor  | 72.27% |

## Detection Image

![Detection Image](resultSample1.png)
