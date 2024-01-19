# YOLO Model Evaluator
Effortlessly run object detection on CPU using only OpenCV with this YOLO model evaluation tool. This tool is designed to evaluate the performance of YOLO models by generating essential metrics, including **confusion matrix**, average **precision**, **recall**, and **mean average precision** (mAP). It provides a comprehensive analysis of your detection model's accuracy. Additionally, the tool offers the capability to generate annotations from the detection model and save them as text files in YOLO format. Simplify your model evaluation process with this straightforward and efficient solution.

## Installation:

<a name="installation"></a>

#### 1. Clone the repo

```shell
https://github.com/loginabhay/YOLO_Model_Evaluator.git
cd YOLO_Model_Evaluator
```

#### 2. Setup a Virtual Environment

This is assuming you have navigated to the `YOLO_Model_Evaluator` root after cloning it.

**NOTE:** This is tested under `python3.10`. For other python versions, you might encounter version conflicts.

```shell
# install required packages from pypi
python3 -m venv <your env name>
source <your env name>/bin/activate
```

#### 3. Intall Requirement to Run the Project

```shell
pip install -r requirements.txt
```

#### 4. Running the Project

- The project is divided in two major
  -   Generating the prediction or annotation from the given images
  -   Comparing the prediction with the ground truth in order to evaluate the metrices
 
#### 4.1 Generating Prediction

Set of arguments used are given below:

| Argument | Description | Flag |
| --- | --- | --- |
| -i | Path to the input image folder | Required |
| -o | Path to the output folder | Default='./image_predictions' |
| -c | Path to YOLO Model CFG file | Required |
| -m | Path to YOLO Model weight file | Required |
| -conf | Change for the confidence value | Default=0.3 |
| -n | Change for the network input shape | Default=416 |
| -nms | Change for the nms value | Default=0.5 |
| -ca | Creates annotations instead of predictions | Default=False |


**Example command**
```shell
python3 yolo_img_infer_write_prediction.py -i <path to input folder>
                    -c <path to input cfg> -m <path to input weights>
```

#### 4.2 Evaluating Metrices

Set of arguments used are given below:

| Argument | Description | Flag |
| --- | --- | --- |
| -ip | Path to the input prediction folder | Required |
| -o | Path to the output folder | Default='./generated_metrices' |
| -ig | Path to the ground truth folder | Required |
| -s | Save the generated metrices in the csv format | Default=True |
| -conf | Change for the confidence value | Default=0.3 |
| -nm | Path to the classes names file | Required |
| -nms | Change for the nms value | Default=0.5 |


**Example command**
```shell
python3 Evaluation_matrix_yolo.py -ip <path to input prediction folder>
                    -ig <path to input ground truth folder> -nm <path to classes names>
   
