# Deep Attentive Time Warping

## How to run
1. Download the UCR Archive from [here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
2. `python3 code/main-pre-training.py dataset.ID=1 dataset_path=../dataset/`
</br>(For dataset.ID, refer to the UCR_dataset_name.json)
2. `python3 code/main-metric-learning.py dataset.ID=1 dataset_path=../dataset/`