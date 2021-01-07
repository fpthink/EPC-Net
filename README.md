## Efficient 3D Point Cloud Feature Learning for Large-Scale Place Recognition

by Le Hui, Mingmei Cheng, Jin Xie, and Jian Yang

### Benchmark Datasets

We use the same benchmark datasets introduced in [PointNetVLAD](https://arxiv.org/abs/1804.03492) for point cloud based place recognition, and they can be downloaded [here](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx).

* Oxford dataset
* NUS (in-house) Datasets
  * university sector (U.S.)
  * residential area (R.A.)
  * business district (B.D.)



### Project Code

#### Pre-requisites

```
Python 3.6+
Tensorflow 1.12
CUDA 9.0
```

#### Dataset set-up

Download the zip file of the benchmark datasets found [here](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx) and extract the folder. Therefore, you have two folders: 1) benchmark_datasets/ and 2) EPC-Net/

#### Generate pickle files
We store the positive and negative point clouds to each anchor on pickle files that are used in our training and evaluation codes. The files only need to be generated once. The generation of these files may take a few minutes.
```
cd generating_queries/ 

# For training tuples in EPC-Net
python generate_training_tuples_baseline.py

# For network evaluation
python generate_test_sets.py
```

#### Model Training and Evaluation

* To train and evaluate EPC-Net, run the following command:

    ```
    # Train
    sh sh_train.sh path_epc-net configs/epc-net.yaml

    # Eval
    python evaluate.py --config configs/epc-net.yaml --log_dir exp/path_epc-net --model_name model_epoch20_iter18101.ckpt
    ```

* To train and evaluate EPC-Net-L, run the following command:

    ```
    # Train
    sh sh_train.sh path_epc-net-l configs/epc-net-l.yaml

    # Eval
    python evaluate.py --config configs/epc-net-l.yaml --log_dir exp/path_epc-net-l --model_name model_epoch20_iter18101.ckpt
    ```

#### Knowledge Distillation

* To transfer EPC-Net model as teacher model, run the following command:

  ```
  python ckpt_transfer.py --old_ckpt exp/path_epc-net/saved_model/model_epoch20_iter18101.ckpt  --new_path exp/path_epc-net-l-d/transfer_teacher --prefix teacher
  ```

* To train and evaluate EPC-Net-L-D, run the following command:

  ```
  # Train
  sh sh_kd_train.sh teacher_model_epoch20_iter18101.ckpt path_epc-net-l configs/epc-net-l-d.yaml
  
  # Eval
  python kd_evaluate.py --config configs/epc-net-l-d.yaml --log_dir exp/path_epc-net-l-d --model_name student_model_epoch20_iter18101.ckpt
  ```

#### Pre-trained Models

The pre-trained models for EPC-Net, EPC-Net-L, and EPC-Net-L-D have been upload in the exp/ folder:
```
# the pre-trained model for EPC-Net
exp/epc-net/saved_model/model_epoch22_iter18101.ckpt

# the pre-trained model for EPC-Net-L
exp/epc-net-l/saved_model/model_epoch13_iter18101.ckpt

# the pre-trained model for EPC-Net-L-D
exp/epc-net-l-d/saved_model/model_epoch20_iter18101.ckpt

# the transfer model for EPC-Net-L-D
exp/epc-net-l-d/transfer_teacher/model_epoch22_iter18101.ckpt
```

You can run the evaluation code to reproduce the results.

#### Acknowledgement

Our code refers to [PointNetVLAD](https://github.com/mikacuy/pointnetvlad).