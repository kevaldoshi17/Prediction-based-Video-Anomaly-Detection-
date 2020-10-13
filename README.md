# Online Anomaly Detection in Surveillance Videos with Asymptotic Bounds on False Alarm Rate

Implementation in Jupyter, Python 3.6, tensorflow. 


## 0. All the files along with the checkpoints can be downloaded from http://doi.org/10.5281/zenodo.3676032. Any changes to the order of files in the directory might raise errors so please follow the uploaded file structure. To get the final results, you can directly run MONAD.ipynb. To train models from scratch, following are the instructions to run the code. 

## 1. Installation (Anaconda with python3.6 installation is recommended)
* Install 3rd-package dependencies of python (listed in requirements.txt)
```
numpy==1.14.1
scipy==1.0.0
matplotlib==2.1.2
tensorflow-gpu==1.4.1
tensorflow==1.4.1
Pillow==5.0.0
pypng==0.0.18
scikit_learn==0.19.1
opencv-python==3.2.0.6
```

## 2. Installation of Darknet (YOLOv3)

Please follow the instructions provided at https://pjreddie.com/darknet/yolo/

## 3. Get Nominal Mean Square Error 

* Running the sript (as ped2 for examples) and cd into **Codes** folder at first.
```shell
python inference.py  --dataset  ped2    \
                    --test_folder  ../Data/ped2/training/frames      \
                    --gpu  1    \
                    --snapshot_dir    checkpoints/pretrains/ped2

The executed results are saved in ped2_train.

## 4. Get Testing Mean Square Error

* Running the sript (as ped2 for examples) and cd into **Codes** folder at first.
```shell
python inference.py  --dataset  ped2    \
                    --test_folder  ../Data/ped2/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    checkpoints/pretrains/ped2

The executed results are saved in ped2_test.    

## 5. Training from scratch (here we use ped2 and avenue datasets for examples)

* Set hyper-parameters
The default hyper-parameters, such as $\lambda_{init}$, $\lambda_{gd}$, $\lambda_{op}$, $\lambda_{adv}$ and the learning rate of G, as well as D, are all initialized in **training_hyper_params/hyper_params.ini**. 
* Running script (as ped2 or avenue for instances) and cd into **Codes** folder at first.
```shell
python train.py  --dataset  ped2    \
                 --train_folder  ../Data/ped2/training/frames     \
                 --test_folder  ../Data/ped2/testing/frames       \
                 --gpu  0       \
                 --iters    80000
```
* Model selection while training
In order to do model selection, a popular way is to testing the saved models after a number of iterations or epochs (Since there are no validation set provided on above all datasets, and in order to compare the performance with other methods, we just choose the best model on testing set). Here, we can use another GPU to listen the **snapshot_dir** folder. When a new model.cpkt.xxx has arrived, then load the model and test. Finnaly, we choose the best model. Following is the script.
```shell
python inference.py  --dataset  ped2    \
                     --test_folder  ../Data/ped2/testing/frames       \
                     --gpu  1
```
Run **python train.py -h** to know more about the flag options or see the detials in **constant.py**.
```shell
Options to run the network.

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU    the device id of gpu.
  -i ITERS, --iters ITERS
                        set the number of iterations, default is 1
  -b BATCH, --batch BATCH
                        set the batch size, default is 4.
  --num_his NUM_HIS    set the time steps, default is 4.
  -d DATASET, --dataset DATASET
                        the name of dataset.
  --train_folder TRAIN_FOLDER
                        set the training folder path.
  --test_folder TEST_FOLDER
                        set the testing folder path.
  --config CONFIG      the path of training_hyper_params, default is
                        training_hyper_params/hyper_params.ini
  --snapshot_dir SNAPSHOT_DIR
                        if it is folder, then it is the directory to save
                        models, if it is a specific model.ckpt-xxx, then the
                        system will load it for testing.
  --summary_dir SUMMARY_DIR
                        the directory to save summaries.
  --psnr_dir PSNR_DIR  the directory to save psnrs results in testing.
  --evaluate EVALUATE  the evaluation metric, default is compute_auc
```

## 6. Run Object Detection 

Run YOLOv3 on the training and testing datasets. Save the bounding boxes and extracted classes along with their confidence levels. Saved bounding boxes are also uploaded as json files.

## 7. Run MONAD

The proposed sequential detector is implemented in MONAD.ipynb. Use Anaconda 3.6 to acheive the final AUC score. 
