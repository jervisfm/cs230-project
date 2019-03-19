# Introduction

A project to build a neural network that can detect fake/altered images from real images.


# Team members
* Jervis
* Raj
* Yash

# Project Setup
We use Conda for package / dependency management.  You can get a copy of conda for your platform here: https://conda.io/en/latest/miniconda.html

One can then create a suitable environment as follows:

```
$ conda env create -f environment.yml
```

You can then activate the environment with:

```
$ source activate cs230Project
```

To pull in any updated dependencies, one can execute
```
$ conda env update
```


We'll be using the CASIA dataset that's available at https://www.kaggle.com/sophatvathana/casia-dataset. Please download this and put under
a data/ folder. E.g. the Casia2 dataset would be at data/CASIA2/...


### Screen Session Management
We also use GNU screen for session management. To check for list of available sessions
run
```
$ screen -ls
```

We usually have a single `cs230` session that we all share. To attach to this session, just
run
```
$ screen -x cs229
```

Some helpful screen commands:
* Open a new window in session - Ctrl + A, c
* Go to next window in session - Ctrl + A, n
* Go to previous window in session - Ctrl + A, p


# Initial baseline results
Running an extremely simple logistic regression with only 10 epochs of training, we got the following result:

```
Acurracy: 0.7757704569606801
Training time(secs): 82.76839876174927
Max training iterations: 10
Training time / Max training iterations: 8.276839876174927
Classification report:               precision    recall  f1-score   support

        Real       0.78      0.99      0.87       735
        Fake       0.33      0.02      0.05       206

   micro avg       0.78      0.78      0.78       941
   macro avg       0.56      0.51      0.46       941
weighted avg       0.68      0.78      0.69       941
```

Running at 100 iterations
```
Acurracy: 0.7513283740701382
Training time(secs): 569.8644268512726
Max training iterations: 100
Training time / Max training iterations: 5.698644268512726
Classification report:               precision    recall  f1-score   support

        Real       0.84      0.84      0.84       735
        Fake       0.43      0.43      0.43       206

   micro avg       0.75      0.75      0.75       941
   macro avg       0.64      0.64      0.64       941
weighted avg       0.75      0.75      0.75       941

```


Running at 400 iterations
```
Acurracy: 0.7120085015940489
Training time(secs): 2077.106256723404
Max training iterations: 400
Training time / Max training iterations: 5.19276564180851
Classification report:               precision    recall  f1-score   support

        Real       0.85      0.77      0.81       735
        Fake       0.38      0.52      0.44       206

   micro avg       0.71      0.71      0.71       941
   macro avg       0.62      0.64      0.62       941
weighted avg       0.75      0.71      0.73       941
```


Running at 1000 iterations
```
Acurracy: 0.6971307120085016
Training time(secs): 9655.94488477707
Max training iterations: 1000
Training time / Max training iterations: 9.65594488477707
Classification report:               precision    recall  f1-score   support

        Real       0.84      0.76      0.80       735
        Fake       0.36      0.47      0.41       206

   micro avg       0.70      0.70      0.70       941
   macro avg       0.60      0.62      0.60       941
weighted avg       0.73      0.70      0.71       941
```

PyTorch Logistic regression

```
-------------------

Dev Acurracy: 74%
Train Acurracy: 96%
Training time(secs): 76004.55631327629
Max training iterations: 10000
Training time / Max training iterations: 7.6004556313276295

```

## SVM baseline

Mosts svm kernels didn't do too well, but poly got reasoanble results. Note however, this 
was sensitive to the number of iterations. We found 10 iterations worked best.

### Poly kernel
```
Acurracy: 0.7619553666312433
Training time(secs): 8.219857215881348
Max training iterations: 10
Training time / Max training iterations: 0.8219857215881348
Classification report:               precision    recall  f1-score   support

        Real       0.78      0.97      0.86       735
        Fake       0.05      0.00      0.01       206

   micro avg       0.76      0.76      0.76       941
   macro avg       0.41      0.49      0.44       941
weighted avg       0.62      0.76      0.68       941
```

## Model training commands

Example to train an inception

```
$ python cnn.py --max_iter 10 --model_name=inception_pretrained --data_folder=data/processed_casia2_224  --cuda=True --l2_regularization=0.0 --experiment_name "l2reg=0.0_iter=10"
```


### Model Image size mappining
* Inception -> 299x299
* Densenet -> 224x224
* VGG16 -> 224x224
* Resnet -> 224x224


## Promising results
* densenet_pretrained_l2reg=0_iter=10 - 67%
* resnet_pretrained_l2reg=0_iter=10 - 68%
* results_resnet_pretrained_l2reg=0_iter=3_trainallweights - 70%
* results_resnet_pretrained_l2reg=0_iter=50_trainallweights - 74% (This was achieved on the 12th epoch)
* resnet_pretrained_l2reg=0.001_iter=20_batchSize=50_learningRate=0.00025_trainallweights - 75% (5th epoch)
* resnet_pretrained_l2reg=0_iter=20_batchSize=50_learningRate=0.0005_trainallweights - 75% (15th epoch)

VGG
```
python cnn.py --batch_size=50 --unfreeze_ratio=0.5 --max_iter 3 --model_name=vgg16_pretrained --data_folder=data/processed_casia2_224 --cuda=True --l2_regularization=0 --unfreeze_all_weights=True --experiment_name l2reg=0_iter=3_unfreezeratio=0.5_trainallweights

Got dev accuracy of 61 / train accuracy 60  but didn't OOM. Could allow training for longer.
```


### AWS instance
VM 1
```
$ ssh -i ~/.ssh/cs230proj.pem ubuntu@ec2-34-220-41-38.us-west-2.compute.amazonaws.com
```
VM 2
```
$ ssh -i ~/.ssh/cs230proj.pem ubuntu@ec2-54-214-145-187.us-west-2.compute.amazonaws.com
```


#### Param tuning

Best dev accuracy of 75% seen at epoch 4.

```
['python', 'cnn.py', '--max_iter', '11', '--batch_size', '70', '--learning_rate', '0.00010113231069171439', '--model_name=resnet_pretrained', '--data_folder=data/processed_casia2_224', '--cuda=True', '--l2_regul
arization=0.014409286623908741', '--unfreeze_all_weights=True', '--experiment_name', 'l2reg=0.014409286623908741_iter=11_trainallweights=True_unfreezeratio=0.5733779454085566']
```
## TODOs

* Add dev loss per epoch of training.
* Add graphs for training loss / dev performance as training continues.
* Try out a simple CNN network.
* Work on Midterm Report.
* Test out SVM baseline: DONE.

## Best Dev Accuracy with ELA
92%
```
$  python cnn.py --max_iter 15 --model_name=resnet_pretrained --data_folder=data/processed_casia2_224_ela --cuda=True --l2_regularization=0 --unfreeze_all_weights=True --experiment_name l2reg=0_iter=15_trainallwe
ights_ela

Dev Acurracy: 90%
Best Dev Acurracy over training: 92% seen at epoch 5
Dev Precision: 0.8685714285714285%
Dev Recall: 0.890625%
Dev F1 Score: 0.8794599807135969%
Train Acurracy: 95%
Training time(secs): 1664.8222482204437
Max training iterations: 15
Training time / Max training iterations: 110.98814988136292
Confusion matrix, without normalization
```


## Final hyper-tuned model.
This one achieved 94% accuracy at the end.
https://github.com/jervisfm/cs230-project/blob/a1a9e5c04c2cb923a4c345fcec65f3ec2ebc60fa/results/cnn_results_resnet_pretrained_l2reg%3D0.005_iter%3D20_batchSize%3D100_learningRate%3D0.00025_trainallweights_raj
```
Model File: results/cnn_checkpoint_resnet_pretrained_l2reg=0.005_iter=20_batchSize=100_learningRate=0.00025_trainallweights_raj.h5

-------------------

Dev Acurracy: 94%
Best Dev Acurracy over training: 94% seen at epoch 5
Dev Precision: 0.9211538461538461%
Dev Recall: 0.935546875%
Dev F1 Score: 0.9282945736434108%
Train Acurracy: 98%
Training time(secs): 2168.5053930282593
Max training iterations: 20
Training time / Max training iterations: 108.42526965141296
```