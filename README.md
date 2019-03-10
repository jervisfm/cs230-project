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


## TODOs

* Add dev loss per epoch of training.
* Add graphs for training loss / dev performance as training continues.
* Try out a simple CNN network.
* Work on Midterm Report.
* Test out SVM baseline: DONE.
