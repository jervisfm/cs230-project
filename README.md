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


We'll be using the CASIA dataset that's available at https://www.kaggle.com/sophatvathana/casia-dataset. Please download this and put under
a data/ folder. E.g. the Casia2 dataset would be at data/CASIA2/...

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

