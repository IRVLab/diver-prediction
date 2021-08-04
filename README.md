# Diver Motion Prediction
This repository implements the paper **"Predicting the Future Motion of Divers for Enhanced Underwater Human-Robot Collaboration" by Tanmay Agarwal, Michael Fulton and Junaed Sattar, published in IROS 2021**

This implementation is based on the implementation of the Social-LSTM by Baran Nama, at this [repository](https://github.com/quancore/social-lstm)

The model directory contains pre-trained Vanilla and Social-LSTM models, both stabilized and unstabilized.
Default parameters in the training and testing scripts are the same as that mentioned in the paper.

It is highly recommended that you use a conda environment created with the environment.yml file. You can do so by running the following command once you have Anaconda installed:
```
conda env create --file envname.yml
conda activate diver_pred_env
```

To train a Social-LSTM, run

`python train.py`

To train a Vanilla-LSTM, run

`python vlstm_train_scuba.py`

You can test either model with the testing scripts, but need to specify the --method option (1 = Social-LSTM, 3 = Vanilla LSTM)

Example: 
```
python test_scuba_stabilized.py --method 1 # For a stabilized Social-LSTM
python test_scuba_unstabilized.py --method 3 # For an unstabilized Vanilla-LSTM
```

