# CoopHance: Cooperative Enhancement for Robustness of Deep Learning Systems

This is repository for paper CoopHance: Cooperative Enhancement for Robustness of Deep Learning Systems.

If you have any problem during using CoopHance, please open an issue to help us improve it.

## Requirements
we mainly require the Python=3.6, keras=2.1.3, tensorflow-gpu=1.12.0, numpy, pandas, scikit-learn,cv2.
Detailed packages are shown in requirements.yaml

## Structure
The "json" directory contains the config files of all experiments.
The generated adversarial examples are saved in "adversarial" directory.
We save all the trained model in "model" directory.
The "data" directory is used for saving dataset.


## Quick Usage
We provide the weights of model.
You can follow the following steps to generate the adversarial examples and detect them.

### Download the Models Weights.
First download model [weights](https://github.com/ZQ-Struggle/CoopHance/releases/download/0.1/model.zip) and unzip them to the "model" directory.

### Get into the 'src' Directory.

``cd src``

### Generate Adversarial Examples on Unprotected Model.
Use ``scripts/attack_existing.py`` to generate the adversarial examples on unprotected model.

``python scripts/attack_existing.py -c svhn_res.json -a``

To change the attack methods, please modify the 'adversarial_type' item in json files. We support five attacks, and please use 'pgd', 'fgsm', 'CW', 'jsma', and 'deepfool' to set the item. The generated adversarial examples are save in 'adversarial' directory, which can be changed by modifying the 'adversarial_dir' item in json files.

### Generate the Adversarial Examples on Enhanced Model.
Use ``scripts/attack_new.py`` to generate the adversarial examples on enhanced.

``python scripts/attack_new.py -c svhn_concat_res.json -a -b``

The json files that contain "concat" means they are used for experiments of defending againist adversarial example generation.

### Defend against Attacks.
Use ``scripts/defend.py`` to defend against attacks.

``python scripts/defend.py -c svhn_res.json -t 0.05``

``python scripts/defend.py -c svhn_concat_res.json -t 0.05``

Above two command can be used to defend against existing adversarial examples and new adversarial examples.
Please generate the corresponding adversarial examples first.
The attack method can also be set by modifying 'adversarial_type' item in json file.

## Detailed Usage.
We provide the detail scripts to train the classifier and regulator.

#### Train Classifier
The classifier can be trained with ``scripts/train_classifier.py``

``python scripts/train_classifier.py -c svhn_res.json``

The weights will be saved in ``model/cifar`` directory.
To use that newly trained classifier, we need to set the 'model_path' item in json files as its save path.

#### Train Regulator
The regulator needs the adaptive noising training, to train it, we need to use the ``scripts/calculate_distance_distribution.py`` and ``scripts/train_regulator.py``

We first collect the $\mu$ and $\sigma$ for confusion layers. To run following scripts, a classiferi is needed, and the 'model_path' item should be set with the path of that classifier.

``python scripts/calculate_distance_distribution.py -c svhn_sta.json -a all``

The json files with 'sta' suffix are the config used to collect $\mu$ and $\sigma$.
The collected data are saved in ``json/svhn(cifar)_statistic.json``.

Except the collected data, the middle weights is also dumped to the 'model' directory.
Before training the regulator, we need set the 'noise_autoencoder_weights' item of json file.

Then, we could train the regulator with collected data to initialize the confusion layers. Ensure that the 'noise_autoencoder_weights' is set in 'svhn_res.json'

``python scripts/train_regulator.py -c svhn_res.json 