# VQVAE As Defense Against Adverserial Attack
#### Attacks:
- FGSM
- I-FGSM
- MI_FGSM

#### Defense:
- KMeans with k=20
- VQVAE with noise 0.25, 0.5,0.75

#### Setup:
- Run these Commands:
```sh
git clone https://github.com/Harsh-Gupta9897/VQVAE_as_Defense_Against_Adverserial_Attack
cd VQVAE_as_Defense_Against_Adverserial_Attack
pip install -r requirements.txt
cd Adverserial_training
python3 train.py
```

### Load Pretrained_Model
- Model is already configured at pre-trained, you can also set it trainable mode by changing hyperparameters mentioned in train.py file for both vqvae and ConvNet for MNIST i.e. **pre_trained_vqvae = False** and **pre_trained_model=False** to train it.
- KMeans is disabled but you can also make it usable by setting **Kmeans_defense_show= True**

### Report:
Report **VQVAE_As_Defense_On_Adversarial_Attacks_final.pdf** is there in folder, which contains the analysis report and Explaination.
