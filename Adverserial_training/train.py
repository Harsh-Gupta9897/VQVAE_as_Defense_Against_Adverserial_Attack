

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,datasets
from tqdm import trange, tqdm
import os
from PIL import Image
np.random.seed(42)
torch.manual_seed(42)


from defenses import KMeans
from models import CNet
from vqvae import VQVAE

from utils import save_img_tensors_as_grid,fit_Dataset, accuracy , train_vqvae
from pipeline import test_pipeline

import warnings
warnings.filterwarnings("ignore")


################## Initializing Directories ############
if not os.path.exists("../losses/"): 
    os.mkdir("../losses/")

if not os.path.exists("../accuracies/"): 
    os.mkdir("../accuracies/")


if not os.path.exists("../images/"): 
    os.mkdir("../images/")

if not os.path.exists("../models/"): 
    os.mkdir("../models/")


if not os.path.exists("../vqvae_outputs/"): 
    os.mkdir("../vqvae_outputs/")
#################################### hyperparameters ###############################################
batch_size= 128
lr = 2e-4
workers = 2
image_size = 64
n_channels = 3
num_epochs = 10
beta = 0.999
weight_decay=0.1
pretrained_model = True    ## if true, it will load pretrained from ../models

############# Attack Hyperparameters #################
epsilons = [0,0.001,0.005,0.2,0.3]        #0,0.006,0.01,0.03,0.05,0.1,
attacks = ["fgsm","ifgsm","mifgsm"]   #,"ifgsm","mifgsm"

############# VQVAE Hyperparameters #################

noises = [0,0.25,0.5,0.75]
pre_trained_vqvae=True   ## if true, it will load pretrained from ../models
vqvae_epochs = 6

######### KMeans Hyperparameters #################
Kmeans_defense_show = False # if true , it will take a day to train on given iteration and all attacks with different epsilon values
num_clusters_list = [20]    # i.e, k List
kmeans_iter = 20

########## cpu config ##############
ngpu = 8
device = torch.device("cuda:4" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



##################  Loss Functions #########################
bce_loss = nn.BCELoss()
nlll_loss = nn.NLLLoss()
mse_loss = nn.MSELoss()



#################################### transforms ###############################################

##### transforms #####
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0),(1.0))])


###### dataset #######
train_dataset = datasets.MNIST('../data', train=True,download=True,transform=transform)
test_dataset = datasets.MNIST('../data', train=False,transform=transform)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])


    
##### dataloader #######
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=True)
print("Training data:",len(train_loader),"Validation data:",len(val_loader),"Test data: ",len(test_loader))





#################################### Training the model ###############################################


###### Model Inititalization ############
model = CNet().to(device)
# model = nn.DataParallel(model, list(range(ngpu)))



###### Optimizer ############
optimizer = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999))

###### Scheduler ###########
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)



if not pretrained_model:
    ###### train_model #######
    loss,val_loss=fit_Dataset(model=model,optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=nlll_loss,
        device=device,
        epochs=num_epochs)
    
    ####### plot Loss #######
    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(1,num_epochs+1), loss, "*-",label="Loss")
    plt.plot(np.arange(1,num_epochs+1), val_loss,"o-",label="Val Loss")
    plt.xlabel("Num of epochs")
    plt.legend()
    plt.savefig("../losses/Loss_MNIST")

    torch.save(model.state_dict(), "../models/MNIST_model_weights.pth")
else:
    model.load_state_dict(torch.load("../models/MNIST_model_weights.pth"))








#### Accuracy ######
train_acc,test_acc = accuracy(train_loader,test_loader,model,device)
print(f'Train Accuracy: {train_acc*100} \t Test Accuracy: {test_acc*100}')








################# Accuracies with different Attacks (without defense) ################
print(" ###############  Different Attacks (without defense)  #######################")
for attack in attacks:
    print(f"\n########## Attack {attack} ##########\n")
    accuracies = []
    examples = []
    for eps in epsilons:
        acc, ex = test_pipeline(model,test_loader,eps,attack,device=device)
        accuracies.append(acc)
        examples.append(ex)
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.title(attack)
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig(f'../accuracies/accuracy_{attack}')

    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
#             
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.savefig(f'../images/pertubed_images_{attack}')






################################# VQVAE Training (for Defense) ########################################

defense_models = {}

#### VQVAE Models ##########
#### hyperparameters VQVAE ########
use_ema = True
model_args = {
    "in_channels": 1,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
    
}


if not pre_trained_vqvae:
    print("\n#####Model Training Started for different versions_of_vqvae##############")
    for noise in noises:
        print("\n#####Training VQVAE with noise={noise}##############")

        defense_models[f'vqvae_{noise}']= VQVAE(**model_args,noise_std=noise,epochs=vqvae_epochs).to(device)
        train_params = [params for params in defense_models[f'vqvae_{noise}'].parameters()]
        lr = lr
        optimizer = optim.Adam(train_params, lr=lr)

        train_vqvae(vqvae_model=defense_models[f'vqvae_{noise}'],
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    loss_function=mse_loss,
                    device=device,
                    use_ema=use_ema
        )
        torch.save(defense_models[f'vqvae_{noise}'].state_dict(), f"../models/vqvae_{noise}_weights.pth")
else:
    print("\n#####Model Loading for different versions_of_vqvae##############")
    for noise in noises:
        defense_models[f'vqvae_{noise}']= VQVAE(**model_args,noise_std=noise).to(device)
        defense_models[f'vqvae_{noise}'].load_state_dict(torch.load(f"../models/vqvae_{noise}_weights.pth"))
    print("Done!\n")

############################### VQVAE (Defense) After Training VQVAE ###################################
for noise in noises:
    print(f"\n\n########## defense using VQVAE & noise={noise} ##########\n")
    for attack in attacks:
        accuracies = []
        examples = []
        print(f"\n\n########## Attack {attack} ##########")

        for eps in epsilons:
            acc, ex = test_pipeline(model,test_loader,eps,attack,defense=defense_models[f'vqvae_{noise}'],have_defense='vqvae',device=device)
            accuracies.append(acc)
            examples.append(ex)

        plt.figure(figsize=(5,5))
        plt.plot(epsilons, accuracies, "*-")
        plt.title(attack)
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.savefig(f'../accuracies/perturbed_VQVAE_{noise}_defense_{attack}_attack.png')

        cnt = 0

        plt.figure(figsize=(8,10))
        for i in range(len(epsilons)):
            for j in range(len(examples[i])):
                cnt += 1
                plt.subplot(len(epsilons),len(examples[0]),cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
                orig,adv,ex = examples[i][j]
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(ex, cmap="gray")
        plt.tight_layout()
        plt.savefig(f"../images/perturbedImage_VQVAE_{noise}_defense_{attack}_attack.png")


########################## VQVAE Defense End #################################




########################### KMeans (Cluster and Defense) ##########################################


if Kmeans_defense_show:
    for k in num_clusters_list:
        defense = KMeans(k,kmeans_iter)
        print(f"\n\n########## defense using KMeans & K={k}##########\n")
        for attack in attacks:
            accuracies = []
            examples = []
            print(f"\n\n########## Attack {attack} using defense KMeans & K={k}##########\n")

            for eps in epsilons:
                acc, ex = test_pipeline(model,test_loader,eps,attack,defense=defense,have_defense='kmeans',device=device)
                accuracies.append(acc)
                examples.append(ex)

            plt.figure(figsize=(5,5))
            plt.plot(epsilons, accuracies, "*-")
            plt.title(attack)
            plt.xlabel("Epsilon")
            plt.ylabel("Accuracy")
            plt.savefig(f'../accuracies/perturbed_KMeans_{k}_defense_{attack}_attack.png')

            cnt = 0

            plt.figure(figsize=(8,10))
            for i in range(len(epsilons)):
                for j in range(len(examples[i])):
                    cnt += 1
                    plt.subplot(len(epsilons),len(examples[0]),cnt)
                    plt.xticks([], [])
                    plt.yticks([], [])
                    if j == 0:
                        plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
                    orig,adv,ex = examples[i][j]
                    plt.title("{} -> {}".format(orig, adv))
                    plt.imshow(ex, cmap="gray")
            plt.tight_layout()
            plt.savefig(f"../images/perturbedImage_KMeans_{k}_defense_{attack}_attack.png")

