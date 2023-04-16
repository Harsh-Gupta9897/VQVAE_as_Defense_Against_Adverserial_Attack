import numpy as np
import torch
from PIL import Image

from tqdm import trange, tqdm
np.random.seed(42)
torch.manual_seed(42)



import torch.nn.functional as F

from attacks import fgsm_attack, ifgsm_attack,mifgsm_attack
from defenses import KMeans
from vqvae import VQVAE

def test_pipeline(model,test_loader,epsilon,attack,defense=None,have_defense=None,device=None):
    correct = 0
    adv_examples = []
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 
        if init_pred.item() != target.item(): continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        if attack == "fgsm": perturbed_data = fgsm_attack(data,epsilon,data_grad)
        elif attack == "ifgsm": perturbed_data = ifgsm_attack(data,epsilon,data_grad)
        elif attack == "mifgsm": perturbed_data = mifgsm_attack(data,epsilon,data_grad)
            
         ###### Defense Layer #####
        if have_defense=="vqvae":
            perturbed_data = defense(perturbed_data)["x_recon"]
        elif have_defense=="kmeans":
            device_temp = perturbed_data.device
            perturbed_data = defense.fit(perturbed_data.cpu().detach())
            perturbed_data = perturbed_data.to(device_temp)
            
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples

