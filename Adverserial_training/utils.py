import numpy as np
import torch
from PIL import Image

from tqdm import trange, tqdm
np.random.seed(42)
torch.manual_seed(42)



def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")

def train_vqvae(vqvae_model,train_loader,val_loader,optimizer,loss_function,device,use_ema,epochs=6,beta=0.25,eval_every = 100):
    
    best_train_loss = float("inf")

    vqvae_model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        total_recon_error = 0
        n_train = 0
        for (batch_idx, train_tensors) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = train_tensors[0].to(device)
            out = vqvae_model(imgs)
            recon_error = loss_function(out["x_recon"], imgs) 
    #         recon_error = criterion(out["x_recon"], imgs) / train_data_variance
            total_recon_error += recon_error.item()
            loss = recon_error + beta * out["commitment_loss"]
            if not use_ema:
                loss += out["dictionary_loss"]

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

            if ((batch_idx + 1) % eval_every) == 0:
                print(f"epoch: {epoch+1}\{epochs} \t batch_id: {batch_idx + 1}", flush=True)
                total_train_loss /= n_train
                if total_train_loss < best_train_loss:
                    best_train_loss = total_train_loss

                print(f"total_train_loss: {total_train_loss:.5f}\tbest_train_loss: {best_train_loss:.5f}\trecon_error: {(total_recon_error / n_train):.5f}\n")
            
                total_train_loss = 0
                total_recon_error = 0
                n_train = 0

    # Generate and save reconstructions.
    vqvae_model.eval()

    with torch.no_grad():
        for valid_tensors in val_loader:
            break
        
        save_img_tensors_as_grid(valid_tensors[0], 4, "../vqvae_outputs/true.png")
        save_img_tensors_as_grid(vqvae_model(valid_tensors[0].to(device))["x_recon"], 4, "../vqvae_outputs/recon.png")




def fit_Dataset(model,optimizer,scheduler,train_loader,val_loader,loss_function,device,epochs=2):
    data_loader = {'train':train_loader,'val':val_loader}
    print("Fitting the model...")
    train_loss,val_loss=[],[]
    for epoch in trange(epochs):
        loss_per_epoch,val_loss_per_epoch=0,0
        for phase in ('train','val'):
            for i,(x,y) in enumerate(data_loader[phase]):
                x,label  = x.to(device),y.to(device)
                output = model(x)
                #calculating loss on the output
                loss = loss_function(output,label)
                if phase == 'train':
                    optimizer.zero_grad()
                    #grad calc w.r.t Loss func
                    loss.backward()
                    #update weights
                    optimizer.step()
                    loss_per_epoch+=loss.cpu().item()
                else:
                    val_loss_per_epoch+=loss.cpu().item()
        scheduler.step(val_loss_per_epoch/len(val_loader))
        print("Epoch: {} Loss: {} Val_Loss: {}".format(epoch+1,loss_per_epoch/len(train_loader),val_loss_per_epoch/len(val_loader)))
        train_loss.append(loss_per_epoch/len(train_loader))
        val_loss.append(val_loss_per_epoch/len(val_loader))
    return train_loss,val_loss

def accuracy(train_loader,test_loader,model,device,reshape=False):
    model.eval()
    with torch.no_grad():
        r_pred=[]
        r=[]
        for _,(x,y) in enumerate(train_loader):
            if reshape==True:
                x = x.reshape(x.shape[0],-1)
            x,y = x.to(device),y.to(device)
            scores = model(x)
            _, y_pred = scores.max(1)
            r = r + list(y.cpu().numpy())
            r_pred = r_pred + list(y_pred.cpu().numpy())

        r_pred = np.array(r_pred)


        ######## test_accuracy #######
        test_r_pred=[]
        test_r=[]
        for _,(x,y) in enumerate(test_loader):
            
            if reshape==True:
                x = x.reshape(x.shape[0],-1)
            
            x,y = x.to(device),y.to(device)
            
            scores = model(x)
            _, y_pred = scores.max(1)
            test_r = test_r + list(y.cpu().numpy())
            test_r_pred = test_r_pred + list(y_pred.cpu().numpy())

        test_r_pred = np.array(test_r_pred)
    return np.mean(r==r_pred) ,np.mean(test_r==test_r_pred)


