from imageclassification import prepare_dataloaders 
from vit import ViT
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

print(os.getcwd())

trainloader, testloader, trainset, testset = prepare_dataloaders(1, classes=[3, 7])

configs = {'image_size':(32,32), 'patch_size':(4,4), 'channels':3, 
         'embed_dim':128, 'num_heads':4, 'num_layers':4, 'num_classes':2,
         'pos_enc':'learnable', 'pool':'cls', 'dropout':0.3, 'fc_dim':None, 
         'num_epochs':20, 'batch_size':16, 'lr':1e-4, 'warmup_steps':625,
         'weight_decay':1e-3, 'gradient_clipping':1}

model = ViT(image_size=configs['image_size'], patch_size=configs['patch_size'], channels=configs['channels'], 
                embed_dim=configs['embed_dim'], num_heads=configs['num_heads'], num_layers=configs['num_layers'],
                pos_enc=configs['pos_enc'], pool=configs['pool'], dropout=configs['dropout'], fc_dim=configs['fc_dim'], 
                num_classes=configs['num_classes'])
model.load_state_dict(torch.load('model.pth'))
with torch.no_grad():
    model.eval()
    for image, label in testloader:
        at = model.half_forward(image)
        num_of_patches = 64
        
        # attention1 = 1 - at[0].mean(axis=2)[0,:64].reshape(int(np.sqrt(num_of_patches)), int(np.sqrt(num_of_patches)))
        # sns.heatmap(attention1.squeeze(0), cmap=sns.color_palette("viridis", as_cmap=True))
        # plt.show()

        # plt.imshow(image.squeeze(0).permute(1,2,0))
        # plt.show()

        attention_mean = torch.stack(at).mean(0)
        attention_mean = 1 - attention_mean.mean(axis=2)[0,1:].reshape(int(np.sqrt(num_of_patches)), int(np.sqrt(num_of_patches)))
        sns.heatmap(attention_mean.squeeze(0), cmap=sns.color_palette("viridis", as_cmap=True))
        plt.show()
        plt.imshow(image.squeeze(0).permute(1,2,0))
        plt.show()

        print('h')