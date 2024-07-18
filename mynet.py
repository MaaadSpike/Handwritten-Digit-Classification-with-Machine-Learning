import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyNet(nn.Module):
    '''
     net = MyNet(img_size=28)
     
     Creates a neural network to do classification on MNIST.
     It assumes the images will be (img_size)x(img_size).
     
     The output of the network is the log of the 10 class probabilities
     (ie. log-softmax). Correspondingly, this network uses the
     negative log-likelihood loss function (nn.NLLLoss).
    '''
    def __init__(self, img_size=28):
        super().__init__()
        self.lyrs = nn.Sequential(
            nn.Linear(img_size**2, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.LogSoftmax(dim=-1),
            )
        self.loss_fcn = nn.NLLLoss()
        self.losses = []
        self.to(device)
        
        
    def forward(self, x):
        return self.lyrs(x)
    
    
    def learn(self, dl, optimizer=None, epochs=10):
        '''
         net.learn(dl, optimizer=None, epochs=10)
         
         Train the network on the dataset represented by the DataLoader dl.
         The default optimizer is Adam().
         
         The targets for the dataset are assumed to be class indices.
        '''
        if optimizer is None:
            print('Need to specify an optimizer and loss function')
            return
        
        for epoch in tqdm(range(epochs)):
            total_loss = 0.
            count = 0.
            for x, t in dl:
                x = x.to(device)   # for use with a GPU
                t = t.to(device)
                y = self(x)
                loss = self.loss_fcn(y, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #total_loss += loss.detach().numpy()
                total_loss += loss.detach().cpu().numpy()
                count += 1.
            self.losses.append(total_loss/len(dl))
            #print(f'Epoch: {epoch}, loss: {total_loss/count}')
        plt.figure(figsize=(4,4))
        plt.plot(self.losses); plt.yscale('log')

def fgsm(net, x, t, eps=0.01, targ=False):
    '''
        x_adv = FGSM(net, x, t, eps=0.01, targ=False)
        
        Performs the Fast Gradient Sign Method, perturbing each input by
        eps (in infinity norm) in an attempt to have it misclassified.
        
        Inputs:
          net    PyTorch Module object
          x      (D,I) tensor containing a batch of D inputs
          t      tensor of D corresponding class indices
          eps    the maximum infinity-norm perturbation from the input
          targ   Boolean, indicating if the FGSM is targetted
                   - if targ is False, then t is considered to be the true
                     class of the input, and FGSM will work to increase the cost
                     for that target
                   - if targ is True, then t is considered to be the target
                     class for the perturbation, and FGSM will work to decrease the
                     cost of the output for that target class
        
        Output:
          x_adv  tensor of a batch of adversarial inputs, the same size as x
    '''

    x_adv = x.clone().to(device)
    
    x_adv.requires_grad = True
    y = net(x_adv)
    loss = net.loss_fcn(y, t)
    net.zero_grad()
    loss.backward()
    x_grad = x_adv.grad.sign()

    x_adv_p = x_adv - eps * x_grad if targ else x_adv + eps * x_grad

    return x_adv_p.detach()     