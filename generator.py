import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import cnn as cnn

from utils import constant

from pytorch_pretrained_bert.modeling import BertEmbeddings, BertConfig

def get_hard_mask(z, return_ind=False):
    '''
        -z: torch Tensor where each element probablity of element
        being selected
        -opt[: experiment level config
        returns: A torch variable that is binary mask of z >= .5
    '''
    max_z, ind = torch.max(z, dim=-1)
    if return_ind:
        del z
        return ind
    masked = torch.ge(z, max_z.unsqueeze(-1)).float()
    del z
    return masked

def gumbel_softmax(input, temperature, cuda, device):
    noise = torch.rand(input.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = autograd.Variable(noise)
    if cuda:
        with torch.cuda.device(device):
            noise = noise.cuda()
    x = (input + noise) / temperature
    x = F.softmax(x.view(-1,  x.size()[-1]), dim=-1)
    return x.view_as(input)



'''
    The generator selects a rationale z from a document x that should be sufficient
    for the encoder to make it's prediction.

    Several froms of Generator are supported. Namely CNN with arbitary number of layers, and @taolei's FastKNN
'''
class Generator(nn.Module):

    def __init__(self, opt, config):
        super(Generator, self).__init__()
        self.embedding_layer = BertEmbeddings(config)
        self.opt = opt
        self.cnn = cnn.CNN(opt, max_pool_over_time = False)

        self.z_dim = 2

        self.hidden = nn.Linear(300, self.z_dim)
        self.dropout = nn.Dropout(constant.DROPOUT_PROB)



    def  __z_forward(self, activ):
        '''
            Returns prob of each token being selected
        '''
        activ = activ.transpose(1,2)
        logits = self.hidden(activ)
        probs = gumbel_softmax(logits, 1, self.opt['cuda'], self.opt['device'])
        z = probs[:,:,1]
        return z


    def forward(self, x_indx):
        '''
            Given input x_indx of dim (batch, length), return z (batch, length) such that z
            can act as element-wise mask on x
        '''
        x = self.embedding_layer(x_indx)
        if self.opt['cuda']:
            with torch.cuda.device(self.opt['device']):
                x = x.cuda()
        x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
        activ = self.cnn(x)
        z = self.__z_forward(F.relu(activ))
        mask = self.sample(z)
        return mask


    def sample(self, z):
        '''
            Get mask from probablites at each token. Use gumbel
            softmax at train time, hard mask at test time
        '''
        mask = z
        if self.training:
            mask = z
        else:
            ## pointwise set <.5 to 0 >=.5 to 1
            mask = get_hard_mask(z)
        return mask


    def loss(self, mask):
        '''
            Compute the generator specific costs, i.e selection cost, continuity cost, and global vocab cost
        '''
        selection_cost = torch.mean( torch.sum(mask, dim=1) )
        l_padded_mask =  torch.cat( [mask[:,0].unsqueeze(1), mask] , dim=1)
        r_padded_mask =  torch.cat( [mask, mask[:,-1].unsqueeze(1)] , dim=1)
        continuity_cost = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=1) )
        return selection_cost, continuity_cost
