import torch
import torch.nn as nn

class Feature_Level_Contrastive_Loss(nn.Module):
    def __init__(self,batch_size, temperature_f, device):
        super(Feature_Level_Contrastive_Loss, self).__init__()
        self.device = device
        self.temperature_f = temperature_f
        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask



    def forword_feature(self, h_i, h_j):
        feature_size, _ = h_i.shape
        N = 2 * feature_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T)/self.temperature_f
        sim_i_j = torch.diag(sim, feature_size)
        sim_j_i = torch.diag(sim, -feature_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N,-1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss_contrast = self.criterion(logits, labels)
        loss_contrast /= N 
        
        return  loss_contrast


