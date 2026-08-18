import torch
from torch import nn
from torch.nn.functional import normalize
from model.encoder import TFencoder

class train_model(nn.Module):
    def __init__(self, configs, device):
        super().__init__()

        self.T_encoder = TFencoder(configs)
        self.F_encoder = TFencoder(configs)
        self.device = device
        self.Shared_Classification_Network = nn.Sequential(
            nn.Linear(configs.projection, 256),
            nn.PReLU(),
            nn.Linear(256, configs.num_class),
            nn.Softmax(dim=1)
        )

        self.Time_Projector = InstanceProject(configs.projection)
        self.Frequency_Projector = InstanceProject(configs.projection)

        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=configs.projection * 2, nhead=configs.head,dim_feedforward=256,dropout=configs.dropout)
        self.Self_Attention_Fusion = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
        self.fusion_proj = nn.Sequential(nn.Linear(configs.projection * 2, configs.projection))

    def forward(self, x_t,x_f):
        hs = []
        h_t = self.T_encoder(x_t)
        h_f = self.F_encoder(x_f)
        hs.append(h_t)
        hs.append(h_f)
        z_t = normalize( self.Time_Projector(h_t), dim=1)
        z_f = normalize( self.Frequency_Projector(h_f), dim=1)

        H= torch.cat(hs, dim=1)
        H= torch.unsqueeze(H, dim=1)
        fusion_fea =self.Self_Attention_Fusion(H)
        fusion_fea = torch.squeeze(fusion_fea, dim=1)
        fusion_fea=self.fusion_proj(fusion_fea)
        z = normalize(fusion_fea, dim=1)
        y1 = self.Shared_Classification_Network(h_t)
        y2 = self.Shared_Classification_Network(h_f)
        y = self.Shared_Classification_Network(fusion_fea)

        return  z_t, z_f,z,y1,y2,y



class finetune_model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.T_encoder = TFencoder(configs)
        self.F_encoder = TFencoder(configs)
        self.projection = build_mlp(2, configs.projection, configs.projection)
        self.fc = build_mlp(configs.num_prediction, configs.projection, configs.num_class)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=configs.projection * 2, nhead=configs.head,dim_feedforward=256,dropout=configs.dropout)
        self.Self_Attention_Fusion = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
        self.fusion_proj = nn.Sequential(nn.Linear(configs.projection * 2, configs.projection))

    def forward(self, x_t,x_f):
        hs = []
        h_t = self.T_encoder(x_t)
        h_f = self.F_encoder(x_f)
        hs.append(h_t)
        hs.append(h_f)
        H = torch.cat(hs, dim=1)
        H = torch.unsqueeze(H, dim=1)
        fusion_fea = self.Self_Attention_Fusion(H)
        fusion_fea = torch.squeeze(fusion_fea, dim=1)
        norm_fusion_fea = normalize(self.fusion_proj(fusion_fea), dim=1)
        self.fea = self.projection(norm_fusion_fea)
        y_hat = self.fc( self.fea)
        return y_hat



def build_mlp(num_layer, input, hidden):

    if num_layer == 0:
        return nn.Identity()
    else:
        mlp = []
        for i in range(num_layer):
            dim1 = input if i==0 else hidden
            dim2 = hidden
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if i < num_layer-1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            else:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)


class InstanceProject(nn.Module):
    def __init__(self, latent_dim):
        super(InstanceProject, self).__init__()
        self._latent_dim = latent_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.instance_projector(x)