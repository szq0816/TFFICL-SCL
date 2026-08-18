import torch
import torch.nn.functional as F
from sklearn.metrics import  accuracy_score, precision_score,f1_score,recall_score
import numpy as np
from loss import Feature_Level_Contrastive_Loss

def Model_Train(model, optimizer, scheduler, data_iter, device, loss, configs):
    model.train()
    loss.reset()
    criterion_f = Feature_Level_Contrastive_Loss(configs.batch_size,configs.tempture, device).to(device)
    for x_t,x_f,_ in data_iter:
        optimizer.zero_grad()
        x_t,x_f= x_t.to(device),x_f.to(device)
        batch_size = x_t.shape[0]
        z_t, z_f, z, y1, y2, y= model(x_t,x_f)
        FeatureLoss = criterion_f.forword_feature(z.T, z_t.T) + criterion_f.forword_feature(z.T, z_f.T)
        KL=F.kl_div(F.log_softmax(y, dim=1), F.softmax(y1, dim=1), reduction='sum')+F.kl_div(F.log_softmax(y, dim=1), F.softmax(y2, dim=1), reduction='sum')
        l=configs.lamda1*FeatureLoss+configs.lamda2*KL
        l.backward()
        optimizer.step()
        with torch.no_grad():
            loss.update((l, batch_size))
    scheduler.step()
    return loss.compute()


def Model_Finetune(model, optimizer, scheduler, data_iter, device, loss, acc):
    model.train()
    loss.reset()
    acc.reset()

    for x_t,x_f, y in data_iter:
        optimizer.zero_grad()
        x_t ,x_f,y= x_t.to(device),x_f.to(device),y.to(device)
        y_hat = model(x_t,x_f)
        batch_size = x_t.shape[0]
        l = F.cross_entropy(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            loss.update((l, batch_size))
            acc.update((y_hat, y))
    scheduler.step()
    total_loss = loss.compute()
    total_acc = acc.compute()

    return total_loss, total_acc




def Model_Test(model, data_iter, device):
    model.eval()
    all_predict = []
    all_labels = []

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for x_t, x_f,y in data_iter:
            x_t, x_f,y = x_t.to(device),x_f.to(device),y.to(device)
            y_hat = model(x_t,x_f)
            all_labels.append(y.cpu().numpy())
            pred_numpy = y_hat.detach().cpu().numpy()
            labels_numpy = y.detach().cpu().numpy()
            predicted = np.argmax(pred_numpy, axis=1)
            pred_numpy = np.argmax(pred_numpy, axis=1)
            all_predict.append(predicted)
            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]


    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

    return acc,precision,recall, F1



