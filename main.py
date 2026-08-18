import torch
import time
import argparse
import os
import math
from config_file.dataset_configs import Config
from model.model import train_model, finetune_model
from worker.worker import Model_Train, Model_Finetune, Model_Test
from utils.utils import Logger, Timer, Loss_Accumulator
import numpy as np
import random
from data.dataloader import  Data_Loader
from ignite.metrics import Accuracy


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', type=str,
                    help='worker mode: train, finetune, test')
parser.add_argument('--signal-length', default=1024, type=int,
                    help='signal length')

parser.add_argument('--source', default='PU_0', type=str,
                    help='source datasets: pretrain')
parser.add_argument('--target', default='PU_0', type=str,
                    help='target datasets: finetune_test')
parser.add_argument('--epochs', default=200,  type=int,
                    help='number of total epochs')
parser.add_argument('--f_epochs', default=100,  type=int,
                    help='number of finetune epochs')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--f_batch_size', type=int, default=32)

parser.add_argument('--train_lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', default=1e-3, type=float,
                    help='weight decay')
parser.add_argument('--lamda1', default=0.3, type=float, help='lamda1 value')
parser.add_argument('--lamda2', default=0.3, type=float, help='lamda2 value')
parser.add_argument('--warmup-epochs', default=5, type=int,
                    help='number of warmup epochs')

parser.add_argument('--percent', default=1, type=float,
                    help='finetune dataset percentage')
parser.add_argument('--seed', default=0, type=int, help='seed value')
parser.add_argument('--classes', default=13, type=int, help='class value')

parser.add_argument('--tempture', default=0.5, type=float, help='tempture value')
parser.add_argument('--projection', default=128, type=int, help='projection')
parser.add_argument('--head', default=1, type=int, help='head')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout')

parser.add_argument('--logs-save-dir', default='./logs_exam', type=str,
                    help='saving experiments logs')
parser.add_argument('--save', default='/test.csv', type=str,
                    help='saving experiments logs')
parser.add_argument('--run', default='test', type=str,
                    help='Experiment Description')

args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    args = parser.parse_args()
    configs = Config()
    configs.num_class=args.classes
    configs.batch_size=args.batch_size
    configs.lamda1 = args.lamda1
    configs.lamda2 = args.lamda2
    configs.tempture = args.tempture
    configs.projection=args.projection
    configs.head = args.head
    configs.dropout = args.dropout

    SEED=args.seed
    set_seed(SEED)
    '''save dir'''
    experiment_log_dir = os.path.join(args.logs_save_dir, args.run)
    if not os.path.exists(experiment_log_dir):   
        os.makedirs(experiment_log_dir)

    '''logger'''
    log_name = os.path.join(experiment_log_dir, '{}_logs_{}.log'.format(args.mode, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())))
    logger = Logger(log_name)



    '''GPU or CPU'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.debug('Device: {}'.format(device))


    '''dataloder'''
    source_path = args.source
    target_path = args.target


    '''load model & data'''
    if args.mode == 'train':
        model = train_model(configs, device)
        data_iter = Data_Loader(source_path, args, args.mode)
    elif args.mode == 'finetune':
        model = finetune_model(configs)
        data_iter = Data_Loader(target_path, args, args.mode)
    else:
        model = finetune_model(configs)
        data_iter = Data_Loader(target_path, args, args.mode)
    model.to(device)



    logger.debug('Model Loaded')
    logger.debug('Data Loaded')

    '''optimizer & scheduler'''
    optimizer = torch.optim.AdamW(model.parameters(),lr = args.train_lr, weight_decay= args.weight_decay) #AdamW 是 Adam 优化算法的一个变种，它引入了权重衰减（weight decay），这是一种正则化技术，用于防止模型过拟合
    warm_up_with_cosine_lr = lambda epoch: ((epoch +1) / args.warmup_epochs ) if epoch < args.warmup_epochs \
    else 0.5 * ( math.cos((epoch - args.warmup_epochs) /(args.epochs - args.warmup_epochs) * math.pi) + 1) \
    if 0.5 * ( math.cos((epoch - args.warmup_epochs) /(args.epochs - args.warmup_epochs) * math.pi) + 1)>0.1 else 0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)
    logger.debug('*' * 50)

    '''work'''
    main_worker(model, optimizer, scheduler, data_iter, device, logger, experiment_log_dir, args, configs)

    logger.debug('*' * 50)

    
def main_worker(model, optimizer, scheduler, data_iter, device, logger, experiment_log_dir, args, configs):
    logger.debug('Worker Started')
    
    '''timer'''
    timer = Timer()

    '''training mode'''
    if args.mode == 'train':
        logger.debug('Training Started')

        '''record loss & acc'''
        loss = Loss_Accumulator()

        '''epochs'''
        for epoch in range(args.epochs):
            timer.start()
            train_loss = Model_Train(model, optimizer, scheduler, data_iter, device, loss, configs )
            timer.stop()
            logger.debug('\nTraining Epoch: {}. Loss: {}. '.format(epoch, train_loss))

        '''save model'''
        save_path = os.path.join(experiment_log_dir, 'saved_model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path,'train_state_dict.pt'))

        logger.debug('\nTrained model is saved')



    '''fientune mode'''
    if args.mode == 'finetune':
        logger.debug('Finetuning Stated')
        
        '''load pre-train parameters'''
        load_path = os.path.join(experiment_log_dir, 'saved_model', 'train_state_dict.pt')
        saved_model = torch.load(load_path)
        model_dict = model.state_dict()
        state_dict = {k:v for k, v in saved_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        '''record loss & acc'''
        loss = Loss_Accumulator()
        acc = Accuracy(device)

        '''epochs'''
        for epoch in range(args.f_epochs):
            timer.start()
            finetune_loss, finetune_acc = Model_Finetune(model, optimizer, scheduler, data_iter, device, loss,acc)
            timer.stop()
            logger.debug('\nFinetuning Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, finetune_loss, finetune_acc))

        '''save model'''
        save_path = os.path.join(experiment_log_dir, 'saved_model')
        if not os.path.exists(save_path):   
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'finetune_{}_state_dict.pt'.format(args.percent)))

        logger.debug('\nFinetuned model is saved')
    
    '''test mode'''
    if args.mode == 'test':
        logger.debug('Testing Stated')

        '''load finetune parameters'''
        load_path = os.path.join(experiment_log_dir, 'saved_model', 'finetune_{}_state_dict.pt'.format(args.percent))
        model.load_state_dict(torch.load(load_path))


        '''testing'''
        timer.start()
        total_acc,precision,recall, F1 = Model_Test(model, data_iter, device)
        timer.stop()

        logger.debug(
            '\nAccuracy: {}. Precision: {}. Recall: {}. F1: {}'.format(total_acc * 100, precision * 100, recall * 100, F1 * 100))


    logger.debug('\nWork Time is: {} sec'.format(timer.sum()))

    logger.debug('\nWoker Finished')



if __name__ == '__main__':
    main()
