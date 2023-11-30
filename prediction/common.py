import json
import time
import numpy as np
import pandas as pd
import torch 

import copy
import argparse
import datetime
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# from datetime import datetime, timedelta

import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import DBSCAN,KMeans
from sklearn.manifold import TSNE

COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CARMINE = "\033[95m"
COLOR_CYAN = "\033[96m"
COLOR_WHITE = "\033[97m"

BACKGROUND_RED = "\033[41m"
BACKGROUND_GREEN = "\033[42m"
BACKGROUND_YELLOW = "\033[43m"
BACKGROUND_BLUE = "\033[44m"
BACKGROUND_CARMINE = "\033[45m"
BACKGROUND_CYAN = "\033[46m"
BACKGROUND_WHITE = "\033[47m"

STYLE_BOLD = "\033[1m"
STYLE_UNDERLINE = "\033[4m"
STYLE_REVERSE = "\033[7m"
COLOR_RESET = "\033[0m"

parser = argparse.ArgumentParser(description='predictive module for fine-grained demand and coarse-grained supply')
parser.add_argument('--hist_len', type=int, default=60)
parser.add_argument('--pred_len', type=int, default=30)
parser.add_argument('--ratio', type=float, default=0.7)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--step', type=int, default=2)
parser.add_argument('--device',type =str, default = 'cuda:0')
parser.add_argument('--poi_type',type = str, default ='full')
parser.add_argument('--model',type = str, default = 'OCAP')
parser.add_argument('--epoch',type = int, default= 150)
parser.add_argument('--lr',type = float, default= 0.001)

parser.add_argument('--seed',type = int, default= 2023)
parser.add_argument('--nn_params', type = dict, default ={})

args = parser.parse_args([])

def get_nn_params(model):
    if model =='OCAP':
        nn_params ={
            'seq_len':args.hist_len,
            'pred_len':args.pred_len,
            'dy_input_dim': 21,
            'dy_embed_dim':8, #[8,16]
            'hidden_dim':32, # [8,16,32]
            'device': args.device,
            'avg_kernel_size':25,
            'group':1,
            'lambda':[5,5,1,1], # [5,10,15]
            
        }
    elif model =='Tide':
        nn_params ={
            'dy_dim': 21, # week 7 + month 12 + event 2
            'st_dim': 4 if args.poi_type == 'short' else 19,
            'dy_embed_dim': 8, #  [8,16,32]
            'decode_dim': 8,  # [8,16,32]
            'hidden_dim': 64, #  hidden encoding e dimension [128,256]
            'enc_layer_nums': 2, #[1,2,3]
            'dec_layer_nums': 2, #[1,2,3]
            'dropout': 0.1, #[0.1,0.2,0.3] 
            'seq_len': 60,
            'pred_len':30,
        }
    elif model =='Dlinear':
        nn_params ={
            'seq_len':args.hist_len,
            'pred_len':args.pred_len,
        }
    elif model =='Transformer':
        nn_params ={
                'hidden':64, # 32,64,128
                'input_size':22, # order +21 tconv
                'seq_len': 60,
                'pred_len': 30,
                'output_size':1,
            }
    elif model =='PatchTST':
        nn_params ={
                'enc_in':300,
                'seq_len': 60,
                'pred_len': 30,
                'd_model':64, # 32,64,128
                'd_ff':32, #32,64,128
                'embed':32,# 32,64,128
            }
    return nn_params

args.nn_params = get_nn_params(args.model)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# step_0 prepare datasets
class DataProcessing:
    def __init__(self,):

        atten = np.load('data/attendance.npy') 
        collective_atten = atten.sum(axis=0)

        with open("data/aoi_poi_infos_manual_selected.json", "r") as file:
            aoi_poi_infos_manual_selected_raw = json.load(file)

        with open("data/aoi_poi_infos.json", "r") as file:
            aoi_poi_infos_raw = json.load(file) # AOI - POIs

        for key,item in aoi_poi_infos_raw.items():
            aoi_poi_infos_raw[key] = [float(x) for x in list(item.values())]

        with open('data/aoi_inner_road.json','r') as f:
            inner = json.load(f) # for selecting AOI within the scope

        self.aoi_poi_infos_manual_selected = dict((key, aoi_poi_infos_manual_selected_raw[key]) for key in inner)
        self.aoi_poi_infos = dict((key, aoi_poi_infos_raw[key]) for key in inner)

        aoi_order_amount = pd.read_csv('data/aoi_order.csv',index_col=0)
        raw_orders = aoi_order_amount.iloc[:,1:].to_numpy()

        self.normed_raw_order, self.x_min, self.x_max = DataProcessing._order_ch_normalize(raw_orders)
        self.normed_collective_atten, self.a_min, self.a_max = DataProcessing._atten_ch_normalize(collective_atten)
        # self.normed_raw_order = DataProcessing._order_normalize(raw_orders)
        self.N, self.T = raw_orders.shape
        self.mask_ratio = 0.7 
        self.n_regions = 31
        self.start_date = "20220801"
        self.end_date = "20230813"
        self.big_sell_event = [["20221107","20221115"],["20221204","20221215"],["20230614","20230623"]]
        self.fes_rest_event = [["20220929","20221005"],["20230120","20230202"],["20230428","20230505"]] 

    def generate_seq_data(self, mode ='short',poi_norm='row_wise'): 
        # order: N=299, T=378, D=1
        # poi_short: N=299, D=5
        # poi_full: N=299, D=20
        # t_enc: N=299, T=378, D=7+12+2
        # atten: T = 378

        assert mode in ['short','full']
        assert poi_norm in ['row_wise', 'col_wise']

        order_data =  self.normed_raw_order

        if mode =='short':
            poi_data = DataProcessing._poi_normalize(np.array(list(self.aoi_poi_infos_manual_selected.values())),mode=poi_norm)
        else:
            poi_data = DataProcessing._poi_normalize(np.array(list(self.aoi_poi_infos.values())),mode=poi_norm)

        t_cov = self._generate_temporal_encoding(order_data.shape[0])

        return order_data, poi_data,t_cov, self.normed_collective_atten
    
    def generate_feature_data(self, mode = 'short', poi_norm='row_wise', mask = True):
        # seq: N = 299, T= 378
        # point_: N = 299, D = 4 or 19
        # seq_point_: N = 299, D = (4 + 4) or (19 + 4)

        assert mode in ['short', 'full']
        assert poi_norm in ['row_wise', 'col_wise']

        if mask:
            lim = int(self.normed_raw_order.shape[0] * self.mask_ratio)
            seq2point = DataProcessing._calc_sequential_feature(self.normed_raw_order[:,:lim])
        else:
            seq2point = DataProcessing._calc_sequential_feature(self.normed_raw_order)
            
        if mode == 'short':
            poi = DataProcessing._poi_normalize(np.array(list(self.aoi_poi_infos_manual_selected.values())),mode=poi_norm)
        else:
            poi = DataProcessing._poi_normalize(np.array(list(self.aoi_poi_infos.values())),mode=poi_norm)
        return  seq2point, poi

    def _generate_temporal_encoding(self,N):
        start = datetime.datetime.strptime(self.start_date, "%Y%m%d")
        end = datetime.datetime.strptime(self.end_date, "%Y%m%d")
        num_days = (end - start).days + 1

        encoding = np.zeros((num_days, 7 + 12 + 2))
        for i in range(num_days):
            date = start + datetime.timedelta(days=i)
            week_index = date.weekday()
            month_index = date.month - 1

            encoding[i, week_index] = 1
            encoding[i, 7 + month_index] = 1
        big_sell_event = DataProcessing._date_to_index(self.start_date,self.end_date,self.big_sell_event)
        for events in big_sell_event:
            encoding[events[0]:events[1]+1,-2] = 1

        fes_rest_event = DataProcessing._date_to_index(self.start_date,self.end_date,self.fes_rest_event)
        for events in fes_rest_event:
            encoding[events[0]:events[1]+1,-1] = 1
        ret = np.tile(encoding,(N,1,1))
        return ret

    def _order_ch_denormalize(self,x):
        # x, B,N,T
        return x*(self.x_max-self.x_min) + self.x_min
    
    def _atten_ch_denormalize(self,x):
        # x, B,N,T
        return x*(self.a_max-self.a_min) + self.a_min
    
    @staticmethod
    def _calc_sequential_feature(x,trimmed_percatage= 0.2):
        # order_amount x : N,T
        N, T = x.shape

        # min,std
        x_sort = np.sort(x, axis=1)
        x_trimmed = x[:,int(T*trimmed_percatage/2):-int(T*trimmed_percatage/2)]
        mean = x_trimmed.mean(axis=1)
        std = x_trimmed.std(axis=1)
        
        # phase   
        m = -np.inf * np.ones(N)
        phase = np.zeros(N)

        for i in range(7):
            ref = np.repeat(np.sin(np.arange(i,T+i) * np.pi *2 /7)[np.newaxis,...], N, axis=0) + 1
            score = (ref *x).sum(axis=1)
            index = (score > m).astype(bool)
            phase[index] = i
            m[index] = score[index]

        # autocorr
        autocorr = np.array([np.correlate(i, i, mode='valid') for i in x]).reshape(-1)
        
        feature = np.array([mean,std,phase,autocorr]).T
        normed_feature = (feature-feature.min(axis=0)) /(feature.max(axis=0)-feature.min(axis=0)) 

        # normed_feature: N,D_f
        return  normed_feature

    @staticmethod
    def _poi_normalize(x, mode='row_wise'):
        # poi_infos x: N, D_p

        if mode == 'row_wise':
            shift = x - x.min(axis=1).reshape(-1, 1)
            norm = x.max(axis=1) - x.min(axis=1)
        elif mode == 'col_wise':
            shift = x - x.min(axis=0).reshape(-1, 1)
            norm = x.max(axis=0) - x.min(axis=0)
        else:
            raise ValueError("Invalid mode. Choose either 'row_wise' or 'col_wise'.")

        norm = np.where(norm == 0, 1, norm) # Handle the case when norm is zero
        ret = shift / norm.reshape(-1, 1)

        return ret

    # def _order_channel_wise_normalize(self,x):
    #     # order x: N,T
    #     ch_min = []
    #     ch_max = []
    #     for 
    #     infty_small = 1e-5 
    #     x_min = x.min(axis=1).reshape(-1,1)
    #     x_max = x.max(axis=1).reshape(-1,1) + infty_small

    @staticmethod
    def _order_normalize(x):
        x_min = x.min()
        x_max = x.max()
        return (x-x_min)/(x_max-x_min)
    @staticmethod
    def _order_ch_normalize(x):
        # order x: N,T
        infty_small = 1e-5 
        x_min = x.min(axis=1).reshape(-1,1)
        x_max = x.max(axis=1).reshape(-1,1) + infty_small

        return (x-x_min)/(x_max-x_min), torch.Tensor(x_min).unsqueeze(0),torch.Tensor(x_max).unsqueeze(0)

    @staticmethod
    def _atten_ch_normalize(x):
        x = torch.Tensor(x)
        x_min = x.min()
        x_max = x.max()
   
        return (x-x_min)/(x_max-x_min), x_min,x_max

    @staticmethod
    def _date_to_index(start_date,end_date,events):
        start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
        event_index = []
        for event in events:
            event_start = datetime.datetime.strptime(event[0], "%Y%m%d")
            event_end = datetime.datetime.strptime(event[1], "%Y%m%d")
            
            if event_start > end_date or event_end < start_date:
                continue
            
            start_index = (event_start - start_date).days
            end_index = (event_end - start_date).days
            
            event_index.append([start_index, end_index])
        return event_index
    
class MyDataset(Dataset):
    def __init__(self, hist_ts,pred_ts,dynamic,hist_at,pred_at,static):
        self.hist_ts = hist_ts
        self.dynamic = dynamic
        self.pred_ts = pred_ts
        self.hist_at = hist_at
        self.pred_at = pred_at
        self.static = static
    def __len__(self):
        return len(self.hist_ts)
    
    def __getitem__(self, idx):
        return self.hist_ts[idx], self.pred_ts[idx], self.dynamic[idx], self.static, self.hist_at[idx], self.pred_at[idx],
    
class DatasetGenerator:
    def __init__(self, ts, static, dynamic, co_atten, ratio, hist_len, pred_len, step, batch_size, mask_event):
        # data: N,T,D
        self.ts = torch.from_numpy(ts).float()
        self.static = torch.from_numpy(static).float()
        self.dynamic = torch.from_numpy(dynamic).float()
        self.co_atten = co_atten
        self.mask = torch.ones_like(self.ts)
        self.ratio = ratio
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.step = step
        self.batch_size = batch_size
        # self.mask_event = mask_event

        # self.start_date = "20220801"
        # self.end_date = "20230813"
        # self.event_range = [["20220929","20221005"],["20221107","20221115"],["20221204","20221215"],["20230120","20230202"],["20230428","20230505"],["20230614","20230623"]]
        
    def generate_dataset(self):
        # if self.mask_event:
        #     self.ts = self._filter_events(self.ts, time_axis =1)
        #     self.dynamic = self._filter_events(self.dynamic, time_axis =1)
        #     self.co_atten = self._filter_events(self.co_atten, time_axis =0)
        
        train_ts, test_ts = self.ts[:,:int(self.ts.shape[1]*self.ratio)], self.ts[:,int(self.ts.shape[1]*self.ratio):]
        train_dy, test_dy = self.dynamic[:,:int(self.ts.shape[1]*self.ratio)], self.dynamic[:,int(self.ts.shape[1]*self.ratio):]
        train_at, test_at = self.co_atten[:int(self.ts.shape[1]*self.ratio)], self.co_atten[int(self.ts.shape[1]*self.ratio):]

        train_samples = self._generate_samples(train_ts, train_dy, train_at)
        test_samples = self._generate_samples(test_ts, test_dy, test_at)
        train_loader = self._build_dataloader(train_samples,self.static)
        test_loader = self._build_dataloader(test_samples,self.static,no_split = True)

        return train_loader, test_loader
    
    # def _filter_events(self,data, time_axis =0):
    #     intervals = DatasetGenerator._date_to_index(self.start_date,self.end_date,self.event_range) 
    #     datalist =[]
    #     index =0

    #     if time_axis == 0:
    #         for interval in intervals:
    #             datalist.append(data[index:interval[0]])
    #             index = interval[1]+1
    #         datalist.append(data[index:])
    #         return torch.cat(datalist,dim=0)
        
    #     elif time_axis == 1:
    #         for interval in intervals:
    #             datalist.append(data[:,index:interval[0]])
    #             index = interval[1]+1
    #         datalist.append(data[:,index:])
    #         return torch.cat(datalist,dim=1)
        
    #     else:
    #         raise

    def _generate_samples(self, ts ,dynamic, at):
        # data: N,T
        # dynamic : N,T,D
        ts_unfolded = ts.unfold(dimension= 1,size = self.hist_len+self.pred_len,step = self.step).transpose(0,1) # bs, N, T
        dynamic_unfolded = dynamic.unfold(dimension= 1,size = self.hist_len+self.pred_len,step = self.step).permute(1,0,-1,-2) # bs,N,T,D
        at_unfolded = at.unfold(dimension = 0,size = self.hist_len+self.pred_len,step = self.step)
        hist_ts = ts_unfolded[...,:self.hist_len]
        pred_ts = ts_unfolded[...,self.hist_len:]
        hist_at = at_unfolded[...,:self.hist_len]
        pred_at = at_unfolded[...,self.hist_len:]
        return (hist_ts,pred_ts,dynamic_unfolded,hist_at,pred_at)

    def _build_dataloader(self, samples, statics, no_split=False):
        dataset = MyDataset(*samples,statics)
        dataloader = DataLoader(dataset,batch_size = int(max(1e5,self.batch_size)) if no_split else self.batch_size)
        return dataloader
    @staticmethod
    def _date_to_index(start_date,end_date,events):
        start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
        event_index = []
        for event in events:
            event_start = datetime.datetime.strptime(event[0], "%Y%m%d")
            event_end = datetime.datetime.strptime(event[1], "%Y%m%d")
            
            if event_start > end_date or event_end < start_date:
                continue
            
            start_index = (event_start - start_date).days
            end_index = (event_end - start_date).days
            
            event_index.append([start_index, end_index])
        return event_index
    
def mape(pred,real,ratio = 0.1):
    mask = np.ones_like(pred.detach().cpu()).reshape(-1)
    _,index = torch.sort(real.reshape(-1))
    lens = mask.shape[0]
    mask[index[:int(lens*ratio)]] = 0

    return (torch.abs(real-pred)/(real+1e-5)).reshape(-1)[mask.astype(bool)].mean()

def match_hidden_dim(dim_1,dim_2,step = 16):
    if dim_1 > dim_2:
        dim_large,dim_small = dim_1,dim_2
    else:
        dim_large,dim_small = dim_2,dim_1

    mid_scale = dim_large/dim_small /2
    return max(int(mid_scale * dim_small // step * step),step)

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True): 
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        
        out = self.ln(out)
        return out

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x