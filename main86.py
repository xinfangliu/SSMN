import json
from collections import deque, defaultdict
import time

import h5py
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from manager_torch import GPUManager
from skimage.transform import resize
from torch.optim import  SGD


def get_iou(groundtruth, predict):
    groundtruth_init = max(0,groundtruth[0])
    groundtruth_end = groundtruth[1]
    predict_init = max(0,predict[0])
    predict_end = predict[1]
    init_min = min(groundtruth_init,predict_init)
    end_max = max(groundtruth_end,predict_end)
    init_max = max(groundtruth_init,predict_init)
    end_min = min(groundtruth_end,predict_end)
    if end_min < init_max:
        return 0
    IOU = ( end_min - init_max ) * 1.0 / ( end_max - init_min)
    return IOU

class CrossAttention(nn.Module):
    def __init__(self,in1_dim,in2_dim,hidden_dim,out_dim,dropout_rate=0.0,weights_only=False):
        super().__init__()
        self.attention_hidden_dim=hidden_dim
        self.in2_dim = in2_dim
        self.in1_dim = in1_dim
        self.weights_only=weights_only
        self.Q = nn.Linear(in_features=in1_dim, out_features=self.attention_hidden_dim,bias=False)
        self.K = nn.Linear(in_features=in2_dim, out_features=self.attention_hidden_dim,bias=False)
        self.V =nn.Linear(in_features=in1_dim,out_features=out_dim)
        self.dropout=nn.Dropout(dropout_rate)
    def forward(self,main_feature,guide_feature):
        q=self.Q(main_feature)
        k=self.K(guide_feature)
        v=self.V(self.dropout(main_feature))
        attention_value=torch.exp(torch.tanh((q*k).sum(dim=-1,keepdim=True)))
        if self.weights_only:
            return attention_value
        return attention_value*v


class Reshape(nn.Module):
 def __init__(self, *args):
  super(Reshape, self).__init__()
  self.shape = args

 def forward(self, x):
  # 如果数据集最后一个batch样本数量小于定义的batch_batch大小，会出现mismatch问题。可以自己修改下，如只传入后面的shape，然后通过x.szie(0)，来输入。
  return x.view(self.shape)

class Permute(nn.Module):
 def __init__(self, *args):
  super().__init__()
  self.order = args

 def forward(self, x:torch.Tensor):
  # 如果数据集最后一个batch样本数量小于定义的batch_batch大小，会出现mismatch问题。可以自己修改下，如只传入后面的shape，然后通过x.szie(0)，来输入。
  return x.permute(self.order)

class VideoConv(nn.Module):
    def __init__(self,**kwargs):
        self.v_dim = 1024
        self.s_dim= 300
        self.time_steps=100
        self.v_att_out_dim=256
        self.att_hidden_dim=128
        self.s_lstm_hidden_dim=256
        self.mix_lstm_hidden_dim=256
        self.predictor_pre_hidden_dim = 128
        self.predictor_hidden_dim = 64
        self.droprate=0.2
        super().__init__()
        self.__dict__.update(kwargs)
        self.drop=nn.Dropout(self.droprate)
        self.guid=CrossAttention(in1_dim=self.v_dim,in2_dim=self.s_lstm_hidden_dim,hidden_dim=self.att_hidden_dim,
                                 out_dim=self.v_att_out_dim,weights_only=False)
        self.s_lstm=nn.LSTM(input_size=self.s_dim,hidden_size=self.s_lstm_hidden_dim)
        self.s_att=nn.Linear(in_features=self.s_lstm_hidden_dim,out_features=1)
        self.mix_lstm=nn.LSTM(input_size=(self.v_att_out_dim+self.s_lstm_hidden_dim),hidden_size=self.mix_lstm_hidden_dim,bidirectional=True)
        self.predictor_pre = nn.Sequential(
            Permute(0, 2, 1),
            nn.BatchNorm1d(2 * self.mix_lstm_hidden_dim),
            nn.ReLU(),
            Permute(0, 2, 1),
            nn.Linear(2 * self.mix_lstm_hidden_dim, self.predictor_pre_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.predictor_pre_hidden_dim, 1),
            nn.Sigmoid()
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.time_steps, self.predictor_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.predictor_hidden_dim, 2)
        )

    def forward(self,vfs:torch.Tensor,sfs:torch.Tensor,vlens,slens):
        vfs=self.drop(vfs)
        sfs=self.drop(sfs)
        s_short_packed,_=self.s_lstm(pack_padded_sequence(sfs.transpose(0,1),slens,enforce_sorted=False))
        s_short,_=pad_packed_sequence(s_short_packed)
        #T,B,D
        s_att_value=self.s_att(s_short).transpose(0, 1)
        #B,T,1
        for i,l in enumerate(slens):
            s_att_value[i][l:][:]=-1e10
        s_att_value=torch.softmax(s_att_value,dim=1)
        s_attned=(s_att_value*s_short.transpose(0,1)).sum(dim=-2,keepdim=True)
        #B,1,D
        v_attned=self.guid(vfs,s_attned)
        #B,T,D
        mix_in=torch.cat([v_attned.transpose(0,1),s_attned.transpose(0,1).repeat([max(vlens),1,1])],dim=-1)
        mix_in=self.drop(mix_in)
        mix_out_packed,_=self.mix_lstm(pack_padded_sequence(mix_in,vlens,enforce_sorted=False))
        mix_out,_=pad_packed_sequence(mix_out_packed)
        mix_out=mix_out.transpose(0,1)
        #mix_out shape (B,T, num_directions * hidden_size):
        y=self.predictor_pre(mix_out).squeeze(-1)
        start_end_index=self.predictor(y)
        return y,start_end_index
    def get_loss(self,yp,sep,yt,set,weight_balance=1e4):
        loss1=F.binary_cross_entropy(yp,yt,reduction='none').sum(dim=-1)
        loss2=torch.nn.MSELoss(reduction='none')(sep,set).sum(-1)
        masks=torch.zeros(len(sep)).float().cuda()
        for i in range(len(masks)):
            iou=get_iou(sep[i],set[i])
            masks[i]=(1-iou)
        loss1=loss1*masks
        loss2=loss2*masks
        return torch.sum(loss1*weight_balance+loss2)/len(yt)

class CharadesDataset(Dataset):
    def __init__(self, v_data_dir, sta_txt, glove_file, charades_info_file, norm_time_steps=100, v_dim=1024, s_dim=300):
        super().__init__()
        self.s_dim = s_dim
        self.v_dim = v_dim
        self.norm_time_steps = norm_time_steps
        print('constructing word db')
        self.db = {}
        with open(glove_file, 'r') as db_file:
            lines = db_file.readlines()
            for line in lines:
                items = line.split(' ')
                k = items[0]
                v = np.array(list(map(float, items[1:])))
                self.db[k] = v
        print('constructing data_list')
        self.sta_list = []
        with open(sta_txt, 'r') as sta_file:
            lines = sta_file.readlines()
            for line in lines:
                com1, sentence = line.split('##')
                vid_str, start_str, end_str = com1.split(" ")
                self.sta_list.append((vid_str, float(start_str), float(end_str), sentence))
        np.random.shuffle(self.sta_list)
        print('get charades info')
        self.charades_info = json.load(open(charades_info_file, 'r'))
        self.i3d_h5 = h5py.File(v_data_dir, 'r',driver='core')

    def __getitem__(self, index: int):
        vid, st, et, sentence = self.sta_list[index]
        vf_src = np.array(self.i3d_h5[vid]['i3d_rgb_features'])
        vf = resize(vf_src, (self.norm_time_steps, self.v_dim))

        words = nltk.word_tokenize(sentence.lower())
        sf = []
        for w in words:
            try:
                sf.append(self.db[w])
            except KeyError:
                pass
        sf = np.array(sf)

        v_duration = self.charades_info[vid]['length']
        start_index = int(st / v_duration * self.norm_time_steps)
        end_index = int(et / v_duration * self.norm_time_steps)
        if start_index >= len(vf):
            start_index = len(vf) - 1
        if end_index >= len(vf):
            end_index = len(vf) - 1
        info = {'start_time': st, 'end_time': et, 'sentence': sentence, 'vname': vid, 'v_duration': v_duration,
                'cut_frames': self.norm_time_steps, 'start_index': start_index, 'end_index': end_index}
        return vf, sf, info

    def __len__(self) -> int:
        return len(self.sta_list)
def norm_batch(data,norm_time_steps=100):
    video_features_batch, sentence_features_batch, info_batch = zip(*data)
    v_dim = video_features_batch[0].shape[-1]
    s_dim = sentence_features_batch[0].shape[-1]
    vlen_batch = [len(x) for x in video_features_batch]
    slen_batch = [len(x) for x in sentence_features_batch]
    yt=np.zeros(shape=(len(info_batch),norm_time_steps))
    for i,x in enumerate(yt):
        x[info_batch[i]['start_index']:info_batch[i]['end_index']]=1
    set= np.zeros(shape=(len(info_batch), 2))
    for i, x in enumerate(set):
        x[0]= info_batch[i]['start_index']
        x[1] = info_batch[i]['end_index']
    empty_v = np.zeros(shape=[len(video_features_batch), max(vlen_batch), v_dim])
    empty_s = np.zeros(shape=[len(sentence_features_batch),max(slen_batch), s_dim])
    for i in range(len(empty_v)):
        empty_v[i][0:vlen_batch[i]] = video_features_batch[i]
    for i in range(len(empty_s)):
        empty_s[i][0:slen_batch[i]] = sentence_features_batch[i]
    return empty_v,empty_s,vlen_batch,slen_batch,yt,set,info_batch

def training_on_Charades(model:VideoConv, opt, data_loader,):
    preds = defaultdict(dict)
    lossq = []
    ious=[]
    p_time = time.time()
    batch_no = 0
    for batch_i, v in enumerate(data_loader):
        video_features_batch,sentence_features_batch,vlen_batch,slen_batch,yt_batch,set_batch,info_batch=v
        video_features_batch=torch.from_numpy(video_features_batch).float().cuda()
        sentence_features_batch=torch.from_numpy(sentence_features_batch).float().cuda()
        vlen_batch=torch.tensor(vlen_batch).long().cuda()
        slen_batch=torch.tensor(slen_batch).long().cuda()
        yt_batch=torch.from_numpy(yt_batch).float().cuda()
        set_batch = torch.from_numpy(set_batch).float().cuda()
        yp,sep = model(video_features_batch, sentence_features_batch, vlen_batch, slen_batch)
        loss = model.get_loss(yp,sep,yt_batch,set_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        batch_no += 1
        c_time = time.time()
        lossq.append(float(loss))
        avg_loss = np.average(lossq[-100:-1])
        sep=sep.detach().cpu().numpy()
        iou_batch=[get_iou(sep[i]/model.time_steps*info_batch[i]['v_duration'],(info_batch[i]['start_time'],info_batch[i]['end_time']) )
                   for i in range(len(sep))]
        print('\033[0;34;0m current_loss:%.4f batch:%d avg_loss_latest100:%.4f time_consume_per_batch:%.2fs' % (
            float(loss), batch_no, avg_loss, c_time - p_time))
        # print('\033[33m first of batch truth and predicted: indexes_truth:{} indexes_predicted:{} \
        # time_truth:{} time_predicted:{}'.format(set_batch[0],sep[0],
        #                                         (info_batch[0]['start_time'],info_batch[0]['end_time']),
        #                                         sep[0]/model.time_steps*info_batch[0]['v_duration']))
        #print('\033[35m first of batch yt:{} and yp：{}'.format(yt_batch[0],yp))

        print('training iou_batch:',iou_batch)
        print('\033[32m mean_iou',np.mean(iou_batch))
        ious.extend(iou_batch)
        p_time = c_time
    ious=np.array(ious)
    print('training')
    print('iou>0.1:%.4f'%np.average(ious>0.1))
    print('iou>0.3:%.4f' % np.average(ious > 0.3))
    print('iou>0.5:%.4f' % np.average(ious > 0.5))
    print('iou>0.7:%.4f' % np.average(ious > 0.7))
    print('iou>0.9:%.4f' % np.average(ious > 0.9))
    return lossq


def testing_on_Charades(model, data_loader,log='log_main86.txt'):
    ious=[]
    preds=defaultdict(dict)
    f=open(log,'a+')
    for batch_i, v in enumerate(data_loader):
        video_features_batch, sentence_features_batch, vlen_batch, slen_batch, yt_batch, set_batch, info_batch = v
        video_features_batch=torch.from_numpy(video_features_batch).float().cuda()
        sentence_features_batch=torch.from_numpy(sentence_features_batch).float().cuda()
        vlen_batch=torch.tensor(vlen_batch).long().cuda()
        slen_batch=torch.tensor(slen_batch).long().cuda()
        yt_batch=torch.from_numpy(yt_batch).float().cuda()
        set_batch = torch.from_numpy(set_batch).float().cuda()
        yp, sep = model(video_features_batch, sentence_features_batch, vlen_batch, slen_batch)
        sep = sep.detach().cpu().numpy()
        iou_batch = [get_iou(sep[i] / model.time_steps * info_batch[i]['v_duration'],
                             (info_batch[i]['start_time'], info_batch[i]['end_time']))
                     for i in range(len(sep))]
        # print('\033[33m first of batch truth and predicted: indexes_truth:{} indexes_predicted:{} \
        #         time_truth:{} time_predicted:{}'.format(set_batch[0], sep[0],
        #                                                 (info_batch[0]['start_time'], info_batch[0]['end_time']),
        #                                                 sep[0] / model.time_steps * info_batch[0]['v_duration']))
        print('testing iou_batch:', iou_batch)
        print('\033[32m mean_iou', np.mean(iou_batch))
        for i,x in enumerate(info_batch):
            x['time_predicted']=sep[i] / model.time_steps * info_batch[i]['v_duration']
            preds[x['vname']+"##"+x['sentence']]['iou']=iou_batch[i]
            preds[x['vname'] + "##" + x['sentence']]['time_predicted'] = x['time_predicted']
        ious.extend(iou_batch)
    ious=np.array(ious)
    print('testing:')
    print('iou>0.1:%.4f'%np.average(ious>0.1))
    print('iou>0.3:%.4f' % np.average(ious > 0.3))
    print('iou>0.5:%.4f' % np.average(ious > 0.5))
    print('iou>0.7:%.4f' % np.average(ious > 0.7))
    print('iou>0.9:%.4f' % np.average(ious > 0.9))
    print('iou>0.1:%.4f'%np.average(ious>0.1),file=f)
    print('iou>0.3:%.4f' % np.average(ious > 0.3),file=f)
    print('iou>0.5:%.4f' % np.average(ious > 0.5),file=f)
    print('iou>0.7:%.4f' % np.average(ious > 0.7),file=f)
    print('iou>0.9:%.4f' % np.average(ious > 0.9),file=f,flush=True)
    f.close()
    return preds

def run_on_Charades(train_epochs, eval_per_epochs, save_path):
    import json
    gm = GPUManager()
    torch.cuda.set_device(gm.auto_choice())
    model = VideoConv().cuda()
    opt = Adam(model.parameters())

    train_data_loader = DataLoader(CharadesDataset('data/Charades/charades_i3d_rgb.hdf5',
                                                   'data/Charades/charades_sta_train.txt',
                                                   'glove_wiki/glove.6B.300d.txt',
                                                   'data/Charades/Charades/charades.json'),
                                   batch_size=32, shuffle=False, collate_fn=norm_batch, num_workers=1,pin_memory=True)

    test_data_loader = DataLoader(CharadesDataset('data/Charades/charades_i3d_rgb.hdf5',
                                                  'data/Charades/charades_sta_test.txt',
                                                  'glove_wiki/glove.6B.300d.txt',
                                                  'data/Charades/Charades/charades.json'),
                                  batch_size=128, shuffle=False, collate_fn=norm_batch, num_workers=1,pin_memory=True)

    for epoch_index in range(train_epochs):
        model.train(True)
        loss = training_on_Charades(model, opt, train_data_loader)
        torch.save(model.state_dict(), save_path + str(epoch_index) + '.pkl')
        if epoch_index % eval_per_epochs == 0:
            model.train(False)
            print('start testing on%d' % (epoch_index // eval_per_epochs))
            testing_on_Charades(model, test_data_loader)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    run_on_Charades(100, 1, 'resources/main86_')

#charades permute pred1-batchnorm drop