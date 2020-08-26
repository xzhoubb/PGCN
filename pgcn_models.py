import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        #output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x


class PGCN(torch.nn.Module):
    def __init__(self, model_configs, graph_configs, test_mode=False):
        super(PGCN, self).__init__()

        self.num_class = model_configs['num_class']
        self.adj_num = graph_configs['adj_num']
        self.child_num = graph_configs['child_num']
        self.child_iou_num = graph_configs['iou_num']
        self.child_dis_num = graph_configs['dis_num']
        self.dropout = model_configs['dropout']
        self.test_mode = test_mode
        self.act_feat_dim = model_configs['act_feat_dim']
        self.comp_feat_dim = model_configs['comp_feat_dim']

        self._prepare_pgcn()
        self.Act_GCN = GCN(self.act_feat_dim, 512, self.act_feat_dim, dropout=model_configs['gcn_dropout'])
        ''' 
            gcn_dropout = 0.7
                        GCN(
            (gc1): GraphConvolution (1024 -> 512)
            (gc2): GraphConvolution (512 -> 1024)
            )
        '''
        self.Comp_GCN = GCN(self.comp_feat_dim, 512, self.comp_feat_dim, dropout=model_configs['gcn_dropout'])
        ''' 
            gcn_dropout = 0.7
                        GCN(
            (gc1): GraphConvolution (3072 -> 512)
            (gc2): GraphConvolution (512 -> 3072)
            )
        '''
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def _prepare_pgcn(self):

        self.activity_fc = nn.Linear(self.act_feat_dim * 2, self.num_class + 1) #(1024*2, 53+1)
        self.completeness_fc = nn.Linear(self.comp_feat_dim * 2, self.num_class) #(3072*2, 53)
        self.regressor_fc = nn.Linear(self.comp_feat_dim * 2, 2 * self.num_class) #(3072*2, 2*53)

        nn.init.normal_(self.activity_fc.weight.data, 0, 0.001) # weight.size -> (54, 2048)
        nn.init.constant_(self.activity_fc.bias.data, 0) # weight.size -> (54)
        nn.init.normal_(self.completeness_fc.weight.data, 0, 0.001) # weight.size -> (53, 6144)
        nn.init.constant_(self.completeness_fc.bias.data, 0) # weight.size -> (53)
        nn.init.normal_(self.regressor_fc.weight.data, 0, 0.001) # weight.size -> (2*53, 6144)
        nn.init.constant_(self.regressor_fc.bias.data, 0)  # weight.size -> (2*53)


    def train(self, mode=True):

        super(PGCN, self).train(mode)


    def get_optim_policies(self):

        normal_weight = []
        normal_bias = []

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, GraphConvolution):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
        ] 
        '''
        normal_weight: weight of linear and graph conv layers
        [torch.Size([54, 2048]), 
        torch.Size([53, 6144]), 
        torch.Size([106, 6144]), 
        torch.Size([1024, 512]), torch.Size([512, 1024]), 
        torch.Size([3072, 512]), torch.Size([512, 3072])]
        '''

    def forward(self, input, target, reg_target, prop_type):
        if not self.test_mode:
            return self.train_forward(input, target, reg_target, prop_type)
        else:
            return self.test_forward(input)


    def train_forward(self, input, target, reg_target, prop_type):
        '''
        input:
            # for 2 gpus, 32 -> 16
            input[0]: act_feature, (16, 168, 1024)
            input[1]: comp_feature, (16, 168, 3072)
            target: cls_id, (16, 168)
            reg_target: reg_target, (16, 168, 2)
            prop_type: type of fg/incomplete/bg, (16, 168)
        
        output:
            raw_act_fc[act_indexer, :]: out_act_ft, (32, 54), 32 = 16(bs) * 2 (1(fg)+1(bg))
            target[act_indexer]: out_target, (32)
            type_data[act_indexer]: out_type, only include 0(fg) and 2(bg), (32)
            raw_comp_fc: out_comp_fc, (112, 53), 112 = 16 * 7 (1(fg) + 6(incomplete))
            comp_target: out_comp_target, root node(fg+incomplete) of target, (112), cls_id
            raw_regress_fc[reg_indexer, :, :]: out_regress_fc, (16, 53, 2), 16 = 16(bs) * 1(fg)
            target[reg_indexer]: (16), 16 = 16(bs) * 1(fg)
            reg_target[reg_indexer, :] (16,2)
        '''

        activity_fts = input[0] # torch.Size([16, 168, 1024])
        completeness_fts = input[1] # torch.Size([16, 168, 3072])
        batch_size = activity_fts.size()[0] # 16

        # construct feature matrix
        act_ft_mat = activity_fts.view(-1, self.act_feat_dim).contiguous() # torch.Size([2688, 1024])
        comp_ft_mat = completeness_fts.view(-1, self.comp_feat_dim).contiguous() # torch.Size([2688, 3072])

        # act cosine similarity
        dot_product_mat = torch.mm(act_ft_mat, torch.transpose(act_ft_mat, 0, 1)) # torch.Size([2688, 2688])
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(act_ft_mat * act_ft_mat, dim=1)), dim=0) # torch.Size([1, 2688])
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec) # torch.Size([2688, 2688])
        act_cos_sim_mat = dot_product_mat / len_mat # # torch.Size([2688, 2688])

        # comp cosine similarity
        dot_product_mat = torch.mm(comp_ft_mat, torch.transpose(comp_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(comp_ft_mat * comp_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        comp_cos_sim_mat = dot_product_mat / len_mat
        
        # self->self is 1 (1 num is 5), node->son is 0.25 (0.25 num is 5*4)
        mask = act_ft_mat.new_zeros(self.adj_num, self.adj_num) # (21,21) all 0
        for stage_cnt in range(self.child_num + 1):
            ind_list = list(range(1 + stage_cnt * self.child_num, 1 + (stage_cnt + 1) * self.child_num))
            for i, ind in enumerate(ind_list):
                mask[stage_cnt, ind] = 1 / self.child_num
            mask[stage_cnt, stage_cnt] = 1

        mask_mat_var = act_ft_mat.new_zeros(act_ft_mat.size()[0], act_ft_mat.size()[0]) # (2688, 2688)
        for row in range(int(act_ft_mat.size(0)/ self.adj_num)): # 128 = 16 * 8, self.adj_num:21
            mask_mat_var[row * self.adj_num : (row + 1) * self.adj_num, row * self.adj_num : (row + 1) * self.adj_num] \
                = mask
        
        # filter cos_sim_mat to only remain self-self * 1, self-son * 0.25, others 0
        act_adj_mat = mask_mat_var * act_cos_sim_mat # (2688, 2688) -> non-zero num is: 5*(1(self)+4(self-son))*8(sample_node)*16(batch_size)
        comp_adj_mat = mask_mat_var * comp_cos_sim_mat # (2688, 2688)

        # normalized by the number of nodes
        act_adj_mat = F.relu(act_adj_mat) # adj matrix
        comp_adj_mat = F.relu(comp_adj_mat)
        
        # GCN, act_gcn ((1024,512),(512,1024)), by 2 gcn, the root node get 4 child and 16 grandchild info(from adj_mat)
        act_gcn_ft = self.Act_GCN(act_ft_mat, act_adj_mat) # input: ft(2688*1024), adj_mat(2688,2688) -> out:(2688*1024)
        comp_gcn_ft = self.Comp_GCN(comp_ft_mat, comp_adj_mat)

        out_act_fts = torch.cat((act_gcn_ft, act_ft_mat), dim=-1) # (2688, 2048)
        act_fts = out_act_fts[:-1: self.adj_num, :] # (128, 2048), /21->get root node of every 21 graph, 128=8(sample/video)*16(batch_size)
        act_fts = self.dropout_layer(act_fts)

        out_comp_fts = torch.cat((comp_gcn_ft, comp_ft_mat), dim=-1)
        comp_fts = out_comp_fts[:-1: self.adj_num, :] # (128, 6144)

        raw_act_fc = self.activity_fc(act_fts) # (128,54), 128= 16(bs) * 8(sample of video)
        raw_comp_fc = self.completeness_fc(comp_fts) # (128,53)

        # keep 7 proposal to calculate completeness, rm bg prop (the 8th)
        raw_comp_fc = raw_comp_fc.view(batch_size, -1, raw_comp_fc.size()[-1])[:, :-1, :].contiguous() # (16, 7, 53)
        raw_comp_fc = raw_comp_fc.view(-1, raw_comp_fc.size()[-1]) # (112, 53)
        comp_target = target.view(batch_size, -1, self.adj_num)[:, :-1, :].contiguous().view(-1).data # (16*168) -> (16*7*21) ->(2352)
        comp_target = comp_target[0: -1: self.adj_num].contiguous() # (2352) -> (112) ,/21->get root node

        # keep the target proposal
        type_data = prop_type.view(-1).data # (16*128)
        type_data = type_data[0: -1: self.adj_num] # (128): 128=16*8,/21->get root node
        target = target.view(-1)
        target = target[0: -1: self.adj_num] # (128): 128=16*8,/21->get root node

        act_indexer = ((type_data == 0) + (type_data == 2)).nonzero().squeeze() # (32), 16*2,only fg(1)+bg(1)

        reg_target = reg_target.view(-1, 2) # (2688, 2)
        reg_target = reg_target[0: -1: self.adj_num] # (128,2), 128=16*8
        reg_indexer = (type_data == 0).nonzero().squeeze() # (16), 16*1, only fg to cal reg
        raw_regress_fc = self.regressor_fc(comp_fts).view(-1, self.completeness_fc.out_features, 2).contiguous() #(128,53,2)

        return raw_act_fc[act_indexer, :], target[act_indexer], type_data[act_indexer], \
               raw_comp_fc, comp_target, \
              raw_regress_fc[reg_indexer, :, :], target[reg_indexer], reg_target[reg_indexer, :]

    def test_forward(self, input):

        activity_fts = input[0]
        completeness_fts = input[1]
        batch_size = activity_fts.size()[0]

        # construct feature matrix
        act_ft_mat = activity_fts.view(-1, self.act_feat_dim).contiguous()
        comp_ft_mat = completeness_fts.view(-1, self.comp_feat_dim).contiguous()

        # act cosine similarity
        dot_product_mat = torch.mm(act_ft_mat, torch.transpose(act_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(act_ft_mat * act_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        act_cos_sim_mat = dot_product_mat / len_mat

        # comp cosine similarity
        dot_product_mat = torch.mm(comp_ft_mat, torch.transpose(comp_ft_mat, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(comp_ft_mat * comp_ft_mat, dim=1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec)
        comp_cos_sim_mat = dot_product_mat / len_mat

        mask = act_ft_mat.new_zeros(self.adj_num, self.adj_num)
        for stage_cnt in range(self.child_num + 1):
            ind_list = list(range(1 + stage_cnt * self.child_num, 1 + (stage_cnt + 1) * self.child_num))
            for i, ind in enumerate(ind_list):
                mask[stage_cnt, ind] = 1 / self.child_num
            mask[stage_cnt, stage_cnt] = 1

        mask_mat_var = act_ft_mat.new_zeros(act_ft_mat.size()[0], act_ft_mat.size()[0])
        for row in range(int(act_ft_mat.size(0)/ self.adj_num)):
            mask_mat_var[row * self.adj_num: (row + 1) * self.adj_num, row * self.adj_num: (row + 1) * self.adj_num] \
                = mask

        act_adj_mat = mask_mat_var * act_cos_sim_mat
        comp_adj_mat = mask_mat_var * comp_cos_sim_mat

        # normalized by the number of nodes
        act_adj_mat = F.relu(act_adj_mat)
        comp_adj_mat = F.relu(comp_adj_mat)

        act_gcn_ft = self.Act_GCN(act_ft_mat, act_adj_mat)
        comp_gcn_ft = self.Comp_GCN(comp_ft_mat, comp_adj_mat)

        out_act_fts = torch.cat((act_gcn_ft, act_ft_mat), dim=-1)
        act_fts = out_act_fts[:-1: self.adj_num, :]

        out_comp_fts = torch.cat((comp_gcn_ft, comp_ft_mat), dim=-1)
        comp_fts = out_comp_fts[:-1: self.adj_num, :]

        raw_act_fc = self.activity_fc(act_fts)
        raw_comp_fc = self.completeness_fc(comp_fts)

        raw_regress_fc = self.regressor_fc(comp_fts).view(-1, self.completeness_fc.out_features * 2).contiguous()

        return raw_act_fc, raw_comp_fc, raw_regress_fc


