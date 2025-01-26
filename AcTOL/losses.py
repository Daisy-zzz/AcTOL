import torch
import torch.nn.functional as F
import random
        

def get_reward_matrix_windowdiff(p_e_s, p_e_text, num_frames, logit_scale=100.):
    diff = []
    for i in range(p_e_s.shape[0]):
        p = p_e_s[i, :num_frames[i], :]  # 当前样本的帧特征序列
        k = num_frames[i]
        # 计算窗口差分的均值
        window_diffs = []
        for window_size in range(k // 2, k):  # 窗口大小从 k / 2 到 k
            for start in range(k - window_size):  # 滑动窗口的开始位置
                window_diff = p[start + window_size] - p[start]  # 窗口差分 
                window_diffs.append(window_diff)
        # 对所有窗口差分取均值
        window_diffs = torch.stack(window_diffs) # L D
        diff.append(window_diffs)  # D
    diff = torch.stack(diff)  # B L D
    diff = diff.permute(1, 0, 2)  # -> L B D
    # 归一化
    diff = diff / diff.norm(dim=-1, keepdim=True)
    p_e_text = p_e_text / p_e_text.norm(dim=-1, keepdim=True)
    # 计算相似度得分
    frame_logits = diff @ p_e_text.t() * logit_scale  # L-1 B D @ D B -> L-1 B B
    reward_matrix = torch.sum(frame_logits, dim=0)
    
    return reward_matrix


def get_reward_matrix_diff(p_e_s, p_e_text, num_frames, logit_scale = 100.):
    diff = []
    # calculate the difference between any two frames
    for i in range(p_e_s.shape[0]):
        p = p_e_s[i, :num_frames[i], :]
        diff_i = []
        for j in range(num_frames[i] - 1, 0, -1):
            diff_i.append(p[j] - p[:j])
        diff_i = torch.concat(diff_i, dim=0) # L D
        diff.append(diff_i)
    diff = torch.stack(diff) # B L D
    diff = diff.permute(1, 0, 2) # -> L B D
    diff = diff / (diff.norm(dim=-1, keepdim=True) + 1e-6)
    p_e_text = p_e_text / p_e_text.norm(dim=-1, keepdim=True)
    frame_logits = diff @ p_e_text.t() * logit_scale # L B D @ D B -> L B B
    reward_matrix = torch.sum(frame_logits, dim=0)
    return reward_matrix

def frame2text(p_e_s, p_e_text, logit_scale = 100.):
    p_e_s = p_e_s.permute(1, 0, 2) # F B D
    p_e_s = p_e_s / p_e_s.norm(dim=-1, keepdim=True)
    p_e_text = p_e_text / p_e_text.norm(dim=-1, keepdim=True)
    frame_logits = p_e_s[-1] @ p_e_text.t() * logit_scale # F B D @ D B -> F B B
    return frame_logits

def avg2text(p_e_s, p_e_text, logit_scale = 100.):
    p_e_s = p_e_s.permute(1, 0, 2) # F B D
    avg = p_e_s.mean(dim=0)
    avg = avg / avg.norm(dim=-1, keepdim=True)
    p_e_text = p_e_text / p_e_text.norm(dim=-1, keepdim=True)
    frame_logits = avg @ p_e_text.t() * logit_scale # B D @ D B -> B B
    return frame_logits

def diff2text(p_e_s, p_e_text, logit_scale = 100.):
    p_e_text = p_e_text / p_e_text.norm(dim=-1, keepdim=True)
    p_e_s = p_e_s.permute(1, 0, 2)
    p_e_s = p_e_s / p_e_s.norm(dim=-1, keepdim=True)
    # let p_e_s = p_e_s[-1]-p_e_s[-2]-...-p_e_s[0]
    # normalize the time interval by the sum of time intervals
    # delta_t = delta_t / delta_t.sum(dim=1, keepdim=True)
    # diff_features = (p_e_s[:, 1:] - p_e_s[:, :-1]) * (t[:, 1:] - t[:, :-1]) 
    # diff_features = diff_features.sum(dim=1) / (t[:, -1] - t[:, 0])
    # 归一化特征向量
    # diff_features = diff_features / diff_features.norm(dim=-1, keepdim=True)
    # diff_features = (p_e_s[:, -1] - p_e_s[:, 0]) / (t[:, -1] - t[:, 0]) # B D
    # diff_features = diff_features / diff_features.norm(dim=-1, keepdim=True)
    frame_logits = (p_e_s[-1] - p_e_s[0]) @ p_e_text.t() * logit_scale # B D @ D B -> B B
    frame_logits = - frame_logits.unsqueeze(-1).norm(2, dim=-1)
    return frame_logits
    

def text2frame(p_e_s, p_e_text, logit_scale = 100.):
    p_e_text = p_e_text / p_e_text.norm(dim=-1, keepdim=True) # B D
    p_e_s = p_e_s / p_e_s.norm(dim=-1, keepdim=True) # B F D
    goal_image = p_e_s[:, -1] # B D
    # frame_logits = p_e_text @ goal_image.t() * logit_scale - p_e_text @ init_image.t() * logit_scale
    frame_logits = []
    for i in range(p_e_text.shape[0]):
        text_i = p_e_text[[i]] # 1 D
        neg_frames = p_e_s[i, [0]] # 1 D
        frames_i = torch.cat((goal_image, neg_frames), dim=0) # B+1 D
        # neg_goal_image = p_e_s[i, [-1]]
        # frames_init_i = torch.cat((init_image, neg_goal_image), dim=0)

        logit_i = text_i @ frames_i.t() * logit_scale #- text_i @ frames_init_i.t() * logit_scale
        frame_logits.append(logit_i)
    frame_logits = torch.cat(frame_logits) # B B+1
    return frame_logits

class DiffLoss(torch.nn.Module):
    def __init__(self, logit_scale = 100, **kwargs):
        
        super().__init__()
        self.logit_scale = logit_scale
    
    def non_decreasing_loss(self, visual_features, text_features, epsilon=0.0, lambda_monotonic=1.0, lambda_trend=1.0, logit_scale=100.0):
        """
        序列非递减约束的损失函数。
        Args:
            sequence (Tensor): 长度为 T 的序列
            epsilon (float): 允许的局部递减阈值
            lambda_monotonic (float): 局部递增约束的权重
            lambda_trend (float): 总体趋势约束的权重
        Returns:
            loss (Tensor): 非递减约束的损失
        """
        # 局部递增约束
        # visual_features = visual_features.permute(1, 0, 2) # F B D
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True) # B F D
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # B D
        sim_seq = visual_features @ text_features.unsqueeze(-1) * logit_scale  # B F D @ B D 1 -> B F 1

        # 计算 loss_monotonic
        sim_seq_diff = sim_seq[:, :-1] - sim_seq[:, 1:]  # B F-1 1
        loss_monotonic = torch.sum(torch.relu(sim_seq_diff - epsilon), dim=(1, 2))  # B

        # 计算 loss_trend
        loss_trend = torch.sum(torch.relu(sim_seq[:, 0] - sim_seq[:, -1]), dim=1)  # B

        # 计算总损失
        loss = lambda_monotonic * loss_monotonic + lambda_trend * loss_trend
        loss = torch.mean(loss)  

        return loss
    
    def get_reward_matrix(self, visual_features, text_features, num_frames, f_labels):

        # frame2text_logtis = []
        # text2frame_logits = []
        # frame2text_logits = avg2text(visual_features, text_features, logit_scale = self.logit_scale)
        # for i in range(1, visual_features.shape[1] + 1):
            # frame2text_logtis.append(frame2text(visual_features[:, :i], text_features, logit_scale = self.logit_scale))
        #     text2frame_logits.append(text2frame(visual_features[:, :i], text_features, logit_scale = self.logit_scale))
        # return frame2text_logits, text2frame_logits
        return frame2text(visual_features, text_features, logit_scale = self.logit_scale), text2frame(visual_features, text_features, logit_scale = self.logit_scale)
        # return diff2text(visual_features, text_features, logit_scale = self.logit_scale)
    
    def forward(self, visual_features, text_features, num_frames, f_labels):
        
        batch_size = visual_features.shape[0]
        reward_matrix, reward_matrix_2 = self.get_reward_matrix(visual_features, text_features, num_frames, f_labels)
        # reward_matrix = self.get_reward_matrix(visual_features, text_features, num_frames, f_labels)

        labels = torch.arange(batch_size, device=reward_matrix[0].device).long()
        # frame2text_logits = 0.
        # frame2text_logits = F.cross_entropy(reward_matrix, labels)
        # text2frame_logits = 0.
        # for reward2 in reward_matrix_2:
            # frame2text_logits += F.cross_entropy(reward1, labels)
            # text2frame_logits += F.cross_entropy(reward2, labels)
        # frame2text_logits /= len(reward_matrix)
        # text2frame_logits /= len(reward_matrix_2)
        # return (frame2text_logits + text2frame_logits) / 2 
        return (F.cross_entropy(reward_matrix, labels) + F.cross_entropy(reward_matrix_2, labels)) / 2 
                
        

class LabelDifference(torch.nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(torch.nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        if self.similarity_type == 'l2':
            # nf, 1, d - 1, nf, d
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'cosine':
            features = features / features.norm(2, dim=-1, keepdim=True)
            return features @ features.T
        else:
            raise ValueError(self.similarity_type)



class RnCLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn_v = FeatureSimilarity(feature_sim)     
        # self.feature_sim_fn_l = FeatureSimilarity('cosine')

    def forward(self, visual_features, text_features, num_frames, f_labels):
        # visual_features: [bs, nf, feat_dim]
        # text_features: [bs, feat_dim]
        # labels: [bs, nf, label_dim]
                        
        total_loss = 0.
        total_bb_loss = 0.
        bs = visual_features.shape[0]            
        for i in range(bs):
            # # Extract features and labels for the current batch element
            current_features = visual_features[i, :num_frames[i]]
            # current_features = cond_features[i]
            current_labels = f_labels[i, :num_frames[i]]
            # add brownian bridge loss
            bb_loss = brownian_bridge_loss(current_features, current_labels)
            
            label_diffs = self.label_diff_fn(current_labels)
            # logits = self.feature_sim_fn_v(current_features).div(self.t) # n, n
            current_features = current_features / current_features.norm(2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(2, dim=-1, keepdim=True)
            logits = current_features @ text_features[[i]].T / self.t # nf 1
            # logits = - torch.abs(logits[:, None, :] - logits[None, :, :]).squeeze(-1)
            logits = - (logits[:, None, :] - logits[None, :, :]).norm(2, dim=-1)
            # logits = - (logits[:, None, :] - logits[None, :, :]).squeeze(-1)
            # # set up triangle of logits to -logits
            # mask = torch.triu(torch.ones_like(logits), diagonal=1).bool()
            # # Apply the mask to set logits[i, j] where i < j to -logits[i, j]
            # logits[mask] = -logits[mask]
            
            
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            logits -= logits_max.detach()
            exp_logits = logits.exp()
            n = logits.shape[0]  
 
            # remove diagonal
            logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
            exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
            label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
            loss = 0.
            for k in range(n - 1): 
                pos_logits = logits[:, k]  # 2bs
                pos_label_diffs = label_diffs[:, k]  # 2bs
                neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
                pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
                loss_k = - (pos_log_probs / (n * (n - 1))).sum()
                loss += loss_k
            total_loss += loss
            total_bb_loss += bb_loss
        # print(total_loss / bs, total_bb_loss / bs)
        return total_loss / bs, total_bb_loss / bs

import torch
def brownian_bridge_loss(current_features, current_labels):
    """
    计算布朗桥约束损失，动态计算方差，适用于非均匀采样帧。

    参数:
        current_features (torch.Tensor): 形状为 (N, d) 的特征张量，其中
                                          N 表示帧的数量，d 表示特征维度。
        current_labels (torch.Tensor): 形状为 (N,) 的帧序号张量，表示帧的时间戳，
                                       必须是从小到大排列的。

    返回:
        torch.Tensor: 标量张量，表示布朗桥约束损失。
    """
    # 起始帧和终止帧特征
    start_feature = current_features[0]   # 形状: (d,)
    end_feature = current_features[-1]    # 形状: (d,)

    # 获取时间范围
    A = current_labels[0]  # 起始时间
    T = current_labels[-1]  # 结束时间
    t = current_labels  # 当前帧时间

    # 计算 alpha
    alpha = (t - A) / (T - A)  # 形状: (N,)

    # 计算线性插值的目标特征
    linear_interpolation = (1 - alpha).unsqueeze(1) * start_feature + alpha.unsqueeze(1) * end_feature  # 形状: (N, d)

    # 计算动态方差
    sigma_squared = alpha * (T - t)  # 形状: (N,)

    # 计算布朗桥约束损失
    squared_diff = torch.sum((current_features[1:-1] - linear_interpolation[1:-1]) ** 2, dim=1)  # 形状: (N,)
    bridge_loss = torch.sum(squared_diff / (2 * sigma_squared[1:-1]))  # 使用动态方差进行归一化

    return bridge_loss

# def brownian_bridge_loss(current_features, current_labels):
#     """
#     计算布朗桥约束损失，适用于非均匀采样帧。

#     参数:
#         current_features (torch.Tensor): 形状为 (N, d) 的特征张量，其中
#                                           N 表示帧的数量，d 表示特征维度。
#         current_labels (torch.Tensor): 形状为 (N,) 的帧序号张量，表示帧的时间戳，
#                                        必须是从小到大排列的。

#     返回:
#         torch.Tensor: 标量张量，表示布朗桥约束损失。
#     """
#     # 起始帧和终止帧特征
#     start_feature = current_features[0]   # 形状: (d,)
#     end_feature = current_features[-1]    # 形状: (d,)

#     # 归一化 current_labels，使其范围在 [0, 1] 之间
#     t = (current_labels - current_labels[0]) / (current_labels[-1] - current_labels[0])  # 形状: (N,)

#     # 生成线性插值的目标特征
#     linear_interpolation = start_feature * (1 - t).unsqueeze(1) + end_feature * t.unsqueeze(1)  # 形状: (N, d)

#     # 计算布朗桥约束损失 (均方误差)
#     bridge_loss = torch.sum((current_features - linear_interpolation) ** 2)

#     return bridge_loss / 100
