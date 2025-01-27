import torch
import torch.nn.functional as F
import random
        

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

def brownian_bridge_loss(current_features, current_labels):
    """
    Computes the Brownian Bridge constraint loss with dynamic variance, suitable for non-uniformly sampled frames.

    Parameters:
        current_features (torch.Tensor): A feature tensor of shape (N, d), where
                                          N represents the number of frames, and d represents the feature dimension.
        current_labels (torch.Tensor): A tensor of shape (N,) representing the frame indices or timestamps,
                                       which must be sorted in ascending order.

    Returns:
        torch.Tensor: A scalar tensor representing the Brownian Bridge constraint loss.
    """
    # Features of the starting and ending frames
    start_feature = current_features[0]   # Shape: (d,)
    end_feature = current_features[-1]    # Shape: (d,)

    # Get the time range
    A = current_labels[0]  # Start time
    T = current_labels[-1]  # End time
    t = current_labels  # Current frame timestamps

    # Compute alpha
    alpha = (t - A) / (T - A)  # Shape: (N,)

    # Compute the target features for linear interpolation
    linear_interpolation = (1 - alpha).unsqueeze(1) * start_feature + alpha.unsqueeze(1) * end_feature  # Shape: (N, d)

    # Compute dynamic variance
    sigma_squared = alpha * (T - t)  # Shape: (N,)

    # Compute the Brownian Bridge constraint loss
    squared_diff = torch.sum((current_features[1:-1] - linear_interpolation[1:-1]) ** 2, dim=1)  # Shape: (N,)
    bridge_loss = torch.sum(squared_diff / (2 * sigma_squared[1:-1]))  # Normalize using dynamic variance

    return bridge_loss

class AcTOL_loss(torch.nn.Module):
    def __init__(self, temperature=0.01, label_diff='l1'):
        super(AcTOL_Loss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)

    def forward(self, visual_features, text_features, num_frames, f_labels):
        # visual_features: [bs, nf, feat_dim]
        # text_features: [bs, feat_dim]
        # labels: [bs, nf, label_dim]
                        
        total_loss_vlo = 0.
        total_bb_loss = 0.
        bs = visual_features.shape[0]            
        for i in range(bs):
            current_features = visual_features[i, :num_frames[i]]
            current_labels = f_labels[i, :num_frames[i]]
            # add brownian bridge loss
            bb_loss = brownian_bridge_loss(current_features, current_labels)
            
            label_diffs = self.label_diff_fn(current_labels)
            current_features = current_features / current_features.norm(2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(2, dim=-1, keepdim=True)
            logits = current_features @ text_features[[i]].T / self.t 
            logits = - (logits[:, None, :] - logits[None, :, :]).norm(2, dim=-1)
            
            
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            logits -= logits_max.detach()
            exp_logits = logits.exp()
            n = logits.shape[0]  
 
            # remove diagonal
            logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
            exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
            label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
            loss_vlo = 0.
            for k in range(n - 1): 
                pos_logits = logits[:, k]  
                pos_label_diffs = label_diffs[:, k]  
                neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  
                pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  
                loss_k = - (pos_log_probs / (n * (n - 1))).sum()
                loss += loss_k
            total_loss_vlo += loss_vlo
            total_bb_loss += bb_loss
        return total_loss_vlo / bs, total_bb_loss / bs




