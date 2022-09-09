import torch
import torch.nn.functional as F


class ContrastiveLoss():
    def __init__(self, tau):
        self.tau = tau

    def __call__(self, pred_path, data1, data2, match):
        pred_path_t = pred_path.transpose(1, 2)
        pred_path = F.softmax(pred_path, dim=2)
        pred_path_t = F.softmax(pred_path_t, dim=2)
        g_data2 = torch.matmul(pred_path_t, data1)
        g_data1 = torch.matmul(pred_path, data2)

        dist1 = (data1 - g_data1).pow(2).view(data1.size(0), -1).mean(dim=1)
        dist2 = (data2 - g_data2).pow(2).view(data2.size(0), -1).mean(dim=1)

        loss1 = (match*dist1+(1-match)*F.relu(self.tau-dist1)).mean()
        loss2 = (match*dist2+(1-match)*F.relu(self.tau-dist2)).mean()

        return (loss1+loss2)/2, (dist1+dist2)/2
