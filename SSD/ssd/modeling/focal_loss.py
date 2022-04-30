import torch.nn as nn
import torch
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def focal_loss(p, y, gamma):
    """
    Args:
        p: [32, 9, 65440]
        y = [32, 65440]
    """
    n_classes = p.size(1)
    y = F.one_hot(y.long(),num_classes=n_classes)
    y = y.transpose(1, 2).contiguous()

    alpha = torch.ones([n_classes]) * 1000.0
    alpha[0] = 10
    # alpha.size() = 9

    p = F.softmax(p,dim=1)
    #p: [32, 9, 65440]

    term1 = torch.pow(1 - p,gamma)
    #term1: [32, 9, 65440]

    term2 = torch.log(p)
    #term2: [32, 9, 65440]

    term3 = term1 * y * term2
    
    
    term3 = torch.einsum('a,bcd->bad', -alpha.to(device), term3.to(device))
    #term3: [32, 9, 65440]
    
    
    return term3

class SSDMultiboxLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)


    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]

            batch_size = 32
            num_anchors = 65440

            bbox_delta: [32, 4, 65440]
            confs: [32, 9, 65440]
            gt_bbox: [32, 65440, 4]
            gt_labels = [32, 65440]
        """

        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]
        classification_loss = focal_loss(confs, gt_labels, gamma=3).sum(dim=1).mean()
        print(classification_loss)

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        total_loss = regression_loss/num_pos + classification_loss
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=classification_loss,
            total_loss=total_loss
        )
        return total_loss, to_log
