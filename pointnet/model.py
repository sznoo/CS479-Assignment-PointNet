import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO 0-1 : Implement point-wise mlp model based on PointNet Architecture.
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64), 
            # nn.BatchNorm1d(2048),
            nn.ReLU(), 
            nn.Linear(64, 64), 
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64), 
            # nn.BatchNorm1d(2048),
            nn.ReLU(), 
            nn.Linear(64, 128), 
            # nn.BatchNorm1d(2048),
            nn.ReLU(), 
            nn.Linear(128, 1024)
        )
    

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """
        device = pointcloud.device
        
        b, n, _ = pointcloud.shape
        # TODO 0-2: Implement forward function.
        if self.input_transform:
            trans3 = self.stn3(pointcloud.permute(0, 2, 1)).to(device)
            x  = torch.bmm(pointcloud, trans3).to(device)

        x = self.mlp1(x)

        if self.feature_transform:
            trans64 = self.stn64(x.permute(0, 2, 1)).to(device)
            x = torch.bmm(x, trans64).to(device)

        global_feature = self.mlp2(x)
        global_feature, _ = torch.max(global_feature, dim =1)
        if self.mode == 'cls' or self.mode == 'ae':
            return global_feature
        if self.mode == 'seg':
            global_feature_expand = global_feature.reshape(b, 1, 1024).expand(-1, n, 1024)
            z = torch.cat([x, global_feature_expand], dim = 2)
            return z


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        self.pointnet_feat.mode = 'cls'
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, self.num_classes)
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        device = pointcloud.device
        # TODO : Implement forward function.
        x = self.pointnet_feat(pointcloud).to(device)
        x = self.mlp(x)
        return x
        


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        input_transform = True
        feature_transform = True
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        self.pointnet_feat.mode = 'seg'

        self.mlp1 = nn.Sequential(
            nn.Linear(1088, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, m), 
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        device = pointcloud.device
        x = self.pointnet_feat(pointcloud).to(device)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = x.permute(0, 2, 1)
        return x
        


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.num_points = num_points

        input_transform = True
        feature_transform = True
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        self.pointnet_feat.mode = 'ae'

        self.fc1 = nn.Sequential(
            nn.Linear(1024, self.num_points//4), 
            nn.BatchNorm1d(self.num_points//4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.num_points//4, self.num_points//2), 
            nn.BatchNorm1d(self.num_points//2), 
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.num_points//2, self.num_points), 
            nn.Dropout(0.25), 
            nn.BatchNorm1d(self.num_points), 
            nn.ReLU()
        )
        self.fc4 = nn.Linear(self.num_points, self.num_points*3)



    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        b, n, _ = pointcloud.shape
        device = pointcloud.device
        x = self.pointnet_feat(pointcloud).to(device)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.reshape(b, n, 3)
        return x

def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
