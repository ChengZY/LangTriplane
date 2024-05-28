import numpy as np
import torch
import torch.nn as nn


def normalize_batch_point_cloud(batch_point_cloud):
    # Find the minimum and maximum values of x, y, and z coordinates
    min_values = torch.min(batch_point_cloud, dim=1).values.unsqueeze(1)
    max_values = torch.max(batch_point_cloud, dim=1).values.unsqueeze(1)

    # Subtract the minimum values to shift the range
    shifted_point_cloud = batch_point_cloud - min_values

    # Divide by the maximum range to scale the values between 0 and 1
    normalized_point_cloud = shifted_point_cloud / (max_values - min_values)

    return normalized_point_cloud

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    # inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    # projections = torch.bmm(coordinates, inv_planes)
    projections = torch.bmm(coordinates, planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates_origin, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'

    if len(coordinates_origin.shape)== 2:
        coordinates = coordinates_origin.unsqueeze(0)
    elif len(coordinates_origin.shape)== 3:
        coordinates = coordinates_origin
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = normalize_batch_point_cloud(coordinates)

    # box_warp = 1
    # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def l1_loss(network_output, gt):
    # return (network_output - gt).mean()
    return torch.abs((network_output - gt)).mean()

if __name__ == "__main__":
    bs = 1
    epoch = 100000
    plane_feature_dim = 8

    loss_fn = nn.MSELoss()
    class Tripplane(nn.Module):
        def __init__(self, plane_feature_dim):
            super(Tripplane, self).__init__()
            self.plane_feature = nn.Parameter(torch.ones((bs, 3, plane_feature_dim, 30, 30),dtype=torch.float32, device='cuda:0').requires_grad_(True))

        def forward(self, plane_axes,points,epoch):
            output = sample_from_planes(plane_axes, self.plane_feature, points)
            # if epoch % 100 == 0:
            #     print('plane_feature_mean:', self.plane_feature.mean())
            return output

    model = Tripplane(plane_feature_dim).to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # plane_feature_gt = torch.ones((3, plane_feature_dim, 32, 32),dtype=torch.float32, device='cuda:0')
    project_feature_gt = torch.ones((bs, 3, 100, plane_feature_dim),dtype=torch.float32, device='cuda:0')
    plane_axes = generate_planes()
    for i in range(epoch):
        points = torch.rand((bs, 100, 3),dtype=torch.float32, device='cuda:0').uniform_(0, 1)
        images = torch.ones((bs, 128, 128, 3),dtype=torch.float32, device='cuda:0')

        plane_axes = plane_axes.to(points.device)
        project_feature = model(plane_axes,points,i)
        # project_feature_gt = torch.ones_like(project_feature, dtype=project_feature.dtype, device=plane_axes.device)

        project_feature_f = project_feature.reshape(-1)
        project_gt_f = project_feature_gt.reshape(-1)
        loss = loss_fn(project_feature_f,project_gt_f)
        loss.backward()
        if i % 100 == 0:
            print('loss:', loss.item())
        optimizer.step()  # optimization step to update model parameters
        optimizer.zero_grad()
