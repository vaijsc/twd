import numpy as np
import torch
import time
import math
from mlp import MLP
from torch import optim
import torch.nn.functional as F

class TW():
    def __init__(self, mlp, nofprojections=1000, nlines=5, p=2, mass_division='uniform', device="cuda"):
        self.nofprojections = nofprojections
        self.device = device
        self.nlines = nlines
        self.p = p
        self.model = mlp
        self.mass_division = mass_division

    def tw(self, X, Y, theta, intercept, subsequent_sources, lr=1e-4, iterations=50):
        torch.autograd.set_detect_anomaly(True)
        X = X.to(self.device)
        Y = Y.to(self.device)

        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        num_trees = self.nofprojections

        if self.mass_division == 'uniform':

            _, combined_axis_coordinate_with_intercept, combined_mass, combined_axis_coordinate = self.get_mass_and_coordinate(X, Y, theta, intercept, subsequent_sources)

            mass_X = combined_mass.clone()
            mass_Y = combined_mass.clone()
            mass_X[:, :, N : ] = 0
            mass_Y[:, :, : N] = 0
            
            mass_X = torch.transpose(mass_X, -2, -1)
            mass_Y = torch.transpose(mass_Y, -2, -1)
            #print("Mass_X.shape", mass_X.shape)

            mass_X = mass_X.flatten(-2, -1).unsqueeze(-2)
            mass_Y = mass_Y.flatten(-2, -1).unsqueeze(-2)
            #print("Mass_X_shape", mass_X.shape)
            #print("Mass_X", mass_X)
            #print("Mass_Y", mass_Y.shape)
        
            # Get H

            combined_axis_coordinate_with_intercept[:, -1, 1] = 1e3
            H = self.get_H_seq_of_line(combined_axis_coordinate_with_intercept)
            #print(H.shape)
            H = H.view(num_trees, H.shape[1] * H.shape[2], H.shape[3] * H.shape[4])
            #print("H", H)
            #print("H.shape", H.shape)
            dt_combined_axis_coordinate_with_intercept = torch.sort(combined_axis_coordinate_with_intercept, dim=-1)
            combined_axis_coordinate_with_intercept_sorted = dt_combined_axis_coordinate_with_intercept.values
            edge_length = combined_axis_coordinate_with_intercept_sorted[:, :, 1:] - combined_axis_coordinate_with_intercept_sorted[:, :, :-1]
            edge_length = edge_length.view(edge_length.size(0), -1).unsqueeze(1).clone()
            #print("Edge_length", edge_length)
            #print("Edge_length_shape", edge_length.shape)
            substract_mass = ((torch.abs((torch.matmul(mass_X - mass_Y, H)))) ** self.p) * edge_length
            substract_mass = substract_mass.view(substract_mass.size(0), -1)
            substract_mass_sum = torch.sum(substract_mass, dim = 1)
            tw = torch.mean(substract_mass_sum) ** (1/self.p)

        elif self.mass_division == 'learnable':
            #self.model.reset_weights()
            #optimizer = optim.Adam(self.model.parameters(), lr=lr)
            #total_loss = np.zeros((iterations,))
            #for i in range(iterations):
                #optimizer.zero_grad()
            _, combined_axis_coordinate_with_intercept, combined_mass, combined_axis_coordinate = self.get_mass_and_coordinate(X, Y, theta, intercept, subsequent_sources)

            mass_X = combined_mass.clone()
            mass_Y = combined_mass.clone()
            mass_X[:, :, N:] = 0
            mass_Y[:, :, :N] = 0

            mass_X = torch.transpose(mass_X, -2, -1).clone()
            mass_Y = torch.transpose(mass_Y, -2, -1).clone()

            mass_X = mass_X.flatten(-2, -1).unsqueeze(-2).clone()
            mass_Y = mass_Y.flatten(-2, -1).unsqueeze(-2).clone()
            #print("Mass_Y", mass_Y)

            combined_axis_coordinate_with_intercept[:, -1, 1] = 1e9
            H = self.get_H_seq_of_line(combined_axis_coordinate_with_intercept).clone()
            H = H.view(num_trees, H.shape[1] * H.shape[2], H.shape[3] * H.shape[4])

            dt_combined_axis_coordinate_with_intercept = torch.sort(combined_axis_coordinate_with_intercept.clone(), dim=-1)
            combined_axis_coordinate_with_intercept_sorted = dt_combined_axis_coordinate_with_intercept.values.clone()
            edge_length = combined_axis_coordinate_with_intercept_sorted[:, :, 1:] - combined_axis_coordinate_with_intercept_sorted[:, :, :-1]

            edge_length = edge_length.view(edge_length.size(0), -1).unsqueeze(1).clone()

            substract_mass = ((torch.abs((torch.matmul(mass_X, H) - torch.matmul(mass_Y, H)))) ** self.p) * edge_length
            substract_mass = substract_mass.view(substract_mass.size(0), -1)
            substract_mass_sum = torch.sum(substract_mass, dim = 1)
            tw = torch.mean(substract_mass_sum) ** (1/self.p)
            #loss = tw
            #total_loss[i] = loss.item()
                #loss.backward(retain_graph=True)
                #optimizer.step()

        return tw

    def get_mass_and_coordinate(self, X, Y, theta, intercept, subsequent_sources):
        # for the last dimension
        # 0, 1, 2, ...., N -1 is of distribution 1
        # N, N + 1, ...., 2N -1 is of distribution 2
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        num_trees = self.nofprojections
        subsequent_sources_translated = subsequent_sources - intercept

        subsequent_sources_coordinate = torch.einsum('abc,abc->ab', subsequent_sources_translated, theta).unsqueeze(-1)
        theta_norm = torch.norm(torch.tensor(theta), dim=-1, keepdim=True)
        theta = torch.tensor(theta) / theta_norm
        _, mass_X, axis_coordinate_X = self.project(X, theta=theta, intercept=intercept)
        _, mass_Y, axis_coordinate_Y = self.project(Y, theta=theta, intercept=intercept)

        combined_mass = torch.cat((mass_X, mass_Y), dim=2)
        intercept_mass = torch.zeros((combined_mass.shape[0], combined_mass.shape[1], 1), device=self.device)
        subsequent_sources_mass = torch.zeros((combined_mass.shape[0], combined_mass.shape[1], 1), device=self.device)
        combined_mass_with_intercept = torch.cat((intercept_mass, subsequent_sources_mass, combined_mass), dim=2)

        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=2)
        intercept_coordinate = torch.zeros((combined_axis_coordinate.shape[0], combined_axis_coordinate.shape[1], 1), device=self.device)
        combined_axis_coordinate_with_intercept = torch.cat((intercept_coordinate, subsequent_sources_coordinate, combined_axis_coordinate), dim=2)

        return combined_mass_with_intercept, combined_axis_coordinate_with_intercept, combined_mass, combined_axis_coordinate      

    def project(self, input, theta, intercept, epsilon=1e-4):
        N, d = input.shape
        num_trees = theta.shape[0]
        num_lines = theta.shape[1]
        if self.mass_division == 'uniform':
            mass_input = torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)
            #print(mass_input)
        elif self.mass_division == 'learnable':
            #print('Executing here')
            mass = self.model(input) # (num_points, num_lines)
            mass_division_expanded = mass.unsqueeze(0).expand(num_trees, -1, -1) # (Num_trees, num_poins, num_lines)
            mass_ratio = mass_division_expanded.permute(0, 2, 1)
            mass_input = (torch.ones((num_trees, num_lines, N), device=self.device) / N)
            mass_input = mass_input * mass_ratio
        input = input.unsqueeze(0).unsqueeze(0).repeat(theta.shape[0], theta.shape[1], 1, 1)
        intercept = intercept.unsqueeze(2).repeat(1, 1, N, 1)
        input_translated = input - intercept
        axis_coordinate = torch.einsum('teld,ted->tel', input_translated, theta)
        input_projected_translated = torch.einsum('tel,ted->teld', axis_coordinate, theta)
        input_projected = input_projected_translated + intercept

        return input_projected, mass_input, axis_coordinate

    def find_indices(self, tensor, values):
        bsz, num_row, num_col = tensor.shape
        temp =  torch.nonzero(tensor[..., None] == values)
        indices = temp[:, :-1]
        index_type = temp[:, -1]
        output = torch.full([values.shape[0], bsz, num_row],
                        float(-1e-9), device=tensor.device, dtype=torch.float)
        output[index_type, indices[:, 0], indices[:, 1]] = indices[:, 2].float()
        return output

    def get_H_seq_of_line(self, coord_matrix):
        num_tree, num_line, num_point_per_line = coord_matrix.shape
        num_projection_point = num_point_per_line - 2
        num_segment = num_point_per_line - 1

        diag = torch.eye(num_line, device=coord_matrix.device)
        strict_lower_diag = torch.eye(num_line, device=coord_matrix.device).flip(-1).cumsum(dim=-1).flip(-1) \
                            - torch.eye(num_line, device=coord_matrix.device)
        diag_indices = torch.nonzero(diag).transpose(0, 1)
        strict_lower_diag_indices = torch.nonzero(strict_lower_diag).transpose(0, 1)

        coord_matrix_sorted, indices = torch.sort(coord_matrix, dim=2)

        mask = indices - 2

        point_to_find = torch.tensor([-2, -1, *list(range(0, num_projection_point))], dtype=torch.int64, device=mask.device)
        indices_source_branch_proj_point = self.find_indices(mask, point_to_find)
        indices_source_point = indices_source_branch_proj_point[0].unsqueeze(0).repeat(num_projection_point+1, 1, 1).unsqueeze(0)
        indices_source_branch_proj_point = indices_source_branch_proj_point[1:].unsqueeze(0)

        indices_source_branch_proj_point, _ = torch.cat([indices_source_point, indices_source_branch_proj_point], dim=0).sort(dim=0)

        source_to_branch_proj_point_left = torch.zeros([num_projection_point+1, num_tree, num_line, num_segment], device=mask.device, dtype=torch.float)
        source_to_branch_proj_point_right = torch.zeros_like(source_to_branch_proj_point_left, device=mask.device, dtype=torch.float)
        ones = torch.ones_like(source_to_branch_proj_point_left, device=mask.device, dtype=torch.float)

        source_to_branch_proj_point_left.scatter_(dim=-1, index=indices_source_branch_proj_point[0].unsqueeze(-1).long(), src=ones)
        source_to_branch_proj_point_right.scatter_(dim=-1, index=(indices_source_branch_proj_point[1].unsqueeze(-1) - 1).long(), src=ones)

        source_to_branch_proj_point_left = torch.cumsum(source_to_branch_proj_point_left, dim=-1)
        source_to_branch_proj_point_right = torch.cumsum(source_to_branch_proj_point_right.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))

        source_to_branch_proj_point = source_to_branch_proj_point_left * source_to_branch_proj_point_right

        H = torch.zeros([num_tree, num_projection_point, num_line, num_line, num_segment], device=mask.device)

        source_to_proj_point = source_to_branch_proj_point[1:].transpose(0, 1)
        H[:, :, diag_indices[0], diag_indices[1], :] = source_to_proj_point

        branch_to_source = source_to_branch_proj_point[0].unsqueeze(1).repeat(1, num_projection_point, 1, 1)
        H[:, :, strict_lower_diag_indices[0], strict_lower_diag_indices[1], :] = branch_to_source[:, :, strict_lower_diag_indices[1], :]

        return H

def generate_trees_frames(L, d, range_root_1=-1.0, range_root_2=1.0, range_source_1=-0.1, range_source_2=0.1, nlines=5, device='cuda', type_lines='sequence_of_lines'):
    root = (range_root_1 - range_root_2) * torch.rand(L, 1, d, device=device) + range_root_2
    if type_lines == 'sequence_of_lines':
        source = (range_source_1 - range_source_2) * torch.rand(L, nlines - 1, device=device) + range_source_2
    else:
        source = torch.zeros(L, nlines - 1, device=device)
    theta = torch.randn(L, nlines, d, device=device)
    theta_mul_source = torch.einsum('tld,tl->tld', theta[:, : nlines - 1, :], source)
    theta_mul_source_cummulative = torch.cumsum(theta_mul_source, dim=1)
    bias = theta_mul_source_cummulative + root
        
    intercept = torch.cat((root, bias), dim=1)
    subsequent_sources = torch.cat((bias, torch.zeros(L, 1, d, device=device)), dim=1)
    return theta, intercept, subsequent_sources
