import torch 

import torch 

def matrices_P1():
    matrices = torch.eye(3).unsqueeze(0)  # (1, 3, 3)
    translations = torch.zeros(1, 3)      # (1, 3)
    return matrices, translations

def matrices_P_minus1():
    matrices = torch.stack([torch.eye(3), -torch.eye(3)], dim=0)  # (2, 3, 3)
    translations = torch.zeros(2, 3)
    return matrices, translations

def matrices_P1211():
    matrices = torch.stack([
        torch.eye(3),
        torch.diag(torch.tensor([-1., 1., -1.]))
    ], dim=0)
    translations = torch.stack([
        torch.zeros(3),
        torch.tensor([0., 0.5, 0.])
    ], dim=0)
    return matrices, translations

def matrices_P22121():
    matrices = torch.stack([
        torch.eye(3),
        torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]),
        torch.tensor([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]),
        torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    ], dim=0)
    translations = torch.stack([
        torch.zeros(3),
        torch.tensor([0., 0., 0.]),
        torch.tensor([0., 0.5, 0.5]),
        torch.tensor([0, 0.5, 0.5])
    ], dim=0)
    return matrices, translations

class Symmetry:
    def __init__(self, space_group):
        self.space_group = space_group.strip().replace(' ','')
        self.matrices, self.translations = self._get_ops(self.space_group)
        self.matrices = self.matrices.to(torch.float64)  # Ensure matrices are float
        self.translations = self.translations.to(torch.float64)  # Ensure translations are float

    def cuda(self):
        self.matrices = self.matrices.cuda()
        self.translations = self.translations.cuda()
        return self
    
    def cpu(self):
        self.matrices = self.matrices.cpu()
        self.translations = self.translations.cpu()
        return self

    def _get_ops(self, space_group):
        if space_group == 'P1':
            return matrices_P1()
        elif space_group == 'P-1':
            return matrices_P_minus1()
        elif space_group == 'P1211' or space_group == 'P21':
            return matrices_P1211()
        elif space_group == 'P22121':
            return matrices_P22121()
        else:
            raise ValueError(f'space group: {space_group} not implemented')

    def apply(self, fractional_coords):
        coords = fractional_coords.reshape(3, -1)  # (3, N)
        coords = coords.unsqueeze(0)  # (1, 3, N)
        transformed = torch.matmul(self.matrices, coords) + self.translations.unsqueeze(2)
        # transformed: (ops, 3, N)
        return transformed.permute(1, 2, 0)  # (3, N, ops)

    def __call__(self, fractional_coords):
        return self.apply(fractional_coords)

    def __repr__(self):
        return f'Symmetry(space_group={self.space_group})'