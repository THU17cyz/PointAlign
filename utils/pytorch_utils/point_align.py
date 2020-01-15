import torch


class PointAlign:
    def __init__(self, normal, azi_vec, nsample):
        """
        normal: normal vector for the first alignment
        azi_vec: azimuthal vector for the second alignment
        """
        batch_s = normal.size()[0]
        num_s = normal.size()[1]

        normal += torch.Tensor([0, 0, 1e-8]).unsqueeze(0).unsqueeze(0).expand(batch_s, num_s, 3).cuda()

        yz = torch.norm(normal[:, :, 1:], dim=2).unsqueeze(-1)
        # yz += torch.Tensor([1e-8]).unsqueeze(0).unsqueeze(0).expand(B, N, 1).cuda()
        selector = torch.LongTensor([0, 1, 2, 1, 0, 0, 2, 0, 0]).cuda()
        mask = torch.Tensor([[0, 1, 1], [-1, 0, 0], [-1, 0, 0]]).unsqueeze(0).unsqueeze(0).expand(batch_s, num_s, 3, 3).cuda()
        first_mat = torch.index_select(normal, 2, selector).view(batch_s, num_s, 3, 3).mul(mask)

        first_row = torch.matmul(first_mat, normal.unsqueeze(-1)).squeeze(-1)
        first_row = first_row / yz

        selector2 = torch.LongTensor([0, 2, 1]).cuda()
        mask2 = torch.Tensor([0, 1, -1]).unsqueeze(0).unsqueeze(0).expand(batch_s, num_s, 3).cuda()
        second_row = torch.index_select(normal, 2, selector2).mul(mask2)
        second_row = second_row / yz

        third_row = normal

        first_align_matrix = torch.stack([first_row, second_row, third_row], dim=2)

        rotated_azi_vec = torch.matmul(first_align_matrix, azi_vec.unsqueeze(-1)).squeeze(-1)

        xy = torch.norm(rotated_azi_vec[:, :, :2], dim=2).unsqueeze(-1)
        tiny = torch.Tensor([1e-8]).unsqueeze(0).unsqueeze(0).expand(batch_s, num_s, 1).cuda()
        xy = xy + tiny
        rotated_rel_divxy = rotated_azi_vec[:, :, :2] / xy
        xs = rotated_rel_divxy[:, :, 0]
        ys = rotated_rel_divxy[:, :, 1]
        zeros = torch.zeros_like(xs)
        ones = torch.ones_like(xs)
        second_align_matrix = torch.stack([xs, ys, zeros, zeros - ys, xs, zeros, zeros, zeros, ones], dim=2).view(batch_s, num_s, 3, 3)
        align_matrix = torch.matmul(second_align_matrix, first_align_matrix)

        self.align_matrix_expand = align_matrix.unsqueeze(1).expand(batch_s, nsample, num_s, 3, 3)

    def align(self, in_fts):
        out_fts = torch.matmul(self.align_matrix_expand, in_fts)
        out_fts = out_fts.squeeze(-1).transpose(1, 3).contiguous()
        return out_fts
