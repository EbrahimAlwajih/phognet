import torch
import torch.nn as nn
import torch.nn.functional as F

class PHOGLayer(nn.Module):
    def __init__(self, num_bins=9, levels=1, in_channels=16):
        super(PHOGLayer, self).__init__()
        self.num_bins = int(num_bins)
        self.levels = int(levels)
        self.in_channels = int(in_channels)

        # Depthwise fixed filters
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([[0, 1, 0],
                          [1,-4, 1],
                          [0, 1, 0]], dtype=torch.float32
            ).view(1,1,3,3).repeat(self.in_channels,1,1,1)
        )
        self.register_buffer(
            'sobel_x_kernel',
            torch.tensor([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=torch.float32
            ).view(1,1,3,3).repeat(self.in_channels,1,1,1)
        )
        self.register_buffer(
            'sobel_y_kernel',
            torch.tensor([[-1,-2,-1],
                          [ 0, 0, 0],
                          [ 1, 2, 1]], dtype=torch.float32
            ).view(1,1,3,3).repeat(self.in_channels,1,1,1)
        )
        self.groups = self.in_channels

        # lightweight Python-side cache for cell-id maps (per device,H,W,level)
        self._cell_id_cache = {}

    def _get_cell_id_map(self, device, H, W, level):
        """
        Returns a tensor [1, H, W] of cell ids in [0, Hc*Wc-1] for the given level.
        Cached by (device,H,W,level) to avoid rebuilding every forward.
        """
        key = (device, H, W, level)
        t = self._cell_id_cache.get(key, None)
        if t is not None and t.device == device:
            return t

        cells_side = 2 ** level
        cell_h = H // cells_side
        cell_w = W // cells_side

        Hc = H // cell_h  # == cells_side
        Wc = W // cell_w  # == cells_side

        # row/col -> cell index
        rows = torch.div(torch.arange(H, device=device), cell_h, rounding_mode='floor')  # [H]
        cols = torch.div(torch.arange(W, device=device), cell_w, rounding_mode='floor')  # [W]
        r_id = rows[:, None].expand(H, W)
        c_id = cols[None, :].expand(W, H).t()  # same as expand(H,W) with ij indexing
        cell_id = (r_id * Wc + c_id).to(torch.long).unsqueeze(0)  # [1,H,W], int64

        self._cell_id_cache[key] = cell_id
        return cell_id

    # @torch.no_grad()
    def forward(self, x):
        """
        Input : x [B,C,H,W]
        Output: [B,1,total_cells_across_levels,num_bins]
        """
        with torch.no_grad():
            B, C, H, W = x.shape
            device = x.device
            eps = 1e-8

            import math
            max_level = int(math.log2(min(H, W)))
            levels = min(self.levels, max_level)
            if self.levels > max_level:
                print(f"Adjusting levels from {self.levels} to {max_level} to fit image size {H}x{W}.")

            # Gradients (depthwise)
            lap = F.conv2d(x, self.laplacian_kernel, padding=1, groups=self.groups)
            gx  = F.conv2d(lap, self.sobel_x_kernel, padding=1, groups=self.groups)
            gy  = F.conv2d(lap, self.sobel_y_kernel, padding=1, groups=self.groups)

            mag = torch.sqrt(gx * gx + gy * gy + eps)              # [B,C,H,W], fp32
            ang = torch.atan2(gy, gx) * (180.0 / torch.pi)         # degrees
            ang = ang.remainder(180.0)                              # [0,180)

            # Hard bin indices (centered bins)
            bin_size = 180.0 / float(self.num_bins)
            bins = ((ang + 0.5 * bin_size) / bin_size).floor().to(torch.long) % self.num_bins  # [B,C,H,W]

            # Flatten per-(B*C)
            BC = B * C
            mag_bc  = mag.reshape(BC, H * W)        # [BC, HW]
            bins_bc = bins.reshape(BC, H * W)       # [BC, HW]

            all_hist = []  # will store [B, cells*C, num_bins] for each level

            # Vectorized per-level accumulation with scatter_add_
            for lvl in range(levels):
                cell_id = self._get_cell_id_map(device, H, W, lvl)         # [1,H,W]
                cell_id = cell_id.view(1, -1).expand(BC, -1)                # [BC, HW]
                cells_side = 2 ** lvl
                num_cells = cells_side * cells_side

                # flat index over cells*bins
                idx_flat = cell_id * self.num_bins + bins_bc               # [BC, HW]

                # Allocate histogram and accumulate magnitudes
                hist_flat = mag_bc.new_zeros((BC, num_cells * self.num_bins))  # [BC, cells*bins]
                hist_flat.scatter_add_(1, idx_flat, mag_bc)                    # in-place accumulate

                # Reshape to [BC, cells, bins]
                hist = hist_flat.view(BC, num_cells, self.num_bins)

                # Normalize per-cell (L1 then L2), same as original
                hist = hist / (hist.sum(dim=-1, keepdim=True) + eps)
                hist = F.normalize(hist, p=2, dim=-1)

                # Back to [B, C*cells, bins]
                hist_B = hist.view(B, C, num_cells, self.num_bins).reshape(B, C * num_cells, self.num_bins)
                all_hist.append(hist_B)

            out = torch.cat(all_hist, dim=1).unsqueeze(1)  # [B,1,total_cells,bins]
        return out
