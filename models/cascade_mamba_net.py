import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class SharedMLP(nn.Module):

    def __init__(self, in_channels, out_channels, activation='leaky_relu', bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not bn)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-06) if bn else nn.Identity()
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def compute_per_point_geo(xyz, neigh_idx):
    (B, N, _) = xyz.shape
    neighbor_xyz = index_points(xyz, neigh_idx)
    centered = neighbor_xyz - xyz.unsqueeze(2)
    cov = (centered.unsqueeze(-1) * centered.unsqueeze(-2)).mean(dim=2).float()
    (eigvals, eigvecs) = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(-1).clamp(min=1e-08)
    (l1, l2, l3) = eigvals.unbind(-1)
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    scattering = l3 / l1
    normal = eigvecs[..., 0]
    verticality = 1.0 - normal[..., 2].abs()
    geo = torch.stack([linearity, planarity, scattering, verticality, normal[..., 0], normal[..., 1], normal[..., 2]], dim=-1)
    return geo

def compute_per_edge_geo(geo_per_point, neigh_idx):
    (B, N, D_geo) = geo_per_point.shape
    K = neigh_idx.shape[-1]
    geo_center = geo_per_point.unsqueeze(2).expand(B, N, K, D_geo)
    geo_neighbor = index_points(geo_per_point, neigh_idx)
    geo_diff = geo_center - geo_neighbor
    edge_geo = torch.cat([geo_center, geo_neighbor, geo_diff], dim=-1)
    return edge_geo.permute(0, 3, 1, 2)

class GADPE(nn.Module):

    def __init__(self, d_out, num_freqs=4):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_out = d_out
        rpe_dim = 7 * 2 * num_freqs
        self.rpe_mlp = nn.Sequential(SharedMLP(rpe_dim, d_out, activation='relu', bn=True), SharedMLP(d_out, d_out, activation='linear', bn=False))
        geo_dim = 21
        self.geo_mlp = nn.Sequential(SharedMLP(geo_dim, max(d_out // 2, 8), activation='relu', bn=True), SharedMLP(max(d_out // 2, 8), d_out, activation='linear', bn=False))
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def sinusoidal_encoding(self, x):
        encodings = []
        for i in range(self.num_freqs):
            freq = 2.0 ** i * math.pi
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
        return torch.cat(encodings, dim=1)

    def forward(self, xyz, neigh_idx, features=None):
        (B, N, _) = xyz.shape
        K = neigh_idx.shape[-1]
        neighbor_xyz = index_points(xyz, neigh_idx)
        xyz_tile = xyz.unsqueeze(2).expand(B, N, K, 3)
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = torch.sqrt((relative_xyz ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-08))
        f_xyz_raw = torch.cat([relative_xyz, relative_dis], dim=-1)
        f_xyz_raw = f_xyz_raw.permute(0, 3, 1, 2)
        dx = relative_xyz[..., 0]
        dy = relative_xyz[..., 1]
        dz = relative_xyz[..., 2]
        r = relative_dis.squeeze(-1)
        r_xy = (dx ** 2 + dy ** 2).clamp(min=1e-08).sqrt()
        theta = torch.atan2(dy, dx)
        cyl = torch.stack([dx, dy, dz, r, r_xy, theta, dz], dim=-1)
        cyl = cyl.permute(0, 3, 1, 2)
        rpe_feat = self.sinusoidal_encoding(cyl)
        rpe_out = self.rpe_mlp(rpe_feat)
        geo_per_pt = compute_per_point_geo(xyz, neigh_idx)
        edge_geo = compute_per_edge_geo(geo_per_pt, neigh_idx)
        edge_geo = edge_geo.to(rpe_out.dtype)
        geo_out = self.geo_mlp(edge_geo)
        alpha = torch.sigmoid(self.alpha)
        f_pe = alpha * rpe_out + (1.0 - alpha) * geo_out
        return (f_pe, f_xyz_raw)

def _morton_encode(x, y, z, num_bits=16):
    code = torch.zeros_like(x)
    for i in range(num_bits):
        code |= (x >> i & 1) << 3 * i
        code |= (y >> i & 1) << 3 * i + 1
        code |= (z >> i & 1) << 3 * i + 2
    return code

def zorder_serialize(xyz, grid_size):
    xyz_min = xyz.min(dim=1, keepdim=True)[0]
    coords = ((xyz - xyz_min) / grid_size).long()
    (x, y, z) = (coords[:, :, 0], coords[:, :, 1], coords[:, :, 2])
    code = _morton_encode(x, y, z)
    return code.argsort(dim=1)

def _hilbert_encode_3d(X, num_bits=16):
    N = X.shape[0]
    num_dims = 3
    X = X.clone()
    M = 1 << num_bits - 1
    Q = M
    while Q > 1:
        P = Q - 1
        for dim in range(num_dims - 1, -1, -1):
            mask = X[:, dim] & Q != 0
            X[mask, 0] ^= P
            not_mask = ~mask
            t = (X[not_mask, 0] ^ X[not_mask, dim]) & P
            X[not_mask, 0] ^= t
            X[not_mask, dim] ^= t
        Q >>= 1
    for i in range(1, num_dims):
        X[:, i] ^= X[:, i - 1]
    t = torch.zeros(N, dtype=torch.long, device=X.device)
    Q = M
    while Q > 1:
        mask = X[:, num_dims - 1] & Q != 0
        t[mask] ^= Q - 1
        Q >>= 1
    for i in range(num_dims):
        X[:, i] ^= t
    h = torch.zeros(N, dtype=torch.long, device=X.device)
    for s in range(num_bits):
        for d in range(num_dims - 1, -1, -1):
            h = h << 1 | X[:, d] >> s & 1
    return h

def hilbert_serialize(xyz, grid_size):
    (B, N, _) = xyz.shape
    xyz_min = xyz.min(dim=1, keepdim=True)[0]
    coords = ((xyz - xyz_min) / grid_size).long()
    coords_flat = coords.reshape(B * N, 3)
    codes = _hilbert_encode_3d(coords_flat)
    codes = codes.reshape(B, N)
    return codes.argsort(dim=1)

def trans_hilbert_serialize(xyz, grid_size, perm=(1, 2, 0)):
    (B, N, _) = xyz.shape
    xyz_min = xyz.min(dim=1, keepdim=True)[0]
    coords = ((xyz - xyz_min) / grid_size).long()
    coords = coords[:, :, list(perm)]
    coords_flat = coords.reshape(B * N, 3)
    codes = _hilbert_encode_3d(coords_flat).reshape(B, N)
    return codes.argsort(dim=1)

def boustrophedon_serialize(xyz, grid_size=0.04, layer_height=None):
    if layer_height is None:
        layer_height = max(grid_size * 4.0, 0.2)
    (B, N, _) = xyz.shape
    xyz_min = xyz.min(dim=1, keepdim=True)[0]
    xyz_norm = xyz - xyz_min
    z_layer = (xyz_norm[:, :, 2] / layer_height).long()
    x_coord = (xyz_norm[:, :, 0] / grid_size).long()
    x_max = x_coord.max() + 1
    is_odd = z_layer % 2 == 1
    x_key = torch.where(is_odd, x_max - x_coord, x_coord)
    y_coord = (xyz_norm[:, :, 1] / grid_size).long()
    y_max = y_coord.max() + 1
    composite = z_layer * (x_max * y_max) + x_key * y_max + y_coord
    return composite.argsort(dim=1)

class LocalAggregation(nn.Module):

    def __init__(self, d_model, config=None):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU(0.2, inplace=True))
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU(0.2, inplace=True))

    def _edge_aggregate(self, features, neigh_idx):
        neighbor_feat = index_points(features, neigh_idx)
        f_i = features.unsqueeze(2).expand_as(neighbor_feat)
        edge = torch.cat([f_i, neighbor_feat - f_i], dim=-1)
        edge = self.edge_mlp(edge)
        f_max = edge.max(dim=2)[0]
        return f_max

    def forward(self, features, neigh_idx):
        if self.training:
            local_feat = cp.checkpoint(self._edge_aggregate, features, neigh_idx, use_reentrant=False)
        else:
            local_feat = self._edge_aggregate(features, neigh_idx)
        return self.mlp(features + local_feat)

class ChannelSSM(nn.Module):

    def __init__(self, d_model, config):
        super().__init__()
        self.d_model = d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.d_inner = int(config.expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=self.d_conv, groups=self.d_inner, padding=self.d_conv - 1, bias=True)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, xyz=None):
        (B_, N, _) = x.shape
        xz = self.in_proj(x).transpose(1, 2)
        (xx, z) = xz.chunk(2, dim=1)
        xx = xx.flip(1).contiguous()
        z = z.flip(1).contiguous()
        xx = self.act(self.conv1d(xx)[..., :N])
        x_dbl = self.x_proj(xx.transpose(1, 2).reshape(B_ * N, self.d_inner))
        (dt, Bmat, Cmat) = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (self.dt_proj.weight @ dt.t()).reshape(self.d_inner, B_, N).permute(1, 0, 2).contiguous()
        Bmat = Bmat.reshape(B_, N, self.d_state).transpose(1, 2).contiguous()
        Cmat = Cmat.reshape(B_, N, self.d_state).transpose(1, 2).contiguous()
        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(xx, dt, A, Bmat, Cmat, self.D.float(), z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y = y.flip(1)
        return self.out_proj(y.transpose(1, 2))

class MambaBlock(nn.Module):

    def __init__(self, d_model, config):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        mamba_kwargs = dict(d_model=d_model, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand)
        self.mamba_fwd = Mamba(**mamba_kwargs)
        self.mamba_bwd = Mamba(**mamba_kwargs)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.chan = ChannelSSM(d_model, config)
        self.chan_gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, xyz=None):
        normed = self.norm(x)
        out_fwd = self.mamba_fwd(normed)
        out_bwd = self.mamba_bwd(normed.flip(1)).flip(1)
        gate = self.gate(torch.cat([out_fwd, out_bwd], dim=-1))
        out = gate * out_fwd + (1.0 - gate) * out_bwd
        out = out + self.chan_gate * self.chan(normed, xyz)
        return x + out

class SerializationFusion(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.att_proj = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.ReLU(inplace=True), nn.Linear(d_model // 4, 1))

    def forward(self, *features):
        stacked = torch.stack(features, dim=2)
        att = self.att_proj(stacked)
        att = F.softmax(att, dim=2)
        fused = (stacked * att).sum(dim=2)
        return fused

class CascadeStage(nn.Module):

    def __init__(self, d_model, n_blocks, num_classes, config, prev_dim=None):
        super().__init__()
        self.d_model = d_model
        in_feat_dim = 3
        if config.use_color:
            in_feat_dim += 3
        in_feat_dim += 3
        in_feat_dim += config.geo_feature_dim
        self.embed = nn.Sequential(nn.Linear(in_feat_dim, d_model), nn.LayerNorm(d_model), nn.LeakyReLU(0.2, inplace=True), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.LeakyReLU(0.2, inplace=True))
        self.has_prev = prev_dim is not None
        if self.has_prev:
            self.cross_fuse = nn.Sequential(nn.Linear(d_model + prev_dim, d_model), nn.LayerNorm(d_model), nn.LeakyReLU(0.2, inplace=True))
        if self.has_prev:
            self.mscf_prev_proj = nn.Linear(prev_dim, d_model)
            self.mscf_gate_mlp = nn.Linear(d_model + prev_dim, d_model)
            self.mscf_gateA = nn.Parameter(torch.zeros(d_model))
        if self.has_prev:
            self.ugcr_prior_proj = nn.Linear(num_classes, d_model)
            self.ugcr_prior_gate = nn.Parameter(torch.zeros(d_model))
            self.ugcr_log_c = math.log(num_classes)
            self.ugcr_unc_scale = nn.Parameter(torch.zeros(1))
        self.local_agg = LocalAggregation(d_model, config=config)
        self.gadpe = GADPE(d_model, config.gadpe_num_freqs)
        self.mamba_zorder = nn.ModuleList([MambaBlock(d_model, config) for _ in range(n_blocks)])
        self.mamba_hilbert = nn.ModuleList([MambaBlock(d_model, config) for _ in range(n_blocks)])
        self.mamba_snake = nn.ModuleList([MambaBlock(d_model, config) for _ in range(n_blocks)])
        self.serial_fusion = SerializationFusion(d_model)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.seg_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3), nn.Linear(d_model // 2, num_classes))

    def _run_mamba_branch(self, features, sort_idx, mamba_blocks, xyz=None):
        (B, N, D) = features.shape
        idx_exp = sort_idx.unsqueeze(-1).expand(B, N, D)
        ordered = features.gather(1, idx_exp)
        ordered_xyz = None
        if xyz is not None:
            ordered_xyz = xyz.gather(1, sort_idx.unsqueeze(-1).expand(B, N, 3))
        for block in mamba_blocks:
            if self.training:
                ordered = cp.checkpoint(block, ordered, ordered_xyz, use_reentrant=False)
            else:
                ordered = block(ordered, ordered_xyz)
        restore_idx = sort_idx.argsort(dim=1).unsqueeze(-1).expand(B, N, D)
        return ordered.gather(1, restore_idx)

    def _apply_gadpe(self, xyz, neigh_idx, features):
        (f_pe, _) = self.gadpe(xyz, neigh_idx, features)
        f_pe = f_pe.max(dim=3)[0]
        f_pe = f_pe.permute(0, 2, 1)
        return features + f_pe

    def forward(self, xyz, neigh_idx, raw_features, grid_size, prev_features=None, prev_pred=None):
        feat = self.embed(raw_features)
        feat_embed = feat
        if self.has_prev and prev_features is not None:
            base = self.cross_fuse(torch.cat([feat, prev_features], dim=-1))
            g = torch.sigmoid(self.mscf_gate_mlp(torch.cat([feat, prev_features], dim=-1)))
            if prev_pred is not None:
                ent = -(prev_pred * torch.log(prev_pred.clamp_min(1e-06))).sum(dim=-1, keepdim=True)
                ent = ent / self.ugcr_log_c
                g = g * (1.0 + self.ugcr_unc_scale * ent)
            refine = g * self.mscf_prev_proj(prev_features)
            base = base + self.mscf_gateA * refine
            feat = base
        if self.has_prev and prev_pred is not None:
            feat = feat + self.ugcr_prior_gate * self.ugcr_prior_proj(prev_pred)
        feat = self.local_agg(feat, neigh_idx)
        feat = self._apply_gadpe(xyz, neigh_idx, feat)
        sort_z = zorder_serialize(xyz, grid_size)
        sort_h = hilbert_serialize(xyz, grid_size)
        sort_snake = boustrophedon_serialize(xyz, grid_size=grid_size)
        f_z = self._run_mamba_branch(feat, sort_z, self.mamba_zorder, xyz=xyz)
        f_h = self._run_mamba_branch(feat, sort_h, self.mamba_hilbert, xyz=xyz)
        f_snake = self._run_mamba_branch(feat, sort_snake, self.mamba_snake, xyz=xyz)
        feat = self.serial_fusion(f_z, f_h, f_snake)
        feat = feat + self.ffn(feat)
        feat = feat + feat_embed
        pred = self.seg_head(feat)
        return (feat, pred)

def knn_interpolate_precomputed(src_feat, src_xyz, tgt_xyz, upsample_idx):
    k = upsample_idx.shape[-1]
    neighbor_xyz = index_points(src_xyz, upsample_idx)
    diff = tgt_xyz.unsqueeze(2) - neighbor_xyz
    dist = (diff ** 2).sum(dim=-1).clamp(min=1e-08).sqrt()
    weight = 1.0 / dist
    weight = weight / weight.sum(dim=-1, keepdim=True)
    neighbor_feat = index_points(src_feat, upsample_idx)
    tgt_feat = (neighbor_feat * weight.unsqueeze(-1)).sum(dim=2)
    return tgt_feat

class PyramidAggregation(nn.Module):

    def __init__(self, stage_dims, d_final, config=None):
        super().__init__()
        self.n = len(stage_dims)
        self.projs = nn.ModuleList([nn.Linear(d, d_final) for d in stage_dims])
        self.weight_mlp = nn.Sequential(nn.Linear(d_final, d_final // 2), nn.ReLU(inplace=True), nn.Linear(d_final // 2, 1))
        self.norm = nn.LayerNorm(d_final)
        self.pyr_gate = nn.Parameter(torch.zeros(d_final))

    def forward(self, feats_at_final, base_feat):
        if self.training:
            return cp.checkpoint(self._forward, base_feat, *feats_at_final, use_reentrant=False)
        return self._forward(base_feat, *feats_at_final)

    def _forward(self, base_feat, *feats_at_final):
        proj = [self.projs[i](f) for (i, f) in enumerate(feats_at_final)]
        stacked = torch.stack(proj, dim=2)
        w = torch.softmax(self.weight_mlp(stacked), dim=2)
        agg = self.norm((stacked * w).sum(dim=2))
        return base_feat + self.pyr_gate * agg

class CascadeMambaNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config.num_classes
        self.stage_points = config.stage_points
        self.stage_dims = config.stage_dims
        self.stage_blocks = config.stage_blocks
        self.stage_grids = config.stage_grids
        self.stage_weights = config.stage_weights
        self.stages = nn.ModuleList()
        for i in range(4):
            prev_dim = self.stage_dims[i - 1] if i > 0 else None
            stage = CascadeStage(d_model=self.stage_dims[i], n_blocks=self.stage_blocks[i], num_classes=num_classes, config=config, prev_dim=prev_dim)
            self.stages.append(stage)
        self.pyramid = PyramidAggregation(self.stage_dims, self.stage_dims[-1], config)

    def _upsample(self, src_feat, src_xyz, tgt_xyz, idx):
        return knn_interpolate_precomputed(src_feat, src_xyz, tgt_xyz, idx)

    def forward(self, inputs):
        preds = []
        feats = []
        prev_feat = None
        prev_pred = None
        for i in range(4):
            xyz = inputs[f'xyz_s{i}']
            neigh_idx = inputs[f'neigh_idx_s{i}']
            raw_feat = inputs[f'features_s{i}']
            grid = self.stage_grids[i]
            prev_up = None
            prev_pred_up = None
            if prev_feat is not None:
                prev_xyz = inputs[f'xyz_s{i - 1}']
                upsample_idx = inputs[f'upsample_idx_s{i}']
                prev_up = self._upsample(prev_feat, prev_xyz, xyz, upsample_idx)
                if prev_pred is not None:
                    prev_prob = torch.softmax(prev_pred, dim=-1)
                    prev_pred_up = knn_interpolate_precomputed(prev_prob, prev_xyz, xyz, upsample_idx)
            (feat, pred) = self.stages[i](xyz, neigh_idx, raw_feat, grid, prev_up, prev_pred_up)
            preds.append(pred)
            feats.append(feat)
            prev_feat = feat
            prev_pred = pred
        feats_final = []
        for i in range(4):
            f = feats[i]
            for j in range(i + 1, 4):
                f = self._upsample(f, inputs[f'xyz_s{j - 1}'], inputs[f'xyz_s{j}'], inputs[f'upsample_idx_s{j}'])
            feats_final.append(f)
        pyr_feat = self.pyramid(feats_final, feats[3])
        preds[-1] = self.stages[3].seg_head(pyr_feat)
        if self.training:
            return preds
        else:
            return preds[-1]
