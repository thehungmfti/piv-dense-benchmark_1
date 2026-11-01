"""Tái hiện kiến trúc RAFT-PIV bằng PyTorch (không dùng mã/trọng số của tác giả).

Dựa trên RAFT (Teed & Deng, ECCV 2020) với các điều chỉnh cho ảnh hạt theo tinh thần RAFT-PIV
(Lagemann et al., Nat. Mach. Intell. 2021): ảnh xám 1 kênh chuẩn hoá theo từng ảnh, bộ mã hoá đặc
trưng giảm mẫu ít hơn (mặc định /4 thay vì /8 để giữ thông tin hạt 1–4 px và lớp cắt mỏng), kim tự
tháp tương quan toàn cặp (all-pairs) nhiều mức, ConvGRU lặp, nâng mẫu lồi (convex upsampling) về độ
phân giải pixel. Mọi kích thước đều cấu hình được qua ``RAFTPIVConfig`` — chi tiết có thể khác bản
gốc của tác giả (bản gốc không truy cập được khi xây dựng).

Quy ước đầu ra: flow[:, 0] = u (theo x / cột), flow[:, 1] = v (theo y / hàng), đơn vị px.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RAFTPIVConfig:
    down: int = 4                    # hệ số giảm mẫu của bộ mã hoá (2, 4 hoặc 8)
    enc_channels: tuple = (32, 48, 64)  # kênh các tầng residual
    feat_dim: int = 128              # số chiều đặc trưng cho tương quan
    hidden_dim: int = 96             # trạng thái ẩn GRU
    context_dim: int = 64            # đặc trưng ngữ cảnh
    corr_levels: int = 4             # số mức kim tự tháp tương quan
    corr_radius: int = 4             # bán kính tra cứu (ô lưới đặc trưng)
    motion_dim: int = 96
    iters: int = 12                  # số vòng cập nhật khi huấn luyện
    dropout: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["enc_channels"] = list(self.enc_channels)
        return d


# ---------------------------------------------------------------------------
# Bộ mã hoá
# ---------------------------------------------------------------------------
def _norm(kind: str, c: int):
    if kind == "instance":
        return nn.InstanceNorm2d(c)
    if kind == "group":
        return nn.GroupNorm(num_groups=max(1, c // 8), num_channels=c)
    return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, norm, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1, self.n2 = _norm(norm, cout), _norm(norm, cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride), _norm(norm, cout))

    def forward(self, x):
        y = self.relu(self.n1(self.conv1(x)))
        y = self.relu(self.n2(self.conv2(y)))
        if self.down is not None:
            x = self.down(x)
        return self.relu(x + y)


class Encoder(nn.Module):
    """conv 7×7 (stride 2) → 3 tầng residual; tổng giảm mẫu = ``down``."""

    def __init__(self, out_dim, channels, down, norm="instance", dropout=0.0):
        super().__init__()
        c1, c2, c3 = channels
        s = {2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2)}[down]
        self.conv1 = nn.Conv2d(1, c1, 7, stride=s[0], padding=3)
        self.n1 = _norm(norm, c1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(ResidualBlock(c1, c1, norm), ResidualBlock(c1, c1, norm))
        self.layer2 = nn.Sequential(ResidualBlock(c1, c2, norm, s[1]), ResidualBlock(c2, c2, norm))
        self.layer3 = nn.Sequential(ResidualBlock(c2, c3, norm, s[2]), ResidualBlock(c3, c3, norm))
        self.out = nn.Conv2d(c3, out_dim, 1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.relu(self.n1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.drop(self.out(x))


# ---------------------------------------------------------------------------
# Tương quan toàn cặp + tra cứu
# ---------------------------------------------------------------------------
def coords_grid(b, h, w, device):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack([x, y], 0).float()[None].repeat(b, 1, 1, 1)


def bilinear_sampler(img, coords):
    """img (N,C,H,W), coords (N,h,w,2) theo pixel (x, y)."""
    H, W = img.shape[-2:]
    x, y = coords.split(1, dim=-1)
    x = 2 * x / (W - 1) - 1
    y = 2 * y / (H - 1) - 1
    return F.grid_sample(img, torch.cat([x, y], -1), align_corners=True)


class CorrBlock:
    def __init__(self, f1, f2, levels, radius):
        self.levels, self.radius = levels, radius
        b, d, h, w = f1.shape
        corr = torch.matmul(f1.view(b, d, h * w).transpose(1, 2), f2.view(b, d, h * w)) / d**0.5
        corr = corr.reshape(b * h * w, 1, h, w)
        self.pyramid = [corr]
        for _ in range(levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        b, h, w, _ = coords.shape
        d = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
        delta = torch.stack(torch.meshgrid(d, d, indexing="ij"), -1)  # (2r+1, 2r+1, 2) = (dy, dx)
        delta = delta.flip(-1).view(1, 2 * r + 1, 2 * r + 1, 2)      # → (dx, dy)
        out = []
        for i, corr in enumerate(self.pyramid):
            c = coords.reshape(b * h * w, 1, 1, 2) / 2**i + delta
            out.append(bilinear_sampler(corr, c).view(b, h, w, -1))
        return torch.cat(out, -1).permute(0, 3, 1, 2).contiguous().float()


# ---------------------------------------------------------------------------
# Khối cập nhật
# ---------------------------------------------------------------------------
class MotionEncoder(nn.Module):
    def __init__(self, corr_planes, out_dim):
        super().__init__()
        self.convc1 = nn.Conv2d(corr_planes, 128, 1)
        self.convc2 = nn.Conv2d(128, 96, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(96 + 32, out_dim - 2, 3, padding=1)

    def forward(self, flow, corr):
        c = F.relu(self.convc2(F.relu(self.convc1(corr))))
        f = F.relu(self.convf2(F.relu(self.convf1(flow))))
        return torch.cat([F.relu(self.conv(torch.cat([c, f], 1))), flow], 1)


class SepConvGRU(nn.Module):
    def __init__(self, hidden, inp):
        super().__init__()
        self.z1 = nn.Conv2d(hidden + inp, hidden, (1, 5), padding=(0, 2))
        self.r1 = nn.Conv2d(hidden + inp, hidden, (1, 5), padding=(0, 2))
        self.q1 = nn.Conv2d(hidden + inp, hidden, (1, 5), padding=(0, 2))
        self.z2 = nn.Conv2d(hidden + inp, hidden, (5, 1), padding=(2, 0))
        self.r2 = nn.Conv2d(hidden + inp, hidden, (5, 1), padding=(2, 0))
        self.q2 = nn.Conv2d(hidden + inp, hidden, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        for cz, cr, cq in ((self.z1, self.r1, self.q1), (self.z2, self.r2, self.q2)):
            hx = torch.cat([h, x], 1)
            z, r = torch.sigmoid(cz(hx)), torch.sigmoid(cr(hx))
            q = torch.tanh(cq(torch.cat([r * h, x], 1)))
            h = (1 - z) * h + z * q
        return h


class UpdateBlock(nn.Module):
    def __init__(self, cfg: RAFTPIVConfig):
        super().__init__()
        planes = cfg.corr_levels * (2 * cfg.corr_radius + 1) ** 2
        self.enc = MotionEncoder(planes, cfg.motion_dim)
        self.gru = SepConvGRU(cfg.hidden_dim, cfg.context_dim + cfg.motion_dim)
        self.flow_head = nn.Sequential(nn.Conv2d(cfg.hidden_dim, 128, 3, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 2, 3, padding=1))
        self.mask = nn.Sequential(nn.Conv2d(cfg.hidden_dim, 128, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 9 * cfg.down * cfg.down, 1))

    def forward(self, h, ctx, corr, flow):
        m = self.enc(flow, corr)
        h = self.gru(h, torch.cat([ctx, m], 1))
        return h, 0.25 * self.mask(h), self.flow_head(h)


# ---------------------------------------------------------------------------
# Mạng
# ---------------------------------------------------------------------------
def normalize_pair(i1, i2, eps=1e-6):
    """Chuẩn hoá từng ảnh về trung bình 0, độ lệch chuẩn 1 (bền với thay đổi độ sáng)."""
    def n(x):
        m = x.mean(dim=(-2, -1), keepdim=True)
        s = x.std(dim=(-2, -1), keepdim=True)
        return (x - m) / (s + eps)
    return n(i1), n(i2)


class RAFTPIV(nn.Module):
    def __init__(self, cfg: RAFTPIVConfig | None = None):
        super().__init__()
        self.cfg = cfg or RAFTPIVConfig()
        c = self.cfg
        self.fnet = Encoder(c.feat_dim, c.enc_channels, c.down, "instance", c.dropout)
        self.cnet = Encoder(c.hidden_dim + c.context_dim, c.enc_channels, c.down, "group", c.dropout)
        self.update = UpdateBlock(c)

    def upsample(self, flow, mask):
        """Nâng mẫu lồi: mỗi pixel mịn = tổ hợp lồi của 3×3 lân cận thô."""
        n, _, h, w = flow.shape
        d = self.cfg.down
        mask = torch.softmax(mask.view(n, 1, 9, d, d, h, w), dim=2)
        up = F.unfold(d * flow, [3, 3], padding=1).view(n, 2, 9, 1, 1, h, w)
        up = torch.sum(mask * up, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up.reshape(n, 2, d * h, d * w)

    def forward(self, img1, img2, iters: int | None = None, flow_init=None, test_mode: bool = False):
        """img1, img2: (N,1,H,W), H và W chia hết cho down·2^(levels−1)."""
        c = self.cfg
        iters = iters or c.iters
        img1, img2 = normalize_pair(img1, img2)
        f1, f2 = self.fnet(torch.cat([img1, img2], 0)).chunk(2, 0)
        corr = CorrBlock(f1, f2, c.corr_levels, c.corr_radius)
        h, ctx = torch.split(self.cnet(img1), [c.hidden_dim, c.context_dim], 1)
        h, ctx = torch.tanh(h), torch.relu(ctx)
        n, _, hh, ww = f1.shape
        coords0 = coords_grid(n, hh, ww, img1.device)
        coords1 = coords0.clone()
        if flow_init is not None:
            coords1 = coords1 + flow_init
        preds = []
        for _ in range(iters):
            coords1 = coords1.detach()
            cv = corr(coords1)
            h, mask, delta = self.update(h, ctx, cv, coords1 - coords0)
            coords1 = coords1 + delta
            if test_mode and _ < iters - 1:
                continue
            preds.append(self.upsample(coords1 - coords0, mask))
        return preds[-1] if test_mode else preds


def sequence_loss(preds, gt, valid=None, gamma: float = 0.8, max_flow: float = 64.0):
    """Σ γ^(N−i−1)·|f_i − gt|₁ (trung bình trên điểm hợp lệ) + EPE của dự đoán cuối."""
    n = len(preds)
    mag = torch.sqrt((gt**2).sum(1, keepdim=True))
    v = (mag < max_flow).float()
    if valid is not None:
        v = v * valid
    denom = v.sum().clamp_min(1.0) * 2
    loss = 0.0
    for i, p in enumerate(preds):
        loss = loss + gamma ** (n - i - 1) * ((p - gt).abs() * v).sum() / denom
    epe = (torch.sqrt(((preds[-1] - gt) ** 2).sum(1, keepdim=True)) * v).sum() / v.sum().clamp_min(1.0)
    return loss, epe


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
