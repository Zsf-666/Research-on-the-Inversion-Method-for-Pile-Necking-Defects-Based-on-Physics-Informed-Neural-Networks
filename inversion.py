

import math
import random
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# ============================================================
# 1. 读取 Abaqus 桩顶速度 RPT
# ============================================================

def read_abaqus_velocity_rpt(
    rpt_path,
    time_scale=1.0,
    velocity_scale=1.0,
    velocity_sign=1.0,
):
    """
    读取两列 Abaqus History Output 报告：
        第1列：时间 X
        第2列：桩顶轴向速度 V3_DOWN_POSITIVE

    当前 abaqus.rpt 已确认包含801个有效点，时间范围为0~8 ms。

    time_scale:
        时间已经是s时取1.0；如果是ms则取1e-3。
    velocity_scale:
        速度已经是m/s时取1.0；如果是mm/s则取1e-3。
    velocity_sign:
        RPT正方向与PINN向下为正一致时取1.0；相反时取-1.0。
    """
    path = Path(rpt_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到Abaqus RPT文件：{path}")

    float_pattern = re.compile(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)"
        r"(?:[EeDd][-+]?\d+)?"
    )
    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            tokens = float_pattern.findall(line)
            if len(tokens) < 2:
                continue

            try:
                time_value = float(
                    tokens[0].replace("D", "E").replace("d", "e")
                )
                velocity_value = float(
                    tokens[1].replace("D", "E").replace("d", "e")
                )
            except ValueError:
                continue

            rows.append((time_value, velocity_value))

    if len(rows) < 2:
        raise ValueError(
            "RPT中未识别出足够的两列数值，请检查报告格式。"
        )

    data = np.asarray(rows, dtype=np.float64)
    time = data[:, 0] * time_scale
    velocity = (
        data[:, 1] * velocity_scale * velocity_sign
    )

    finite = np.isfinite(time) & np.isfinite(velocity)
    time = time[finite]
    velocity = velocity[finite]

    order = np.argsort(time)
    time = time[order]
    velocity = velocity[order]

    # 删除可能的重复时刻，仅保留第一次出现的数据。
    time, unique_indices = np.unique(
        time, return_index=True
    )
    velocity = velocity[unique_indices]

    time = time - time[0]
    if len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("处理后的RPT时间序列不是严格递增的。")

    print(
        f"Abaqus RPT loaded: N={len(time)}, "
        f"t=[{time[0]:.3e}, {time[-1]:.3e}] s, "
        f"v=[{velocity.min():.3e}, "
        f"{velocity.max():.3e}] m/s"
    )

    return time, velocity


def read_and_align_top_rpts(
    rpt_paths,
    point_weights=None,
    time_scale=1.0,
    velocity_scale=1.0,
    velocity_sign=1.0,
):
    """
    读取同一桩顶截面多个点位的RPT速度并对齐到公共时间轴。

    所有对齐后的点位速度都会直接进入 PINN 数据损失。这里计算的
    截面加权平均速度仅用于诊断、绘图和与一维网络输出进行比较，
    不会替代各点位原始训练数据。

    参数
    ----
    rpt_paths : sequence of str
        每个桩顶点位对应一个两列RPT文件，第一列为时间，第二列为V3。
        所有文件必须来自同一次Abaqus分析和同一输出变量。
    point_weights : sequence of float or None
        各桩顶点位在数据损失中的权重。None表示所有点位等权。
        若给定，长度必须与rpt_paths相同，且所有权重非负、总和大于0。
    time_scale, velocity_scale, velocity_sign : float
        与read_abaqus_velocity_rpt中的定义相同。

    返回
    ----
    time : ndarray, shape (Nt,)
        对齐后的公共时间轴。
    velocity_mean : ndarray, shape (Nt,)
        M个桩顶点位速度的加权平均，仅用于诊断与绘图。
    velocity_points : ndarray, shape (M, Nt)
        M个桩顶点位在公共时间轴上的原始速度。
    normalized_point_weights : ndarray, shape (M,)
        归一化后的点位权重，所有元素之和为1。
    """
    rpt_paths = [Path(path) for path in rpt_paths]

    if len(rpt_paths) == 0:
        raise ValueError("rpt_paths不能为空，至少需要一个桩顶RPT文件。")

    time_list = []
    velocity_list = []

    for point_index, path in enumerate(rpt_paths, start=1):
        print(
            f"\nLoading pile-head point {point_index}: {path}"
        )
        time_i, velocity_i = read_abaqus_velocity_rpt(
            rpt_path=path,
            time_scale=time_scale,
            velocity_scale=velocity_scale,
            velocity_sign=velocity_sign,
        )
        time_list.append(time_i)
        velocity_list.append(velocity_i)

    # 只保留所有RPT共同覆盖的时间区间，避免在数据范围外进行外推。
    common_start = max(time_i[0] for time_i in time_list)
    common_end = min(time_i[-1] for time_i in time_list)

    if common_end <= common_start:
        raise ValueError("多个RPT文件之间没有共同时间区间。")

    reference_mask = (
        (time_list[0] >= common_start)
        & (time_list[0] <= common_end)
    )
    time = time_list[0][reference_mask]

    if len(time) < 2:
        raise ValueError("多个RPT的公共时间区间内数据点过少。")

    aligned_velocities = []
    interpolation_used = False

    for time_i, velocity_i in zip(time_list, velocity_list):
        same_grid = (
            len(time_i) == len(time)
            and np.allclose(
                time_i,
                time,
                rtol=1e-9,
                atol=1e-12,
            )
        )

        if same_grid:
            velocity_aligned = velocity_i
        else:
            interpolation_used = True
            velocity_aligned = np.interp(
                time,
                time_i,
                velocity_i,
            )

        aligned_velocities.append(velocity_aligned)

    velocity_points = np.vstack(aligned_velocities)
    number_of_points = velocity_points.shape[0]

    if point_weights is None:
        normalized_point_weights = np.full(
            number_of_points,
            1.0 / number_of_points,
            dtype=float,
        )
        weighting_description = "equal weights"
    else:
        normalized_point_weights = np.asarray(
            point_weights,
            dtype=float,
        ).reshape(-1)

        if normalized_point_weights.size != number_of_points:
            raise ValueError(
                "point_weights的数量必须与rpt_paths中的文件数量相同。"
            )
        if np.any(normalized_point_weights < 0.0):
            raise ValueError("point_weights不能包含负数。")

        weight_sum = float(normalized_point_weights.sum())
        if weight_sum <= 0.0:
            raise ValueError("point_weights的总和必须大于0。")

        normalized_point_weights = (
            normalized_point_weights / weight_sum
        )
        weighting_description = "user-defined normalized weights"

    velocity_mean = np.sum(
        normalized_point_weights[:, None] * velocity_points,
        axis=0,
    )

    # 节点离散程度仅用于诊断，不进入PINN损失。
    point_deviation = (
        velocity_points - velocity_mean[None, :]
    )
    rms_point_spread = math.sqrt(
        float(np.mean(
            np.sum(
                normalized_point_weights[:, None]
                * point_deviation**2,
                axis=0,
            )
        ))
    )
    mean_response_rms = math.sqrt(
        float(np.mean(velocity_mean**2))
    )
    relative_rms_spread = (
        rms_point_spread / max(mean_response_rms, 1e-15)
    )

    print("\n===== Pile-head multi-point direct training =====")
    print(f"Number of top points: {number_of_points}")
    print(f"Point weighting: {weighting_description}")
    print(
        "All aligned point histories enter loss_data directly; "
        "the mean curve is diagnostic only."
    )
    print(f"Interpolation used: {interpolation_used}")
    print(
        f"Common time range: "
        f"[{time[0]:.6e}, {time[-1]:.6e}] s"
    )
    print(
        f"RMS point spread: "
        f"{rms_point_spread:.6e} m/s"
    )
    print(
        f"Relative RMS point spread: "
        f"{100.0 * relative_rms_spread:.3f}%"
    )

    if relative_rms_spread > 0.10:
        print(
            "WARNING: 桩顶不同节点的RMS离散度超过平均响应的10%。"
            "这可能表示局部三维振动、偏心加载或弯曲效应较明显，"
            "请检查一维模型适用性。"
        )

    return (
        time,
        velocity_mean,
        velocity_points,
        normalized_point_weights,
    )


# ============================================================
# 2. PINN：三子网 + 无量纲 PDE 残差
# ============================================================

class SubNet(nn.Module):
    """
    子网络：输入 (xi, tau)，输出无量纲位移 w(xi, tau)。
    """

    def __init__(self, in_dim=2, hidden_dim=50, n_hidden=4):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            ])
        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, xi, tau):
        x = torch.cat([xi, tau], dim=1)
        return self.model(x)


def pde_residual_nd(net, xi, tau):
    """
    无量纲 PDE 残差：
        r = w_tau_tau - w_xi_xi
    """
    w = net(xi, tau)

    w_tau = autograd.grad(
        w,
        tau,
        grad_outputs=torch.ones_like(w),
        create_graph=True,
        retain_graph=True,
    )[0]

    w_tautau = autograd.grad(
        w_tau,
        tau,
        grad_outputs=torch.ones_like(w_tau),
        create_graph=True,
        retain_graph=True,
    )[0]

    w_xi = autograd.grad(
        w,
        xi,
        grad_outputs=torch.ones_like(w),
        create_graph=True,
        retain_graph=True,
    )[0]

    w_xixi = autograd.grad(
        w_xi,
        xi,
        grad_outputs=torch.ones_like(w_xi),
        create_graph=True,
        retain_graph=True,
    )[0]

    return w_tautau - w_xixi


def moving_average(values, window=500):
    """对一维序列进行移动平均。"""
    values = np.asarray(values, dtype=float)

    if window <= 1 or len(values) < window:
        return values.copy()

    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")
def set_random_seed(seed=2026):
    """
    固定网络初始化、配点采样和数据抽样的随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 尽量保证GPU计算具有可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

"""三子网、突变截面桩模型的独立残差分布检查。

用法：将本文件放在主程序同目录并导入 check_residual_distributions，
或者将本文件全部内容粘贴到主程序 def main(): 之前。
调用必须在 main 内、训练结束后，不能放在 torch.inference_mode() 中。
本函数不更新网络、损失权重、配点或优化器，不消耗随机数。
"""


def check_residual_distributions(
    subnetworks, get_parameters, pde_residual, top_load,
    impact_data, fixed_points, L, T0, tau_max,
    out_dir="residual_distribution_checks",
    tag="final", n_z=240, n_tau=801, n_curve=8001,
    batch_size=512, show=True,
):
    """所有残差为未乘损失权重的无量纲残差，支持CPU/CUDA与float32/64。

    get_parameters: 原 get_alpha_xi1_xi2。
    pde_residual: 原 pde_residual_nd(net, xi, tau)。
    top_load: 原 P_star_tau(tau, P0, Tp)。
    impact_data: 当前单次加载的数据字典（不是桩顶观测速度数组）。
    fixed_points: 当前训练配点字典，用于同一模型状态下的训练/检查对比。
    PDE检查网格固定在物理空间；在每个位置选择当前所在子域的网络。
    恰好落在两处界面上的PDE网格点留空，界面另用独立条件检查。
    """
    import json
    from datetime import datetime
    from pathlib import Path
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FixedLocator

    if len(subnetworks) != 3:
        raise ValueError("此函数用于三个子网的突变截面模型。")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("请在 torch.inference_mode() 外调用残差检查。")
    if min(n_z, n_tau, n_curve) < 2 or batch_size < 1:
        raise ValueError("网格每个方向至少2点，batch_size至少1。")
    if L <= 0 or T0 <= 0 or tau_max <= 0:
        raise ValueError("L、T0和tau_max必须为正数。")
    nets = list(subnetworks)
    reference = next(nets[0].parameters())
    device, dtype = reference.device, reference.dtype

    def array(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().astype(float, copy=True).reshape(-1)
        return np.asarray(value, dtype=float).reshape(-1).copy()

    def tensor(value):
        return torch.as_tensor(value, dtype=dtype, device=device).reshape(-1, 1)

    def partial(output, coordinate):
        if not output.requires_grad:
            return torch.zeros_like(coordinate)
        result = torch.autograd.grad(
            output, coordinate, torch.ones_like(output),
            create_graph=False, allow_unused=True,
        )[0]
        return torch.zeros_like(coordinate) if result is None else result

    def pde_values(net, xi, tau):
        xi, tau = array(xi), array(tau)
        if xi.size != tau.size:
            raise ValueError("空间和时间坐标必须成对，长度必须相同。")
        values = []
        for start in range(0, xi.size, batch_size):
            end = start + batch_size
            x = tensor(xi[start:end]).clone().requires_grad_(True)
            t = tensor(tau[start:end]).clone().requires_grad_(True)
            values.append(array(pde_residual(net, x, t)))
        return np.concatenate(values) if values else np.empty(0)

    def trace(net, xi_value, tau):
        """界面/桩顶处的位移与空间导数；不对网络参数累计grad。"""
        tau = array(tau)
        w_values, wx_values = [], []
        for start in range(0, tau.size, batch_size):
            t = tensor(tau[start:start + batch_size])
            x = torch.full_like(t, xi_value, requires_grad=True)
            w = net(x, t)
            wx = partial(w, x)
            w_values.append(array(w))
            wx_values.append(array(wx))
        if not w_values:
            return np.empty(0), np.empty(0)
        return np.concatenate(w_values), np.concatenate(wx_values)

    def stats(values):
        values = array(values)
        if values.size == 0:
            return {"count": 0, "rms": None, "p95_abs": None, "max_abs": None}
        if not np.isfinite(values).all():
            raise FloatingPointError("残差出现NaN/Inf，请先检查模型数值状态。")
        return {
            "count": int(values.size),
            "rms": float(np.sqrt(np.mean(values ** 2))),
            "p95_abs": float(np.quantile(np.abs(values), 0.95)),
            "max_abs": float(np.max(np.abs(values))),
        }

    def interface_values(index, tau, alpha, xi1, xi2):
        xi_value = xi1 if index == 1 else xi2
        wl, dl = trace(nets[index - 1], xi_value, tau)
        wr, dr = trace(nets[index], xi_value, tau)
        ru = wl - wr
        rn = dl - alpha * dr if index == 1 else alpha * dl - dr
        return ru, rn

    def load_values(tau):
        with torch.no_grad():
            return array(top_load(tensor(tau), impact_data["P0"], impact_data["Tp"]))

    # 只暂时切换eval模式；完整恢复各层原先的状态。
    modes = [(module, module.training) for net in nets for module in net.modules()]
    try:
        for net in nets:
            net.eval()
        with torch.enable_grad():
            alpha, xi1, xi2 = [float(array(v)[0]) for v in get_parameters()[:3]]
            if not (0 < xi1 < xi2 < 1) or not np.isfinite(alpha):
                raise ValueError("当前界面参数必须满足 0 < xi1 < xi2 < 1。")
            bounds = (0.0, xi1, xi2, 1.0)
            # 固定全域空间中点，不随当前xi1、xi2移动；时间包含0及末时刻。
            xi_axis = (np.arange(n_z, dtype=float) + 0.5) / n_z
            tau_axis = np.linspace(0.0, tau_max, n_tau)
            xx, tt = np.meshgrid(xi_axis, tau_axis, indexing="ij")
            residual_grid = np.full(xx.shape, np.nan)
            tolerance = 4 * torch.finfo(dtype).eps
            at_interface = ((np.abs(xx - xi1) <= tolerance)
                            | (np.abs(xx - xi2) <= tolerance))
            records, raw = {}, {}
            train_space, train_tau, train_domain = [], [], []
            pde_check_mse, pde_train_mse = [], []

            print("\n[Residual check] PDE grid and current training points...")
            for j, net in enumerate(nets, start=1):
                selected = ((xx > bounds[j - 1]) & (xx < bounds[j]) & ~at_interface)
                if not selected.any():
                    raise ValueError("某子域未被PDE检查网格覆盖，请增大n_z。")
                values = pde_values(net, xx[selected], tt[selected])
                residual_grid[selected] = values
                s = array(fixed_points[f"s_f{j}"])
                t = array(fixed_points[f"tau_f{j}"])
                x = bounds[j - 1] + (bounds[j] - bounds[j - 1]) * s
                trained = pde_values(net, x, t)
                peak = int(np.argmax(np.abs(values)))
                records[f"pde_domain_{j}"] = {
                    "check": stats(values), "train": stats(trained),
                    "peak_z_m": float(xx[selected][peak] * L),
                    "peak_time_ms": float(tt[selected][peak] * T0 * 1000),
                }
                pde_check_mse.append(float(np.mean(values ** 2)))
                pde_train_mse.append(float(np.mean(trained ** 2)))
                train_space.append(x)
                train_tau.append(t)
                train_domain.append(np.full(x.shape, j))
                raw[f"pde_train_residual_{j}"] = trained
            raw.update(
                xi=xi_axis, tau=tau_axis, pde_signed=residual_grid,
                pde_train_xi=np.concatenate(train_space),
                pde_train_tau=np.concatenate(train_tau),
                pde_train_domain=np.concatenate(train_domain),
            )

            print("[Residual check] Separate interface and pile-top traces...")
            tau_curve = np.linspace(0.0, tau_max, n_curve)
            raw["tau_curve"] = tau_curve
            for index in (1, 2):
                key = f"tau_interface{index}"
                tau_train = array(fixed_points[key] if key in fixed_points
                                  else fixed_points["tau_interface"])
                ru, rn = interface_values(index, tau_curve, alpha, xi1, xi2)
                ru_train, rn_train = interface_values(index, tau_train, alpha, xi1, xi2)
                raw[f"if{index}_tau_train"] = tau_train
                for kind, check, trained in (("u", ru, ru_train), ("N", rn, rn_train)):
                    name = f"if{index}_{kind}"
                    records[name] = {
                        "check": stats(check), "train": stats(trained),
                        "peak_time_ms": float(tau_curve[np.argmax(np.abs(check))] * T0 * 1000),
                    }
                    raw[name] = check
                    raw[name + "_train"] = trained
            impact_index = int(impact_data.get("impact_index", 0))
            tau_top_train = array(fixed_points["tau_top_list"][impact_index])
            _, wx_top = trace(nets[0], 0.0, tau_curve)
            _, wx_train = trace(nets[0], 0.0, tau_top_train)
            pressure = load_values(tau_curve)
            top_residual = wx_top + pressure
            top_train = wx_train + load_values(tau_top_train)
            records["bc_top"] = {
                "check": stats(top_residual), "train": stats(top_train),
                "peak_time_ms": float(tau_curve[np.argmax(np.abs(top_residual))] * T0 * 1000),
            }
            raw.update(top_w_xi=wx_top, top_load=pressure,
                       top_residual=top_residual, top_tau_train=tau_top_train,
                       top_train=top_train)
    finally:
        for module, training in modes:
            module.training = training

    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(tag)) or "final"
    folder = Path(out_dir) / (safe_tag + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True, exist_ok=False)
    summary = {
        "alpha": alpha, "xi1": xi1, "xi2": xi2, "L": float(L), "T0": float(T0),
        "tau_max": float(tau_max), "n_z": int(n_z), "n_tau": int(n_tau),
        "n_curve": int(n_curve),
        "pde_check_equal_domain_mse": float(np.mean(pde_check_mse)),
        "pde_train_equal_domain_mse": float(np.mean(pde_train_mse)),
        "records": records,
        "notes": [
            "Residuals are dimensionless and unweighted; all scores use the same model snapshot.",
            "Check grids do not train the model or choose adaptive points.",
            "Check and training sampling distributions differ; a ratio is not a proof of overfitting.",
            "PDE points exactly at an interface are masked; interface conditions are checked separately.",
            "Pile-top BC residual is not the observed pile-top velocity data residual.",
            "Finite-grid maxima may miss narrower peaks; verify suspicious regions on a finer grid.",
        ],
    }
    (folder / "residual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(folder / "residual_arrays.npz", **raw)

    times_ms = tau_axis * T0 * 1000
    curve_ms = tau_curve * T0 * 1000
    magnitude = np.abs(residual_grid)
    largest = float(np.nanmax(magnitude))
    upper = max(largest, 1e-12)
    lower = max(upper * 1e-6, 1e-14)
    if lower >= upper:
        lower = upper * 0.01
    fig_pde, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True,
                                 constrained_layout=True)
    # 空间中点及完整时间网格的边缘，避免imshow半个单元错位。
    tau_edges = np.r_[tau_axis[0], 0.5 * (tau_axis[:-1] + tau_axis[1:]), tau_axis[-1]]
    mesh = axes[0].pcolormesh(
        tau_edges * T0 * 1000, np.linspace(0, L, n_z + 1),
        np.ma.masked_invalid(np.maximum(magnitude, lower)),
        cmap="magma", norm=LogNorm(vmin=lower, vmax=upper), shading="flat")
    fig_pde.colorbar(mesh, ax=axes[0], label="Unweighted |PDE residual|")
    axes[0].set_title("Independent PDE check grid")
    for j in (1, 2, 3):
        selected = raw["pde_train_domain"] == j
        # 显示全部点，以免抽稀产生虚假的覆盖空洞。
        axes[1].scatter(raw["pde_train_tau"][selected] * T0 * 1000,
                        raw["pde_train_xi"][selected] * L, s=3, alpha=0.4,
                        label=f"Domain {j}")
    axes[1].set_title("Current PDE training points (all)")
    axes[1].legend(loc="upper right", markerscale=2)
    for ax in axes:
        ax.axhline(xi1 * L, color="cyan", linestyle="--", linewidth=1)
        ax.axhline(xi2 * L, color="cyan", linestyle="--", linewidth=1)
        ax.set_xlabel("Time (ms)")
        ax.set_xlim(0, times_ms[-1])
        ax.set_ylim(L, 0)
    axes[0].set_ylabel("Depth z (m)")
    fig_pde.suptitle(f"PDE residual and sampling | alpha={alpha:.5f}")

    def residual_plot(ax, check, train, train_times, title):
        ax.plot(curve_ms, check, color="tab:blue", linewidth=1, label="Independent time grid")
        ax.scatter(train_times * T0 * 1000, train, s=3, color="tab:orange", alpha=0.4,
                   label="Current training points", rasterized=True)
        scale = max(float(np.max(np.abs(check))), float(np.max(np.abs(train))), 1e-12)
        threshold = max(scale * 1e-3, 1e-14)
        ax.set_yscale("symlog", linthresh=threshold)
        # 在线性核心区域只标0，避免0和相邻小量级刻度挤在一起。
        powers = 10.0 ** np.arange(np.ceil(np.log10(10 * threshold)),
                                   np.floor(np.log10(scale)) + 1)
        ax.yaxis.set_major_locator(FixedLocator(np.r_[-powers[::-1], 0.0, powers]))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Signed residual (dimensionless, symlog)")
        ax.set_xlim(0, curve_ms[-1])
        ax.grid(True, alpha=0.25)

    fig_if, if_axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, index in enumerate((1, 2)):
        for column, kind in enumerate(("u", "N")):
            name = f"if{index}_{kind}"
            title = f"Interface {index}: " + ("displacement continuity" if kind == "u" else "axial-force continuity")
            residual_plot(if_axes[row, column], raw[name], raw[name + "_train"],
                          raw[f"if{index}_tau_train"], title)
            peak = records[name]["peak_time_ms"]
            if_axes[row, column].axvline(peak, color="red", linestyle=":", linewidth=0.9)
    if_axes[0, 0].legend(fontsize=8)
    fig_if.suptitle(f"Separate interface residuals | z1={xi1 * L:.4f} m, z2={xi2 * L:.4f} m")

    fig_top, top_axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True)
    top_axes[0].plot(curve_ms, wx_top, label="PINN w_xi at pile top")
    top_axes[0].plot(curve_ms, -pressure, "--", label="Required -p*(t)")
    top_axes[0].set_ylabel("Dimensionless spatial derivative")
    top_axes[0].set_title("Pile-top boundary target and prediction")
    top_axes[0].legend()
    top_axes[0].grid(True, alpha=0.25)
    top_axes[0].set_xlim(0, curve_ms[-1])
    top_axes[0].set_xlabel("Time (ms)")
    residual_plot(top_axes[1], top_residual, top_train, tau_top_train,
                  "Pile-top BC residual: w_xi + p*(t) (not velocity-data error)")
    top_axes[1].axvline(records["bc_top"]["peak_time_ms"], color="red", linestyle=":")
    for ax in top_axes:
        ax.axvline(impact_data["Tp"] * 1000, color="gray", linestyle="--", linewidth=0.8)
    figures = [fig_pde, fig_if, fig_top]
    for figure, name in zip(figures, ["01_pde_residual.png", "02_interface_residuals.png", "03_top_bc_residual.png"]):
        figure.savefig(folder / name, dpi=180, facecolor="white")

    print("\nResidual component         check RMS    train RMS    check P95    check MAX")
    for name, item in records.items():
        check, train = item["check"], item["train"]
        print(f"{name:<26s} {check['rms']:10.3e}   {train['rms']:10.3e}   "
              f"{check['p95_abs']:10.3e}   {check['max_abs']:10.3e}")
        location = f"t={item['peak_time_ms']:.4f} ms"
        if "peak_z_m" in item:
            location += f", z={item['peak_z_m']:.4f} m"
        print("  check-grid peak: " + location)
    print("Saved residual checks to:", folder.resolve())
    if show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)
    return {"output_dir": str(folder.resolve()), "summary": summary, "arrays": raw}

# ============================================================
# 3. 主程序
# ============================================================

def main():
    set_random_seed(2026)
    torch.set_default_dtype(torch.float32)

    # --------------------------------------------------------
    # 3.1 物理与几何参数
    # --------------------------------------------------------
    L = 10.0
    D0 = 0.8
    A0 = math.pi * D0**2 / 4.0

    # Abaqus模型中的缺陷真值（仅用于合成算例误差评价）
    z1_true = 4.0
    z2_true = 5.0
    # alpha is the cross-sectional AREA ratio A_defect / A0.
    alpha_true = 0.6

    xi1_true = z1_true / L
    xi2_true = z2_true / L
    xi_c_true = 0.5 * (xi1_true + xi2_true)
    lam_true = xi2_true - xi1_true

    E = 3e10
    rho = 2500.0
    c = math.sqrt(E / rho)

    T_max = 0.008

    # --------------------------------------------------------
    # 3.2 Abaqus RPT与桩顶锤击设置
    # --------------------------------------------------------
    # 每个桩顶点位单独导出一个两列RPT：时间X、轴向速度V3。
    # 目前只有一个RPT时可以保持下面的单文件列表，程序等价于原版本。
    # 有更多点位后，只需继续在列表中增加文件路径。
    top_rpt_paths = [
        r"/kaggle/input/datasets/fhgsswrg/youxian666/10m0.6.rpt",
        #r"/kaggle/input/datasets/zhangshifan/10m0-8/V3-5-9.rpt",
    ]

    # 所有桩顶点位都会直接进入数据损失。
    # None：所有RPT点位等权；例如4个点位时每个权重均为1/4。
    # 若已知各点位代表的控制面积，可改为：
    # top_point_weights = [A1, A2, A3, A4]
    # 程序会自动归一化，且每个正权重点位仍直接参与训练。
    top_point_weights = None

    # Abaqus载荷类型为“压力”：750 N/m^2，而不是750 N总力。
    pressure0 = 750.0
    Tp = 1.0e-3

    # 将均布压力换算成桩顶合力，供现有轴力边界形式使用：
    #     F(t) = pressure(t) * A0
    P0 = pressure0 * A0

    # 当前RPT已按s和m/s导出，且表头说明向下为正。
    time_scale = 1.0
    velocity_scale = 1.0
    velocity_sign = 1.0

    n_impacts = 1
    P0_ref = P0

    print(f"Wave speed c ≈ {c:.2f} m/s")

    # --------------------------------------------------------
    # 3.3 读取并对齐多个桩顶点位
    #     原始点位速度全部用于训练；平均速度仅用于诊断和绘图。
    # --------------------------------------------------------
    (
        t_np,
        v_top_np,
        v_top_points_np,
        normalized_point_weights_np,
    ) = read_and_align_top_rpts(
        rpt_paths=top_rpt_paths,
        point_weights=top_point_weights,
        time_scale=time_scale,
        velocity_scale=velocity_scale,
        velocity_sign=velocity_sign,
    )

    time_mask = (t_np >= 0.0) & (t_np <= T_max)
    t_np = t_np[time_mask]
    v_top_np = v_top_np[time_mask]
    v_top_points_np = v_top_points_np[:, time_mask]

    if len(t_np) < 10:
        raise ValueError("0~8 ms范围内的Abaqus数据点过少。")

    dt_values = np.diff(t_np)
    print(
        f"Abaqus data used: N={len(t_np)}, "
        f"top_points={v_top_points_np.shape[0]}, "
        f"direct_training_pairs="
        f"{v_top_points_np.shape[0] * len(t_np)}, "
        f"dt_median={np.median(dt_values):.3e} s, "
        f"pressure0={pressure0:.1f} Pa, "
        f"resultant_force0={P0:.3f} N, "
        f"Tp={Tp*1e3:.1f} ms"
    )

    # --------------------------------------------------------
    # 3.4 无量纲化
    # --------------------------------------------------------
    T0 = L / c
    U0 = L * P0_ref / (E * A0)
    v_scale = U0 * c / L

    tau_np = t_np / T0
    tau_max = float(tau_np[-1])

    print(
        f"T0={T0:.4e} s, tau_max={tau_max:.3f}, "
        f"U0={U0:.4e} m, v_scale={v_scale:.4e} m/s"
    )
    print(
        f"True parameters: alpha={alpha_true:.3f}, "
        f"xi1={xi1_true:.3f}, xi2={xi2_true:.3f}, "
        f"xi_c={xi_c_true:.3f}, lam={lam_true:.3f}"
    )

    # --------------------------------------------------------
    # 3.5 转为 Torch 数据
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    impact_data_list = [{
        "impact_index": 0,
        "name": "Abaqus",
        "P0": P0,
        "Tp": Tp,
        "tau": torch.tensor(
            tau_np,
            dtype=torch.float32,
            device=device,
        ).view(-1, 1),
        # shape = (M, Nt)。所有M个桩顶点位的原始速度直接进入loss_data。
        "v_star_points": torch.tensor(
            v_top_points_np / v_scale,
            dtype=torch.float32,
            device=device,
        ),
        # shape = (M, 1)。默认等权，也可由节点控制面积给定。
        "point_weights": torch.tensor(
            normalized_point_weights_np,
            dtype=torch.float32,
            device=device,
        ).view(-1, 1),
        # 以下平均曲线只用于诊断和画图，不作为训练标签替代原始点位。
        "v_top_np": v_top_np,
        "v_top_points_np": v_top_points_np,
    }]

    # --------------------------------------------------------
    # 3.6 三子网
    # --------------------------------------------------------
    # subnetworks[0]：上部健康段网络
    # subnetworks[1]：缺陷段网络
    # subnetworks[2]：下部健康段网络
    # 三个网络均以 (xi, tau) 为输入。
    subnetworks = nn.ModuleList([
        SubNet(in_dim=2).to(device),
        SubNet(in_dim=2).to(device),
        SubNet(in_dim=2).to(device),
    ])

    # --------------------------------------------------------
    # 3.7 待反演缺陷参数
    # --------------------------------------------------------
    logit_alpha = nn.Parameter(
        torch.tensor(-2.3, dtype=torch.float32, device=device)
    )
    theta_c = nn.Parameter(
        torch.tensor(0.0, dtype=torch.float32, device=device)
    )
    theta_l = nn.Parameter(
        torch.tensor(0.0, dtype=torch.float32, device=device)
    )

    def get_alpha_xi1_xi2():
        """
        保留原代码中的参数范围：
            alpha_hat in (0.40, 0.95)
            lam_hat   in (0.05, 0.20)
            xi_c_hat  in (0.30, 0.50)
        """
        alpha_hat = 0.4 + 0.55 * torch.sigmoid(logit_alpha)
        lam_hat = 0.05 + 0.15 * torch.sigmoid(theta_l)
        xi_c_hat = 0.2 + 0.4 * torch.sigmoid(theta_c)

        xi1_hat = xi_c_hat - 0.5 * lam_hat
        xi2_hat = xi_c_hat + 0.5 * lam_hat

        xi1_hat = torch.clamp(xi1_hat, 0.02, 0.98)
        xi2_hat = torch.clamp(xi2_hat, 0.02, 0.98)
        xi2_hat = torch.maximum(xi2_hat, xi1_hat + 1e-3)

        return alpha_hat, xi1_hat, xi2_hat, xi_c_hat, lam_hat

    def detached_inverse_parameter_values():
        """Read the current inverse parameters without building a graph."""
        with torch.no_grad():
            (
                alpha_hat,
                xi1_hat,
                xi2_hat,
                xi_c_hat,
                lam_hat,
            ) = get_alpha_xi1_xi2()

            return {
                "logit_alpha": float(logit_alpha.detach().item()),
                "theta_c": float(theta_c.detach().item()),
                "theta_l": float(theta_l.detach().item()),
                "alpha_hat": float(alpha_hat.detach().item()),
                "xi1_hat": float(xi1_hat.detach().item()),
                "xi2_hat": float(xi2_hat.detach().item()),
                "xi_c_hat": float(xi_c_hat.detach().item()),
                "lam_hat": float(lam_hat.detach().item()),
            }

    def empty_inverse_gradient_info():
        """Return placeholders when an exact gradient is not evaluated."""
        nan = float("nan")
        return {
            "grad_logit_alpha": nan,
            "grad_theta_c": nan,
            "grad_theta_l": nan,
            "map_dalpha_dlogit": nan,
            "map_dxic_dtheta_c": nan,
            "map_dlam_dtheta_l": nan,
            "grad_alpha_phys": nan,
            "grad_xi_c_phys": nan,
            "grad_lam_phys": nan,
            "grad_xi1_phys": nan,
            "grad_xi2_phys": nan,
        }

    def collect_inverse_gradient_info():
        """
        Collect raw and physical inverse-parameter gradients.

        This function must be called after backward() and before gradients are
        cleared.  alpha_hat is not a leaf tensor, so the optimizer gradient is
        read from logit_alpha.grad and then converted by the chain rule.
        """

        def grad_value(parameter):
            if parameter.grad is None:
                return float("nan")
            return float(parameter.grad.detach().item())

        grad_logit_alpha = grad_value(logit_alpha)
        grad_theta_c = grad_value(theta_c)
        grad_theta_l = grad_value(theta_l)

        with torch.no_grad():
            sigmoid_alpha = torch.sigmoid(logit_alpha)
            sigmoid_c = torch.sigmoid(theta_c)
            sigmoid_l = torch.sigmoid(theta_l)

            # These coefficients must stay consistent with
            # get_alpha_xi1_xi2().
            d_alpha_d_logit = float(
                (
                    0.55
                    * sigmoid_alpha
                    * (1.0 - sigmoid_alpha)
                ).item()
            )
            d_xic_d_theta = float(
                (
                    0.40
                    * sigmoid_c
                    * (1.0 - sigmoid_c)
                ).item()
            )
            d_lam_d_theta = float(
                (
                    0.15
                    * sigmoid_l
                    * (1.0 - sigmoid_l)
                ).item()
            )

        tiny = 1.0e-14

        def physical_gradient(raw_gradient, mapping_slope):
            if not math.isfinite(raw_gradient):
                return float("nan")
            if abs(mapping_slope) <= tiny:
                return float("nan")
            return raw_gradient / mapping_slope

        grad_alpha_phys = physical_gradient(
            grad_logit_alpha,
            d_alpha_d_logit,
        )
        grad_xi_c_phys = physical_gradient(
            grad_theta_c,
            d_xic_d_theta,
        )
        grad_lam_phys = physical_gradient(
            grad_theta_l,
            d_lam_d_theta,
        )

        if (
            math.isfinite(grad_xi_c_phys)
            and math.isfinite(grad_lam_phys)
        ):
            # xi1 = xi_c - lambda/2; xi2 = xi_c + lambda/2.
            grad_xi1_phys = (
                0.5 * grad_xi_c_phys - grad_lam_phys
            )
            grad_xi2_phys = (
                0.5 * grad_xi_c_phys + grad_lam_phys
            )
        else:
            grad_xi1_phys = float("nan")
            grad_xi2_phys = float("nan")

        return {
            "grad_logit_alpha": grad_logit_alpha,
            "grad_theta_c": grad_theta_c,
            "grad_theta_l": grad_theta_l,
            "map_dalpha_dlogit": d_alpha_d_logit,
            "map_dxic_dtheta_c": d_xic_d_theta,
            "map_dlam_dtheta_l": d_lam_d_theta,
            "grad_alpha_phys": grad_alpha_phys,
            "grad_xi_c_phys": grad_xi_c_phys,
            "grad_lam_phys": grad_lam_phys,
            "grad_xi1_phys": grad_xi1_phys,
            "grad_xi2_phys": grad_xi2_phys,
        }

    def inverse_update_info(values_before, values_after):
        """Record the actual accepted parameter changes."""
        return {
            "delta_alpha_hat": (
                values_after["alpha_hat"]
                - values_before["alpha_hat"]
            ),
            "delta_xi1_hat": (
                values_after["xi1_hat"]
                - values_before["xi1_hat"]
            ),
            "delta_xi2_hat": (
                values_after["xi2_hat"]
                - values_before["xi2_hat"]
            ),
            "delta_xi_c_hat": (
                values_after["xi_c_hat"]
                - values_before["xi_c_hat"]
            ),
            "delta_lam_hat": (
                values_after["lam_hat"]
                - values_before["lam_hat"]
            ),
            "delta_logit_alpha": (
                values_after["logit_alpha"]
                - values_before["logit_alpha"]
            ),
            "delta_theta_c": (
                values_after["theta_c"]
                - values_before["theta_c"]
            ),
            "delta_theta_l": (
                values_after["theta_l"]
                - values_before["theta_l"]
            ),
        }

    # 神经网络参数
    network_params = list(subnetworks.parameters())
    
    # 待反演参数
    inverse_params = [logit_alpha, theta_c, theta_l]
    
    # 全部参数，供 L-BFGS 使用
    params = network_params + inverse_params

    mse = nn.MSELoss()

    # --------------------------------------------------------
    # 3.8 基于梯度范数的有界自适应损失权重
    # --------------------------------------------------------
    # 保留已经验证过的固定权重作为基础尺度，再乘以训练过程中自动更新的
    # 无量纲修正系数：
    #     lambda_i(epoch) = base_weight_i * multiplier_i(epoch)
    #
    # 修正系数只在 Adam 阶段、预热结束后定期更新；进入 L-BFGS 前冻结。
    # 梯度范数仅相对于三个子网的最后一个线性层计算，不包括
    # logit_alpha、theta_c 和 theta_l，避免数据损失对这些参数没有直接梯度时
    # 产生不合理的权重放大。
    loss_component_keys = (
        "pde",
        "data",
        "bc_top",
        "bc_bottom",
        "ic",
        "if_u",
        "if_N",
    )

    base_loss_weights = {
        "pde": 2.0,
        "data": 10.0,
        "bc_top": 1.0,
        "bc_bottom": 1.0,
        "ic": 1.0,
        "if_u": 15.0,
        "if_N": 15.0,
    }

    # 保留旧变量名，便于后续代码和论文参数表识别；这些变量现在表示基础权重。
    w_pde = base_loss_weights["pde"]
    w_data = base_loss_weights["data"]
    w_bc_top = base_loss_weights["bc_top"]
    w_bc_bottom = base_loss_weights["bc_bottom"]
    w_ic = base_loss_weights["ic"]
    w_if_u = base_loss_weights["if_u"]
    w_if_N = base_loss_weights["if_N"]

    use_gradient_adaptive_weights = True
    gradient_weight_warmup_epochs = 3000
    gradient_weight_update_interval = 1000
    gradient_weight_alpha = 0.5
    gradient_weight_ema = 0.90
    gradient_weight_eps = 1.0e-12
    gradient_weight_max_change = 1.50

    # data 是唯一观测来源，if_N 是 alpha 的关键识别通道，因此二者采用
    # 较高的下限，防止自适应算法将其权重降得过低。
    adaptive_multiplier_bounds = {
        "pde": (0.10, 10.00),
        "data": (0.10, 100.00),
        "bc_top": (0.10, 10.00),
        "bc_bottom": (0.10, 10.00),
        "ic": (0.10, 10.00),
        "if_u": (0.10, 10.00),
        "if_N": (0.10, 10.00),
    }

    adaptive_weight_multipliers = {
        key: 1.0 for key in loss_component_keys
    }
    gradient_reference_losses = {}
    latest_raw_gradient_norms = {
        key: float("nan") for key in loss_component_keys
    }
    latest_effective_gradient_norms = {
        key: float("nan") for key in loss_component_keys
    }
    adaptive_weights_frozen = False

    # 使用每个子网最后一个线性层作为 GradNorm 的参考层。相比对全部网络
    # 参数计算七次梯度，这种做法显著降低计算量，同时仍能反映各损失项
    # 对三个子网输出的训练强度。
    gradient_balance_parameters = []
    for network in subnetworks:
        linear_layers = [
            layer
            for layer in network.model
            if isinstance(layer, nn.Linear)
        ]
        if not linear_layers:
            raise RuntimeError("SubNet 中未找到线性层，无法计算梯度范数。")

        last_linear = linear_layers[-1]
        # 只使用 weight。对于 PDE、速度和空间导数残差，输出层 bias 的
        # 梯度经常严格为零，将其纳入范数只会增加无效分量。
        gradient_balance_parameters.append(last_linear.weight)

    def current_effective_loss_weights():
        return {
            key: (
                base_loss_weights[key]
                * adaptive_weight_multipliers[key]
            )
            for key in loss_component_keys
        }

    # 桩顶缺陷反射宽时间窗加权
    # 该窗口覆盖较宽的缺陷反射波段，不针对某个精确缺陷位置。
    use_defect_window_weight = True
    defect_window_start = 2.0e-3
    defect_window_end = 4.2e-3
    defect_window_edge = 0.20e-3
    defect_window_gain = 10.0

    def P_star_tau(tau_nd, P0_k, Tp_k):
        """
        无量纲桩顶等效合力：
            F*(tau) = F(t) / F0

        由于F(t)=p(t)A0且F0=p0A0，因此也等价于p(t)/p0。
        """
        t_phys = tau_nd * T0
        amplitude_ratio = P0_k / P0_ref

        active_force = (
            amplitude_ratio
            * torch.sin(math.pi * t_phys / Tp_k)
        )

        return torch.where(
            (t_phys >= 0.0) & (t_phys <= Tp_k),
            active_force,
            torch.zeros_like(t_phys),
        )

    def defect_window_weight(t_phys):
        """
        平滑矩形时间窗：
            2.0 ms ~ 4.2 ms 内权重提高，
        避免使用精确反射峰位置作为强先验。
        """
        if not use_defect_window_weight:
            return torch.ones_like(t_phys)

        left_gate = torch.sigmoid(
            (t_phys - defect_window_start) / defect_window_edge
        )
        right_gate = torch.sigmoid(
            (defect_window_end - t_phys) / defect_window_edge
        )
        window = left_gate * right_gate

        return 1.0 + defect_window_gain * window

    # ========================================================
    # 3.9 通用：计算单次激励的各项损失
    # ========================================================

    def compute_single_impact_losses(
        net_group,
        impact_data,
        fixed_points,
        data_indices=None,
    ):
        """
        使用三子网计算单次锤击对应的 PDE、数据、边界、
        接口和初始损失。

        net_group:
            上部健康段、缺陷段和下部健康段网络
        impact_data:
            当前单次锤击的桩顶观测、P0 和 Tp
        fixed_points:
            固定采样点字典
        data_indices:
            Adam 阶段的固定数据索引；None 表示使用全数据。
        """
        net1, net2, net3 = net_group

        (
            alpha_hat,
            xi1_hat,
            xi2_hat,
            _,
            _,
        ) = get_alpha_xi1_xi2()

        # ----------------------------------------------------
        # (1) PDE 残差：三个子域采用独立的可微局部坐标映射
        # ----------------------------------------------------
        # s_fj 始终固定在 (0, 1)；实际坐标 xi_fj 随当前 xi1_hat、
        # xi2_hat 连续移动。不能对 xi_fj 再执行 detach，否则会再次
        # 切断 PDE 损失到界面位置参数的梯度。
        s_f1 = fixed_points["s_f1"].clone().detach()
        s_f2 = fixed_points["s_f2"].clone().detach()
        s_f3 = fixed_points["s_f3"].clone().detach()

        tau_f1 = (
            fixed_points["tau_f1"]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        tau_f2 = (
            fixed_points["tau_f2"]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        tau_f3 = (
            fixed_points["tau_f3"]
            .clone()
            .detach()
            .requires_grad_(True)
        )

        # 子域1: [0, xi1_hat]
        # 子域2: [xi1_hat, xi2_hat]
        # 子域3: [xi2_hat, 1]
        xi_f1 = s_f1 * xi1_hat
        xi_f2 = xi1_hat + s_f2 * (xi2_hat - xi1_hat)
        xi_f3 = xi2_hat + s_f3 * (1.0 - xi2_hat)

        r1 = pde_residual_nd(
            net1,
            xi_f1,
            tau_f1,
        )
        r2 = pde_residual_nd(
            net2,
            xi_f2,
            tau_f2,
        )
        r3 = pde_residual_nd(
            net3,
            xi_f3,
            tau_f3,
        )

        # 三个子域等权，避免优化器通过压缩某个子域长度来人为降低
        # 该子域对 PDE 损失的贡献。
        loss_pde = (
            r1.pow(2).mean()
            + r2.pow(2).mean()
            + r3.pow(2).mean()
        ) / 3.0

        # ----------------------------------------------------
        # (2) 桩顶速度数据损失
        #
        # 当前一维网络在同一时刻只输出一条截面等效速度，但所有
        # Abaqus桩顶点位的原始速度均直接进入损失：
        #   L_data = (1/Nt) sum_j q_j sum_m omega_m
        #            [w_tau(0,tau_j) - v_m^*(tau_j)]^2
        # 其中 sum_m omega_m = 1。单RPT时严格退化为原始数据损失。
        # ----------------------------------------------------
        if data_indices is None:
            tau_data = impact_data["tau"]
            v_star_points = impact_data["v_star_points"]
        else:
            tau_data = impact_data["tau"][data_indices]
            # 第一维保留全部桩顶点位，只在第二维抽取时间点。
            v_star_points = (
                impact_data["v_star_points"][:, data_indices]
            )

        tau_data = (
            tau_data
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi_top_data = torch.zeros_like(tau_data)

        w_top_data = net1(
            xi_top_data,
            tau_data,
        )
        w_top_tau = autograd.grad(
            w_top_data,
            tau_data,
            grad_outputs=torch.ones_like(w_top_data),
            create_graph=True,
        )[0]

        # w_top_tau: (Nd, 1) -> (1, Nd)，再广播到(M, Nd)。
        v_star_prediction = (
            w_top_tau
            .transpose(0, 1)
            .expand_as(v_star_points)
        )
        data_residual = (
            v_star_prediction - v_star_points
        )

        t_data_phys = tau_data * T0
        # (Nd, 1) -> (1, Nd)，对全部桩顶点位使用相同时间窗权重。
        time_weight = (
            defect_window_weight(t_data_phys)
            .transpose(0, 1)
        )
        point_weights = impact_data["point_weights"]

        # 分母仅为时间点数：点位权重已归一化为总和1。
        # 这样单点时与原始 mean(time_weight * residual^2) 完全一致，
        # 多点等权时等价于所有M*Nd个原始观测直接参加训练。
        loss_data = (
            point_weights
            * time_weight
            * data_residual.pow(2)
        ).sum() / tau_data.shape[0]

        # ----------------------------------------------------
        # (3) 桩底自由边界
        # ----------------------------------------------------
        tau_bottom = (
            fixed_points["tau_bottom"]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi_bottom = torch.ones_like(
            tau_bottom,
            requires_grad=True,
        )

        w_bottom = net3(
            xi_bottom,
            tau_bottom,
        )
        w_bottom_xi = autograd.grad(
            w_bottom,
            xi_bottom,
            grad_outputs=torch.ones_like(w_bottom),
            create_graph=True,
        )[0]

        loss_bc_bottom = mse(
            w_bottom_xi,
            torch.zeros_like(w_bottom_xi),
        )

        # ----------------------------------------------------
        # (4) 桩顶锤击边界
        # ----------------------------------------------------
        tau_top = (
            fixed_points["tau_top_list"][
                impact_data["impact_index"]
            ]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi_top_bc = torch.zeros_like(
            tau_top,
            requires_grad=True,
        )

        w_top_bc = net1(
            xi_top_bc,
            tau_top,
        )
        w_top_xi = autograd.grad(
            w_top_bc,
            xi_top_bc,
            grad_outputs=torch.ones_like(w_top_bc),
            create_graph=True,
        )[0]

        p_star = P_star_tau(
            tau_top,
            impact_data["P0"],
            impact_data["Tp"],
        )
        loss_bc_top = mse(
            w_top_xi + p_star,
            torch.zeros_like(w_top_xi),
        )

        # ----------------------------------------------------
        # (5) 两处接口：位移连续 + 轴力连续
        # ----------------------------------------------------
        tau_interface_base = fixed_points["tau_interface"]

        # z = z1
        tau_if1 = (
            tau_interface_base
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi1_interface = xi1_hat.expand_as(tau_if1)

        w1_at_xi1 = net1(
            xi1_interface,
            tau_if1,
        )
        w2_at_xi1 = net2(
            xi1_interface,
            tau_if1,
        )

        w1_xi_at_xi1 = autograd.grad(
            w1_at_xi1,
            xi1_interface,
            grad_outputs=torch.ones_like(w1_at_xi1),
            create_graph=True,
        )[0]
        w2_xi_at_xi1 = autograd.grad(
            w2_at_xi1,
            xi1_interface,
            grad_outputs=torch.ones_like(w2_at_xi1),
            create_graph=True,
        )[0]

        loss_if_u1 = mse(w1_at_xi1, w2_at_xi1)
        loss_if_N1 = mse(
            w1_xi_at_xi1,
            alpha_hat * w2_xi_at_xi1,
        )

        # z = z2
        tau_if2 = (
            tau_interface_base
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi2_interface = xi2_hat.expand_as(tau_if2)

        w2_at_xi2 = net2(
            xi2_interface,
            tau_if2,
        )
        w3_at_xi2 = net3(
            xi2_interface,
            tau_if2,
        )

        w2_xi_at_xi2 = autograd.grad(
            w2_at_xi2,
            xi2_interface,
            grad_outputs=torch.ones_like(w2_at_xi2),
            create_graph=True,
        )[0]
        w3_xi_at_xi2 = autograd.grad(
            w3_at_xi2,
            xi2_interface,
            grad_outputs=torch.ones_like(w3_at_xi2),
            create_graph=True,
        )[0]

        loss_if_u2 = mse(w2_at_xi2, w3_at_xi2)
        loss_if_N2 = mse(
            alpha_hat * w2_xi_at_xi2,
            w3_xi_at_xi2,
        )

        loss_if_u = 0.5 * (loss_if_u1 + loss_if_u2)
        loss_if_N = 0.5 * (loss_if_N1 + loss_if_N2)

        # ----------------------------------------------------
        # (6) 初始条件：w=0, w_tau=0
        #     与 PDE 配点相同，采用三个子域的可微局部坐标映射。
        # ----------------------------------------------------
        s_ic1 = fixed_points["s_ic1"].clone().detach()
        s_ic2 = fixed_points["s_ic2"].clone().detach()
        s_ic3 = fixed_points["s_ic3"].clone().detach()

        xi_ic1 = s_ic1 * xi1_hat
        xi_ic2 = xi1_hat + s_ic2 * (xi2_hat - xi1_hat)
        xi_ic3 = xi2_hat + s_ic3 * (1.0 - xi2_hat)

        tau_ic1 = torch.zeros_like(s_ic1, requires_grad=True)
        tau_ic2 = torch.zeros_like(s_ic2, requires_grad=True)
        tau_ic3 = torch.zeros_like(s_ic3, requires_grad=True)

        w_ic1 = net1(
            xi_ic1,
            tau_ic1,
        )
        w_ic2 = net2(
            xi_ic2,
            tau_ic2,
        )
        w_ic3 = net3(
            xi_ic3,
            tau_ic3,
        )

        w_ic1_tau = autograd.grad(
            w_ic1,
            tau_ic1,
            grad_outputs=torch.ones_like(w_ic1),
            create_graph=True,
        )[0]
        w_ic2_tau = autograd.grad(
            w_ic2,
            tau_ic2,
            grad_outputs=torch.ones_like(w_ic2),
            create_graph=True,
        )[0]
        w_ic3_tau = autograd.grad(
            w_ic3,
            tau_ic3,
            grad_outputs=torch.ones_like(w_ic3),
            create_graph=True,
        )[0]

        loss_ic1 = 0.5 * (
            w_ic1.pow(2).mean()
            + w_ic1_tau.pow(2).mean()
        )
        loss_ic2 = 0.5 * (
            w_ic2.pow(2).mean()
            + w_ic2_tau.pow(2).mean()
        )
        loss_ic3 = 0.5 * (
            w_ic3.pow(2).mean()
            + w_ic3_tau.pow(2).mean()
        )

        loss_ic = (loss_ic1 + loss_ic2 + loss_ic3) / 3.0

        return {
            "pde": loss_pde,
            "data": loss_data,
            "bc_bottom": loss_bc_bottom,
            "bc_top": loss_bc_top,
            "if_u": loss_if_u,
            "if_N": loss_if_N,
            "ic": loss_ic,
        }

    def aggregate_loss_components(component_dicts):
        """
        对多次激励的同类损失取平均。

        当前 Abaqus 文件通常只有一次激励，但保留该通用写法，便于后续
        加入多次激励时继续使用同一套自适应权重。
        """
        if not component_dicts:
            raise ValueError("component_dicts 不能为空。")

        return {
            key: torch.stack([
                components[key]
                for components in component_dicts
            ]).mean()
            for key in loss_component_keys
        }

    def component_gradient_norm(loss_value):
        """
        计算一个损失项相对于三个子网最后线性层参数的 L2 梯度范数。

        retain_graph=True 是必要的：七个分量依次计算梯度后，还要对总损失
        执行一次正式的 backward()。
        """
        gradients = autograd.grad(
            loss_value,
            gradient_balance_parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        squared_norm = torch.zeros((), device=device)
        has_gradient = False

        for gradient in gradients:
            if gradient is None:
                continue
            has_gradient = True
            squared_norm = squared_norm + gradient.detach().pow(2).sum()

        if not has_gradient:
            return torch.zeros((), device=device)

        return torch.sqrt(
            squared_norm + gradient_weight_eps
        )

    def update_gradient_adaptive_weights(
        loss_components,
        epoch,
    ):
        """
        GradNorm-like 有界权重更新。

        1. 计算 base_weight_i * L_i 对参考层的原始梯度范数；
        2. 根据各损失相对参考时刻的下降速度构造目标梯度范数；
        3. 由目标梯度/原始梯度得到目标修正系数；
        4. 经过均值归一化、单次变化限制、EMA 和全局上下限后更新。

        权重不是 nn.Parameter，不参与主优化器；这样可以避免权重与网络
        同方向最小化时退化到零。
        """
        nonlocal gradient_reference_losses

        if not gradient_reference_losses:
            gradient_reference_losses = {
                key: max(
                    float(loss_components[key].detach().item()),
                    gradient_weight_eps,
                )
                for key in loss_component_keys
            }

        raw_gradient_norm_tensors = []
        for key in loss_component_keys:
            base_weighted_component = (
                base_loss_weights[key]
                * loss_components[key]
            )
            raw_gradient_norm_tensors.append(
                component_gradient_norm(base_weighted_component)
            )

        raw_gradient_norms = torch.stack(
            raw_gradient_norm_tensors
        )
        current_multipliers = torch.tensor(
            [
                adaptive_weight_multipliers[key]
                for key in loss_component_keys
            ],
            dtype=raw_gradient_norms.dtype,
            device=device,
        )
        current_losses = torch.tensor(
            [
                max(
                    float(loss_components[key].detach().item()),
                    gradient_weight_eps,
                )
                for key in loss_component_keys
            ],
            dtype=raw_gradient_norms.dtype,
            device=device,
        )
        reference_losses = torch.tensor(
            [
                gradient_reference_losses[key]
                for key in loss_component_keys
            ],
            dtype=raw_gradient_norms.dtype,
            device=device,
        )

        # r_i > 1 表示该项相对其他项下降更慢，应获得更大的目标梯度。
        relative_training_rates = (
            current_losses
            / reference_losses.clamp_min(gradient_weight_eps)
        )
        relative_training_rates = (
            relative_training_rates
            / relative_training_rates.mean().clamp_min(
                gradient_weight_eps
            )
        )

        current_effective_gradient_norms = (
            current_multipliers * raw_gradient_norms
        )
        mean_effective_gradient_norm = (
            current_effective_gradient_norms
            .mean()
            .clamp_min(gradient_weight_eps)
        )
        target_gradient_norms = (
            mean_effective_gradient_norm
            * relative_training_rates.pow(gradient_weight_alpha)
        )

        valid_gradient_mask = (
            raw_gradient_norms > 1.0e-10
        )
        target_multipliers = current_multipliers.clone()
        target_multipliers[valid_gradient_mask] = (
            target_gradient_norms[valid_gradient_mask]
            / raw_gradient_norms[valid_gradient_mask]
        )
        target_multipliers = torch.nan_to_num(
            target_multipliers,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )

        # 保持七个修正系数的平均值约为1，避免总损失整体尺度无控制漂移。
        target_multipliers = (
            target_multipliers
            / target_multipliers.mean().clamp_min(
                gradient_weight_eps
            )
        )

        proposed_multipliers = {}
        for index, key in enumerate(loss_component_keys):
            current_value = adaptive_weight_multipliers[key]
            target_value = float(
                target_multipliers[index].detach().item()
            )

            lower_bound, upper_bound = (
                adaptive_multiplier_bounds[key]
            )
            if not bool(valid_gradient_mask[index].detach().item()):
                target_value = current_value

            target_value = min(
                max(target_value, lower_bound),
                upper_bound,
            )

            # 防止一次更新跨越过大；随后再用 EMA 平滑。
            target_value = min(
                max(
                    target_value,
                    current_value / gradient_weight_max_change,
                ),
                current_value * gradient_weight_max_change,
            )
            updated_value = (
                gradient_weight_ema * current_value
                + (1.0 - gradient_weight_ema) * target_value
            )
            proposed_multipliers[key] = updated_value

        # EMA 后再次归一化，使修正系数整体尺度保持稳定。
        proposed_mean = (
            sum(proposed_multipliers.values())
            / len(proposed_multipliers)
        )
        proposed_mean = max(
            proposed_mean,
            gradient_weight_eps,
        )

        for index, key in enumerate(loss_component_keys):
            lower_bound, upper_bound = (
                adaptive_multiplier_bounds[key]
            )
            adaptive_weight_multipliers[key] = min(
                max(
                    proposed_multipliers[key] / proposed_mean,
                    lower_bound,
                ),
                upper_bound,
            )
            latest_raw_gradient_norms[key] = float(
                raw_gradient_norms[index].detach().item()
            )
            latest_effective_gradient_norms[key] = (
                adaptive_weight_multipliers[key]
                * latest_raw_gradient_norms[key]
            )

        effective_weights = current_effective_loss_weights()
        print(
            f"[Gradient weight update] epoch={epoch} | "
            + ", ".join(
                (
                    f"{key}: a={adaptive_weight_multipliers[key]:.3f}, "
                    f"lambda={effective_weights[key]:.3e}"
                )
                for key in loss_component_keys
            )
        )

    def weighted_total_loss(loss_components):
        effective_weights = current_effective_loss_weights()
        return sum(
            effective_weights[key] * loss_components[key]
            for key in loss_component_keys
        )

    def empty_loss_accumulator():
        return {
            "loss": 0.0,
            "loss_pde": 0.0,
            "loss_data": 0.0,
            "loss_bc_bottom": 0.0,
            "loss_bc_top": 0.0,
            "loss_if_u": 0.0,
            "loss_if_N": 0.0,
            "loss_ic": 0.0,
        }

    def add_detached_losses(accumulator, total_loss, components, scale):
        accumulator["loss"] += (
            float(total_loss.detach().item()) * scale
        )
        accumulator["loss_pde"] += (
            float(components["pde"].detach().item()) * scale
        )
        accumulator["loss_data"] += (
            float(components["data"].detach().item()) * scale
        )
        accumulator["loss_bc_bottom"] += (
            float(components["bc_bottom"].detach().item()) * scale
        )
        accumulator["loss_bc_top"] += (
            float(components["bc_top"].detach().item()) * scale
        )
        accumulator["loss_if_u"] += (
            float(components["if_u"].detach().item()) * scale
        )
        accumulator["loss_if_N"] += (
            float(components["if_N"].detach().item()) * scale
        )
        accumulator["loss_ic"] += (
            float(components["ic"].detach().item()) * scale
        )

    def append_parameter_info(info):
        with torch.no_grad():
            (
                alpha_hat,
                xi1_hat,
                xi2_hat,
                xi_c_hat,
                lam_hat,
            ) = get_alpha_xi1_xi2()

        info.update({
            "logit_alpha": float(logit_alpha.detach().item()),
            "theta_c": float(theta_c.detach().item()),
            "theta_l": float(theta_l.detach().item()),
            "alpha_hat": float(alpha_hat.detach().item()),
            "xi1_hat": float(xi1_hat.detach().item()),
            "xi2_hat": float(xi2_hat.detach().item()),
            "xi_c_hat": float(xi_c_hat.detach().item()),
            "lam_hat": float(lam_hat.detach().item()),
        })

        effective_weights = current_effective_loss_weights()
        for key in loss_component_keys:
            info[f"adaptive_multiplier_{key}"] = (
                adaptive_weight_multipliers[key]
            )
            info[f"effective_weight_{key}"] = (
                effective_weights[key]
            )
            info[f"gradient_norm_{key}"] = (
                latest_effective_gradient_norms[key]
            )

        return info

    # ========================================================
    # 3.10 Adam 阶段固定采样
    # ========================================================
    def split_count_three(total_count):
        """将总配点数尽可能均匀地分配给三个子域。"""
        base = total_count // 3
        return base, base, total_count - 2 * base

    # --------------------------------------------------------
    # 自适应采样参数
    # --------------------------------------------------------
    # 是否启用残差自适应采样
    # False：训练开始时生成一次配点，后续不再更新
    # True ：按照下面的设置定期更新配点
    use_residual_adaptive_sampling = True
    adaptive_warmup_epochs = 10000
    adaptive_update_interval = 5000
    adaptive_stop_epoch = 100000
    adaptive_candidate_factor = 3
    adaptive_high_fraction = 0.70
    adaptive_batch_size = 512
    # 每次自适应更新只替换 30%，保留上一周期的 70%
    adaptive_replace_fraction = 0.5

    # 桩顶边界分层比例：锤击作用段、缺陷反射段、完整时域。
    top_force_fraction = 0.30
    top_reflection_fraction = 0.40

    def allocate_top_counts(total_count):
        n_force = int(round(total_count * top_force_fraction))
        n_reflection = int(
            round(total_count * top_reflection_fraction)
        )
        n_global = total_count - n_force - n_reflection
        return n_force, n_reflection, n_global

    def sample_tau_interval(n_points, t_start, t_end):
        """在给定物理时间区间均匀采样，并转换为无量纲时间。"""
        if n_points <= 0:
            return torch.empty(0, 1, device=device)

        t_final = float(t_np[-1])
        t_start = max(0.0, min(float(t_start), t_final))
        t_end = max(t_start, min(float(t_end), t_final))

        if t_end <= t_start:
            t_phys = torch.full(
                (n_points, 1),
                fill_value=t_start,
                device=device,
            )
        else:
            t_phys = (
                t_start
                + torch.rand(n_points, 1, device=device)
                * (t_end - t_start)
            )

        return t_phys / T0

    def shuffled_concat(tensor_list):
        result = torch.cat(tensor_list, dim=0)
        if result.shape[0] > 1:
            permutation = torch.randperm(
                result.shape[0], device=device
            )
            result = result[permutation]
        return result.detach()

    def make_stratified_top_times(impact_data, total_count):
        """生成桩顶边界的初始分层时间点。"""
        n_force, n_reflection, n_global = allocate_top_counts(
            total_count
        )

        return shuffled_concat([
            sample_tau_interval(
                n_force, 0.0, impact_data["Tp"]
            ),
            sample_tau_interval(
                n_reflection,
                defect_window_start,
                defect_window_end,
            ),
            sample_tau_interval(
                n_global, 0.0, float(t_np[-1])
            ),
        ])

    def select_mixed_indices(scores, n_select):
        """选择高残差点，同时保留均匀随机点保证全域覆盖。"""
        scores = scores.reshape(-1)
        n_candidates = int(scores.numel())
        n_select = min(int(n_select), n_candidates)

        if n_select <= 0:
            return torch.empty(
                0, dtype=torch.long, device=device
            )

        n_high = int(round(n_select * adaptive_high_fraction))
        n_high = max(0, min(n_high, n_select))
        n_uniform = n_select - n_high

        if n_high > 0:
            high_indices = torch.topk(
                scores, k=n_high, largest=True
            ).indices
        else:
            high_indices = torch.empty(
                0, dtype=torch.long, device=device
            )

        if n_uniform > 0:
            available_mask = torch.ones(
                n_candidates, dtype=torch.bool, device=device
            )
            available_mask[high_indices] = False
            available = torch.nonzero(
                available_mask, as_tuple=False
            ).reshape(-1)
            random_order = torch.randperm(
                available.numel(), device=device
            )
            uniform_indices = available[
                random_order[:n_uniform]
            ]
        else:
            uniform_indices = torch.empty(
                0, dtype=torch.long, device=device
            )

        selected = torch.cat(
            [high_indices, uniform_indices], dim=0
        )
        if selected.numel() > 1:
            selected = selected[
                torch.randperm(selected.numel(), device=device)
            ]
        return selected
    def partial_refresh_counts(n_target, n_available):
        """
        计算本次需要保留的旧点数和需要补充的新点数。

        n_target:
            更新后要求保持的配点总数。
        n_available:
            当前旧配点的实际数量。
        """
        n_target = int(n_target)
        n_available = int(n_available)

        n_replace = int(round(
            n_target * adaptive_replace_fraction
        ))
        n_replace = max(0, min(n_replace, n_target))

        n_keep = n_target - n_replace

        # 防止旧点数量小于计划保留数量
        if n_keep > n_available:
            n_keep = n_available
            n_replace = n_target - n_keep

        return n_keep, n_replace


    def random_keep_indices(n_available, n_keep):
        """从旧配点中无放回地随机选择需要保留的索引。"""
        if n_keep <= 0:
            return torch.empty(
                0,
                dtype=torch.long,
                device=device,
            )

        if n_keep >= n_available:
            return torch.arange(
                n_available,
                dtype=torch.long,
                device=device,
            )

        return torch.randperm(
            n_available,
            device=device,
        )[:n_keep]


    def shuffle_paired(first, second):
        """
        使用同一个随机顺序打乱两个成对张量。

        用于保证 PDE 配点中的 s 和 tau 始终一一对应。
        """
        if first.shape[0] != second.shape[0]:
            raise ValueError(
                "成对配点的数量不一致，不能进行联合打乱。"
            )

        if first.shape[0] > 1:
            order = torch.randperm(
                first.shape[0],
                device=device,
            )
            first = first[order]
            second = second[order]

        return first.detach(), second.detach()

    def map_reference_to_subdomain(s_value, domain_index, xi1, xi2):
        if domain_index == 1:
            return s_value * xi1
        if domain_index == 2:
            return xi1 + s_value * (xi2 - xi1)
        if domain_index == 3:
            return xi2 + s_value * (1.0 - xi2)
        raise ValueError("domain_index 必须为 1、2 或 3。")

    def score_pde_candidates(domain_index, s_candidates, tau_candidates):
        """计算单次锤击下的 PDE 候选点残差。"""
        with torch.no_grad():
            _, xi1_now, xi2_now, _, _ = get_alpha_xi1_xi2()
            xi1_value = xi1_now.detach()
            xi2_value = xi2_now.detach()

        net = subnetworks[domain_index - 1]
        score_chunks = []

        for start in range(0, s_candidates.shape[0], adaptive_batch_size):
            end = min(
                start + adaptive_batch_size,
                s_candidates.shape[0],
            )
            s_batch = s_candidates[start:end]
            tau_base = tau_candidates[start:end]
            batch_score = torch.zeros_like(s_batch)

            for impact_data in impact_data_list:
                # 候选点评分不参与参数优化，因此此处可以把实际坐标
                # 变为叶子张量；正式训练使用的映射仍保持可微。
                xi_batch = map_reference_to_subdomain(
                    s_batch,
                    domain_index,
                    xi1_value,
                    xi2_value,
                ).detach().requires_grad_(True)
                tau_batch = (
                    tau_base.clone().detach().requires_grad_(True)
                )
                residual = pde_residual_nd(
                    net, xi_batch, tau_batch
                )
                batch_score = (
                    batch_score + residual.detach().pow(2)
                )

            batch_score = batch_score / n_impacts
            score_chunks.append(batch_score)

        return torch.cat(score_chunks, dim=0).reshape(-1)

    def score_top_bc_candidates(impact_data, tau_candidates):
        """计算某次激励的桩顶力边界残差。"""
        score_chunks = []
        net1 = subnetworks[0]

        for start in range(0, tau_candidates.shape[0], adaptive_batch_size):
            end = min(
                start + adaptive_batch_size,
                tau_candidates.shape[0],
            )
            tau_batch = tau_candidates[start:end].detach()
            xi_batch = torch.zeros_like(
                tau_batch, requires_grad=True
            )
            w_top = net1(xi_batch, tau_batch)
            w_top_xi = autograd.grad(
                w_top,
                xi_batch,
                grad_outputs=torch.ones_like(w_top),
                create_graph=False,
            )[0]
            p_star = P_star_tau(
                tau_batch,
                impact_data["P0"],
                impact_data["Tp"],
            )
            score_chunks.append(
                (w_top_xi + p_star).detach().pow(2)
            )

        return torch.cat(score_chunks, dim=0).reshape(-1)

    def score_interface_candidates(tau_candidates):
        """计算单次锤击、两个界面的综合连续条件残差。"""
        with torch.no_grad():
            alpha_now, xi1_now, xi2_now, _, _ = (
                get_alpha_xi1_xi2()
            )
            alpha_value = alpha_now.detach()
            xi1_value = float(xi1_now.detach().cpu())
            xi2_value = float(xi2_now.detach().cpu())

        net1, net2, net3 = subnetworks
        score_chunks = []

        for start in range(0, tau_candidates.shape[0], adaptive_batch_size):
            end = min(
                start + adaptive_batch_size,
                tau_candidates.shape[0],
            )
            tau_batch = tau_candidates[start:end].detach()
            batch_score = torch.zeros_like(tau_batch)

            for impact_data in impact_data_list:
                xi1_batch = torch.full_like(
                    tau_batch,
                    fill_value=xi1_value,
                    requires_grad=True,
                )
                xi2_batch = torch.full_like(
                    tau_batch,
                    fill_value=xi2_value,
                    requires_grad=True,
                )

                w1_xi1 = net1(xi1_batch, tau_batch)
                w2_xi1 = net2(xi1_batch, tau_batch)
                dw1_xi1 = autograd.grad(
                    w1_xi1,
                    xi1_batch,
                    grad_outputs=torch.ones_like(w1_xi1),
                    create_graph=False,
                )[0]
                dw2_xi1 = autograd.grad(
                    w2_xi1,
                    xi1_batch,
                    grad_outputs=torch.ones_like(w2_xi1),
                    create_graph=False,
                )[0]

                w2_xi2 = net2(xi2_batch, tau_batch)
                w3_xi2 = net3(xi2_batch, tau_batch)
                dw2_xi2 = autograd.grad(
                    w2_xi2,
                    xi2_batch,
                    grad_outputs=torch.ones_like(w2_xi2),
                    create_graph=False,
                )[0]
                dw3_xi2 = autograd.grad(
                    w3_xi2,
                    xi2_batch,
                    grad_outputs=torch.ones_like(w3_xi2),
                    create_graph=False,
                )[0]

                # 用正式损失中的界面权重进行候选点评分，避免位移
                # 连续残差与轴力连续残差的重要性在选点阶段失配。
                displacement_score = 0.5 * (
                    (w1_xi1 - w2_xi1).detach().pow(2)
                    + (w2_xi2 - w3_xi2).detach().pow(2)
                )
                force_score = 0.5 * (
                    (dw1_xi1 - alpha_value * dw2_xi1)
                    .detach()
                    .pow(2)
                    + (alpha_value * dw2_xi2 - dw3_xi2)
                    .detach()
                    .pow(2)
                )
                interface_effective_weights = (
                    current_effective_loss_weights()
                )
                residual_score = (
                    interface_effective_weights["if_u"]
                    * displacement_score
                    + interface_effective_weights["if_N"]
                    * force_score
                )
                batch_score = batch_score + residual_score

            batch_score = batch_score / n_impacts
            score_chunks.append(batch_score)

        return torch.cat(score_chunks, dim=0).reshape(-1)

    def adapt_pde_points(fixed_points, domain_counts):
        """
        更新三个子域的 PDE 配点。

        每个子域保留70%的旧(s, tau)配点，
        仅生成并替换30%的新配点。
        """
        for domain_index, n_target in enumerate(
            domain_counts,
            start=1,
        ):
            s_key = f"s_f{domain_index}"
            tau_key = f"tau_f{domain_index}"

            old_s = fixed_points[s_key].detach()
            old_tau = fixed_points[tau_key].detach()

            if old_s.shape[0] != old_tau.shape[0]:
                raise ValueError(
                    f"子域{domain_index}的旧s和tau数量不一致。"
                )

            n_keep, n_replace = partial_refresh_counts(
                n_target=n_target,
                n_available=old_s.shape[0],
            )

            # --------------------------------------------
            # 1. 使用同一组索引保留70%的旧(s, tau)点
            # --------------------------------------------
            keep_indices = random_keep_indices(
                old_s.shape[0],
                n_keep,
            )

            kept_s = old_s[keep_indices]
            kept_tau = old_tau[keep_indices]

            # --------------------------------------------
            # 2. 只为剩余30%生成新的候选点
            # --------------------------------------------
            if n_replace > 0:
                n_candidates = max(
                    n_replace,
                    adaptive_candidate_factor * n_replace,
                )

                s_candidates = torch.rand(
                    n_candidates,
                    1,
                    device=device,
                )

                tau_candidates = (
                    torch.rand(
                        n_candidates,
                        1,
                        device=device,
                    )
                    * tau_max
                )

                scores = score_pde_candidates(
                    domain_index,
                    s_candidates,
                    tau_candidates,
                )

                selected = select_mixed_indices(
                    scores,
                    n_replace,
                )

                # 新点也必须使用同一组选中索引
                new_s = s_candidates[selected].detach()
                new_tau = tau_candidates[selected].detach()

            else:
                new_s = torch.empty(
                    0, 1, device=device
                )
                new_tau = torch.empty(
                    0, 1, device=device
                )

            # --------------------------------------------
            # 3. 合并70%旧点和30%新点
            # --------------------------------------------
            merged_s = torch.cat(
                [kept_s, new_s],
                dim=0,
            )

            merged_tau = torch.cat(
                [kept_tau, new_tau],
                dim=0,
            )

            # s和tau必须使用相同的随机顺序
            merged_s, merged_tau = shuffle_paired(
                merged_s,
                merged_tau,
            )

            if merged_s.shape[0] != n_target:
                raise RuntimeError(
                    f"子域{domain_index}更新后的PDE点数错误："
                    f"{merged_s.shape[0]} != {n_target}"
                )

            fixed_points[s_key] = merged_s
            fixed_points[tau_key] = merged_tau


    def adapt_top_boundary_points(
        fixed_points,
        total_count,
    ):
        """
        更新桩顶边界时间点。

        每个激励分别保留70%的旧桩顶时间点，
        其余30%仍按以下三层时间窗生成：
        1. 锤击作用时间段；
        2. 缺陷反射时间窗；
        3. 完整时间域。
        """
        old_tau_top_list = fixed_points["tau_top_list"]

        if len(old_tau_top_list) != len(impact_data_list):
            raise ValueError(
                "桩顶旧配点组数与激励数量不一致。"
            )

        updated_tau_top_list = []

        for impact_index, impact_data in enumerate(
            impact_data_list
        ):
            old_tau = old_tau_top_list[
                impact_index
            ].detach()

            n_keep, n_replace = partial_refresh_counts(
                n_target=total_count,
                n_available=old_tau.shape[0],
            )

            # --------------------------------------------
            # 1. 保留该激励70%的旧桩顶时间点
            # --------------------------------------------
            keep_indices = random_keep_indices(
                old_tau.shape[0],
                n_keep,
            )
            kept_tau = old_tau[keep_indices]

            # --------------------------------------------
            # 2. 为30%的新点进行分层残差采样
            # --------------------------------------------
            if n_replace > 0:
                new_layer_counts = allocate_top_counts(
                    n_replace
                )

                intervals = [
                    (
                        0.0,
                        impact_data["Tp"],
                    ),
                    (
                        defect_window_start,
                        defect_window_end,
                    ),
                    (
                        0.0,
                        float(t_np[-1]),
                    ),
                ]

                new_layers = []

                for n_layer, (
                    t_start,
                    t_end,
                ) in zip(
                    new_layer_counts,
                    intervals,
                ):
                    if n_layer <= 0:
                        continue

                    n_candidates = max(
                        n_layer,
                        adaptive_candidate_factor
                        * n_layer,
                    )

                    tau_candidates = sample_tau_interval(
                        n_candidates,
                        t_start,
                        t_end,
                    )

                    scores = score_top_bc_candidates(
                        impact_data,
                        tau_candidates,
                    )

                    selected = select_mixed_indices(
                        scores,
                        n_layer,
                    )

                    new_layers.append(
                        tau_candidates[
                            selected
                        ].detach()
                    )

                if len(new_layers) > 0:
                    new_tau = shuffled_concat(
                        new_layers
                    )
                else:
                    new_tau = torch.empty(
                        0, 1, device=device
                    )

            else:
                new_tau = torch.empty(
                    0, 1, device=device
                )

            # --------------------------------------------
            # 3. 合并旧点和新点，并整体打乱
            # --------------------------------------------
            merged_tau = shuffled_concat([
                kept_tau,
                new_tau,
            ])

            if merged_tau.shape[0] != total_count:
                raise RuntimeError(
                    f"激励{impact_index + 1}更新后的"
                    f"桩顶点数错误："
                    f"{merged_tau.shape[0]} "
                    f"!= {total_count}"
                )

            updated_tau_top_list.append(
                merged_tau
            )

        fixed_points["tau_top_list"] = (
            updated_tau_top_list
        )


    def adapt_interface_points(
        fixed_points,
        n_target,
    ):
        """
        更新界面条件时间点。

        保留70%的旧界面时间点，
        只替换30%的高残差/随机新点。
        """
        old_tau = fixed_points[
            "tau_interface"
        ].detach()

        n_keep, n_replace = partial_refresh_counts(
            n_target=n_target,
            n_available=old_tau.shape[0],
        )

        # --------------------------------------------
        # 1. 保留70%的旧界面点
        # --------------------------------------------
        keep_indices = random_keep_indices(
            old_tau.shape[0],
            n_keep,
        )
        kept_tau = old_tau[keep_indices]

        # --------------------------------------------
        # 2. 生成30%的新界面点
        # --------------------------------------------
        if n_replace > 0:
            n_candidates = max(
                n_replace,
                adaptive_candidate_factor * n_replace,
            )

            tau_candidates = (
                torch.rand(
                    n_candidates,
                    1,
                    device=device,
                )
                * tau_max
            )

            scores = score_interface_candidates(
                tau_candidates
            )

            selected = select_mixed_indices(
                scores,
                n_replace,
            )

            new_tau = tau_candidates[
                selected
            ].detach()

        else:
            new_tau = torch.empty(
                0, 1, device=device
            )

        # --------------------------------------------
        # 3. 合并并打乱
        # --------------------------------------------
        merged_tau = shuffled_concat([
            kept_tau,
            new_tau,
        ])

        if merged_tau.shape[0] != n_target:
            raise RuntimeError(
                "更新后的界面点数错误："
                f"{merged_tau.shape[0]} != {n_target}"
            )

        fixed_points["tau_interface"] = merged_tau
    def refresh_adaptive_points(
        fixed_points,
        domain_counts,
        n_top,
        n_interface,
        phase_label,
    ):
        """依次更新 PDE、桩顶边界和界面采样点。"""
        adapt_pde_points(fixed_points, domain_counts)
        adapt_top_boundary_points(fixed_points, n_top)
        adapt_interface_points(fixed_points, n_interface)
        print(
            f"[Adaptive sampling] {phase_label}: "
            f"保留上一周期的"
            f"{(1.0 - adaptive_replace_fraction) * 100:.0f}%配点，"
            f"替换"
            f"{adaptive_replace_fraction * 100:.0f}%新配点。"
        )

    N_f_adam = 8000
    N_bc_adam = 2000
    N_if_adam = 5000
    N_ic_adam = 2000
    N_data_adam = 5000

    # 保持原代码的 PDE/IC 总配点量基本不变，只是改为三个子域
    # 分别持有自己的固定参考坐标 s∈(0,1)。
    N_f1_adam, N_f2_adam, N_f3_adam = split_count_three(
        N_f_adam
    )
    N_ic1_adam, N_ic2_adam, N_ic3_adam = split_count_three(
        N_ic_adam
    )

    with torch.no_grad():
        adam_fixed_points = {
            "s_f1": torch.rand(
                N_f1_adam, 1, device=device
            ),
            "s_f2": torch.rand(
                N_f2_adam, 1, device=device
            ),
            "s_f3": torch.rand(
                N_f3_adam, 1, device=device
            ),
            "tau_f1": torch.rand(
                N_f1_adam, 1, device=device
            ) * tau_max,
            "tau_f2": torch.rand(
                N_f2_adam, 1, device=device
            ) * tau_max,
            "tau_f3": torch.rand(
                N_f3_adam, 1, device=device
            ) * tau_max,
            "tau_bottom": torch.rand(
                N_bc_adam, 1, device=device
            ) * tau_max,
            # 桩顶边界分层采样：
            # 锤击作用段 + 缺陷反射时间窗 + 全时间域。
            "tau_top_list": [
                make_stratified_top_times(
                    impact_data,
                    N_bc_adam,
                )
                for impact_data in impact_data_list
            ],
            "tau_interface": torch.rand(
                N_if_adam, 1, device=device
            ) * tau_max,
            "s_ic1": torch.rand(
                N_ic1_adam, 1, device=device
            ),
            "s_ic2": torch.rand(
                N_ic2_adam, 1, device=device
            ),
            "s_ic3": torch.rand(
                N_ic3_adam, 1, device=device
            ),
        }

        adam_data_indices = []
        for impact_data in impact_data_list:
            total_points = impact_data["tau"].shape[0]

            if N_data_adam >= total_points:
                indices = torch.arange(
                    total_points,
                    device=device,
                )
            else:
                # 不放回抽样，避免同一个时间点重复过多
                indices = torch.randperm(
                    total_points,
                    device=device,
                )[:N_data_adam]

            adam_data_indices.append(indices)

    # --------------------------------------------------------
    # 梯度权重更新专用的小型固定点集
    # --------------------------------------------------------
    # 这批点只用于估计七个损失分量的梯度范数，不替代正式训练点。
    # 若直接在8000个PDE点的大计算图上连续计算七次梯度范数，显存和时间
    # 开销会显著增加，因此采用独立的小型代表性点集。
    N_f_balance = 768
    N_bc_balance = 384
    N_if_balance = 512
    N_ic_balance = 384
    N_data_balance = 1024

    (
        N_f1_balance,
        N_f2_balance,
        N_f3_balance,
    ) = split_count_three(N_f_balance)
    (
        N_ic1_balance,
        N_ic2_balance,
        N_ic3_balance,
    ) = split_count_three(N_ic_balance)

    with torch.no_grad():
        gradient_balance_fixed_points = {
            "s_f1": torch.rand(
                N_f1_balance, 1, device=device
            ),
            "s_f2": torch.rand(
                N_f2_balance, 1, device=device
            ),
            "s_f3": torch.rand(
                N_f3_balance, 1, device=device
            ),
            "tau_f1": torch.rand(
                N_f1_balance, 1, device=device
            ) * tau_max,
            "tau_f2": torch.rand(
                N_f2_balance, 1, device=device
            ) * tau_max,
            "tau_f3": torch.rand(
                N_f3_balance, 1, device=device
            ) * tau_max,
            "tau_bottom": torch.rand(
                N_bc_balance, 1, device=device
            ) * tau_max,
            "tau_top_list": [
                make_stratified_top_times(
                    impact_data,
                    N_bc_balance,
                )
                for impact_data in impact_data_list
            ],
            "tau_interface": torch.rand(
                N_if_balance, 1, device=device
            ) * tau_max,
            "s_ic1": torch.rand(
                N_ic1_balance, 1, device=device
            ),
            "s_ic2": torch.rand(
                N_ic2_balance, 1, device=device
            ),
            "s_ic3": torch.rand(
                N_ic3_balance, 1, device=device
            ),
        }

        gradient_balance_data_indices = []
        for impact_data in impact_data_list:
            total_points = impact_data["tau"].shape[0]
            if N_data_balance >= total_points:
                indices = torch.arange(
                    total_points,
                    device=device,
                )
            else:
                indices = torch.randperm(
                    total_points,
                    device=device,
                )[:N_data_balance]
            gradient_balance_data_indices.append(indices)

    # 不同类型参数采用不同学习率
    lr_network = 1.0e-3
    lr_alpha = 3.0e-4
    lr_xi_c = 3.0e-5
    lr_lambda = 5.0e-5
    optimizer_adam = optim.Adam(
        [
            {
                "params": network_params,
                "lr": lr_network,
            },
            {
                "params": [logit_alpha],
                "lr": lr_alpha,
            },
            {
                "params": [theta_c],
                "lr": lr_xi_c,
            },
            {
                "params": [theta_l],
                "lr": lr_lambda,
            },
        ]
    )
    history = []

    def train_step_adam(epoch):
        optimizer_adam.zero_grad(set_to_none=True)

        # 权重更新使用独立的小型点集和独立计算图。更新完成后释放该图，
        # 再重新计算正式训练损失，避免七次梯度范数计算长期占用正式大图。
        if (
            use_gradient_adaptive_weights
            and not adaptive_weights_frozen
            and epoch >= gradient_weight_warmup_epochs
            and epoch % gradient_weight_update_interval == 0
        ):
            balance_component_dicts = []
            for k in range(n_impacts):
                balance_components = compute_single_impact_losses(
                    subnetworks,
                    impact_data_list[k],
                    gradient_balance_fixed_points,
                    data_indices=gradient_balance_data_indices[k],
                )
                balance_component_dicts.append(balance_components)

            aggregate_balance_components = (
                aggregate_loss_components(
                    balance_component_dicts
                )
            )
            update_gradient_adaptive_weights(
                aggregate_balance_components,
                epoch,
            )

            del aggregate_balance_components
            del balance_component_dicts
            del balance_components
            optimizer_adam.zero_grad(set_to_none=True)

        accumulator = empty_loss_accumulator()
        impact_data_loss_values = []
        component_dicts = []

        # 先构造所有激励的损失图，再对同类损失取平均。只有这样，梯度权重
        # 更新看到的才是与最终总损失一致的聚合分量。
        for k in range(n_impacts):
            components = compute_single_impact_losses(
                subnetworks,
                impact_data_list[k],
                adam_fixed_points,
                data_indices=adam_data_indices[k],
            )
            component_dicts.append(components)
            impact_data_loss_values.append(
                float(components["data"].detach().item())
            )

        aggregate_components = aggregate_loss_components(
            component_dicts
        )

        total_loss = weighted_total_loss(aggregate_components)

        # Read the accepted physical parameters before the Adam update.
        parameter_values_before = (
            detached_inverse_parameter_values()
        )

        total_loss.backward()

        # Gradients must be read after backward() and before step().
        inverse_gradient_info = collect_inverse_gradient_info()

        optimizer_adam.step()

        # Measure the actual physical-parameter change made by Adam.
        parameter_values_after = (
            detached_inverse_parameter_values()
        )
        parameter_update_info = inverse_update_info(
            parameter_values_before,
            parameter_values_after,
        )

        add_detached_losses(
            accumulator,
            total_loss,
            aggregate_components,
            scale=1.0,
        )

        for k, value in enumerate(impact_data_loss_values, start=1):
            accumulator[f"loss_data_impact_{k}"] = value

        info = append_parameter_info(accumulator)
        info.update(inverse_gradient_info)
        info.update(parameter_update_info)

        info["lr_network"] = optimizer_adam.param_groups[0]["lr"]
        info["lr_alpha"] = optimizer_adam.param_groups[1]["lr"]
        info["lr_xi_c"] = optimizer_adam.param_groups[2]["lr"]
        info["lr_lambda"] = optimizer_adam.param_groups[3]["lr"]
        info["lbfgs_closure_evals"] = 0

        return info

    # 正式训练轮数。如需先检查程序，可临时改小。
    n_epochs_adam = 150000
    gradient_weight_freeze_epoch = 100000

    for epoch in range(1, n_epochs_adam + 1):
        if epoch == gradient_weight_freeze_epoch + 1:
            adaptive_weights_frozen = True
        
            print(
                f"\n===== Loss weights frozen after epoch "
                f"{gradient_weight_freeze_epoch} ====="
            )
        
            frozen_weights = current_effective_loss_weights()
        
            for key in loss_component_keys:
                print(
                    f"{key:>9s}: "
                    f"multiplier="
                    f"{adaptive_weight_multipliers[key]:.6g}, "
                    f"effective={frozen_weights[key]:.6g}"
                )
        # 必须放在学习率调整之后
        info = train_step_adam(epoch)
        info["epoch"] = epoch
        info["phase"] = "adam"
        history.append(info)

        if epoch % 1000 == 0:
            print(
                f"[Adam] Epoch {epoch:6d} | "
                f"loss={info['loss']:.3e}, "
                f"pde={info['loss_pde']:.3e}, "
                f"data={info['loss_data']:.3e}, "
                f"bc_top={info['loss_bc_top']:.3e}, "
                f"if_u={info['loss_if_u']:.3e}, "
                f"if_N={info['loss_if_N']:.3e}, "
                f"ic={info['loss_ic']:.3e}, "
                f"alpha={info['alpha_hat']:.4f}, "
                f"grad_alpha={info['grad_alpha_phys']:+.3e}, "
                f"delta_alpha={info['delta_alpha_hat']:+.3e}, "
                f"map_grad={info['map_dalpha_dlogit']:.3e}, "
                f"xi1={info['xi1_hat']:.4f}, "
                f"xi2={info['xi2_hat']:.4f}"
            )

        # Adam 训练若干轮后，根据当前网络残差重新选择三类关键点。
        # 更新之后的点在下一个更新周期内保持固定，避免每轮随机换点。
        if (
            use_residual_adaptive_sampling
            and epoch >= adaptive_warmup_epochs
            and epoch % adaptive_update_interval == 0
            and epoch <= adaptive_stop_epoch
            and epoch < n_epochs_adam
        ):
            refresh_adaptive_points(
                adam_fixed_points,
                (
                    N_f1_adam,
                    N_f2_adam,
                    N_f3_adam,
                ),
                N_bc_adam,
                N_if_adam,
                phase_label=f"Adam epoch {epoch}",
            )
       
    # L-BFGS 的 strong-Wolfe 线搜索要求每次 closure 面对同一个目标函数。
    # 因此进入该阶段前冻结 Adam 最终得到的自适应权重，closure 内绝不更新。
    adaptive_weights_frozen = True
    frozen_effective_weights = current_effective_loss_weights()
    print("\n===== Frozen loss weights for L-BFGS =====")
    for key in loss_component_keys:
        print(
            f"{key:>9s}: "
            f"base={base_loss_weights[key]:.6g}, "
            f"multiplier={adaptive_weight_multipliers[key]:.6g}, "
            f"effective={frozen_effective_weights[key]:.6g}"
        )

    # ========================================================
    # 3.11 L-BFGS 继承 Adam 最后一套固定配点
    # ========================================================
    # 不再重新随机生成配点，也不再进行切换前自适应刷新。
    # 对 Adam 最终配点进行深拷贝，确保 L-BFGS 全阶段目标固定。
    lbfgs_fixed_points = {}

    for key, value in adam_fixed_points.items():
        if isinstance(value, list):
            # tau_top_list 是 Tensor 列表，需要逐项复制
            lbfgs_fixed_points[key] = [
                item.detach().clone()
                for item in value
            ]
        elif torch.is_tensor(value):
            lbfgs_fixed_points[key] = (
                value.detach().clone()
            )
        else:
            raise TypeError(
                f"Unsupported fixed-point type for "
                f"{key}: {type(value)}"
            )

    # 同时继承 Adam 使用的桩顶数据索引，
    # 避免切换优化器时数据目标也发生变化。
    lbfgs_data_indices = [
        indices.detach().clone()
        for indices in adam_data_indices
    ]

    print(
        "\n[L-BFGS] Directly inherited the final "
        "Adam collocation points and data indices."
    )

    def compute_aggregate_fixed(backward=False):
        accumulator = empty_loss_accumulator()
        impact_data_loss_values = []
        total_value = torch.zeros((), device=device)

        for k in range(n_impacts):
            components = compute_single_impact_losses(
                subnetworks,
                impact_data_list[k],
                lbfgs_fixed_points,
                data_indices=lbfgs_data_indices[k]
            )
            total_loss_k = weighted_total_loss(components)
            scaled_loss = total_loss_k / n_impacts

            if backward:
                scaled_loss.backward()

            total_value = (
                total_value
                + scaled_loss.detach()
            )

            add_detached_losses(
                accumulator,
                total_loss_k,
                components,
                scale=1.0 / n_impacts,
            )
            impact_data_loss_values.append(
                float(components["data"].detach().item())
            )

        for k, value in enumerate(impact_data_loss_values, start=1):
            accumulator[f"loss_data_impact_{k}"] = value

        accumulator = append_parameter_info(accumulator)
        return total_value, accumulator

    optimizer_lbfgs = optim.LBFGS(
        params,
        lr=0.05,
        max_iter=20,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    # 正式 L-BFGS 精修步数。如需先检查程序，可临时改小。
    n_lbfgs_steps = 2000
    # Re-evaluating the accepted-point gradient costs one extra backward
    # pass. Recording every 10 outer steps matches the print interval.
    lbfgs_gradient_record_interval = 10
    start_epoch = n_epochs_adam
    info_holder = {}

    for step in range(1, n_lbfgs_steps + 1):

        parameter_values_before = (
            detached_inverse_parameter_values()
        )
        closure_counter = {"count": 0}

        def closure():
            closure_counter["count"] += 1
            optimizer_lbfgs.zero_grad()

            loss_value, closure_info = compute_aggregate_fixed(
                backward=True
            )
            info_holder["info"] = closure_info

            # LBFGS 已从参数中取得梯度；返回当前标量损失即可
            return loss_value

        optimizer_lbfgs.step(closure)

        # step 完成后重新评估当前参数状态
        parameter_values_after = (
            detached_inverse_parameter_values()
        )
        parameter_update_info = inverse_update_info(
            parameter_values_before,
            parameter_values_after,
        )

        record_lbfgs_gradient = (
            step == 1
            or step % lbfgs_gradient_record_interval == 0
            or step == n_lbfgs_steps
        )

        if record_lbfgs_gradient:
            # Recompute the gradient at the final accepted point. Do not
            # append history inside closure(), since one outer L-BFGS step
            # calls closure several times.
            optimizer_lbfgs.zero_grad(set_to_none=True)
            _, info = compute_aggregate_fixed(backward=True)
            inverse_gradient_info = collect_inverse_gradient_info()
            optimizer_lbfgs.zero_grad(set_to_none=True)
        else:
            _, info = compute_aggregate_fixed(backward=False)
            inverse_gradient_info = empty_inverse_gradient_info()

        info.update(inverse_gradient_info)
        info.update(parameter_update_info)

        lbfgs_lr = optimizer_lbfgs.param_groups[0]["lr"]
        info["lr_network"] = lbfgs_lr
        info["lr_alpha"] = lbfgs_lr
        info["lr_xi_c"] = lbfgs_lr
        info["lr_lambda"] = lbfgs_lr
        info["lbfgs_closure_evals"] = closure_counter["count"]

        epoch = start_epoch + step
        info["epoch"] = epoch
        info["phase"] = "lbfgs"
        history.append(info)

        if step % 10 == 0:
            print(
                f"[L-BFGS] Step {step:5d} "
                f"(epoch~{epoch}) | "
                f"loss={info['loss']:.3e}, "
                f"pde={info['loss_pde']:.3e}, "
                f"data={info['loss_data']:.3e}, "
                f"bc_top={info['loss_bc_top']:.3e}, "
                f"if_u={info['loss_if_u']:.3e}, "
                f"if_N={info['loss_if_N']:.3e}, "
                f"ic={info['loss_ic']:.3e}, "
                f"alpha={info['alpha_hat']:.4f}, "
                f"grad_alpha={info['grad_alpha_phys']:+.3e}, "
                f"delta_alpha={info['delta_alpha_hat']:+.3e}, "
                f"xi1={info['xi1_hat']:.4f}, "
                f"xi2={info['xi2_hat']:.4f}"
            )

    # ========================================================
    # 4. 结果与可视化
    # ========================================================
    residual_check = check_residual_distributions(
        subnetworks=subnetworks,
        get_parameters=get_alpha_xi1_xi2,
        pde_residual=pde_residual_nd,
        top_load=P_star_tau,
        impact_data=impact_data_list[0],
        fixed_points=adam_fixed_points,
        L=L,
        T0=T0,
        tau_max=tau_max,
        n_z=240,
        n_tau=801,
        n_curve=8001,
        batch_size=512,
        tag="training_end",
        show=True,
    )
    last = history[-1]

    alpha_est = last["alpha_hat"]
    xi1_est = last["xi1_hat"]
    xi2_est = last["xi2_hat"]
    xi_c_est = last["xi_c_hat"]
    lam_est = last["lam_hat"]

    print("\n===== Inversion result =====")
    print(
        f"True alpha={alpha_true:.3f}, "
        f"Estimated alpha={alpha_est:.4f}"
    )
    print(
        f"True xi1={xi1_true:.3f}, "
        f"Estimated xi1={xi1_est:.4f}"
    )
    print(
        f"True xi2={xi2_true:.3f}, "
        f"Estimated xi2={xi2_est:.4f}"
    )
    print(
        f"True xi_c={xi_c_true:.3f}, "
        f"Estimated xi_c={xi_c_est:.4f}"
    )
    print(
        f"True lam={lam_true:.3f}, "
        f"Estimated lam={lam_est:.4f}"
    )
    print(
        f"True z1,z2=({z1_true:.2f},{z2_true:.2f}) m"
    )
    print(
        f"Estimated z1,z2="
        f"({xi1_est*L:.2f},{xi2_est*L:.2f}) m"
    )

    epochs = np.asarray(
        [item["epoch"] for item in history],
        dtype=int,
    )

    def history_array(key):
        return np.asarray(
            [item[key] for item in history],
            dtype=float,
        )

    total_losses = history_array("loss")
    pde_losses = history_array("loss_pde")
    data_losses = history_array("loss_data")
    bc_top_losses = history_array("loss_bc_top")
    bc_bottom_losses = history_array("loss_bc_bottom")
    if_u_losses = history_array("loss_if_u")
    if_N_losses = history_array("loss_if_N")
    ic_losses = history_array("loss_ic")

    effective_weight_histories = {
        key: history_array(f"effective_weight_{key}")
        for key in loss_component_keys
    }
    adaptive_multiplier_histories = {
        key: history_array(f"adaptive_multiplier_{key}")
        for key in loss_component_keys
    }

    eps_plot = 1e-16
    total_losses = np.maximum(total_losses, eps_plot)
    pde_losses = np.maximum(pde_losses, eps_plot)
    data_losses = np.maximum(data_losses, eps_plot)
    bc_top_losses = np.maximum(bc_top_losses, eps_plot)
    bc_bottom_losses = np.maximum(
        bc_bottom_losses,
        eps_plot,
    )
    if_u_losses = np.maximum(if_u_losses, eps_plot)
    if_N_losses = np.maximum(if_N_losses, eps_plot)
    ic_losses = np.maximum(ic_losses, eps_plot)

    # --------------------------------------------------------
    # 图1：未加权损失分量
    # --------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.semilogy(
        epochs,
        total_losses,
        label="total",
        linewidth=1.8,
    )
    plt.semilogy(epochs, pde_losses, label="pde")
    plt.semilogy(epochs, data_losses, label="data")
    plt.semilogy(epochs, bc_top_losses, label="bc_top")
    plt.semilogy(
        epochs,
        bc_bottom_losses,
        label="bc_bottom",
    )
    plt.semilogy(
        epochs,
        if_u_losses,
        label="interface_u",
    )
    plt.semilogy(
        epochs,
        if_N_losses,
        label="interface_N",
    )
    plt.semilogy(epochs, ic_losses, label="initial")
    plt.axvline(
        n_epochs_adam,
        linestyle="--",
        linewidth=1.0,
        label="Adam / L-BFGS",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Training loss components "
        "(all pile-head points)"
    )
    plt.legend(ncol=2)
    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )
    plt.tight_layout()

    # --------------------------------------------------------
    # 图2：加权损失分量
    # --------------------------------------------------------
    weighted_pde = (
        effective_weight_histories["pde"] * pde_losses
    )
    weighted_data = (
        effective_weight_histories["data"] * data_losses
    )
    weighted_bc_top = (
        effective_weight_histories["bc_top"] * bc_top_losses
    )
    weighted_bc_bottom = (
        effective_weight_histories["bc_bottom"]
        * bc_bottom_losses
    )
    weighted_if_u = (
        effective_weight_histories["if_u"] * if_u_losses
    )
    weighted_if_N = (
        effective_weight_histories["if_N"] * if_N_losses
    )
    weighted_ic = (
        effective_weight_histories["ic"] * ic_losses
    )

    plt.figure(figsize=(12, 5))
    plt.semilogy(
        epochs,
        total_losses,
        label="total",
        linewidth=1.8,
    )
    plt.semilogy(
        epochs,
        weighted_pde,
        label="weighted pde",
    )
    plt.semilogy(
        epochs,
        weighted_data,
        label="weighted data",
    )
    plt.semilogy(
        epochs,
        weighted_bc_top,
        label="weighted bc_top",
    )
    plt.semilogy(
        epochs,
        weighted_bc_bottom,
        label="weighted bc_bottom",
    )
    plt.semilogy(
        epochs,
        weighted_if_u,
        label="weighted interface_u",
    )
    plt.semilogy(
        epochs,
        weighted_if_N,
        label="weighted interface_N",
    )
    plt.semilogy(
        epochs,
        weighted_ic,
        label="weighted initial",
    )
    plt.axvline(
        n_epochs_adam,
        linestyle="--",
        linewidth=1.0,
        label="Adam / L-BFGS",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Weighted loss")
    plt.title(
        "Weighted training loss components "
        "(gradient-adaptive weights)"
    )
    plt.legend(ncol=2)
    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )
    plt.tight_layout()

    # -------------------------------------------------------

    # --------------------------------------------------------
    # 图4：移动平均总损失
    # --------------------------------------------------------
    ma_window = 500
    total_ma = moving_average(total_losses, ma_window)

    if len(total_losses) >= ma_window:
        epochs_ma = epochs[ma_window - 1:]
    else:
        epochs_ma = epochs

    plt.figure(figsize=(12, 5))
    plt.semilogy(
        epochs,
        total_losses,
        linewidth=0.6,
        alpha=0.40,
        label="Raw total loss",
    )
    plt.semilogy(
        epochs_ma,
        total_ma,
        linewidth=2.0,
        label=f"Moving average, window={ma_window}",
    )
    plt.axvline(
        n_epochs_adam,
        linestyle="--",
        linewidth=1.0,
        label="Adam / L-BFGS",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("Raw and moving-average total loss")
    plt.legend()
    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
    )
    plt.tight_layout()

    # --------------------------------------------------------
    # 图5：反演参数演化
    # --------------------------------------------------------
    alpha_history = history_array("alpha_hat")
    xi1_history = history_array("xi1_hat")
    xi2_history = history_array("xi2_hat")
    xi_c_history = history_array("xi_c_hat")
    lam_history = history_array("lam_hat")

    logit_alpha_history = history_array("logit_alpha")
    theta_c_history = history_array("theta_c")
    theta_l_history = history_array("theta_l")

    grad_logit_alpha_history = history_array(
        "grad_logit_alpha"
    )
    grad_theta_c_history = history_array("grad_theta_c")
    grad_theta_l_history = history_array("grad_theta_l")

    grad_alpha_history = history_array("grad_alpha_phys")
    grad_xi_c_history = history_array("grad_xi_c_phys")
    grad_lam_history = history_array("grad_lam_phys")
    grad_xi1_history = history_array("grad_xi1_phys")
    grad_xi2_history = history_array("grad_xi2_phys")

    map_alpha_gradient_history = history_array(
        "map_dalpha_dlogit"
    )
    map_xic_gradient_history = history_array(
        "map_dxic_dtheta_c"
    )
    map_lam_gradient_history = history_array(
        "map_dlam_dtheta_l"
    )

    delta_alpha_history = history_array("delta_alpha_hat")
    delta_xi1_history = history_array("delta_xi1_hat")
    delta_xi2_history = history_array("delta_xi2_hat")
    delta_xi_c_history = history_array("delta_xi_c_hat")
    delta_lam_history = history_array("delta_lam_hat")
    delta_logit_alpha_history = history_array(
        "delta_logit_alpha"
    )
    delta_theta_c_history = history_array("delta_theta_c")
    delta_theta_l_history = history_array("delta_theta_l")

    lr_network_history = history_array("lr_network")
    lr_alpha_history = history_array("lr_alpha")
    lr_xi_c_history = history_array("lr_xi_c")
    lr_lambda_history = history_array("lr_lambda")
    lbfgs_closure_history = history_array(
        "lbfgs_closure_evals"
    )

    phase_code = np.asarray(
        [
            0 if item["phase"] == "adam" else 1
            for item in history
        ],
        dtype=int,
    )

    parameter_history_columns = [
        epochs,
        phase_code,
        total_losses,
        alpha_history,
        xi1_history,
        xi2_history,
        xi_c_history,
        lam_history,
        logit_alpha_history,
        theta_c_history,
        theta_l_history,
        grad_logit_alpha_history,
        grad_theta_c_history,
        grad_theta_l_history,
        grad_alpha_history,
        grad_xi_c_history,
        grad_lam_history,
        grad_xi1_history,
        grad_xi2_history,
        map_alpha_gradient_history,
        map_xic_gradient_history,
        map_lam_gradient_history,
        delta_alpha_history,
        delta_xi1_history,
        delta_xi2_history,
        delta_xi_c_history,
        delta_lam_history,
        delta_logit_alpha_history,
        delta_theta_c_history,
        delta_theta_l_history,
        lr_network_history,
        lr_alpha_history,
        lr_xi_c_history,
        lr_lambda_history,
        lbfgs_closure_history,
    ]

    parameter_history_names = [
        "epoch",
        "phase_code",
        "total_loss",
        "alpha_hat",
        "xi1_hat",
        "xi2_hat",
        "xi_c_hat",
        "lam_hat",
        "logit_alpha",
        "theta_c",
        "theta_l",
        "grad_logit_alpha",
        "grad_theta_c",
        "grad_theta_l",
        "grad_alpha_phys",
        "grad_xi_c_phys",
        "grad_lam_phys",
        "grad_xi1_phys",
        "grad_xi2_phys",
        "map_dalpha_dlogit",
        "map_dxic_dtheta_c",
        "map_dlam_dtheta_l",
        "delta_alpha_hat",
        "delta_xi1_hat",
        "delta_xi2_hat",
        "delta_xi_c_hat",
        "delta_lam_hat",
        "delta_logit_alpha",
        "delta_theta_c",
        "delta_theta_l",
        "lr_network",
        "lr_alpha",
        "lr_xi_c",
        "lr_lambda",
        "lbfgs_closure_evals",
    ]

    np.savetxt(
        "inverse_parameter_training_history.csv",
        np.column_stack(parameter_history_columns),
        delimiter=",",
        header=",".join(parameter_history_names),
        comments="",
    )

    plt.figure(figsize=(8, 3))
    plt.plot(
        epochs,
        alpha_history,
        label="alpha_hat",
        linewidth=1.2,
    )
    plt.axhline(
        alpha_true,
        color="k",
        linestyle="--",
        linewidth=1.0,
        label="true alpha",
    )
    plt.xlabel("Epoch")
    plt.ylabel("alpha_hat")
    plt.title("Evolution of alpha_hat")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(8, 3))
    plt.plot(
        epochs,
        xi1_history,
        label="Z1_hat",
        linewidth=1.2,
    )
    plt.plot(
        epochs,
        xi2_history,
        label="Z2_hat",
        linewidth=1.2,
    )
    plt.axhline(
        xi1_true,
        color="k",
        linestyle="--",
        linewidth=1.0,
        label="Z1_true",
    )
    plt.axhline(
        xi2_true,
        color="k",
        linestyle=":",
        linewidth=1.0,
        label="Z2_true",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Zi")
    plt.title("Evolution of Z1_hat and Z2_hat")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 图6：Abaqus桩顶各点位速度与加权平均速度
    # --------------------------------------------------------
    t_ms = t_np * 1000.0

    fig, axis = plt.subplots(figsize=(9, 4.2))

    for point_index, point_velocity in enumerate(
        v_top_points_np,
        start=1,
    ):
        axis.plot(
            t_ms,
            point_velocity,
            linewidth=0.85,
            alpha=0.62,
            label=f"Top point {point_index}",
        )

    axis.plot(
        t_ms,
        v_top_np,
        color="black",
        linewidth=2.0,
        label="Weighted point mean (diagnostic)",
        zorder=10,
    )
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel(r"$v_{\mathrm{top}}$ (m/s)")
    axis.set_title(
        "Abaqus "
    )
    axis.grid(True, linestyle="--", alpha=0.40)
    axis.legend(
        loc="best",
        ncol=2 if v_top_points_np.shape[0] > 4 else 1,
    )
    fig.tight_layout()
    fig.savefig(
        "abaqus.png",
        dpi=600,
        bbox_inches="tight",
    )

    # 保存诊断平均曲线和各节点原始曲线，便于复核。
    # 注意：训练直接使用各节点原始曲线，而不是只使用这里的平均曲线。
    output_columns = [
        t_np,
        v_top_np,
    ]
    output_header_names = [
        "time_s",
        "diagnostic_weighted_mean_m_per_s",
    ]

    for point_index, point_velocity in enumerate(
        v_top_points_np,
        start=1,
    ):
        output_columns.append(point_velocity)
        output_header_names.append(
            f"top_point_{point_index}_m_per_s"
        )

    np.savetxt(
        "abaqus_all_pile_head_point_velocities.csv",
        np.column_stack(output_columns),
        delimiter=",",
        header=",".join(output_header_names),
        comments="",
    )

    # --------------------------------------------------------
    # 图7：全部桩顶点位、诊断平均曲线与PINN等效速度
    # --------------------------------------------------------
    for net in subnetworks:
        net.eval()

    velocity_predictions = []
    net1 = subnetworks[0]

    for k in range(n_impacts):
        tau_eval = (
            impact_data_list[k]["tau"]
            .clone()
            .detach()
            .requires_grad_(True)
        )
        xi_top_eval = torch.zeros_like(tau_eval)
        w_top = net1(xi_top_eval, tau_eval)
        w_top_tau = autograd.grad(
            w_top,
            tau_eval,
            grad_outputs=torch.ones_like(w_top),
            create_graph=False,
        )[0]

        v_prediction = (
            w_top_tau
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            * v_scale
        )
        velocity_predictions.append(v_prediction)

    # 桩顶响应图
    fig, axes = plt.subplots(
        n_impacts,
        1,
        figsize=(9, 3.0 * n_impacts),
        sharex=True,
    )

    if n_impacts == 1:
        axes = [axes]

    for k, axis in enumerate(axes):
        forward_velocity = impact_data_list[k]["v_top_np"]
        point_velocities = impact_data_list[k]["v_top_points_np"]
        predicted_velocity = velocity_predictions[k]

        for point_index, point_velocity in enumerate(
            point_velocities,
            start=1,
        ):
            axis.plot(
                t_ms,
                point_velocity,
                color="black",
                linewidth=1.5,
                label=(
                    "Abaqus top points used in loss"
                    if point_index == 1
                    else None
                ),
                zorder=3,
            )
        axis.plot(
            t_ms,
            predicted_velocity,
            "--",
            color="C3",
            label="PINN",
            linewidth=1.3,
            zorder=4,
        )
        axis.set_ylabel(r"$v_{\mathrm{top}}$ (m/s)")
        axis.set_title(
            f"{impact_data_list[k]['name']}: "
            f"Tp={impact_data_list[k]['Tp']*1e3:.1f} ms"
        )
        axis.grid(True, linestyle="--", alpha=0.45)
        axis.legend(loc="upper left")

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(
        "All pile-head point velocities used in PINN training"
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(
        "all_pile_head_points_vs_pinn.png",
        dpi=600,
        bbox_inches="tight",
    )

    # 输出带局部放大窗口的桩顶响应图
    x1_ms = 2.8
    x2_ms = 3.4

    for k in range(n_impacts):
        forward_velocity = impact_data_list[k]["v_top_np"]
        point_velocities = impact_data_list[k]["v_top_points_np"]
        predicted_velocity = velocity_predictions[k]

        fig, axis = plt.subplots(figsize=(6.0, 4.0))

        for point_index, point_velocity in enumerate(
            point_velocities,
            start=1,
        ):
            axis.plot(
                t_ms,
                point_velocity,
                color="black",
                linewidth=1.6,
                label=(
                    "Abaqus top points used in loss"
                    if point_index == 1
                    else None
                ),
                zorder=3,
            )

        axis.plot(
            t_ms,
            predicted_velocity,
            "--",
            color="C3",
            label="PINN equivalent velocity",
            linewidth=1.4,
            zorder=4,
        )

        axis.set_xlabel("Time (ms)")
        axis.set_ylabel(r"$v_{\mathrm{top}}$ (m/s)")
        axis.set_title(
            f"{impact_data_list[k]['name']}: "
            "pile-head velocity"
        )
        axis.grid(True, linestyle="--", alpha=0.45)
        axis.legend(loc="upper left")

        mask = (t_ms >= x1_ms) & (t_ms <= x2_ms)

        # 只有当时间范围内确实存在数据点时才绘制局部放大图
        if np.any(mask):
            # 手动定位到中上部，避免遮挡后期大反射峰
            inset_axis = axis.inset_axes([
                0.30,
                0.47,
                0.35,
                0.30,
            ])

            inset_axis.plot(
                t_ms,
                point_velocities.T,
                color="black",
                linewidth=1.2,
                zorder=3,
            )
            inset_axis.plot(
                t_ms,
                predicted_velocity,
                "--",
                color="C3",
                linewidth=1.1,
                zorder=4,
            )
            inset_axis.set_xlim(x1_ms, x2_ms)

            y_min = min(
                point_velocities[:, mask].min(),
                forward_velocity[mask].min(),
                predicted_velocity[mask].min(),
            )
            y_max = max(
                point_velocities[:, mask].max(),
                forward_velocity[mask].max(),
                predicted_velocity[mask].max(),
            )
            delta_y = max(y_max - y_min, 1e-12)

            inset_axis.set_ylim(
                y_min - 0.08 * delta_y,
                y_max + 0.08 * delta_y,
            )
            inset_axis.grid(
                True,
                linestyle="--",
                alpha=0.35,
            )
            inset_axis.tick_params(labelsize=8)

            mark_inset(
                axis,
                inset_axis,
                loc1=1,
                loc2=3,
                fc="none",
                ec="0.35",
                linewidth=0.8,
            )

        fig.tight_layout()
        fig.savefig(
            "all_pile_head_points_vs_pinn_with_inset.png",
            dpi=600,
            bbox_inches="tight",
        )
    # ========================================================
    # 截面积剖面对比：真实值 vs PINN反演值
    # ========================================================

    # 将无量纲反演位置转换为物理深度
    z1_est_profile = xi1_est * L
    z2_est_profile = xi2_est * L

    def build_abrupt_area_profile(
        pile_length,
        defect_start,
        defect_end,
        area_ratio,
    ):
        """
        构造突变型截面积剖面。

        在 z1 和 z2 处重复深度坐标，使绘图时形成垂直跳变，
        不会把突变界面错误画成平滑或倾斜过渡。
        """
        depth_profile = np.asarray(
            [
                0.0,
                defect_start,
                defect_start,
                defect_end,
                defect_end,
                pile_length,
            ],
            dtype=float,
        )

        normalized_area_profile = np.asarray(
            [
                1.0,
                1.0,
                area_ratio,
                area_ratio,
                1.0,
                1.0,
            ],
            dtype=float,
        )

        return normalized_area_profile, depth_profile

    # 真实面积剖面
    (
        true_area_ratio_profile,
        true_depth_profile,
    ) = build_abrupt_area_profile(
        pile_length=L,
        defect_start=z1_true,
        defect_end=z2_true,
        area_ratio=alpha_true,
    )

    # PINN反演面积剖面
    (
        estimated_area_ratio_profile,
        estimated_depth_profile,
    ) = build_abrupt_area_profile(
        pile_length=L,
        defect_start=z1_est_profile,
        defect_end=z2_est_profile,
        area_ratio=alpha_est,
    )

        # 横向画布
    fig_area, axis_area = plt.subplots(
        figsize=(6,6)
    )

    # 真实面积剖面：横轴深度，纵轴面积比
    axis_area.plot(
        true_depth_profile,
        true_area_ratio_profile,
        color="black",
        linestyle="-",
        linewidth=2.0,
        label="True profile",
        zorder=3,
    )

    # PINN反演面积剖面
    axis_area.plot(
        estimated_depth_profile,
        estimated_area_ratio_profile,
        color="#D62728",
        linestyle="--",
        linewidth=2.0,
        label="PINN inversion",
        zorder=4,
    )

    # 横轴为桩身深度：桩顶0 m，桩底L m
    axis_area.set_xlim(
        0.0,
        L,
    )

    # 纵轴为归一化截面积
    minimum_area_ratio = min(
        alpha_true,
        alpha_est,
    )
    y_lower = max(
        0.0,
        minimum_area_ratio - 0.10,
    )

    axis_area.set_ylim(
        y_lower,
        1.05,
    )

    axis_area.set_xlabel(
        r"Depth $z$ (m)",
        fontsize=11,
    )
    axis_area.set_ylabel(
        r"Normalized cross-sectional area $A(z)/A_0$",
        fontsize=11,
    )
    axis_area.set_title(
        "Cross-sectional area profile",
        fontsize=12,
    )

    axis_area.tick_params(
        axis="both",
        labelsize=10,
    )
    axis_area.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
    )
    axis_area.legend(
        loc="lower right",
        fontsize=10,
        frameon=True,
    )

    # 参数说明框
    parameter_text = (
        rf"True: $\alpha={alpha_true:.3f}$, "
        rf"$z_1={z1_true:.2f}$ m, "
        rf"$z_2={z2_true:.2f}$ m"
        "\n"
        rf"PINN: $\alpha={alpha_est:.3f}$, "
        rf"$z_1={z1_est_profile:.2f}$ m, "
        rf"$z_2={z2_est_profile:.2f}$ m"
    )

    axis_area.text(
        0.02,
        0.05,
        parameter_text,
        transform=axis_area.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.60",
            "alpha": 0.90,
            "boxstyle": "round,pad=0.35",
        },
    )


    plt.show()


if __name__ == "__main__":
    main()