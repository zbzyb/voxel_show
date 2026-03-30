import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib.colors import ListedColormap
import tkinter.messagebox as messagebox
import pandas as pd
import pyvista as pv
import lib


file_path = lib.PathUtl(__file__).parent
lib.load_env(file_path / ".env")


def calculate_plane_parameters(center, dip_deg, strike_deg):
    """计算切面参数"""
    dip_rad = np.radians(dip_deg)
    # strike_rad = np.radians(strike_deg)
    dip_dir = (strike_deg + 90) % 360
    dip_dir_rad = np.radians(dip_dir)
    normal = np.array(
        [
            np.sin(dip_rad) * np.sin(dip_dir_rad),
            np.sin(dip_rad) * np.cos(dip_dir_rad),
            np.cos(dip_rad),
        ]
    )
    normal = normal / np.linalg.norm(normal)
    return normal, np.array(center)


def parse_option(value, option):
    """解析选项，验证值是否符合要求"""
    if option.get("equal") is not None:
        if value not in option["equal"]:
            return False, 0
    if option.get("not_equal") is not None:
        if value in option["not_equal"]:
            return False, 0
    if option.get("max") is not None and isinstance(value, (int, float)):
        if value > option["max"]:
            return False, 0
    if option.get("min") is not None and isinstance(value, (int, float)):
        if value < option["min"]:
            return False, 0
    return True, value


def l_map(idx: int, options: list[dict]):
    """映射索引到选项"""
    visible, re_value = False, 0
    for option in options:
        value = option["db"].search(idx)[0]
        if value == 0:
            break
        visible, re_value = parse_option(value, option)
        if not visible:
            break
    return visible, re_value


def create_labels(grid: lib.Grid3D):
    dtype = np.int64
    nx = int((grid.info.x_max - grid.info.x_min) // grid.info.x_step)
    ny = int((grid.info.y_max - grid.info.y_min) // grid.info.y_step)
    nz = int((grid.info.z_max - grid.info.z_min) // grid.info.z_step)
    cell_label_ids = np.zeros((nx, ny, nz), dtype=dtype)
    return cell_label_ids, nx, ny, nz


def create_volume(
    cell_label_ids: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    grid: lib.Grid3D,
):
    volume = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(grid.info.x_step, grid.info.y_step, grid.info.z_step),
        origin=(
            grid.info.x_min - grid.info.x_step / 2,
            grid.info.y_min - grid.info.y_step / 2,
            grid.info.z_min - grid.info.z_step / 2,
        ),
    )
    volume.cell_data["label_id"] = cell_label_ids.flatten(order="F")
    return volume.threshold(0.5, scalars="label_id")


def show_model(
    context: lib.Context,
    options: list[dict],
    label_map,
    grid: lib.Grid3D,
    opacity: float = 1,
    edge_color: str = "black",
    cmap: str = "viridis",
    line_width: float = 0.5,
    show_edges: bool = False,
    show_scalar_bar: bool = False,
    stype: str = "mesh",
    crinkle: bool = False,
    slices: list = None,
    show_loop_sum: int = 1000000,
):
    struct_labels, cell_nx, cell_ny, cell_nz = create_labels(grid)
    vtk = lib.engine()
    struct_sum_voxel = 0
    for idx, index, _ in grid.enumerate():
        if idx % show_loop_sum == 0:
            context.echo(
                f"结构进度: {idx}/{grid.sum()} ({idx / (grid.sum()) * 100:.2f}%)"
            )
        visible, label = l_map(idx, options)
        if not visible:
            continue
        x_i, y_i, z_i = index
        if 0 <= x_i < cell_nx and 0 <= y_i < cell_ny and 0 <= z_i < cell_nz:
            struct_labels[x_i, y_i, z_i] = label
            struct_sum_voxel += 1

    if struct_sum_voxel == 0:
        context.echo("没有可见地质结构体素")
    else:
        context.echo(f"共发现 {struct_sum_voxel} 个可见地质结构体素")
        struct_volume = create_volume(struct_labels, cell_nx, cell_ny, cell_nz, grid)

        def pick_callback(point):
            """鼠标点击回调：显示被点击位置最近体素的信息"""
            # print("=" * 50)
            # print("🔍 体素点击事件触发")
            # print(f"📍 点击坐标: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
            try:
                label_idx = struct_volume.find_closest_cell(point)
                label = struct_volume.cell_data["label_id"][label_idx]
                root = tk.Tk()
                root.withdraw()
                info = label_map[label]
                show_info = ""
                for key, val in info.items():
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        show_info += f"{key}: {val:.2f}\n"
                    else:
                        show_info += f"{key}: {val}\n"
                messagebox.showinfo(
                    "体素信息",
                    f"标签索引: {label}\n{show_info}",
                )
                root.destroy()

            except Exception as e:
                print(f"\n❌ 获取体素信息失败: {e}\n")

        if stype == "mesh":
            vtk.add_mesh(
                struct_volume,
                cmap=cmap,
                show_edges=show_edges,
                edge_color=edge_color,
                line_width=line_width,
                opacity=opacity,
                interpolate_before_map=False,
                show_scalar_bar=show_scalar_bar,
            )
        elif stype == "box":
            vtk.add_mesh_clip_box(
                struct_volume,
                cmap=cmap,
                show_edges=show_edges,
                edge_color=edge_color,
                line_width=line_width,
                opacity=opacity,
                interpolate_before_map=False,
                show_scalar_bar=show_scalar_bar,
                crinkle=crinkle,
            )
        elif stype == "plane":
            plane_params = {
                "cmap": cmap,
                "show_edges": show_edges,
                "edge_color": edge_color,
                "line_width": line_width,
                "opacity": opacity,
                "interpolate_before_map": False,
                "show_scalar_bar": show_scalar_bar,
                "crinkle": crinkle,
            }
            vtk.add_mesh_clip_plane(struct_volume, **plane_params)
        elif stype == "slice":
            slice_params = {
                "cmap": cmap,
                "show_edges": show_edges,
                "edge_color": edge_color,
                "line_width": line_width,
                "opacity": opacity,
                "interpolate_before_map": False,
                "show_scalar_bar": show_scalar_bar,
            }
            # 如果提供了切片位置参数，添加到参数列表中
            vtk.add_mesh_slice(struct_volume, **slice_params)
        elif stype == "orthogonal":
            orthogonal_params = {
                "cmap": cmap,
                "show_edges": show_edges,
                "edge_color": edge_color,
                "line_width": line_width,
                "opacity": opacity,
                "interpolate_before_map": False,
                "show_scalar_bar": show_scalar_bar,
            }
            vtk.add_mesh_slice_orthogonal(struct_volume, **orthogonal_params)
        elif stype == "slices":
            if slices is None:
                raise ValueError("slices 参数不能为空")
            for slice_info in slices:
                normal, origin = calculate_plane_parameters(
                    [slice_info["x"], slice_info["y"], slice_info["z"]],
                    slice_info["dip"],
                    slice_info["strike"],
                )
                if slice_info["stype"] == "plane":
                    plane_params = {
                        "cmap": cmap,
                        "show_edges": show_edges,
                        "edge_color": edge_color,
                        "line_width": line_width,
                        "opacity": opacity,
                        "interpolate_before_map": False,
                        "show_scalar_bar": show_scalar_bar,
                        "crinkle": crinkle,
                        "normal": normal,
                        "origin": origin,
                    }
                    vtk.add_mesh_clip_plane(struct_volume, **plane_params)
                elif slice_info["stype"] == "slice":
                    slice_params = {
                        "cmap": cmap,
                        "show_edges": show_edges,
                        "edge_color": edge_color,
                        "line_width": line_width,
                        "opacity": opacity,
                        "interpolate_before_map": False,
                        "show_scalar_bar": show_scalar_bar,
                        "normal": normal,
                        "origin": origin,
                    }
                    vtk.add_mesh_slice(struct_volume, **slice_params)
                elif slice_info["stype"] == "orthogonal":
                    orthogonal_params = {
                        "cmap": cmap,
                        "show_edges": show_edges,
                        "edge_color": edge_color,
                        "line_width": line_width,
                        "opacity": opacity,
                        "interpolate_before_map": False,
                        "show_scalar_bar": show_scalar_bar,
                        "normal": normal,
                        "origin": origin,
                    }
                    vtk.add_mesh_slice_orthogonal(struct_volume, **orthogonal_params)
                else:
                    raise ValueError(f"未知的切片类型: {slice_info['stype']}")
        else:
            raise ValueError(f"未知的显示类型: {stype}")
    vtk.enable_point_picking(callback=pick_callback, show_message=True)


def get_file_path(title: str = "选择文件"):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = tk.filedialog.askopenfilename(
        parent=root,
        title=title,
        filetypes=[("Excel 文件", "*.xlsx")],
        initialdir="./",
    )
    root.destroy()
    if not file_path:
        raise FileNotFoundError("未选择文件")
    return file_path


def get_save_file_path(title: str = "选择文件"):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = tk.filedialog.asksaveasfilename(
        parent=root,
        title=title,
        defaultextension=".xlsx",
        filetypes=[("Excel 文件", "*.xlsx")],
        initialdir="./",
    )
    root.destroy()
    if not file_path:
        raise FileNotFoundError("未选择文件")
    if not str(file_path).lower().endswith(".xlsx"):
        file_path = str(file_path) + ".xlsx"
    return file_path


cli = lib.Group(
    "cli",
    desc=(
        "岩性数据处理工具"
        "\n>>> 模型计算的一般过程："
        "\n>>> \t (1)准备“数据表”和“岩性编码对照表”；“数据表”的岩性信息必须是岩性编码"
        "\n>>> \t (2)运行“体素模型计算”，根据“数据表”和“岩性编码对照表”计算体素模型"
        "\n>>> \t (3)根据计算好的模型文件，进行可视化"
    ),
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if s in {"false", "f", "no", "n", "0", "off"}:
        return False
    try:
        return bool(int(s))
    except Exception:
        return False


class ParamsWindows:
    def __init__(self, title, fields, files=None, alias=None, verify=None):
        self.title = title
        self.fields = fields
        self.files = files or {}
        self.alias = alias or {}
        self.values = {}
        self.verify = verify

    def show(self):
        root = tk.Tk()
        root.title(self.title)
        root.attributes("-topmost", True)
        vars = {}
        confirmed = {"ok": False}

        def cfg(container):
            try:
                container.columnconfigure(0, weight=0)
                container.columnconfigure(1, weight=1)
                container.columnconfigure(2, weight=0)
            except Exception:
                pass

        cfg(root)

        def build(container, subfields, path=()):
            row = 0
            for k, v in subfields.items():
                if isinstance(v, dict):
                    lf = tk.LabelFrame(container, text=self.alias.get(k, k))
                    lf.grid(
                        row=row, column=0, padx=8, pady=6, sticky="nsew", columnspan=3
                    )
                    cfg(lf)
                    build(lf, v, path + (k,))
                    row += 1
                else:
                    tk.Label(container, text=self.alias.get(k, k)).grid(
                        row=row, column=0, padx=8, pady=6, sticky="w"
                    )
                    var = tk.StringVar(value=str(v))
                    entry = tk.Entry(container, textvariable=var)
                    entry.grid(row=row, column=1, padx=8, pady=6, sticky="nsew")
                    vars[path + (k,)] = var
                    if k in self.files:
                        mode = self.files[k].get("mode", "open")
                        title = self.files[k].get("title", "选择文件")

                        def browse(var=var, mode=mode, title=title):
                            if mode == "open":
                                p = filedialog.askopenfilename(
                                    parent=root,
                                    title=title,
                                    filetypes=[("Excel 文件", "*.xlsx")],
                                    initialdir="./",
                                )
                            else:
                                p = filedialog.asksaveasfilename(
                                    parent=root,
                                    title=title,
                                    defaultextension=".xlsx",
                                    filetypes=[("Excel 文件", "*.xlsx")],
                                    initialdir="./",
                                )
                            if p:
                                if mode == "save":
                                    if not str(p).lower().endswith(".xlsx"):
                                        p = str(p) + ".xlsx"
                                var.set(p)

                        tk.Button(container, text="浏览", command=browse).grid(
                            row=row, column=2, padx=8, pady=6
                        )
                    row += 1
            return row

        total_rows = build(root, self.fields)

        def on_ok():
            def collect(subfields, path=()):
                out = {}
                for k, v in subfields.items():
                    if isinstance(v, dict):
                        out[k] = collect(v, path + (k,))
                    else:
                        out[k] = vars[path + (k,)].get()
                return out

            self.values = collect(self.fields)
            confirmed["ok"] = True
            root.destroy()

        footer = tk.Frame(root)
        footer.grid(row=total_rows, column=0, columnspan=3, pady=12, sticky="ew")
        footer.columnconfigure(0, weight=1)
        footer.columnconfigure(1, weight=0)
        btn = tk.Button(footer, text="确定", command=on_ok)
        btn.grid(row=0, column=1, padx=(0, 12), sticky="e")
        root.mainloop()
        if not confirmed["ok"]:
            return None
        if self.verify:
            try:
                ok = self.verify(self.values)
                if ok is False:
                    return None
            except Exception as e:
                messagebox.showerror("参数错误", str(e))
                return None
        return self.values


class BuildIndexWindows:
    def show(self):
        """
        build_index:
        - index_option
          - index_file_path
          - index_col
          - index_name_col
        - data_option
          - data_file_path
          - x_col
          - y_col
          - z_col
          - data_name_col
        - BUILD_LOOP_SUM"""
        defaults = {
            "index_option": {
                "index_file_path": "",
                "index_col": lib.getenv("index_col", "岩性编码"),
                "index_name_col": lib.getenv("name_col", "岩性名称"),
            },
            "data_option": {
                "data_file_path": "",
                "x_col": lib.getenv("x_col", "x_local"),
                "y_col": lib.getenv("y_col", "y_local"),
                "z_col": lib.getenv("z_col", "z_local"),
                "data_name_col": lib.getenv("name_col", "岩性名称"),
            },
            "BUILD_LOOP_SUM": str(lib.getenv("BUILD_LOOP_SUM", 1000, int)),
        }
        files = {
            "index_file_path": {"mode": "open", "title": "选择岩性编码对照表"},
            "data_file_path": {"mode": "open", "title": "选择数据文件"},
        }
        alias = {
            "index_option": "岩性编码配置",
            "data_option": "数据表配置",
            "index_file_path": "岩性编码对照表",
            "index_col": "岩性编码列",
            "index_name_col": "岩性名称列",
            "data_file_path": "数据文件",
            "x_col": "X坐标列",
            "y_col": "Y坐标列",
            "z_col": "Z坐标列",
            "data_name_col": "岩性名称列",
            "BUILD_LOOP_SUM": "打印间隔",
        }
        return ParamsWindows(
            "岩性名称转为索引", defaults, files=files, alias=alias, verify=self.verify
        ).show()

    def verify(self, values):
        def v(p, k):
            if isinstance(p, dict) and k in p:
                return p[k]
            for x in p.values():
                if isinstance(x, dict):
                    r = v(x, k)
                    if r is not None:
                        return r
            return None

        required = [
            "index_file_path",
            "index_col",
            "index_name_col",
            "data_file_path",
            "x_col",
            "y_col",
            "z_col",
            "data_name_col",
            "BUILD_LOOP_SUM",
        ]
        missing = [k for k in required if not str(v(values, k) or "").strip()]
        if missing:
            messagebox.showerror("参数错误", f"缺少参数: {', '.join(missing)}")
            return False
        try:
            int(str(v(values, "BUILD_LOOP_SUM") or "1000").strip())
        except Exception:
            messagebox.showerror("参数错误", "打印间隔必须为整数")
            return False
        return True


class CompedParamsWindows:
    def show(self):
        """
        comped
        - grid_option
          - grid_file_path
        - data_option
          - data_file_path
          - x_col
          - y_col
          - z_col
          - index_col
        - COMPED_LOOP_SUM
        """
        defaults = {
            "grid_option": {
                "grid_file_path": "",
            },
            "data_option": {
                "data_file_path": "",
                "x_col": lib.getenv("x_col", "x_local"),
                "y_col": lib.getenv("y_col", "y_local"),
                "z_col": lib.getenv("z_col", "z_local"),
                "index_col": lib.getenv("index_col", "岩性编码"),
            },
            "COMPED_LOOP_SUM": str(lib.getenv("COMPED_LOOP_SUM", 1000000, int)),
            "tol": "0.1",
        }
        files = {
            "grid_file_path": {"mode": "open", "title": "选择网格定义文件"},
            "data_file_path": {"mode": "open", "title": "选择数据文件"},
        }
        alias = {
            "grid_option": "网格定义配置",
            "data_option": "数据配置",
            "grid_file_path": "网格定义文件",
            "data_file_path": "数据文件",
            "x_col": "X坐标列",
            "y_col": "Y坐标列",
            "z_col": "Z坐标列",
            "index_col": "岩性编码列",
            "COMPED_LOOP_SUM": "打印间隔",
            "tol": "容差",
        }
        return ParamsWindows(
            "体素模型计算", defaults, files=files, alias=alias, verify=self.verify
        ).show()

    def verify(self, values):
        def v(p, k):
            if isinstance(p, dict) and k in p:
                return p[k]
            for x in p.values():
                if isinstance(x, dict):
                    r = v(x, k)
                    if r is not None:
                        return r
            return None

        required = [
            "grid_file_path",
            "data_file_path",
            "x_col",
            "y_col",
            "z_col",
            "index_col",
            "COMPED_LOOP_SUM",
            "tol",
        ]
        missing = [k for k in required if not str(v(values, k) or "").strip()]
        if missing:
            messagebox.showerror("参数错误", f"缺少参数: {', '.join(missing)}")
            return False
        try:
            int(str(v(values, "COMPED_LOOP_SUM") or "1000000").strip())
        except Exception:
            messagebox.showerror("参数错误", "打印间隔必须为整数")
            return False
        try:
            float(str(v(values, "tol") or "0.1").strip())
        except Exception:
            messagebox.showerror("参数错误", "容差必须为数值")
            return False
        return True


class BuildColorWindows:
    def show(self):
        """
        build_color:
        - data_option
          - data_file_path
          - name_col
          - color_col
          - index_col(创建)
        - index_option
          - color_save_path(创建)
        """
        defaults = {
            "data_option": {
                "data_file_path": "",
                "name_col": lib.getenv("name_col", "岩性名称"),
                "color_col": lib.getenv("color_col", "颜色"),
                "index_col": lib.getenv("index_col", "岩性编码"),
            },
            "index_option": {
                "color_save_path": "",
            },
            "BUILD_LOOP_SUM": str(lib.getenv("BUILD_LOOP_SUM", 1000, int)),
        }
        files = {
            "data_file_path": {"mode": "open", "title": "选择数据表"},
            "color_save_path": {"mode": "save", "title": "保存颜色编码对照表"},
        }
        alias = {
            "data_option": "数据配置",
            "data_file_path": "数据表",
            "name_col": "岩性名称列",
            "color_col": "颜色列",
            "index_col": "岩性编码列",
            "index_option": "索引配置",
            "color_save_path": "保存颜色编码对照表",
            "BUILD_LOOP_SUM": "打印间隔",
        }
        return ParamsWindows(
            "根据已有数据构建岩性编码对照表",
            defaults,
            files=files,
            alias=alias,
            verify=self.verify,
        ).show()

    def verify(self, values):
        def v(p, k):
            if isinstance(p, dict) and k in p:
                return p[k]
            for x in p.values():
                if isinstance(x, dict):
                    r = v(x, k)
                    if r is not None:
                        return r
            return None

        required = [
            "data_file_path",
            "name_col",
            "color_col",
            "index_col",
            "color_save_path",
            "BUILD_LOOP_SUM",
        ]
        missing = [k for k in required if not str(v(values, k) or "").strip()]
        if missing:
            messagebox.showerror("参数错误", f"缺少参数: {', '.join(missing)}")
            return False
        return True


class ShowWCWindows:
    def show(self):
        """
        show_wc
        - grid_option
          - grid_file_path
        - index_option
          - index_file_path
          - index_col
          - color_col
        - slice_option
          - slice_file_path
          - stype
          - pass_x
          - pass_y
          - pass_z
          - dip_deg
          - strike_deg
        - model_option
          - vmin
          - vmax
          - edges
          - crinkle
        - COMPED_LOOP_SUM
        """
        defaults = {
            "grid_option": {
                "grid_file_path": "",
            },
            "index_option": {
                "index_file_path": "",
                "index_col": lib.getenv("index_col", "岩性编码"),
                "color_col": lib.getenv("color_col", "颜色"),
            },
            "slice_option": {
                "slice_file_path": "",
                "stype": lib.getenv("stype", "stype"),
                "pass_x": lib.getenv("pass_x", "x"),
                "pass_y": lib.getenv("pass_y", "y"),
                "pass_z": lib.getenv("pass_z", "z"),
                "dip_deg": lib.getenv("dip_deg", "dip"),
                "strike_deg": lib.getenv("strike_deg", "strike"),
            },
            "model_option": {
                "vmin": "-1",
                "vmax": "-1",
                "edges": "false",
                "crinkle": "false",
            },
            "COMPED_LOOP_SUM": str(lib.getenv("COMPED_LOOP_SUM", 1000, int)),
        }
        files = {
            "grid_file_path": {"mode": "open", "title": "选择网格定义文件"},
            "index_file_path": {"mode": "open", "title": "选择岩性编码对照表"},
            "slice_file_path": {"mode": "open", "title": "选择切片定义文件"},
        }
        alias = {
            "grid_option": "网格配置",
            "grid_file_path": "网格定义文件",
            "index_option": "索引配置",
            "index_file_path": "岩性编码对照表",
            "index_col": "岩性编码列",
            "color_col": "颜色列",
            "slice_option": "切片配置",
            "slice_file_path": "切片定义文件",
            "stype": "切片表-类型列",
            "pass_x": "切片表-x列",
            "pass_y": "切片表-y列",
            "pass_z": "切片表-z列",
            "dip_deg": "切片表-倾角列",
            "strike_deg": "切片表-方位角列",
            "model_option": "模型配置",
            "vmin": "最小值",
            "vmax": "最大值",
            "edges": "显示体素边框",
            "crinkle": "不剖切体素",
            "COMPED_LOOP_SUM": "打印间隔",
        }
        return ParamsWindows(
            "显示带切片的体素模型",
            defaults,
            files=files,
            alias=alias,
            verify=self.verify,
        ).show()

    def verify(self, values):
        def v(p, k):
            if isinstance(p, dict) and k in p:
                return p[k]
            for x in p.values():
                if isinstance(x, dict):
                    r = v(x, k)
                    if r is not None:
                        return r
            return None

        required = [
            "grid_file_path",
            "index_file_path",
            "index_col",
            "color_col",
            "slice_file_path",
            "stype",
            "pass_x",
            "pass_y",
            "pass_z",
            "dip_deg",
            "strike_deg",
            "vmin",
            "vmax",
            "edges",
            "crinkle",
            "COMPED_LOOP_SUM",
        ]
        missing = [k for k in required if not str(v(values, k) or "").strip()]
        if missing:
            messagebox.showerror("参数错误", f"缺少参数: {', '.join(missing)}")
            return False
        try:
            int(str(v(values, "vmin") or "-1").strip())
            int(str(v(values, "vmax") or "-1").strip())
        except Exception:
            messagebox.showerror("参数错误", "最小值/最大值必须为整数")
            return False
        parse_bool(str(v(values, "edges") or "false"))
        parse_bool(str(v(values, "crinkle") or "true"))
        return True


class ShowOriWindows:
    def show(self):
        """
        show_ori
        - grid_option
          - grid_file_path
        - index_option
          - index_file_path
          - index_col
          - color_col
        - model_option
          - vmin
          - vmax
          - edges
          - crinkle
        - COMPED_LOOP_SUM
        """
        defaults = {
            "grid_option": {
                "grid_file_path": "",
            },
            "index_option": {
                "index_file_path": "",
                "index_col": lib.getenv("index_col", "岩性编码"),
                "color_col": lib.getenv("color_col", "颜色"),
            },
            "model_option": {
                "vmin": "-1",
                "vmax": "-1",
                "edges": "false",
                "crinkle": "false",
            },
            "COMPED_LOOP_SUM": str(lib.getenv("COMPED_LOOP_SUM", 1000, int)),
        }
        files = {
            "grid_file_path": {"mode": "open", "title": "选择网格定义文件"},
            "index_file_path": {"mode": "open", "title": "选择岩性编码对照表"},
        }
        alias = {
            "grid_option": "网格配置",
            "grid_file_path": "网格定义文件",
            "index_option": "索引配置",
            "index_file_path": "岩性编码对照表",
            "index_col": "岩性编码列",
            "color_col": "颜色列",
            "model_option": "模型配置",
            "vmin": "最小值",
            "vmax": "最大值",
            "edges": "显示体素边框",
            "crinkle": "不剖切体素",
            "COMPED_LOOP_SUM": "打印间隔",
        }
        return ParamsWindows(
            "显示体素模型",
            defaults,
            files=files,
            alias=alias,
            verify=self.verify,
        ).show()

    def verify(self, values):
        def v(p, k):
            if isinstance(p, dict) and k in p:
                return p[k]
            for x in p.values():
                if isinstance(x, dict):
                    r = v(x, k)
                    if r is not None:
                        return r
            return None

        required = [
            "grid_file_path",
            "index_file_path",
            "index_col",
            "color_col",
            "vmin",
            "vmax",
            "edges",
            "crinkle",
            "COMPED_LOOP_SUM",
        ]
        missing = [k for k in required if not str(v(values, k) or "").strip()]
        if missing:
            messagebox.showerror("参数错误", f"缺少参数: {', '.join(missing)}")
            return False
        try:
            int(str(v(values, "vmin") or "-1").strip())
            int(str(v(values, "vmax") or "-1").strip())
        except Exception:
            messagebox.showerror("参数错误", "最小值/最大值必须为整数")
            return False
        parse_bool(str(v(values, "edges") or "false"))
        parse_bool(str(v(values, "crinkle") or "true"))
        return True


@cli.command("build_index")
def build_index(context: lib.Context):
    """(工具)根据岩性名称，依照岩性编码对照表，将岩性名称转为岩性编码，并在数据文件中添加“岩性编码列”"""
    params = BuildIndexWindows().show()
    if params is None:
        return

    def v(p, k):
        if isinstance(p, dict) and k in p:
            return p[k]
        for x in p.values():
            if isinstance(x, dict):
                r = v(x, k)
                if r is not None:
                    return r
        return None

    # index_file
    index_file_path = v(params, "index_file_path")
    lib.Verify.include_sheet(index_file_path, sheet_name=["info"])
    index_col = v(params, "index_col")
    index_name_col = v(params, "index_name_col")
    index_df = pd.read_excel(index_file_path, sheet_name="info")
    lib.Verify.include_column(index_df, column_name=[index_col, index_name_col])
    idx_map = {
        item[index_name_col]: {**item} for item in index_df.to_dict(orient="records")
    }
    # data_file
    data_file_path = lib.PathUtl(v(params, "data_file_path"))
    x_col = v(params, "x_col")
    y_col = v(params, "y_col")
    z_col = v(params, "z_col")
    data_name_col = v(params, "data_name_col")
    lib.Verify.include_sheet(str(data_file_path), sheet_name=["info"])
    data_df = pd.read_excel(str(data_file_path), sheet_name="info")
    lib.Verify.include_column(data_df, column_name=[x_col, y_col, z_col, data_name_col])
    idx_col_data = []
    sum_len = len(data_df)
    build_loop_sum = int(v(params, "BUILD_LOOP_SUM"))
    for index, row in data_df.iterrows():
        if index % build_loop_sum == 0:
            context.echo(f"已处理 {index} / {sum_len} 行数据")
        name = row[data_name_col]
        if name in idx_map:
            idx_col_data.append(idx_map[name][index_col])
        else:
            context.echo(f"数据文件中第{index + 1}行名称{name}不在索引文件中")
            idx_col_data.append(None)
    data_df[index_col] = idx_col_data
    output_path = data_file_path.parent / (data_file_path.stem + "_index.xlsx")
    data_df.to_excel(str(output_path), sheet_name="info", index=False)
    context.echo(f"已保存数据文件到 {output_path}")


@cli.command("build_color")
def build_color(context: lib.Context):
    """(工具)根据数据文件的岩性名称列和颜色列，构建岩性编码对照表"""
    params = BuildColorWindows().show()
    if params is None:
        return

    def v(p, k):
        if isinstance(p, dict) and k in p:
            return p[k]
        for x in p.values():
            if isinstance(x, dict):
                r = v(x, k)
                if r is not None:
                    return r
        return None

    # data_file
    data_file_path = lib.PathUtl(v(params, "data_file_path"))
    lib.Verify.include_sheet(str(data_file_path), sheet_name=["info"])
    data_df = pd.read_excel(str(data_file_path), sheet_name="info")
    name_col = v(params, "name_col")
    color_col = v(params, "color_col")
    lib.Verify.include_column(data_df, column_name=[name_col, color_col])
    color_map = []
    label_index = []
    build_loop_sum = int(v(params, "BUILD_LOOP_SUM"))
    sum_len = len(data_df)
    for index, item in enumerate(data_df.to_dict(orient="records")):
        if index % build_loop_sum == 0:
            context.echo(f"已处理 {index} / {sum_len} 行数据")
        if (item[name_col], item[color_col]) not in color_map:
            color_map.append((item[name_col], item[color_col]))
        label_index.append(color_map.index((item[name_col], item[color_col])) + 1)
    index_col = v(params, "index_col")
    if index_col in data_df.columns:
        index_col = f"{index_col}_index"
    data_df[index_col] = label_index
    color_map_df = pd.DataFrame(
        [
            {index_col: index + 1, name_col: item[0], color_col: item[1]}
            for index, item in enumerate(color_map)
        ],
        columns=[index_col, name_col, color_col],
    )
    color_path = v(params, "color_save_path")
    color_map_df.to_excel(str(color_path), sheet_name="info", index=False)
    data_df.to_excel(str(data_file_path), sheet_name="info", index=False)


@cli.command("comped")
def comped(
    context: lib.Context,
):
    """(模型计算)体素模型计算"""
    params = CompedParamsWindows().show()
    if params is None:
        return

    def v(p, k):
        if isinstance(p, dict) and k in p:
            return p[k]
        for x in p.values():
            if isinstance(x, dict):
                r = v(x, k)
                if r is not None:
                    return r
        return None

    # grid_file
    grid_file_path = v(params, "grid_file_path")
    lib.Verify.include_sheet(str(grid_file_path), sheet_name=["info"])
    grid_df = pd.read_excel(str(grid_file_path), sheet_name="info").to_dict(
        orient="records"
    )
    if len(grid_df) == 0:
        context.echo(f"{grid_file_path} 文件中没有数据")
        return
    grid_info = grid_df[0]
    grid_info["x_min"] -= grid_info["x_step"]
    grid_info["y_min"] -= grid_info["y_step"]
    grid_info["z_min"] -= grid_info["z_step"]
    grid_info["x_max"] += grid_info["x_step"]
    grid_info["y_max"] += grid_info["y_step"]
    grid_info["z_max"] += grid_info["z_step"]
    grid = lib.Grid3D.from_step(**grid_info)

    print(grid.sum())
    context.echo("成功加载网格模型")

    x_col = v(params, "x_col")
    y_col = v(params, "y_col")
    z_col = v(params, "z_col")
    index_col = v(params, "index_col")

    # data_file
    data_file_path = v(params, "data_file_path")
    lib.Verify.include_sheet(str(data_file_path), sheet_name=["info"])
    data_df = pd.read_excel(str(data_file_path), sheet_name="info")
    lib.Verify.include_column(data_df, column_name=[x_col, y_col, z_col, index_col])
    data_df[x_col] = pd.to_numeric(data_df[x_col], errors="coerce")
    data_df[y_col] = pd.to_numeric(data_df[y_col], errors="coerce")
    data_df[z_col] = pd.to_numeric(data_df[z_col], errors="coerce")
    if data_df[[x_col, y_col, z_col, index_col]].isnull().values.any():
        context.echo(f"{data_file_path} 文件中存在空值，请检查数据")
        return
    # 按照xyz组合查询数据(误差tol)
    file_db = lib.FileDB(
        str(lib.PathUtl(grid_file_path).parent / "model.cube"), fmt="=q"
    )("wb")
    sum_len = grid.sum()
    show_sum = 0
    comped_loop_sum = int(v(params, "COMPED_LOOP_SUM"))
    tol_s = v(params, "tol")
    try:
        tol = float(str(tol_s).strip())
    except Exception:
        tol = 0.1
    for idx, _, point in grid.enumerate():
        if idx % comped_loop_sum == 0:
            context.echo(f"已处理 {idx} / {sum_len} 个点")
        x, y, z = point
        value = 0
        cond = (
            ((data_df[x_col] - x).abs() <= tol)
            & ((data_df[y_col] - y).abs() <= tol)
            & ((data_df[z_col] - z).abs() <= tol)
        )
        df3 = data_df[cond]
        if len(df3) != 0:
            value = df3[index_col].values[0]
            show_sum += 1
        file_db.append([value])
    context.echo(f"有{show_sum}个点有数据")
    context.echo("计算完成")


@cli.command("show_wc", short=False)
def show_wc(
    context: lib.Context,
):
    """(可视化)显示体素模型"""
    params = ShowWCWindows().show()
    if params is None:
        return

    def v(p, k):
        if isinstance(p, dict) and k in p:
            return p[k]
        for x in p.values():
            if isinstance(x, dict):
                r = v(x, k)
                if r is not None:
                    return r
        return None

    # grid_file
    grid_file_path = v(params, "grid_file_path")
    lib.Verify.include_sheet(str(grid_file_path), sheet_name=["info"])
    grid_df = pd.read_excel(str(grid_file_path), sheet_name="info").to_dict(
        orient="records"
    )
    if len(grid_df) == 0:
        context.echo(f"{grid_file_path} 文件中没有数据")
        return
    grid_info = grid_df[0]
    grid_info["x_min"] -= grid_info["x_step"]
    grid_info["y_min"] -= grid_info["y_step"]
    grid_info["z_min"] -= grid_info["z_step"]
    grid_info["x_max"] += grid_info["x_step"]
    grid_info["y_max"] += grid_info["y_step"]
    grid_info["z_max"] += grid_info["z_step"]
    grid = lib.Grid3D.from_step(**grid_info)
    context.echo("成功加载网格模型")
    options = [
        {
            "db": lib.FileDB(
                str(lib.PathUtl(grid_file_path).parent / "model.cube"), fmt="=q"
            )("rb"),
        }
    ]
    try:
        vmin = int(str(v(params, "vmin")).strip())
    except Exception:
        vmin = -1
    try:
        vmax = int(str(v(params, "vmax")).strip())
    except Exception:
        vmax = -1
    if vmin != -1:
        options[0]["min"] = vmin
    if vmax != -1:
        options[0]["max"] = vmax
    # index_file
    index_file_path = v(params, "index_file_path")
    lib.Verify.include_sheet(str(index_file_path), sheet_name=["info"])
    df3 = pd.read_excel(str(index_file_path), sheet_name="info")
    if len(df3) == 0:
        context.echo(f"{index_file_path} 文件中没有数据")
        return
    index_col = v(params, "index_col")
    color_col = v(params, "color_col")
    lib.Verify.include_column(df3, column_name=[index_col, color_col])
    index_map = {item[index_col]: {**item} for item in df3.to_dict(orient="records")}
    # 使用matplotlib创建颜色映射
    cmap = ListedColormap([item[color_col] for item in df3.to_dict(orient="records")])

    slice_file_path = v(params, "slice_file_path")
    lib.Verify.include_sheet(str(slice_file_path), sheet_name=["info"])
    slice_df = pd.read_excel(str(slice_file_path), sheet_name="info")
    stype = v(params, "stype")
    pass_x = v(params, "pass_x")
    pass_y = v(params, "pass_y")
    pass_z = v(params, "pass_z")
    dip_deg = v(params, "dip_deg")
    strike_deg = v(params, "strike_deg")
    lib.Verify.include_column(
        slice_df, column_name=[stype, pass_x, pass_y, pass_z, dip_deg, strike_deg]
    )
    edges = parse_bool(v(params, "edges"))
    crinkle = parse_bool(v(params, "crinkle"))
    comped_loop_sum = int(v(params, "COMPED_LOOP_SUM"))
    show_model(
        context,
        options,
        index_map,
        grid,
        cmap=cmap,
        show_scalar_bar=True,
        show_edges=edges,
        stype="slices",
        crinkle=crinkle,
        slices=slice_df.to_dict(orient="records"),
        show_loop_sum=comped_loop_sum,
    )
    context.echo("显示完成")


@cli.command("show_ori", short=False)
def show_ori(
    context: lib.Context,
):
    """(可视化)显示体素模型的方向"""
    params = ShowOriWindows().show()
    if params is None:
        return

    def v(p, k):
        if isinstance(p, dict) and k in p:
            return p[k]
        for x in p.values():
            if isinstance(x, dict):
                r = v(x, k)
                if r is not None:
                    return r
        return None

    # grid_file
    grid_file_path = v(params, "grid_file_path")
    lib.Verify.include_sheet(str(grid_file_path), sheet_name=["info"])
    grid_df = pd.read_excel(str(grid_file_path), sheet_name="info").to_dict(
        orient="records"
    )
    if len(grid_df) == 0:
        context.echo(f"{grid_file_path} 文件中没有数据")
        return
    grid_info = grid_df[0]
    grid_info["x_min"] -= grid_info["x_step"]
    grid_info["y_min"] -= grid_info["y_step"]
    grid_info["z_min"] -= grid_info["z_step"]
    grid_info["x_max"] += grid_info["x_step"]
    grid_info["y_max"] += grid_info["y_step"]
    grid_info["z_max"] += grid_info["z_step"]
    grid = lib.Grid3D.from_step(**grid_info)
    context.echo("成功加载网格模型")
    options = [
        {
            "db": lib.FileDB(
                str(lib.PathUtl(grid_file_path).parent / "model.cube"), fmt="=q"
            )("rb"),
        }
    ]
    try:
        vmin = int(str(v(params, "vmin")).strip())
    except Exception:
        vmin = -1
    try:
        vmax = int(str(v(params, "vmax")).strip())
    except Exception:
        vmax = -1
    if vmin != -1:
        options[0]["min"] = vmin
    if vmax != -1:
        options[0]["max"] = vmax
    # index_file
    index_file_path = v(params, "index_file_path")
    lib.Verify.include_sheet(str(index_file_path), sheet_name=["info"])
    index_df = pd.read_excel(str(index_file_path), sheet_name="info")
    if len(index_df) == 0:
        context.echo(f"{index_file_path} 文件中没有数据")
        return
    index_col = v(params, "index_col")
    color_col = v(params, "color_col")
    lib.Verify.include_column(index_df, column_name=[index_col, color_col])
    index_map = {
        item[index_col]: {**item} for item in index_df.to_dict(orient="records")
    }
    # 使用matplotlib创建颜色映射
    cmap = ListedColormap(
        [item[color_col] for item in index_df.to_dict(orient="records")]
    )
    edges = parse_bool(v(params, "edges"))
    stype = "mesh"
    crinkle = parse_bool(v(params, "crinkle"))
    comped_loop_sum = int(v(params, "COMPED_LOOP_SUM"))
    show_model(
        context,
        options,
        index_map,
        grid,
        cmap=cmap,
        show_scalar_bar=True,
        show_edges=edges,
        stype=stype,
        crinkle=crinkle,
        show_loop_sum=comped_loop_sum,
    )
    context.echo("显示完成")


is_fixed = False


@cli.command("fixed", short=False)
def fixed(
    context: lib.Context,
):
    """(视图操作)固定/取消固定旋转"""
    global is_fixed
    vtk = lib.engine()
    try:
        if is_fixed:
            if hasattr(vtk, "enable_trackball_style"):
                vtk.enable_trackball_style()
            is_fixed = False
            context.echo("已启用旋转")
        else:
            if hasattr(vtk, "enable_image_style"):
                vtk.enable_image_style()
            is_fixed = True
            context.echo("已禁用旋转")
    except Exception:
        pass


is_hidden_widget = False


@cli.command("hidden_widget", short=False)
def hidden_widget(
    context: lib.Context,
):
    """(视图操作)关闭/显示切面交互部件"""
    global is_hidden_widget
    # 隐藏切面交互线框
    vtk = lib.engine()
    try:
        if hasattr(vtk, "plane_widgets"):
            for widget in vtk.plane_widgets:
                opacity = 0 if not is_hidden_widget else 1
                widget.GetNormalProperty().SetOpacity(opacity)
                widget.GetSelectedNormalProperty().SetOpacity(opacity)
                widget.GetOutlineProperty().SetOpacity(opacity)
                widget.GetSelectedOutlineProperty().SetOpacity(opacity)
                if hasattr(widget, "GetEdgesProperty"):
                    widget.GetEdgesProperty().SetOpacity(opacity)
                if hasattr(widget, "GetSelectedEdgesProperty"):
                    widget.GetSelectedEdgesProperty().SetOpacity(opacity)
    except Exception:
        pass

    if not is_hidden_widget:
        is_hidden_widget = True
        context.echo("已隐藏切面交互部件")
    else:
        is_hidden_widget = False
        context.echo("已显示切面交互部件")


@cli.command("get_view", short=False)
def get_view(
    context: lib.Context,
):
    """(视图操作)获取当前视角参数"""
    vtk = lib.engine()
    try:
        cam = None
        if hasattr(vtk, "camera") and vtk.camera is not None:
            cam = vtk.camera
        elif hasattr(vtk, "renderer") and vtk.renderer is not None:
            cam = vtk.renderer.GetActiveCamera()
        elif hasattr(vtk, "plotter") and hasattr(vtk.plotter, "camera"):
            cam = vtk.plotter.camera
        if cam is not None:
            # 计算真实的 azimuth 和 elevation
            fp = np.array(cam.GetFocalPoint())
            pos = np.array(cam.GetPosition())
            view_vec = fp - pos
            dist = np.linalg.norm(view_vec)
            if dist > 0:
                view_vec /= dist
            elevation = np.rad2deg(np.arcsin(view_vec[2]))
            azimuth = np.rad2deg(np.arctan2(view_vec[1], view_vec[0]))
            context.echo(f"当前视角 - 水平角: {azimuth:.2f}, 垂直角: {elevation:.2f}")

    except Exception as e:
        context.echo(f"获取视角失败: {e}")


@cli.command("to_view", short=False)
def to_view(
    context: lib.Context,
):
    """(视图操作)设置当前视角参数"""
    params = ParamsWindows(
        "设置视角",
        {"model_option": {"azimuth": "0", "elevation": "0"}},
        alias={
            "model_option": "显示配置",
            "azimuth": "水平角(度)",
            "elevation": "垂直角(度)",
        },
    ).show()
    if params is None:
        return

    def v(p, k):
        if isinstance(p, dict) and k in p:
            return p[k]
        for x in p.values():
            if isinstance(x, dict):
                r = v(x, k)
                if r is not None:
                    return r
        return None

    vtk = lib.engine()
    try:
        az = float(str(v(params, "azimuth") or "0").strip())
    except Exception:
        az = 0.0
    try:
        el = float(str(v(params, "elevation") or "0").strip())
    except Exception:
        el = 0.0
    try:
        cam = None
        if hasattr(vtk, "camera") and vtk.camera is not None:
            cam = vtk.camera
        elif hasattr(vtk, "renderer") and vtk.renderer is not None:
            cam = vtk.renderer.GetActiveCamera()
        elif hasattr(vtk, "plotter") and hasattr(vtk.plotter, "camera"):
            cam = vtk.plotter.camera
        if cam is not None:
            try:
                fp = np.array(cam.GetFocalPoint(), dtype=float)
                pos = np.array(cam.GetPosition(), dtype=float)
                r = float(np.linalg.norm(pos - fp))
                if not np.isfinite(r) or r <= 1e-9:
                    r = 1.0
                azr = np.deg2rad(az)
                elr = np.deg2rad(el)
                dirx = np.cos(elr) * np.cos(azr)
                diry = np.cos(elr) * np.sin(azr)
                dirz = np.sin(elr)
                dirv = np.array([dirx, diry, dirz], dtype=float)
                new_pos = fp - r * dirv
                cam.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
                if abs(dirz) > 0.99:
                    cam.SetViewUp(0.0, 1.0, 0.0)
                else:
                    cam.SetViewUp(0.0, 0.0, 1.0)
                if hasattr(vtk, "renderer") and vtk.renderer is not None:
                    try:
                        vtk.renderer.ResetCameraClippingRange()
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    context.echo("已设置视角")


if __name__ == "__main__":
    lib.run_cmd(cli, ignore_start=True)
