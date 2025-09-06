
# This script uses Python 3.10.11

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'generate_color_mesh',
    'plt_GBZ_slice',
    'plt_GBZ_xy',
    'plt_record_loss',
]

# change default dtype of torch.tensor to float64
torch.set_default_dtype(torch.float64)











def generate_color_mesh(L1, L2, center_gray=0.85):
    """
    Generate a 2D RGB mesh with size (L1, L2, 3).
    This RGB mesh represents the projection of an RGB cube from the viewpoint of (1, 1, 1).
    The original projection forms a hexagon shape. 
    The mesh is created by dividing a rectangle into six parts, with each part mapping to 1/6 of the hexagon. 
    The center color is set to a gray value (not pure white) to ensure better visual distinction in plots.

    Params:
        L1/L2 (int): Size of the first/second dimension of the mesh.
        center_gray (float): Intensity of the gray color at the center of the mesh.
            Should be a float in the range [0, 1].

    Returns:
        (torch.tensor): The RGB mesh tensor of shape (L1, L2, 3).
    """

    X, Y = torch.meshgrid(torch.linspace(0, 1, L1),
                          torch.linspace(0, 1, L2), indexing='ij')
    R = torch.zeros_like(X)
    G = torch.zeros_like(X)
    B = torch.zeros_like(X)

    lam = center_gray

    mask1 = (X <= 0.5) & (Y <= X)
    R[mask1] = 1 - 2*(1-lam)*Y[mask1]
    G[mask1] = 2*X[mask1] - 2*(1-lam)*Y[mask1]
    B[mask1] = 2*lam*Y[mask1]

    mask2 = (X <= 0.5) & (Y > X) & (Y <= -X+1)
    R[mask2] = 1 - 2*(1-lam)*X[mask2]
    G[mask2] = 2*lam*X[mask2]
    B[mask2] = (2*lam-1)*X[mask2] + Y[mask2]

    mask3 = (X <= 0.5) & (Y > -X+1)
    R[mask3] = 1+2*lam - 2*X[mask3] - 2*lam*Y[mask3]
    G[mask3] = 2*lam - 2*lam*Y[mask3]
    B[mask3] = 2*lam-1 + 2*(1-lam)*Y[mask3]

    mask4 = (X > 0.5) & (Y <= -X+1)
    R[mask4] = 2 - 2*X[mask4] - 2*(1-lam)*Y[mask4]
    G[mask4] = 1 - 2*(1-lam)*Y[mask4]
    B[mask4] = 2*lam*Y[mask4]

    mask5 = (X > 0.5) & (Y > -X+1) & (Y <= X)
    R[mask5] = 2*lam - 2*lam*X[mask5]
    G[mask5] = 2*lam-1 + 2*(1-lam)*X[mask5]
    B[mask5] = 2*lam-1 + (1-2*lam)*X[mask5] + Y[mask5]

    mask6 = (X > 0.5) & (Y > X)
    R[mask6] = 2*lam - 2*lam*Y[mask6]
    G[mask6] = 2*lam-1 + 2*X[mask6] - 2*lam*Y[mask6]
    B[mask6] = 2*lam-1 + 2*(1-lam)*Y[mask6]

    RGB = torch.stack([R, G, B], dim=-1)
    RGB = torch.clip(RGB, 0, 1)
    
    return RGB








def init_ax(ax, pane_color, view_params):
    """
    Initialize the appearance of a 3D axis.

    Params:
        ax (matplotlib.axes._axes.Axes): The 3D axis object to initialize.
        pane_color (str or tuple): The color to set for the axis panes.
        view_params (tuple): A tuple controling the view of the axes.
            Should have the form (elev, azim)
    """

    for axis in ['xaxis', 'yaxis', 'zaxis']:
        getattr(ax, axis).pane.set_color(pane_color)
    ax.view_init(elev=view_params[0], azim=view_params[1])









def retouch_ax(ax, labels, limits, ticksteps, title):
    """
    Retouch the appearance of a 3D axis.

    Params:
        ax (matplotlib.axes._axes.Axes): The 3D axis object to retouch.
        labels (dict): Dictionary with keys 'xlabel', 'ylabel', 'zlabel'.
        limits (dict): Dictionary with keys 'xlim', 'ylim', 'zlim', each as (min, max).
        ticksteps (dict): Dictionary with keys 'xstep', 'ystep', 'zstep'
        title (str): Title of the axis.
    """
    
    labelsize = 12
    ticksize = 9
    titlesize = 12

    ax.set_xlabel(labels['xlabel'], fontsize=labelsize)
    ax.set_ylabel(labels['ylabel'], fontsize=labelsize)
    ax.set_zlabel(labels['zlabel'], fontsize=labelsize)

    def change_lim(lim, get_lim):
        min_lim = get_lim()[0] if lim[0] is None else lim[0]
        max_lim = get_lim()[1] if lim[1] is None else lim[1]
        return (min_lim, max_lim)
    limits['xlim'] = change_lim(limits['xlim'], ax.get_xlim)
    limits['ylim'] = change_lim(limits['ylim'], ax.get_ylim)
    limits['zlim'] = change_lim(limits['zlim'], ax.get_zlim)

    xlim, ylim, zlim = limits['xlim'], limits['ylim'], limits['zlim']
    ax.set_xticks(np.arange(np.floor(xlim[0]), np.ceil(xlim[1])+1, ticksteps['xstep']))
    ax.set_yticks(np.arange(np.floor(ylim[0]), np.ceil(ylim[1])+1, ticksteps['ystep']))
    ax.set_zticks(np.arange(np.floor(zlim[0]), np.ceil(zlim[1])+1, ticksteps['zstep']))
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.set_title(title, fontdict={'fontsize': titlesize})
    # ax.axis('equal')
    ax.set(xlim=xlim, ylim=ylim, zlim=zlim)








def plt_GBZ_slice(GBZ, Spectra, slice_dir='t1', color_scheme='1D', cropping=True):
    """
    Plot a slice of the GBZ tensor along a specified direction.

    Params:
        GBZ (torch.tensor): Tensor of shape (Nt1, Nt2, 4). 
            GBZ is typically the output from the encoder.
        slice_dir (str): Direction of the slice to plot, which can be either:
            - 't1': Slice along the first dimension, plotting with a fixed t1 value and varying t2.
            - 't2': Slice along the second dimension, plotting with a fixed t2 value and varying t1.
        color_scheme (str): Scheme for coloring the scatter plot, which can be:
            - '1D': Use a rainbow color scheme to represent varying directions.
            - '2D': Use a 2D color mesh to represent different values of t1 and t2.
        cropping (bool): Whether to crop the GBZ tensor by removing the last row and column. 
            This is useful because the GBZ (output of the encoder) is periodic in t1 and t2 dimensions,
            causing the first and last rows/columns to overlap.
            Cropping helps avoid color overlap in the plot.
    """

    assert GBZ.dim() == 3
    assert GBZ.size(2) == 4
    assert slice_dir.lower() in ['t1', 't2'], "Parameter slice_dir can only be 't1' or 't2'."
    assert color_scheme.lower() in ['1d', '2d'], "Parameter color_scheme can only be '1D' and '2D'."

    Nt1, Nt2 = GBZ.size(0)-1, GBZ.size(1)-1
    if cropping:
        GBZ = GBZ[:-1, :-1, :]
        Spectra = Spectra[:-1, :-1, :]
    Lt1, Lt2 = GBZ.size(0), GBZ.size(1)
    cmap = plt.get_cmap('rainbow')

    if slice_dir.lower() == 't1':
        Z = torch.tensor([[i] * Lt2 for i in range(Lt1)]) / Nt1
        dir_label = r'$t_1$'
        colors = cmap(np.linspace(0, 1, Lt2))[:, :3]
        RGB = torch.tensor(colors).unsqueeze(0).repeat(Lt1, 1, 1)
    else:
        Z = torch.tensor([[i] * Lt1 for i in range(Lt2)]).T / Nt2
        dir_label = r'$t_2$'
        colors = cmap(np.linspace(0, 1, Lt1))[:, :3]
        RGB = torch.tensor(colors).unsqueeze(1).repeat(1, Lt2, 1)

    if color_scheme.lower() == '2d':
        RGB = generate_color_mesh(Lt1, Lt2)

    fig = plt.figure(figsize=(14.6, 4.2))
    ax1 = fig.add_axes([0.12, 0.05, 0.25, 0.84], projection='3d')
    ax2 = fig.add_axes([0.41, 0.05, 0.25, 0.84], projection='3d')
    ax3 = fig.add_axes([0.70, 0.05, 0.25, 0.84], projection='3d')
    axc = fig.add_axes([0.04, 0.30, 0.05, 0.40])

    init_ax(ax1, pane_color='white', view_params=(25, -65))
    init_ax(ax2, pane_color='white', view_params=(25, -65))
    init_ax(ax3, pane_color='white', view_params=(25, -65))

    # betax
    ax1.scatter(GBZ[:, :, 0].ravel(), GBZ[:, :, 1].ravel(), Z.ravel(),
                s=15, c=RGB.reshape(-1, 3), edgecolors=None, alpha=1)
    retouch_ax(ax1,
               labels={'xlabel': r'Re$[\beta_x]$', 'ylabel': r'Im$[\beta_x]$', 'zlabel': dir_label},
               limits={'xlim': (-2, 2), 'ylim': (-2, 2), 'zlim': (0, 1)},
               ticksteps={'xstep': 1, 'ystep': 1, 'zstep': 0.2},
               title=r'$\beta_x$ with fixed ' + dir_label)

    
    # betay
    ax2.scatter(GBZ[:, :, 2].ravel(), GBZ[:, :, 3].ravel(), Z.ravel(),
                s=15, c=RGB.reshape(-1, 3), edgecolors=None, alpha=1)
    retouch_ax(ax2,
               labels={'xlabel': r'Re$[\beta_y]$', 'ylabel': r'Im$[\beta_y]$', 'zlabel': dir_label},
               limits={'xlim': (-2, 2), 'ylim': (-2, 2), 'zlim': (0, 1)},
               ticksteps={'xstep': 1, 'ystep': 1, 'zstep': 0.2},
               title=r'$\beta_y$ with fixed ' + dir_label)
    
    # Spectra
    ax3.scatter(Spectra[:, :, 0].ravel(), Spectra[:, :, 1].ravel(), Z.ravel(),
                s=15, c=RGB.reshape(-1, 3), edgecolors=None, alpha=1)
    min_xlim = np.min( (-1, np.floor(ax3.get_xlim()[0])-0.5))
    max_xlim = np.max( (1, np.floor(ax3.get_xlim()[1])+0.5))
    min_ylim = np.min( (-1, np.floor(ax3.get_ylim()[0])-0.5))
    max_ylim = np.max( (1, np.floor(ax3.get_ylim()[1])+0.5))
    xstep = np.max( (1, np.floor( (ax3.get_xlim()[1] - ax3.get_xlim()[0]) / 5)) )
    ystep = np.max( (1, np.floor( (ax3.get_ylim()[1] - ax3.get_ylim()[0]) / 5)) )
    retouch_ax(ax3,
               labels={'xlabel': r'Re$[E]$', 'ylabel': r'Im$[E]$', 'zlabel': dir_label},
               limits={'xlim': (min_xlim, max_xlim), 'ylim': (min_ylim, max_ylim), 'zlim': (0, 1)},
               ticksteps={'xstep': xstep, 'ystep': ystep, 'zstep': 0.2},
               title=r'$E_{obc}$ with fixed ' + dir_label)
    
    # colors
    axc.imshow(RGB.transpose(1, 0), origin='lower', 
               aspect=Lt1/Lt2)
    axc.set_xlabel(r'$t_1$', fontsize=12)
    axc.set_ylabel(r'$t_2$', fontsize=12)
    axc.set(xticks=(-0.5, Lt1-0.5), xticklabels=['0', '1'])
    axc.set(yticks=(-0.5, Lt2-0.5), yticklabels=['0', '1'])
    axc.tick_params(axis='both', labelsize=9)

    plt.show()










def plt_GBZ_xy(GBZ, Spectra, LogAbs=False, cropping=True, bg_surface=False):
    """
    Plot the scatter GBZ data in the form of |beta|-kx-ky.

    Params:
        GBZ (torch.tensor): Tensor of shape (Nt1, Nt2, 4). 
            GBZ is typically the output from the encoder.
        LogAbs (bool): Whether to use log scale for the z axis.
        cropping (bool): Whether to crop the GBZ tensor by removing the last row and column. 
            This is useful because the GBZ (output of the encoder) is periodic in t1 and t2 dimensions,
                causing the first and last rows/columns to overlap.
            Cropping helps avoid color overlap in the plot.
        bg_surface (bool): Whether to plot a eye-guiding surface for the scatter.
            This is optional because that the encoder may not map [t1, t2] to the correct order of (kx, ky),
                which may cause overlap to the surface.
    """

    assert GBZ.dim() == 3
    assert GBZ.size(2) == 4

    if cropping:
        GBZ = GBZ[:-1, :-1, :]
        Spectra = Spectra[:-1, :-1, :]
    Lt1, Lt2 = GBZ.size(0), GBZ.size(1)
    RGB = generate_color_mesh(Lt1, Lt2)

    betax = GBZ[:, :, 0] + 1.j*GBZ[:, :, 1]
    betay = GBZ[:, :, 2] + 1.j*GBZ[:, :, 3]
    argx = torch.atan2(GBZ[:, :, 1], GBZ[:, :, 0]) / torch.pi
    argy = torch.atan2(GBZ[:, :, 3], GBZ[:, :, 2]) / torch.pi

    if LogAbs:
        Zx = torch.log( torch.abs( betax))
        Zy = torch.log( torch.abs( betay))
        zlabelx = r'$\log{|\beta_x|}$'
        zlabely = r'$\log{|\beta_y|}$'
        zlimx = (torch.min(Zx)-0.1, torch.max(Zx)+0.1)
        zlimy = (torch.min(Zy)-0.1, torch.max(Zy)+0.1)
    else:
        Zx = torch.abs( betax)
        Zy = torch.abs( betay)
        zlabelx = r'$|\beta_x|$'
        zlabely = r'$|\beta_y|$'
        zlimx = (0, torch.max(Zx).item()+0.5) if (torch.max(Zx) > 2) else (0, 2)
        zlimy = (0, torch.max(Zy).item()+0.5) if (torch.max(Zy) > 2) else (0, 2)

    fig = plt.figure(figsize=(14.6, 4.2))
    ax1 = fig.add_axes([0.12, 0.05, 0.25, 0.84], projection='3d')
    ax2 = fig.add_axes([0.41, 0.05, 0.25, 0.84], projection='3d')
    ax3 = fig.add_axes([0.75, 0.15, 0.21, 0.72])
    axc = fig.add_axes([0.04, 0.30, 0.05, 0.40])

    init_ax(ax1, pane_color='white', view_params=(25, -65))
    init_ax(ax2, pane_color='white', view_params=(25, -65))

    # background surface
    if bg_surface:
        argx_flat = argx.numpy().ravel()
        argy_flat = argy.numpy().ravel()
        id = np.lexsort((argx_flat, argy_flat))
        argx_sorted = argx_flat[id].reshape(argx.shape)
        argy_sorted = argy_flat[id].reshape(argy.shape)

        Zx_flat = Zx.numpy().ravel()
        Zy_flat = Zy.numpy().ravel()
        Zx_sorted = Zx_flat[id].reshape(Zx.shape)
        Zy_sorted = Zy_flat[id].reshape(Zy.shape)

        ax1.plot_surface(argx_sorted, argy_sorted, Zx_sorted,
                     rstride=1, cstride=1, color='black', alpha=0.05)
        ax2.plot_surface(argx_sorted, argy_sorted, Zy_sorted,
                     rstride=1, cstride=1, color='black', alpha=0.05)

    # betex scatter
    ax1.scatter(argx.ravel(), argy.ravel(), Zx.ravel(),
                s=5, c=RGB.reshape(-1, 3))
    retouch_ax(ax1,
               labels={'xlabel': r'Arg$[\beta_x]/\pi$', 'ylabel': r'Arg$[\beta_y]/\pi$', 'zlabel': zlabelx},
               limits={'xlim': (-1, 1), 'ylim': (-1, 1), 'zlim': zlimx},
               ticksteps={'xstep': 1, 'ystep': 1, 'zstep': 0.5},
               title=r'$GBZ_x$')
    
    # betay scatter
    ax2.scatter(argx.ravel(), argy.ravel(), Zy.ravel(),
                s=5, c=RGB.reshape(-1, 3))
    retouch_ax(ax2,
               labels={'xlabel': r'Arg$[\beta_x]/\pi$', 'ylabel': r'Arg$[\beta_y]/\pi$', 'zlabel': zlabely},
               limits={'xlim': (-1, 1), 'ylim': (-1, 1), 'zlim': zlimy},
               ticksteps={'xstep': 1, 'ystep': 1, 'zstep': 0.5},
               title=r'$GBZ_y$')
    
    # Spectra
    ax3.scatter(Spectra[:, :, 0].ravel(), Spectra[:, :, 1].ravel(),
                s=5, c=RGB.reshape(-1, 3))
    min_xlim = np.min( (-1, np.floor(ax3.get_xlim()[0])-0.5))
    max_xlim = np.max( (1, np.floor(ax3.get_xlim()[1])+0.5))
    min_ylim = np.min( (-1, np.floor(ax3.get_ylim()[0])-0.5))
    max_ylim = np.max( (1, np.floor(ax3.get_ylim()[1])+0.5))
    ax3.set_xlabel(r'Re$[E]$', fontsize=12)
    ax3.set_ylabel(r'Im$[E]$', fontsize=12)
    ax3.set_xticks(np.arange(-10, 10.5, 2))
    ax3.set_yticks(np.arange(-10, 10.5, 2))
    ax3.tick_params(axis='both', labelsize=9)
    ax3.set_title(r'$E_{obc}$')
    # ax3.axis('equal')
    ax3.set(xlim=(min_xlim, max_xlim), ylim=(min_ylim, max_ylim))

    # colors
    axc.imshow(RGB.transpose(1, 0), origin='lower', 
               aspect=Lt1/Lt2)
    axc.set_xlabel(r'$t_1$', fontsize=12)
    axc.set_ylabel(r'$t_2$', fontsize=12)
    axc.set(xticks=(-0.5, Lt1-0.5), xticklabels=['0', '1'])
    axc.set(yticks=(-0.5, Lt2-0.5), yticklabels=['0', '1'])
    axc.tick_params(axis='both', labelsize=9)

    plt.show()









def plt_record_loss(h5path: str, id_start: int = 0):
    """
    Plot the recorded loss data in the training process.
    The loss data includes loss_fg, 2 loss_areas, and 3 loss_symms.
    The loss data is stored in an h5 file.

    Params:
        h5path (str): path of the data file.
        id_start (int): only loss after epoch id_start will  be ploted.
    """
    try:
        with h5py.File(h5path, 'r') as f:
            loss_fg_lst = f['loss_fg'][:]
            loss_area_x_avg_lst = f['loss_area_x_avg'][:]
            loss_area_y_avg_lst = f['loss_area_y_avg'][:]
            loss_symm_x_avg_lst = f['loss_symm_x_avg'][:]
            loss_symm_y_avg_lst = f['loss_symm_y_avg'][:]
            loss_symm_xy_ex_lst = f['loss_symm_xy_ex'][:]
    except:
        raise Exception(f'Cannot find recorded loss at {h5path}!')
    
    num_epochs = len(loss_fg_lst)
    xData = [i for i in range(num_epochs)]

    fig = plt.figure(figsize=(12.0, 5.4))
    ax1 = fig.add_axes([0.06, 0.70, 0.42, 0.26])
    ax2 = fig.add_axes([0.06, 0.40, 0.42, 0.26])
    ax3 = fig.add_axes([0.06, 0.10, 0.42, 0.26])
    ax4 = fig.add_axes([0.56, 0.70, 0.42, 0.26])
    ax5 = fig.add_axes([0.56, 0.40, 0.42, 0.26])
    ax6 = fig.add_axes([0.56, 0.10, 0.42, 0.26])

    def major_step(x):
        a = 10 ** np.floor(np.log10(x))
        return a if x < 5*a else 2*a
    
    ax1.plot(xData[id_start:], loss_fg_lst[id_start:], linewidth=1,
                color=(0, 0.18, 0.82),
                label=r'$L_1=MSE\{z, g(\beta)\}$')
    ax1.set_xticks(np.arange(0, num_epochs+1, num_epochs/5), labels=[])
    ax1.yaxis.set_major_locator(plt.MultipleLocator(major_step(ax1.get_ylim()[1])))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
    ax1.set(xlim=(id_start, len(xData)), ylim=(-ax1.get_ylim()[1]/50, None))
    ax1.set_ylabel(r'$L_1$', fontsize=12)
    ax1.legend()


    ax2.plot(xData[id_start:], loss_area_x_avg_lst[id_start:], linewidth=1,
                color=(0.95, 0.1, 0.1),
                label=r'$L_2=S(\mathcal{L}_E(t_2=const))$')
    ax2.set_xticks(np.arange(0, num_epochs+1, num_epochs/5), labels=[])
    ax2.yaxis.set_major_locator(plt.MultipleLocator(major_step(ax2.get_ylim()[1])))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
    ax2.set(xlim=(id_start, len(xData)), ylim=(-ax2.get_ylim()[1]/50, None))
    ax2.set_ylabel(r'$L_2$', fontsize=12)
    ax2.legend()


    ax3.plot(xData[id_start:], loss_area_y_avg_lst[id_start:], linewidth=1,
                color=(1.0, 0.55, 0.0),
                label=r'$L_3=S(\mathcal{L}_E(t_1=const))$')
    ax3.set_xticks(np.arange(0, num_epochs+1, num_epochs/5))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(major_step(ax3.get_ylim()[1])))
    ax3.ticklabel_format(style='sci', axis='both', scilimits=(-1, 1))
    ax3.set(xlim=(id_start, len(xData)), ylim=(-ax3.get_ylim()[1]/50, None))
    ax3.set_xlabel(r'epoch', fontsize=12)
    ax3.set_ylabel(r'$L_3$', fontsize=12)
    ax3.legend()


    ax4.plot(xData[id_start:], loss_symm_x_avg_lst[id_start:], linewidth=1,
                color=(0, 0.40, 0),
                label=r'$L_4=D\{\bar{\beta_1}, \mathcal{F}(\bar{z_1}, z_2)\}$')
    ax4.set_xticks(np.arange(0, num_epochs+1, num_epochs/5), labels=[])
    ax4.yaxis.set_major_locator(plt.MultipleLocator(major_step(ax4.get_ylim()[1])))
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
    ax4.set(xlim=(id_start, len(xData)), ylim=(-ax4.get_ylim()[1]/50, None))
    ax4.set_ylabel(r'$L_4$', fontsize=12)
    ax4.legend()


    ax5.plot(xData[id_start:], loss_symm_y_avg_lst[id_start:], linewidth=1,
                color=(0.45, 0.8, 0),
                label=r'$L_5=D\{\bar{\beta_2}, \mathcal{F}(z_1, \bar{z_2})\}$')
    ax5.set_xticks(np.arange(0, num_epochs+1, num_epochs/5), labels=[])
    ax5.yaxis.set_major_locator(plt.MultipleLocator(major_step(ax5.get_ylim()[1])))
    ax5.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
    ax5.set(xlim=(id_start, len(xData)), ylim=(-ax5.get_ylim()[1]/50, None))
    ax5.set_ylabel(r'$L_5$', fontsize=12)
    ax5.legend()


    ax6.plot(xData[id_start:], loss_symm_xy_ex_lst[id_start:], linewidth=1,
                color=(0.59, 0.14, 0.8),
                label=r'$L_6=D(|\beta_1|, |\beta_2|)$')
    ax6.set_xticks(np.arange(0, num_epochs+1, num_epochs/5))
    ax6.yaxis.set_major_locator(plt.MultipleLocator(major_step(ax6.get_ylim()[1])))
    ax6.ticklabel_format(style='sci', axis='both', scilimits=(-1, 1))
    ax6.set(xlim=(id_start, len(xData)), ylim=(-ax6.get_ylim()[1]/50, None))
    ax6.set_xlabel(r'epoch', fontsize=12)
    ax6.set_ylabel(r'$L_6$', fontsize=12)
    ax6.legend()

    plt.show()