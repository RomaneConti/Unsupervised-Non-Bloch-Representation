
# This script uses Python 3.10.11

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle

from typing import Any, Union, Optional, Type, List, Dict, Tuple, Callable

__all__ = [
    'plt_GBZ_slice',
    'plt_GBZ_quiver_and_spectra',
    'plt_record_loss',
]

# change default dtype of torch.tensor to float64
torch.set_default_dtype(torch.float64)









#%%
### helper functions

def get_xyzGBZ_cform(GBZ: torch.Tensor):
    """
    Get βx, βy and βz, return as their complex forms.

    Params:
        GBZ (tensor): a tensor with shape (..., 6). Usually the output of the encoder.
    """
    return tuple(GBZ[..., 2*i] + 1.j*GBZ[..., 2*i+1] for i in range(3))





def init_ax(ax, pane_color='w', view_params=(25, -65)):
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











#%%
### sliced plot
DIM_MAPPING = {'x': 0, 'y': 1, 'z': 2}

def plt_single_slice(
        ax, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, C: Optional[np.ndarray] = None,
        slice_dim: Union[str, int] = 'x', slice_id: int = 0, *args, **kwargs):
    if C is None:
        C = Z
    slice_dim = DIM_MAPPING.get(slice_dim, slice_dim)
    X_slice = X.take(slice_id, axis=slice_dim)
    Y_slice = Y.take(slice_id, axis=slice_dim)
    Z_slice = Z.take(slice_id, axis=slice_dim)
    C_slice = C.take(slice_id, axis=slice_dim)
    ax.scatter(X_slice, Y_slice, Z_slice, c=C_slice, *args, **kwargs)

    def get_border(arr):
        return np.r_[arr[0, :], arr[-1, :], arr[1:-1, 0], arr[1:-1, -1]]
    X_border = get_border(X_slice)
    Y_border = get_border(Y_slice)
    Z_border = get_border(Z_slice)
    ax.scatter(X_border, Y_border, Z_border, c='k', s=0.1)





def retouch_ax(ax, labels: Dict = {'x': 'X', 'y': 'Y', 'z': 'Z'}, title: str = 'title'):
    """
    Retouch the appearance of a 3D axis.

    Params:
        ax (matplotlib.axes._axes.Axes): The 3D axis object to retouch.
        labels (dict): Dictionary with keys 'xlabel', 'ylabel', 'zlabel'.
        title (str): Title of the axis.
    """
    labelsize = 12
    ticksize = 9
    titlesize = 12

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel(labels.get('x', 'X'), fontsize=labelsize)
    ax.set_ylabel(labels.get('y', 'Y'), fontsize=labelsize)
    ax.set_zlabel(labels.get('z', 'Z'), fontsize=labelsize)

    ax.set_title(title, fontsize=titlesize)

    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_yticks(np.linspace(-1, 1, 5))
    ax.set_zticks(np.linspace(-1, 1, 5))
    ax.tick_params(axis='both', labelsize=ticksize)

    ax.set(xlim=[-1, 1], ylim=[-1, 1], zlim=[-1, 1])





def plt_GBZ_slice(GBZ, cropping=True, clim=(0, 2)):
    """
    Get kx, ky, kz from the GBZ and color it based on abs(βx,y,z), respectively.

    Params:
        GBZ (torch.tensor): Tensor of shape (N1, N2, N3, 6). 
            GBZ is typically the output from the encoder.
        cropping (bool): Whether to crop the GBZ tensor by removing the last row and column. 
            This is useful because the GBZ (output of the encoder) is periodic in each dimension,
            Cropping helps avoid color overlap in the plot.
        clim (tuple): limit of the colorbar.
    """

    if cropping:
        GBZ = GBZ[:-1, :-1, :-1]

    xyzGBZ_cform = get_xyzGBZ_cform(GBZ)
    absx = torch.abs(xyzGBZ_cform[0]).numpy()
    absy = torch.abs(xyzGBZ_cform[1]).numpy()
    absz = torch.abs(xyzGBZ_cform[2]).numpy()
    argx = torch.atan2(GBZ[..., 1], GBZ[..., 0]).numpy() / torch.pi
    argy = torch.atan2(GBZ[..., 3], GBZ[..., 2]).numpy() / torch.pi
    argz = torch.atan2(GBZ[..., 5], GBZ[..., 4]).numpy() / torch.pi

    idx = argx.shape[0] // 2
    idy = argx.shape[1] // 2
    idz = argx.shape[2] // 2

    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    scatter_params = {'s': 1, 'cmap': 'rainbow', 'alpha': 0.2, 'norm': norm}

    fig = plt.figure(figsize=(14.6, 4.2))
    ax1 = fig.add_axes([0.01, 0.08, 0.25, 0.84], projection='3d')
    ax2 = fig.add_axes([0.32, 0.08, 0.25, 0.84], projection='3d')
    ax3 = fig.add_axes([0.63, 0.08, 0.25, 0.84], projection='3d')
    init_ax(ax1, pane_color='w', view_params=(30, -65))
    init_ax(ax2, pane_color='w', view_params=(30, -65))
    init_ax(ax3, pane_color='w', view_params=(30, -65))

    plt_single_slice(ax1, argx, argy, argz, C=absx, slice_dim='x', slice_id=idx, **scatter_params)
    plt_single_slice(ax1, argx, argy, argz, C=absx, slice_dim='y', slice_id=idy, **scatter_params)
    plt_single_slice(ax1, argx, argy, argz, C=absx, slice_dim='z', slice_id=idz, **scatter_params)

    plt_single_slice(ax2, argx, argy, argz, C=absy, slice_dim='x', slice_id=idx, **scatter_params)
    plt_single_slice(ax2, argx, argy, argz, C=absy, slice_dim='y', slice_id=idy, **scatter_params)
    plt_single_slice(ax2, argx, argy, argz, C=absy, slice_dim='z', slice_id=idz, **scatter_params)

    plt_single_slice(ax3, argx, argy, argz, C=absz, slice_dim='x', slice_id=idx, **scatter_params)
    plt_single_slice(ax3, argx, argy, argz, C=absz, slice_dim='y', slice_id=idy, **scatter_params)
    plt_single_slice(ax3, argx, argy, argz, C=absz, slice_dim='z', slice_id=idz, **scatter_params)

    labels = {'x': r'$\arg{\beta_1}$',
              'y': r'$\arg{\beta_2}$',
              'z': r'$\arg{\beta_3}$',}
    retouch_ax(ax1, labels=labels, title=r'$|\beta_1|$')
    retouch_ax(ax2, labels=labels, title=r'$|\beta_2|$')
    retouch_ax(ax3, labels=labels, title=r'$|\beta_3|$')

    ax_c = fig.add_axes([0.94, 0.30, 0.01, 0.40])
    sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
    sm.set_array([])
    hc = fig.colorbar(sm, cax=ax_c)
    hc.set_label(r'$|\beta|$', fontsize=12, rotation=0, labelpad=15)
    ax_c.tick_params(axis='y', labelsize=10, direction='in', length=5)

    plt.show()











#%%
### quiver and spectra plot
def plt_quiver(ax, argx, argy, argz, absx, absy, absz,
               num_slice=(7, 7, 7), *args, **kwargs):
    """
    lot the GBZ as quivers, which start from (kx, ky, kz) and vector as (|βx, βy, βz|).

    Params:
        argx/y/z (np.ndarray): arg[β] got from the GBZ.
        absx/y/z (np.ndarray): abs[β] got from the GBZ.
        num_slice: determine the number of quiver that will be plot for each dimension.
    """
    assert len(num_slice) == 3
    idx = np.linspace(0, argx.shape[0]-1, num_slice[0]).astype(int)
    idy = np.linspace(0, argx.shape[1]-1, num_slice[1]).astype(int)
    idz = np.linspace(0, argx.shape[2]-1, num_slice[2]).astype(int)
    id = np.ix_(idx, idy, idz)
    ax.quiver(argx[id], argy[id], argz[id], absx[id], absy[id], absz[id],
               length=0.3, normalize=True, color='blue', *args, **kwargs)
    
    labels = {'x': r'$\arg{\beta_1}$',
              'y': r'$\arg{\beta_2}$',
              'z': r'$\arg{\beta_3}$',}
    retouch_ax(ax, labels=labels, title=r'$|\beta|$')





def get_rgb_from_dict(default_shape: Tuple, rgb_dict: Optional[Dict] = None):
    """
    Get r, g, b from a possibly given dict (with possible keys 'r', 'g', 'b').
    If not given, generate a default data.

    Params:
        default_shape (tuple): tuple with 3 int elements (N1, N2, N3).
            The shape of r/g/b should be the same as the default shape.
            Default r/g/b is generated by treating the 1st/2nd/3rd dim as r/g/b form 0 to 1.
        rgb_dict (dict): possibly given dict.
    """
    assert len(default_shape) == 3
    r_default, g_default, b_default = np.meshgrid(
        np.linspace(0, 1, default_shape[0]), 
        np.linspace(0, 1, default_shape[1]),
        np.linspace(0, 1, default_shape[2]), indexing='ij')
    rgb_dict = rgb_dict or {}

    r, g, b = (rgb_dict.get('r', r_default),
               rgb_dict.get('g', g_default),
               rgb_dict.get('b', b_default))
    assert all(c.shape == default_shape for c in [r, g, b])

    return (r, g, b)





def plt_spectra(ax, Spectra: torch.Tensor, num_slice=(20, 20, 20),
                rgb: Dict = None, *args, **kwargs):
    """
    Plot the GBZ mapped Spectra, colored by RGB.

    Params:
        Spectra (torch.tensor): spectra got from the GBZ, has shape (N1, N2, N3, 2).
        num_slice (tuple): determine the number of spectra that will be ploted for each dimension.
        rgb (dict): determine the color scheme for the spectra
    """
    assert Spectra.dim() == 4 and Spectra.size(-1) == 2
    
    realE = Spectra[..., 0].numpy()
    imagE = Spectra[..., 1].numpy()

    assert len(num_slice) == 3
    idx_s = np.linspace(0, realE.shape[0]-1, num_slice[0]).astype(int)
    idy_s = np.linspace(0, realE.shape[1]-1, num_slice[1]).astype(int)
    idz_s = np.linspace(0, realE.shape[2]-1, num_slice[2]).astype(int)
    id_s = np.ix_(idx_s, idy_s, idz_s)

    r, g, b = get_rgb_from_dict(default_shape=realE.shape, rgb_dict=rgb)
    r, g, b = r[id_s], g[id_s], b[id_s]

    r = (r - np.min(r)) / (np.max(r) - np.min(r))
    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))
    colors = np.stack([r.flatten(), g.flatten(), b.flatten()], axis=1)

    ax.scatter(realE[id_s].flatten(), imagE[id_s].flatten(),
               c=colors, s=1, alpha=1, *args, **kwargs)
    
    ax.set_xlabel(r'$\Re[E]$', fontsize=12)
    ax.set_ylabel(r'$\Im[E]$', fontsize=12)
    ax.set_title(r'Spectra', fontsize=12)
    




def plt_GBZ_quiver_and_spectra(
        GBZ: torch.Tensor, Spectra: torch.Tensor, cropping=True,
        num_quiver=(7, 7, 7), num_spectra=(20, 20, 20)):
    """
    Plot the GBZ as quivers, which start from (kx, ky, kz) and vector as (|βx, βy, βz|).
    Plot the GBZ mapped Spectra, colored by RGB.

    Params:
        GBZ (torch.tensor): Tensor of shape (N1, N2, N3, 6). 
            GBZ is typically the output from the encoder.
        Spectra (torch.tensor): spectra got from the GBZ, has shape (N1, N2, N3, 2).
        cropping (bool): Whether to crop the GBZ tensor by removing the last row and column. 
            This is useful because the GBZ (output of the encoder) is periodic in each dimension,
            Cropping helps avoid color overlap in the plot.
        num_quiver (tuple): determine the number of quiver that will be ploted for each dimension.
        num_spectra (tuple): determine the number of spectra that will be ploted for each dimension.
    """
    assert GBZ.dim() == 4 and GBZ.size(-1) == 6
    assert Spectra.dim() == 4 and Spectra.size(-1) == 2

    if cropping:
        GBZ = GBZ[:-1, :-1, :-1]
        Spectra = Spectra[:-1, :-1, :-1]

    xyzGBZ_cform = get_xyzGBZ_cform(GBZ)
    absx = torch.abs(xyzGBZ_cform[0]).numpy()
    absy = torch.abs(xyzGBZ_cform[1]).numpy()
    absz = torch.abs(xyzGBZ_cform[2]).numpy()
    argx = torch.atan2(GBZ[..., 1], GBZ[..., 0]).numpy() / torch.pi
    argy = torch.atan2(GBZ[..., 3], GBZ[..., 2]).numpy() / torch.pi
    argz = torch.atan2(GBZ[..., 5], GBZ[..., 4]).numpy() / torch.pi

    fig = plt.figure(figsize=(9.2, 4.2))
    ax1 = fig.add_axes([0.01, 0.08, 0.45, 0.84], projection='3d')
    ax2 = fig.add_axes([0.58, 0.12, 0.38, 0.80])
    init_ax(ax1, pane_color='w', view_params=(30, -65))

    plt_quiver(ax1, argx, argy, argz, absx, absy, absz, num_slice=num_quiver)
    plt_spectra(ax2, Spectra, rgb={'r': argx, 'g': argy, 'b': argz}, num_slice=num_spectra)
    plt.show()










#%%
### loss plot
def major_step(x):
        a = 10 ** np.floor(np.log10(x))
        return a if x < 5 * a else 2 * a





def plt_single_loss(ax, h5_path, dset_name, id_start=0, *args, **kwargs):
    """
    Plot the loss in a single dataset.

    Params:
        h5path (str): path of the data file.
        dset_name (str): target dataset name
        id_start (int): only loss after epoch id_start will  be ploted.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            loss_data = f[dset_name][:]
    except KeyError:
        print(f"Error: Cannot find dataset '{dset_name}' in {h5_path}. Skipping this plot.")
        return
    except Exception as e:
        print(f"Error in loading dataset '{dset_name}': {e}")
        return
    
    num_epochs = loss_data.shape[-1]
    xData = np.arange(num_epochs)

    colors = kwargs.pop('colors', plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors = cycle(colors)

    legends = kwargs.pop('legends', ['x', 'y', 'z'])
    legends = cycle(legends)

    if loss_data.ndim == 1:
        loss_data = loss_data[np.newaxis, :]

    for i in range(loss_data.shape[0]):
        ax.plot(xData[id_start:], loss_data[i, id_start:], *args, **kwargs,
                color=next(colors), label=next(legends))
            
    ax.set_xticks(np.arange(0, num_epochs + 1, num_epochs / 5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(major_step(ax.get_ylim()[1])))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
    ax.set_xlim(id_start, num_epochs)
    ax.set_ylim(-ax.get_ylim()[1] / 50, None)
    ax.set_xlabel(r'epoch', fontsize=12)
    ax.set_ylabel(kwargs.pop('ylabel', 'L'), fontsize=12)
    ax.legend(loc='upper right')





def plt_record_loss(h5path: str):
    """
    Plot all the recorded loss data in the training process.

    Params:
        h5path (str): path of the data file.
    """
    fig = plt.figure(figsize=(5.4, 7.5))
    ax1 = fig.add_axes([0.06, 0.81, 0.90, 0.16])
    ax2 = fig.add_axes([0.06, 0.625, 0.90, 0.16])
    ax3 = fig.add_axes([0.06, 0.44, 0.90, 0.16])
    ax4 = fig.add_axes([0.06, 0.255, 0.90, 0.16])
    ax5 = fig.add_axes([0.06, 0.06, 0.90, 0.16])

    id_start = 0
    plt_single_loss(ax1, h5path, dset_name='loss_fg', id_start=id_start,
                    colors=[(0.80, 0.18, 0.22)],
                    legends=[r'$L_\mathrm{fg}$'])
    plt_single_loss(ax2, h5path, dset_name='loss_area', id_start=id_start,
                    legends=[rf'$L_{{\mathrm{{area}},{d}}}$' for d in ['x', 'y', 'z']])
    plt_single_loss(ax3, h5path, dset_name='loss_symm', id_start=id_start,
                    legends=[rf'$L_{{\mathrm{{symm}},{d}}}$' for d in ['x', 'y', 'z']])
    plt_single_loss(ax4, h5path, dset_name='loss_symm_ex', id_start=id_start,
                    colors=[(0.59, 0.14, 0.8)],
                    legends=[r'$L_\mathrm{symm,ex}$'])
    plt_single_loss(ax5, h5path, dset_name='loss_arg', id_start=id_start,
                    legends=[rf'$L_{{\mathrm{{arg}},{d}}}$' for d in ['x', 'y', 'z']])
    
    def remove_xlabel(ax_lst):
        for ax in ax_lst:
            ax.set_xticklabels([])
            ax.set_xlabel('')
    remove_xlabel([ax1, ax2, ax3, ax4])

    plt.show()
# %%
