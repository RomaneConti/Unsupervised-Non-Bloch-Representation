
# This script uses Python 3.10.11

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'plt_GFS',
    'plt_record_loss',
]

# change default dtype of torch.tensor to float64
torch.set_default_dtype(torch.float64)




def generate_input_vec(N, rand_input=False):
    """
    Generate the input data [cos(x), sin(x)] for the encoder.
    """
    if rand_input:
        t = torch.rand(1) + torch.linspace(0, 1, N).unsqueeze(1)
    else:
        t = torch.linspace(0, 1, N).unsqueeze(1)
    
    return torch.cat((torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t)), dim=1)










def plt_GFS(betax, rhoy, ref_point_lst=None, 
            Spectra_major_step=(2, 2),
            xyRange=(2, 2)):
    """
    Plot the GBZ and related spectra.

    Params:
        betax (torch.tensor): Tensor of shape (Nt, 2). 
            betax is typically the output from the encoder.
        rhoy (torch.tensor): Tensor of shape (Nt, 2). 
            rhoy is typically obtained by the gfs_model.
        ref_point_lst (torch.tensor): Tensor of shape (M, 2). 
            The reference points on the GBZ. Obtained by solving a short OBC chain.
        Spectra_major_step (tuple): tuple with two (float) scalars.
            The major step of xtick and ytick of the Spectra plot.
        xyRange (tuple): tuple with two (float) scalars.
            The xlim of the plot is (-xyRange[0], xyRange[0]).
            The ylim of the plot is (-xyRange[1], xyRange[1]).
    """
    assert betax.dim() == 2 and betax.size(1) == 2
    assert rhoy.dim() == 2 and rhoy.size(1) == 2

    fig = plt.figure(figsize=(10.0, 4.6))
    ax1 = fig.add_axes([0.08, 0.12, 0.37, 0.80])
    ax2 = fig.add_axes([0.54, 0.12, 0.37, 0.80])

    h_betax = ax1.scatter(betax[:, 0], betax[:, 1], 8,
                        c=range(len(betax)), cmap='rainbow')
    if ref_point_lst is not None:
        ax1.scatter(ref_point_lst[:, 0], ref_point_lst[:, 1], 8)
    ax1.grid(True)
    ax1.set_xlabel(r'Re$[\beta]$', fontsize=12)
    ax1.set_ylabel(r'Im$[\beta]$', fontsize=12)
    ax1.set_xticks(np.arange(-4, 4.5, 0.5))
    ax1.set_yticks(np.arange(-4, 4.5, 0.5))
    ax1.tick_params(axis='both', labelsize=9)
    ax1.set_title(r'Predicted GBZ')
    ax1.axis('equal')
    ax1.set_xlim(-xyRange[0], xyRange[0])
    ax1.set_ylim(-xyRange[1], xyRange[1])

    # ax2.plot(Spectra[:, 0], Spectra[:, 1], 'k-', linewidth=0.5)
    ax2.scatter(rhoy[:, 0], rhoy[:, 1], 8,
                c=range(len(rhoy)), cmap='rainbow')
    ax2.set_xlabel(r'Re$[E]$', fontsize=12)
    ax2.set_ylabel(r'Im$[E]$', fontsize=12)
    ax2.set_xticks(np.arange(-10, 10.5, Spectra_major_step[0]))
    ax2.set_yticks(np.arange(-10, 10.5, Spectra_major_step[1]))
    ax2.tick_params(axis='both', labelsize=9)
    ax2.set_title(r'Predicted $E_{obc}$')
    ax2.axis('equal')

    ax2_c = ax2.inset_axes([1.07, 0.3, 0.04, 0.4])
    hc = fig.colorbar(h_betax, cax=ax2_c)
    hc.set_label(r'$t$', fontsize=12, rotation=0)
    hc.set_ticks([0, len(rhoy)-1])
    hc.set_ticklabels(['0', '1'])
    plt.show()






def plt_record_loss(h5path: str, id_start: int = 0):
    """
    Plot the recorded loss data in the training process.
    The loss data includes loss_fg, loss_area, loss_symm, and loss_uniform.
    The loss data is stored in an h5 file.

    Params:
        h5path (str): path of the data file.
        id_start (int): only loss after epoch id_start will  be ploted.
    """
    try:
        with h5py.File(h5path, 'r') as f:
            loss_fg_lst = f['loss_fg'][:]
            loss_area_lst = f['loss_area'][:]
            loss_uniform_lst = f['loss_uniform'][:]
    except:
        raise Exception(f'Cannot find recorded loss at {h5path}!')
    
    num_epochs = len(loss_fg_lst)
    xData = [i for i in range(num_epochs)]

    fig = plt.figure(figsize=(5.4, 5.8))
    ax1 = fig.add_axes([0.10, 0.70, 0.85, 0.27])
    ax2 = fig.add_axes([0.10, 0.39, 0.85, 0.27])
    ax3 = fig.add_axes([0.10, 0.08, 0.85, 0.27])

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


    ax2.plot(xData[id_start:], loss_area_lst[id_start:], linewidth=1,
                color=(0.95, 0.1, 0.1),
                label=r'$L_2=S(\mathcal{L}_E)$')
    ax2.set_xticks(np.arange(0, num_epochs+1, num_epochs/5), labels=[])
    ax2.yaxis.set_major_locator(plt.MultipleLocator(major_step(ax2.get_ylim()[1])))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
    ax2.set(xlim=(id_start, len(xData)), ylim=(-ax2.get_ylim()[1]/50, None))
    ax2.set_ylabel(r'$L_2$', fontsize=12)
    ax2.legend()

    ax3.plot(xData[id_start:], loss_uniform_lst[id_start:], linewidth=1,
                color=(0.59, 0.14, 0.8),
                label=r'$L_3=U(\Delta\beta, \Delta\arg{\beta})$')
    ax3.set_xticks(np.arange(0, num_epochs+1, num_epochs/5))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(major_step(ax3.get_ylim()[1])))
    ax3.ticklabel_format(style='sci', axis='both', scilimits=(-1, 1))
    ax3.set(xlim=(id_start, len(xData)), ylim=(-ax3.get_ylim()[1]/50, None))
    ax3.set_xlabel(r'epoch', fontsize=12)
    ax3.set_ylabel(r'$L_3$', fontsize=12)
    ax3.legend()

    plt.show()