
# This script uses Python 3.10.11

#%%
import os
import logging
import shutil

import h5py
import textwrap
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Any, Union, Optional, Type, List, Dict, Tuple, Callable

from plot_helpers import*
from Area_Iter import PolyArea


# change default dtype of torch.tensor to float64
torch.set_default_dtype(torch.float64)











#%%
### helper functions
def generate_input_mesh(num_input, rand_input=False, input_radius: Tuple[float, float]=(1, 1)):
    """
    Generate the input data [cos(kx), sin(kx), cos(ky), sin(ky)] for the encoder.
    Return a 3 dimensional tensor, the 1st/2nd dimension coresponds to t1(kx)/t2(ky). 
    So, data[id, :, :] means fix kx and slice along ky while
        data[:, id, :] means fix ky and slice along kx.

    Params:
        num_input (tuple): tuple with 2 ints, number of mesh points along t1(kx) and t2(ky) direction.
        rand_input (bool): whether to add a small offset to the mesh.
        input_radius (tuple): should have form (rx, ry). The mesh can be seen as 
            traversing two circles with radius rx and ry.
    """
    assert len(num_input) == 2 and len(input_radius) == 2
    N1, N2 = num_input
    rx, ry = input_radius
    t1 = (2*torch.rand(1)/N1 if rand_input else 0) + torch.linspace(-1, 1, N1)
    t2 = (2*torch.rand(1)/N2 if rand_input else 0) + torch.linspace(-1, 1, N2)
    t1_grid, t2_grid = torch.meshgrid(t1, t2, indexing='ij')
    
    return {'kx': torch.pi * (-1 + (t1 + 1) % 2),
            'ky': torch.pi * (-1 + (t2 + 1) % 2),
            'z': torch.stack(
                [torch.cos(torch.pi * t1_grid) * rx,
                 torch.sin(torch.pi * t1_grid) * rx,
                 torch.cos(torch.pi * t2_grid) * ry,
                 torch.sin(torch.pi * t2_grid) * ry], dim=-1)}





def get_xyGBZ(GBZ: torch.Tensor):
    """
    Get βx and βy, return as tensor with shape (..., 2).
    With the 1st layer Re[β] and the 2nd layer Im[β].

    Params:
        GBZ (tensor): a tensor with shape (..., 4). Usually the output of the encoder.
    """
    # assert GBZ.size(-1) == 4, "GBZ should be an (..., 4) tensor."
    return GBZ[..., :2], GBZ[..., 2:]





def get_xyGBZ_cform(GBZ: torch.Tensor):
    """
    Get βx and βy, return as their complex forms.

    Params:
        GBZ (tensor): a tensor with shape (..., 4). Usually the output of the encoder.
    """
    # assert GBZ.size(-1) == 4, "GBZ should be an (..., 4) tensor."
    return GBZ[..., 0]+1.j*GBZ[..., 1], GBZ[..., 2]+1.j*GBZ[..., 3]





def return_zero(*args, **kwargs):
    """
    Return zero for all inputs. This will be useful in those loss configs.
    """
    return torch.tensor(0.)












#%%
### 2D single-band model
class MonoArray2D:
    def __init__(self, model_name: str, model_params: Optional[Tuple[float,...]], scale_factor: float = 1.):
        """
        Class of the 2D single-band model.
        For simplicity, we do not need a general approach to define the 2D model.
        Concrete strs will be used to label different models.

        Params:
            model_name (str): a unique tag for different models.
            model_params (tuple): possible hopping parameters.
                Should be given in order, or one may get the wrong model.
            scale_factor (float): a number that will be multiplied onto the 
                model_params. Sometimes it may be beneficial for training.
        """
        self._model_name = model_name
        self._scale_factor = float(scale_factor)
        if model_params is not None:
            params = tuple(x if isinstance(x, complex) else float(x) for x in model_params)
            self._model_params = tuple(x * self._scale_factor for x in params)
            params_tag = '_' + '_'.join([f'{num:.4g}' for num in params])
        else:
            self._model_params = None
            params_tag = ''

        self._tag = f'{model_name}{params_tag}_x{self._scale_factor:.4g}'

    
    @ staticmethod
    def _spectra_cform_HN2D(xGBZ_cform, yGBZ_cform, params):
        # params should be given with form (JL, JU, JR, JD)
        assert len(params) == 4, "4 params (JL, JU, JR, JD) are needed for an HN2D model."
        return (params[0] / xGBZ_cform + params[1] * yGBZ_cform +
                params[2] * xGBZ_cform + params[3] / yGBZ_cform)
    

    @staticmethod
    def _spectra_cform_L1R2_HN90(xGBZ_cform, yGBZ_cform, params):
        # params should be given with form (L1, R1, R2, J1, J2)
        assert len(params) == 5, "5 params (L1, R1, R2, J1, J2) are needed for an L1R2_HN90 model."
        L1, R1, R2, J1, J2 = params
        return (L1 / xGBZ_cform + R1 * xGBZ_cform + R2 * xGBZ_cform.pow(2) +
                J1 / yGBZ_cform + J2 * yGBZ_cform)

    
    @staticmethod
    def _spectra_cform_L1R2_HN45(xGBZ_cform, yGBZ_cform, params):
        # params should be given with form (L1, R1, R2, J1, J2)
        assert len(params) == 5, "5 params (L1, R1, R2, J1, J2) are needed for an L1R2_HN45 model."
        L1, R1, R2, J1, J2 = params
        return (L1 / xGBZ_cform + R1 * xGBZ_cform + R2 * xGBZ_cform.pow(2) +
                J1 / (xGBZ_cform * yGBZ_cform) + J2 * xGBZ_cform * yGBZ_cform)

        
    @staticmethod
    def _spectra_cform_L2R2_HN45(xGBZ_cform, yGBZ_cform, params):
        # params should be given with form (L1, L2, R1, R2, J1, J2)
        assert len(params) == 6, "6 params (L1, L2, R1, R2, J1, J2) are needed for an L2R2_HN45 model."
        L1, L2, R1, R2, J1, J2 = params
        return (L1 / xGBZ_cform + L2 / xGBZ_cform.pow(2) +
                R1 * xGBZ_cform + R2 * xGBZ_cform.pow(2) +
                J1 / (xGBZ_cform * yGBZ_cform) + J2 * xGBZ_cform * yGBZ_cform)

    
    @staticmethod
    def _spectra_cform_L1R2_D2U1(xGBZ_cform, yGBZ_cform, params):
        # params should be given with form (L1, R1, R2, D1, D2, U1)
        assert len(params) == 6, "6 params (L1, R1, R2, D1, D2, U1) are needed for an L1R2_D2U1 model."
        L1, R1, R2, D1, D2, U1 = params
        return (L1 / xGBZ_cform + R1 * xGBZ_cform + R2 * xGBZ_cform.pow(2) +
                D1 / yGBZ_cform + D2 / yGBZ_cform.pow(2) + U1 * yGBZ_cform)


    @staticmethod
    def _spectra_cform_Amoeba(xGBZ_cform, yGBZ_cform, params):
        # params should be given with form (t=1, gamma=0.2, t1=0.5)
        assert len(params) == 3, "3 params (t, gamma, t1) are needed for an Amoeba-like model."
        Jsum, Jdiff, t = (params[0] + params[1], params[0] - params[1],  params[2])
        return (Jsum * xGBZ_cform + Jdiff / xGBZ_cform +
                Jsum * yGBZ_cform + Jdiff / yGBZ_cform +
                t * (xGBZ_cform + 1/xGBZ_cform)*(yGBZ_cform + 1/yGBZ_cform))


    def _get_spectra_from_GBZ(self, GBZ: torch.Tensor):
        """
        Get the spectra from the GBZ.
        
        Params:
            GBZ (tensor): a tensor with shape (..., 4).
                Usually the output of the encoder and have shape (Nx, Ny, 4).
                The last dimension corresponds to Re/Im[βx], Re/Im[βy] in order.
        """
        assert GBZ.size(-1) == 4, "GBZ should be an (..., 4) tensor."

        xGBZ_cform, yGBZ_cform = get_xyGBZ_cform(GBZ)
        registered_model = {
            'HN2D': self._spectra_cform_HN2D,
            'L1R2_HN90': self._spectra_cform_L1R2_HN90,
            'L1R2_HN45': self._spectra_cform_L1R2_HN45,
            'L2R2_HN45': self._spectra_cform_L2R2_HN45,
            'L1R2_D2U1': self._spectra_cform_L1R2_D2U1,
            'Amoeba': self._spectra_cform_Amoeba,
        }

        if self._model_name not in registered_model:
            raise ValueError(f"Undefined model: '{self._model_name}'.")
        E_cform = registered_model[self._model_name](xGBZ_cform, yGBZ_cform, self._model_params)
        return torch.stack((E_cform.real, E_cform.imag), dim=-1)
        

    def __repr__(self):
        return f"MonoArray2D(name={self._model_name}, params={self._model_params})"












#%%
### encoder-decoder loss functions
class FGLossConfig:
    def __init__(self,
                 loss_weights: float = 100.,
                 func_fg: Callable = nn.MSELoss(reduction='mean')):
        """
        This class determines how to calculate the encoder-decoder loss.

        Params:
            loss_weights (float): weights of loss_fg.
            func_fg (function): loss function about the encoder and decoder.
                Should accept z_in and z_reconstructed as inputs, return a scalar tensor.
        """
        self._loss_weights = float(loss_weights)
        self._func_fg = func_fg

    def _fg_loss(self, z_in, z_reconstructed):
        return self._loss_weights * self._func_fg(z_in, z_reconstructed)












#%% 
### area loss functions
polyarea = PolyArea





def polyarea_simple(polygon: torch.Tensor):
    """
    Area of a simple polygon. S = sum_n (x[n]*y[n+1] - x[n+1]*y[n]).
    """
    assert polygon.dim() == 2 and polygon.size(1) == 2, "polygon should be an (N, 2) tensor."

    return torch.abs(torch.sum(polygon[:, 0] * (polygon[:, 1].roll(-1)) - 
                               polygon[:, 0].roll(-1) * polygon[:, 1]))





def polyarea_sumabs(polygon: torch.Tensor, ref_point: torch.Tensor = None):
    """
    Area of a simple polygon by summing up the absolute value of each term.
        S = sum_n abs(x[n]*y[n+1] - x[n+1]*y[n]).
    This is a wrong formula, but works when the polygon is consist of
        several straight lines intersecting at a single point.
    """
    assert polygon.dim() == 2 and polygon.size(1) == 2, "polygon should be an (N, 2) tensor."
    assert ref_point is None or ref_point.shape == (1, 2), "ref_point should be a (1, 2) tensor."

    if ref_point is None:
        ref_point = polygon.mean(dim=0)
    polygon = polygon - ref_point
    return torch.sum(torch.abs(polygon[:, 0] * (polygon[:, 1].roll(-1)) - 
                               polygon[:, 0].roll(-1) * polygon[:, 1]))

class PolyArea_SumAbs:
    def __init__(self, ref_point = None):
        self._ref_point = ref_point

    def __call__(self, polygon):
        return polyarea_sumabs(polygon, self._ref_point)





class AreaLoss:
    def __init__(self, farea = polyarea):
        self._farea = farea

    def __call__(self, polygon: torch.Tensor):
        return self._farea(polygon)





class AreaLossConfig:
    def __init__(self,
                 loss_weights: Tuple[float,...] = (1, 1),
                 func_area_x: Optional[Callable] = AreaLoss(),
                 func_area_y: Optional[Callable] = AreaLoss()):
        """
        This class determines how to calculate the area loss.

        Params:
            loss_weights (tuple): weights of loss_area along kx and ky in order.
            func_area_x (function): determine how to calculate the area along kx (ky fixed).
            func_area_y (function): determine how to calculate the area along ky (kx fixed).
        """
        assert len(loss_weights) == 2, "loss_weights should be a tuple with 2 float elements."
        self._loss_weights = tuple(map(float, loss_weights))
        self._func_area_x = func_area_x or return_zero
        self._func_area_y = func_area_y or return_zero
    
    
    def _area_loss_x_avg(self, Spectra: torch.Tensor, slice_id = None):
        assert Spectra.dim() == 3 and Spectra.size(2) == 2, "Spectra should be an (Nx, Ny, 2) tensor."
        # assert hasattr(slice_id, '__iter__') and hasattr(slice_id, '__len__')
        slice_id = slice_id or range(Spectra.size(1))
        total_area = sum(
            self._func_area_x(Spectra[:, id, :])
            for id in slice_id)
        return self._loss_weights[0] * total_area / len(slice_id)
    

    def _area_loss_y_avg(self, Spectra: torch.Tensor, slice_id = None):
        assert Spectra.dim() == 3 and Spectra.size(2) == 2, "Spectra should be an (Nx, Ny, 2) tensor."
        # assert hasattr(slice_id, '__iter__') and hasattr(slice_id, '__len__')
        slice_id = slice_id or range(Spectra.size(0))
        total_area = sum(
            self._func_area_y(Spectra[id, :, :])
            for id in slice_id)
        return self._loss_weights[1] * total_area / len(slice_id)












#%%
### symmetric loss functions
def symm_loss_xyexchange(GBZ: torch.Tensor, fdist = nn.L1Loss(reduction='mean')):
    """
    A loss making |βx|=|βy|.
    This loss can be used when the model is symmetric for exchanging x and y.
    """
    xGBZ_cform, yGBZ_cform = get_xyGBZ_cform(GBZ)
    return fdist(xGBZ_cform.abs(), yGBZ_cform.abs())

class SymmLoss_XYExchange:
    def __init__(self, fdist = nn.L1Loss(reduction='mean')):
        self._fdist = fdist

    def __call__(self, GBZ):
        return symm_loss_xyexchange(GBZ, self._fdist)





def symm_loss_x(GBZ: torch.Tensor, GBZ_xconj_input: torch.Tensor,
                full_dim: bool = False, fdist = nn.L1Loss(reduction='mean')):
    """
    A loss making βx symmetric about kx=0.

    Params:
        GBZ (tensor): a tensor with shape (..., 4). Usually the output of the encoder.
        GBZ_conj_input (torch.tensor): also a tensor with shape (..., 4).
            Obtained using the x-conj input mesh (multiply the 2nd layer by -1) for the encoder.
        full_dim (bool): whether to include βy(the last 2 dims) when calculating the distance.
        fdist (function): a function measuring the distance between two tensors.
    """
    # assert GBZ.size(-1) == 4, "GBZ should be a (..., 4) tensor."
    assert GBZ_xconj_input.shape == GBZ.shape
    GBZ_xconj = GBZ * torch.tensor([1, -1, 1, 1])
    calc_dim = slice(None) if full_dim else slice(None, 2)
    return fdist(GBZ_xconj[..., calc_dim], GBZ_xconj_input[..., calc_dim])

class SymmLoss_X:
    def __init__(self, full_dim: bool = False, fdist = nn.L1Loss(reduction='mean')):
        self._full_dim = full_dim
        self._fdist = fdist

    def __call__(self, GBZ, GBZ_conj_input):
        return symm_loss_x(GBZ, GBZ_conj_input, self._full_dim, self._fdist)
    




def symm_loss_y(GBZ: torch.Tensor, GBZ_yconj_input: torch.Tensor,
                full_dim: bool = False, fdist = nn.L1Loss(reduction='mean')):
    """
    A loss making βy symmetric about ky=0.

    Params:
        GBZ (tensor): a tensor with shape (..., 4). Usually the output of the encoder.
        GBZ_yconj_input (torch.tensor): also a tensor with shape (..., 4).
            Obtained using the y-conj input mesh (multiply the 4nd layer by -1) for the encoder.
        full_dim (bool): whether to include βx(the first 2 dims) when calculating the distance.
        fdist (function): a function measuring the distance between two tensors.
    """
    # assert GBZ.size(-1) == 4, "GBZ should be an (..., 4) tensor."
    assert GBZ_yconj_input.shape == GBZ.shape
    GBZ_yconj = GBZ * torch.tensor([1, 1, 1, -1])
    calc_dim = slice(None) if full_dim else slice(2, None)
    return fdist(GBZ_yconj[..., calc_dim], GBZ_yconj_input[..., calc_dim])

class SymmLoss_Y:
    def __init__(self, full_dim: bool = False, fdist = nn.L1Loss(reduction='mean')):
        self._full_dim = full_dim
        self._fdist = fdist

    def __call__(self, GBZ, GBZ_conj_input):
        return symm_loss_y(GBZ, GBZ_conj_input, self._full_dim, self._fdist)
    




class SymmLossConfig:
    def __init__(self,
                 loss_weights: Tuple[float,...] = (50, 50, 0),
                 func_symm_x: Optional[Callable] = None,
                 func_symm_y: Optional[Callable] = None,
                 func_symm_xy_ex: Optional[Callable] = None):
        """
        This class determines how to calculate the symm loss.

        Params:
            loss_weights (tuple): weights of loss_symm for x_sliced, y_sliced, and exchanged in order.
            func_symm_x (function): determine how to calculate the symm loss along kx (ky fixed).
            func_symm_y (function): determine how to calculate the symm loss along ky (kx fixed).
            func_symm_xy_exchange (function): determine how to calculate the exchanged symm loss.
        """
        assert len(loss_weights) == 3, "loss_weights should be a tuple with 3 float elements."
        self._loss_weights = tuple(map(float, loss_weights))
        self._func_symm_x = func_symm_x or return_zero
        self._func_symm_y = func_symm_y or return_zero
        self._func_symm_xy_ex = func_symm_xy_ex or return_zero
    
    
    def _symm_loss_x_avg(self, GBZ, GBZ_xconj_input, slice_id = None):
        assert GBZ.dim() == 3 and GBZ.size(-1) == 4, "GBZ should be an (Nx, Ny, 4) tensor."
        # assert GBZ.shape == GBZ_xconj_input.shape
        slice_id = slice_id or range(GBZ.size(1))
        # assert hasattr(slice_id, '__len__')
        loss = sum(
            self._func_symm_x(GBZ[:, id, :], GBZ_xconj_input[:, id, :])
            for id in slice_id)
        return self._loss_weights[0] * loss / len(slice_id)
    
    
    def _symm_loss_y_avg(self, GBZ, GBZ_yconj_input, slice_id = None):
        assert GBZ.dim() == 3 and GBZ.size(-1) == 4, "GBZ should be an (Nx, Ny, 4) tensor."
        # assert GBZ.shape == GBZ_yconj_input.shape
        slice_id = slice_id or range(GBZ.size(0))
        # assert hasattr(slice_id, '__len__')
        loss = sum(
            self._func_symm_y(GBZ[id,...], GBZ_yconj_input[id,...])
            for id in slice_id)
        return self._loss_weights[1] * loss / len(slice_id)
    

    def _symm_loss_xy_ex(self, GBZ):
        return self._loss_weights[2] * self._func_symm_xy_ex(GBZ)












#%%
### arg loss functions
def arg_loss_x(GBZ: torch.Tensor, kx_lst: Optional[torch.Tensor] = None,
               fdist: Callable = nn.MSELoss(reduction='mean')):
    """
    A loss making arg[βx] constant along y-slices (fixed x, t1).

    Params:
        GBZ (tensor): a tensor with shape (Nx, Ny, 4). Usually the output of the encoder.
        kx_lst (tensor): if given, should be a tensor with Nx elements.
            We make arg[βx] equals to given kx's along each slice.
        fdist (function): function measuring the distance between arg[βx] and given kx.
    """
    # assert GBZ.dim() == 3 and GBZ.size(-1) == 4, "GBZ should be an (Nx, Ny, 4) tensor."
    argx = torch.atan2(GBZ[..., 1], GBZ[..., 0])
    if kx_lst is None:
        return torch.mean(argx.max(dim=1).values - argx.min(dim=1).values)
    else:
        assert kx_lst.shape == (argx.size(0),), f"kx_lst should be a vector tensor with length {argx.size(0)}."
        return fdist(argx, kx_lst.unsqueeze(1).repeat(1, argx.size(1)))
        # return torch.mean((kx.max(dim=1).values - kx_lst).abs() + (kx.min(dim=1).values - kx_lst).abs())
    
class ArgLoss_X:
    def __init__(self, fdist = nn.MSELoss(reduction='mean')):
        self._fdist = fdist

    def __call__(self, GBZ: torch.Tensor, kx_lst = None):
        return arg_loss_x(GBZ, kx_lst, self._fdist)
    





def arg_loss_y(GBZ: torch.Tensor, ky_lst: Optional[torch.Tensor] = None,
               fdist: Callable = nn.MSELoss(reduction='mean')):
    """
    A loss making arg[βy] constant along x-slices (fixed y, t2).

    Params:
        GBZ (tensor): a tensor with shape (Nx, Ny, 4). Usually the output of the encoder.
        ky_lst (tensor): if given, should be a tensor with Ny elements.
            We make arg[βy] equals to given ky's along each slice.
        fdist (function): function measuring the distance between arg[βy] and given ky.
    """
    # assert GBZ.dim() == 3 and GBZ.size(-1) == 4, "GBZ should be an (Nx, Ny, 4) tensor."
    argy = torch.atan2(GBZ[..., 3], GBZ[..., 2])
    if ky_lst is None:
        return torch.mean(argy.max(dim=0).values - argy.min(dim=0).values)
    else:
        assert ky_lst.shape == (argy.size(1),), f"ky_lst should be a vector tensor with length {argy.size(1)}."
        return fdist(argy, ky_lst.repeat(argy.size(0), 1))
        # return torch.mean((ky.max(dim=0).values - ky_lst).abs() + (ky.min(dim=0).values - ky_lst).abs())
    
class ArgLoss_Y:
    def __init__(self, fdist = nn.MSELoss(reduction='mean')):
        self._fdist = fdist

    def __call__(self, GBZ: torch.Tensor, ky_lst = None):
        return arg_loss_y(GBZ, ky_lst, self._fdist)
    




class ArgLossConfig:
    def __init__(self,
                 loss_weights: Tuple[float,...] = (1, 1),
                 func_arg_x: Optional[Callable] = None,
                 func_arg_y: Optional[Callable] = None,
                 use_target: bool = False):
        """
        This class determines how to calculate the arg loss.

        Params:
            loss_weights (tuple): weights of loss_arg for x and y direction in order.
            func_arg_x (function): determine how to calculate the arg loss for kx.
            func_arg_y (function): determine how to calculate the arg loss for ky.
            use_target (bool): if use_target, kx,y_lst are needed in the calculation.
        """
        assert len(loss_weights) == 2, "loss_weights should be a tuple with 2 float elements."
        self._loss_weights = tuple(map(float, loss_weights))
        self._func_arg_x = func_arg_x or return_zero
        self._func_arg_y = func_arg_y or return_zero
        self._use_target = use_target


    def _arg_loss_x(self, GBZ, kx_lst):
        kx_lst = kx_lst if self._use_target else None
        assert not self._use_target or kx_lst is not None, "kx_lst is needed. Or set _use_target as False to avoid this."
        return self._loss_weights[0] * self._func_arg_x(GBZ, kx_lst)
    

    def _arg_loss_y(self, GBZ, ky_lst):
        ky_lst = ky_lst if self._use_target else None
        assert not self._use_target or ky_lst is not None, "ky_lst is needed. Or set _use_target as False to avoid this."
        return self._loss_weights[1] * self._func_arg_y(GBZ, ky_lst)

        












#%%
### assistant training functions
def init_models(f_net: nn.Module, g_net: nn.Module, init_fold: Optional[str] = None):
    """
    Initialize the encoder and decoder by saved weights.
    There should be files 'Encoder.pt' and 'Decoder.pt' under init_path.
    """
    if init_fold is not None:
        if not os.path.exists(init_fold):
            raise ValueError(f'Initialization path {init_fold} does not exist!')
        
        try:
            f_load_path = os.path.join(init_fold, 'Encoder.pt').replace('\\', '/')
            g_load_path = os.path.join(init_fold, 'Decoder.pt').replace('\\', '/')
            f_state_dict = torch.jit.load(f_load_path).state_dict()
            g_state_dict = torch.jit.load(g_load_path).state_dict()
            f_net.load_state_dict(f_state_dict)
            g_net.load_state_dict(g_state_dict)
        except FileNotFoundError as e:
            raise Exception(f'No available file: {e.filename}!')
        except RuntimeError as e:
            raise Exception(f'Loading state_dict failed: {e}!')
        




def save_models(f_net: nn.Module, g_net: nn.Module, f_save_path, g_save_path):
    """
    Save the encoder and decoder to the given path.
    """
    f_net_jit = torch.jit.trace(f_net, example_inputs=torch.rand(1, 4))
    g_net_jit = torch.jit.trace(g_net, example_inputs=torch.rand(1, 4))
    f_net_jit.save(f_save_path)
    g_net_jit.save(g_save_path)





def plt_training_results(encoder: nn.Module, monoarray: MonoArray2D, input_radius: Tuple = (1, 1),
                         N_slice: int = 7, N_loop: int=101, LogAbs: bool = False, cropping: bool = True):
    """
    Plot the current results in the training process.
    Including slice along x, y, and the whole |βx,y| plot.

    Params:
        encoder (nn.Module): the encoder.
        monoarray (MonoArray2D): target 2D model.
        N_slice (int): determine how many fixed k's will be chosen when slice along another k.
        N_loop (int): determine how many k's will be sampled along the sliced direction.
        LogAbs (bool): whether to plot log|β| or |β| in the abs plot.
        cropping (bool): Whether to crop the GBZ tensor by removing the last row and column. 
            This is useful because the GBZ (output of the encoder) is periodic in t1 and t2 dimensions,
            causing the first and last rows/columns to overlap.
    """
    zin_slice_x = generate_input_mesh((N_loop, N_slice), rand_input=True, input_radius=input_radius)['z']
    zin_slice_y = generate_input_mesh((N_slice, N_loop), rand_input=True, input_radius=input_radius)['z']
    zin_xy = generate_input_mesh((N_loop, N_loop), rand_input=True, input_radius=input_radius)['z']
    GBZ_slice_x = encoder(zin_slice_x)
    GBZ_slice_y = encoder(zin_slice_y)
    GBZ_xy = encoder(zin_xy)
    Spectra_slice_x = monoarray._get_spectra_from_GBZ(GBZ_slice_x)
    Spectra_slice_y = monoarray._get_spectra_from_GBZ(GBZ_slice_y)
    Spectra_xy = monoarray._get_spectra_from_GBZ(GBZ_xy)
    plt_GBZ_slice(GBZ_slice_x, Spectra_slice_x, slice_dir='t2', cropping=cropping)
    plt_GBZ_slice(GBZ_slice_y, Spectra_slice_y, slice_dir='t1', cropping=cropping)
    plt_GBZ_xy(GBZ_xy, Spectra_xy, LogAbs=LogAbs, cropping=cropping)





def test_models(monoarray: MonoArray2D, load_fold: str, input_radius: Tuple = (1, 1)):
    """
    Test the trained encoder and decoder.
    There should be files 'Encoder.pt' and 'Decoder.pt' under load_fold.
    """
    try:
        f_net_test = torch.jit.load(os.path.join(load_fold, 'Encoder.pt').replace('\\', '/'))
    except:
        raise Exception(f'No available file in folder {load_fold}')

    with torch.no_grad():
        plt_training_results(f_net_test, monoarray, input_radius)












#%%
### train config
class InputConfig:
    def __init__(self,
                 num_input: Tuple[int, int] = (101, 101),
                 rand_input: bool = True,
                 input_radius: Tuple[float, float] = (1, 1)):
        """
        This calss determine the properties of the input mesh.

        Params:
            num_input (tuple): number of sampling points of the input mesh. Should have form (N1, N2).
            rand_input (bool): whether to add an offset to the input tensor.
            input_radius (tuple): scale factor of the input mesh. Should have form (rx, ry).
        """
        self._num_input = num_input
        self._rand_input = rand_input
        self._input_radius = tuple(map(float, input_radius))


    def _create_mesh(self):
        return generate_input_mesh(self._num_input, self._rand_input, self._input_radius)





class SliceConfig:
    def __init__(self,
                 x_slice_id: Optional[Any] = None, 
                 y_slice_id: Optional[Any] = None,
                 slice_ratio: Tuple[float, float] = (None, None),
                 random_slice: bool = True):
        """
        This class determine how to slice the output.
        This may help to reduce the computational cost.

        Params:
            x_slice_id: indices along y, for each index an x-slice will be chosen.
            y_slice_id: indices along x, for each index a y-slice will be chosen.  
        [Note] If x,y_slice_id are given, ignore the following parameters.
            slice_ratio (tuple): a tuple with 2 elements, each element determines how many slices
                will be calculated for kx/ky. None is equivalent to ratio=1 (all the slices are included).
            random_slice (bool): whether to choose slice randomly when slice_ratio is not 1.
                If False, choose the first several slices based on slice_ratio.
        """
        assert x_slice_id is None or (hasattr(x_slice_id, '__iter__') and hasattr(x_slice_id, '__len__'))
        assert y_slice_id is None or (hasattr(y_slice_id, '__iter__') and hasattr(y_slice_id, '__len__'))
        self._x_slice_id = x_slice_id
        self._y_slice_id = y_slice_id
        self._slice_ratio = slice_ratio
        self._random_slice = random_slice


    @staticmethod
    def _create_slice_id_single(given_id, num_total: int, ratio: float, random_slice: bool):
        if given_id is not None:
            return given_id
        elif ratio is None:
            return range(num_total - 1)
        else:
            num_slct = int( float(ratio) * (num_total-1))
            if random_slice:
                return torch.randperm(num_total - 1)[:num_slct].tolist()
            else:
                return list(range(num_slct))
    

    def _create_slice_id(self, num_input: Tuple[int, int]):
        Nx, Ny = num_input
        ratio_x, ratio_y = self._slice_ratio
        lstx = self._create_slice_id_single(
            self._x_slice_id, Ny, ratio_y, self._random_slice)
        lsty = self._create_slice_id_single(
            self._y_slice_id, Nx, ratio_x, self._random_slice)
        return lstx, lsty





class OptimConfig:
    def __init__(self,
                 optimizer_class: Type[optim.Optimizer] = optim.Adam,
                 optimizer_params: Optional[Dict] = {'lr': 1e-3},
                 scheduler_class: Optional[Type[lr_scheduler._LRScheduler]] = lr_scheduler.StepLR,
                 scheduler_params: Optional[Dict] = {'step_size': 100, 'gamma': 0.9}):
        """
        This class determine the optimizer and scheduler.
        
        Params:
            optimizer_class (Type): optimizer class, this parameter cannot be None.
            optimizer_params (dict): parameters for the optimizer.
            scheduler_class (Type): learning rate scheduler class.
            scheduler_params (dict): parameters for the scheduler.
        """
        if optimizer_class is None:
            raise ValueError("optimizer_class must be specified and cannot be None.")
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params or {}

        self._scheduler_class = scheduler_class
        self._scheduler_params = scheduler_params or {}


    def _create_optimizer(self, model_params):
        return self._optimizer_class(model_params, **self._optimizer_params)
    

    def _create_scheduler(self, optimizer):
        if self._scheduler_class is not None:
            return self._scheduler_class(optimizer, **self._scheduler_params)
        return None





class TrainConfig:
    def __init__(self,
                 num_epochs: int = 3000,
                 init_path: Optional[str] = None,
                 input_config: InputConfig = InputConfig(),
                 slice_config: SliceConfig = SliceConfig(),
                 optim_config: OptimConfig = OptimConfig(),
                 update_each_slice: bool = True, 
                 output_step: Tuple[int, int] = (50, 200)):
        """
        This class determines the hyper parameters in training.

        Params:
            num_epochs (int): total number of epochs of training.
            init_path (str): path for initializing model weights.
            input_config (InputCofig): determine the properties of the input mesh.
            slice_config (SliceConfig): determine how to slice the GBZ to reduce computational cost.
            optim_config (OptimConfig): determine the optimizer and scheduler.
            update_each_slice (bool): determine how to update the weights.
                True: update the weights after every sliced area is calculated.
                False: only update the weights after the total(average) loss is calculated.
            output_step (tuple): a tuple with 2 ints, the 1st/2nd one is the output step for logging/plot.
            
        """
        self._num_epochs = num_epochs
        self._init_path = init_path
        self._input_config = input_config
        self._slice_config = slice_config
        self._optim_config = optim_config
        self._update_each_slice = update_each_slice
        self._output_step = output_step
        self._validate_range()


    def _validate_range(self):
        num_input = self._input_config._num_input
        id_x, id_y = self._slice_config._create_slice_id(num_input)
        assert max(id_x) <= num_input[1] and min(id_x) >= 0
        assert max(id_y) <= num_input[0] and min(id_y) >= 0












#%%
### main training functions
def train_models(f_net: nn.Module, g_net: nn.Module,
                 monoarray: MonoArray2D,
                 train_config: TrainConfig = TrainConfig(),
                 fg_config: FGLossConfig = FGLossConfig(),
                 area_config: AreaLossConfig = AreaLossConfig(),
                 symm_config: SymmLossConfig = SymmLossConfig(),
                 arg_config: ArgLossConfig = ArgLossConfig()):
    """
    Train the model with given conditions.

    Params:
        f_net, g_net (nn.Module): encoder and decoder.
        monoarray (MonoArray2D): target single-band system.
        train_config (TrainConfig): the configuration parameters for training.
        fg_config (FGLossConfig): configuration of the encoder-decoder loss.
        area_config (AreaLossConfig): configuration of the area loss.
        symm_config (SymmLossConfig): configuration of the symmetric loss.
    """
    for handler in logging.getLogger().handlers:
        handler.close()
        logging.getLogger().removeHandler(handler)

    init_models(f_net, g_net, train_config._init_path)

    input_config = train_config._input_config
    slice_config = train_config._slice_config
    optim_config = train_config._optim_config

    temp_paths = init_temp_save_path(
        temp_save_fold= f'./temp_save_weights/',
        temp_save_tag= monoarray._tag)
    f_save_path = temp_paths['f_save_path']
    g_save_path = temp_paths['g_save_path']
    
    f_optimizer = optim_config._create_optimizer(f_net.parameters())
    g_optimizer = optim_config._create_optimizer(g_net.parameters())

    f_scheduler = optim_config._create_scheduler(f_optimizer)
    g_scheduler = optim_config._create_scheduler(g_optimizer)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(temp_paths['logging_path'], mode='a'),
                  logging.StreamHandler()])
    
    num_epochs = train_config._num_epochs
    loss_path = temp_paths['loss_path']
    
    loss_file = H5LossFile(loss_path, num_epochs)
    try:
        for epoch in range(num_epochs):
            x_slice_id, y_slice_id = slice_config._create_slice_id(input_config._num_input)

            if train_config._update_each_slice:
                update_slice_losses(
                    f_net, g_net, monoarray, train_config,
                    fg_config, area_config, symm_config, arg_config,
                    f_optimizer, g_optimizer, x_slice_id, dir='x')
                update_slice_losses(
                    f_net, g_net, monoarray, train_config,
                    fg_config, area_config, symm_config, arg_config,
                    f_optimizer, g_optimizer, y_slice_id, dir='y')

            losses = update_total_losses(
                f_net, g_net, monoarray, train_config,
                fg_config, area_config, symm_config, arg_config,
                f_optimizer, g_optimizer, x_slice_id, y_slice_id)

            if f_scheduler is not None:
                f_scheduler.step()
            if g_scheduler is not None:
                g_scheduler.step()

            loss_file._write(epoch, **losses)

            if (epoch+1) % train_config._output_step[0] == 0:
                logging.info(textwrap.dedent(
                    f"""
                    Epoch [{epoch+1}/{num_epochs}]:
                        loss_fg: {losses['loss_fg']:.8g},
                        loss_area: ({losses['loss_area_x_avg']:.8g}, {losses['loss_area_y_avg']:.8g}),
                        loss_symm: ({losses['loss_symm_x_avg']:.8g}, {losses['loss_symm_y_avg']:.8g}, {losses['loss_symm_xy_ex']:.8g}),
                        loss_arg: ({losses['loss_arg_x']:.8g}, {losses['loss_arg_y']:.8g})
                    """
                    ))

            if (epoch+1) % train_config._output_step[1] == 0:
                save_models(f_net, g_net, f_save_path, g_save_path)
                with torch.no_grad():
                    plt_training_results(f_net, monoarray, input_config._input_radius, LogAbs=True)
    finally:
        loss_file.close()

    plt_record_loss(loss_path)        
    save_models(f_net, g_net, f_save_path, g_save_path)

    for handler in logging.getLogger().handlers:
        handler.close()
        logging.getLogger().removeHandler(handler)





class H5LossFile:
    def __init__(self, file_path: str, num_epochs: int):
        """
        This class is used to save the losses in the training process.

        Params:
            file_path (str): path of the loss file, should be an h5 file.
            num_epochs (int): determine the size of the datasets.
        """
        self._file_path = file_path
        self._num_epochs = num_epochs
        self._h5file = h5py.File(file_path, 'w')

        self.datasets = {
            'loss_fg': self._h5file.create_dataset('loss_fg', (num_epochs,)),
            'loss_area_x_avg': self._h5file.create_dataset('loss_area_x_avg', (num_epochs,)),
            'loss_area_y_avg': self._h5file.create_dataset('loss_area_y_avg', (num_epochs,)),
            'loss_symm_x_avg': self._h5file.create_dataset('loss_symm_x_avg', (num_epochs,)),
            'loss_symm_y_avg': self._h5file.create_dataset('loss_symm_y_avg', (num_epochs,)),
            'loss_symm_xy_ex': self._h5file.create_dataset('loss_symm_xy_ex', (num_epochs,)),
            'loss_arg_x': self._h5file.create_dataset('loss_arg_x', (num_epochs,)),
            'loss_arg_y': self._h5file.create_dataset('loss_arg_y', (num_epochs,)),
            'lr_f': self._h5file.create_dataset('lr_f', (num_epochs,)),
            'lr_g': self._h5file.create_dataset('lr_g', (num_epochs,)),
            }
        

    def _write(self, epoch: int, **kwargs):
        for key, value in kwargs.items():
            if key in self.datasets:
                self.datasets[key][epoch] = value
            else:
                raise KeyError(f"Dataset '{key}' not found in H5LossFile.")

    
    def close(self):
        self._h5file.close()





def init_temp_save_path(temp_save_fold: str, temp_save_tag: str):
    """
    Clear all the files under temp_save_fold, make a fold named temp_save_tag
        and return the path for saving the encoder and decoder.

    Params:
        temp_save_fold (str): target fold to create the tag-named fold.
        temp_save_tag (str): label of the current model. It will contain all the
            files for the model, including saved weights, loss data, logging...
    """
    for root, dirs, files in os.walk(temp_save_fold):
        for file in files:
            os.remove(os.path.join(root, file).replace('\\', '/'))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir).replace('\\', '/'))

    temp_save_path = os.path.join(temp_save_fold, temp_save_tag).replace('\\', '/')
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    f_save_path = os.path.join(temp_save_path, 'Encoder.pt').replace('\\', '/')
    g_save_path = os.path.join(temp_save_path, 'Decoder.pt').replace('\\', '/')
    loss_path = os.path.join(temp_save_path, 'loss_data.h5').replace('\\', '/')
    logging_path = os.path.join(temp_save_path, 'training_log.log').replace('\\', '/')

    return {'f_save_path': f_save_path, 
            'g_save_path': g_save_path,
            'loss_path': loss_path,
            'logging_path': logging_path}




def update_slice_losses(
        f_net: nn.Module, g_net: nn.Module, monoarray: MonoArray2D,
        train_config: TrainConfig, fg_config: FGLossConfig, area_config: AreaLossConfig,
        symm_config: SymmLossConfig, arg_config: ArgLossConfig,
        f_optimizer, g_optimizer, slice_id, dir: str):
    """
    Update the network for each slice along dir = 'x' or 'y'.
    """
    assert dir in ['x', 'y']

    inputs = train_config._input_config._create_mesh()
    zin = inputs['z']
    for id in slice_id:
        f_optimizer.zero_grad()
        g_optimizer.zero_grad()

        GBZ = f_net(zin)
        z_reconstructed = g_net(GBZ)
        Spectra = monoarray._get_spectra_from_GBZ(GBZ)
        
        match dir:
            case 'x':
                zin_conj = zin.clone() * torch.tensor([1, -1, 1, 1])
                GBZ_conj_input = f_net(zin_conj)

                loss_area_sliced = area_config._area_loss_x_avg(Spectra, [id])
                loss_symm_sliced = symm_config._symm_loss_x_avg(GBZ, GBZ_conj_input, [id])
                loss_arg = arg_config._arg_loss_x(GBZ, kx_lst=inputs['kx'])
            case 'y':
                zin_conj = zin.clone() * torch.tensor([1, 1, 1, -1])
                GBZ_conj_input = f_net(zin_conj)

                loss_area_sliced = area_config._area_loss_y_avg(Spectra, [id])
                loss_symm_sliced = symm_config._symm_loss_y_avg(GBZ, GBZ_conj_input, [id])
                loss_arg = arg_config._arg_loss_y(GBZ, ky_lst=inputs['ky'])

        loss_fg = fg_config._fg_loss(zin, z_reconstructed)
        loss_sliced = loss_fg + loss_area_sliced + loss_symm_sliced + loss_arg
        
        loss_sliced.backward()
        f_optimizer.step()
        g_optimizer.step()





def update_total_losses(
        f_net: nn.Module, g_net: nn.Module, monoarray: MonoArray2D,
        train_config: TrainConfig, fg_config: FGLossConfig, area_config: AreaLossConfig,
        symm_config: SymmLossConfig, arg_config: ArgLossConfig,
        f_optimizer, g_optimizer, x_slice_id, y_slice_id):
    """
    Update the network for the total losses.
    """
    f_optimizer.zero_grad()
    g_optimizer.zero_grad()

    inputs = train_config._input_config._create_mesh()
    zin = inputs['z']
    GBZ = f_net(zin)
    z_reconstructed = g_net(GBZ)
    Spectra = monoarray._get_spectra_from_GBZ(GBZ)

    zin_xconj = zin * torch.tensor([1, -1, 1, 1])
    zin_yconj = zin * torch.tensor([1, 1, 1, -1])
    GBZ_xconj_input = f_net(zin_xconj)
    GBZ_yconj_input = f_net(zin_yconj)

    loss_fg = fg_config._fg_loss(zin, z_reconstructed)
    loss_area_x_avg = area_config._area_loss_x_avg(Spectra, x_slice_id)
    loss_area_y_avg = area_config._area_loss_y_avg(Spectra, y_slice_id)
    loss_symm_x_avg = symm_config._symm_loss_x_avg(GBZ, GBZ_xconj_input)
    loss_symm_y_avg = symm_config._symm_loss_y_avg(GBZ, GBZ_yconj_input)
    loss_symm_xy_ex = symm_config._symm_loss_xy_ex(GBZ)
    loss_arg_x = arg_config._arg_loss_x(GBZ, kx_lst=inputs['kx'])
    loss_arg_y = arg_config._arg_loss_y(GBZ, ky_lst=inputs['ky'])

    loss_total = (loss_fg + loss_area_x_avg + loss_area_y_avg +
                  loss_symm_x_avg + loss_symm_y_avg + loss_symm_xy_ex +
                  loss_arg_x + loss_arg_y)
    
    loss_total.backward()
    f_optimizer.step()
    g_optimizer.step()

    return {
        'loss_fg': loss_fg.item(),
        'loss_area_x_avg': loss_area_x_avg.item(),
        'loss_area_y_avg': loss_area_y_avg.item(),
        'loss_symm_x_avg': loss_symm_x_avg.item(),
        'loss_symm_y_avg': loss_symm_y_avg.item(),
        'loss_symm_xy_ex': loss_symm_xy_ex.item(),
        'loss_arg_x': loss_arg_x.item(),
        'loss_arg_y': loss_arg_y.item(),
        'lr_f': f_optimizer.param_groups[0]['lr'],
        'lr_g': g_optimizer.param_groups[0]['lr']}












#%%
### Residual block
class ResBlock(nn.Module):
    def __init__(self, feature_lst, ln_affine=True, zero_init=False):
        """
        Residual network with tunable features.
            -Use LayerNorm after each Linear layer.
            -Use relu as the active function.

        Params:
            feature_lst (list/tuple): features for each layer, should have form 
                [in_feature, hidden_feature1, hidden_feature2,... out_feature]
            ln_affine (bool): 'elementwise_affine' for LayerNoem layers.
            zero_init (bool): if zero_init, use zero to initialize the output layer
                to make sure the output equal to the input.
        """
        super(ResBlock, self).__init__()
        self._feature_lst = feature_lst
        self._ln_params = ln_affine
        self._zero_init = zero_init

        self._fc_lst = nn.ModuleList(
            [nn.Linear(feature_lst[i], feature_lst[i+1]) 
            for i in range(len(feature_lst)-1)]
        )

        self._ln_lst = nn.ModuleList(
            [nn.LayerNorm(feature_lst[i+1], elementwise_affine=ln_affine)
            for i in range(len(feature_lst)-2)]
        )
            
        if zero_init:
            self.zero_initialization()


    def forward(self, x):
        res = x

        for fc, ln in zip(self._fc_lst[:-1], self._ln_lst):
            res = fc.forward(res)
            res = ln.forward(res)
            res = torch.relu(res)

        res = self._fc_lst[-1](res)
        return x + res


    def zero_initialization(self):
        nn.init.zeros_(self._fc_lst[-1].weight)
        nn.init.zeros_(self._fc_lst[-1].bias)












#%%
### encoder & decoder
class FNet(nn.Module):
    def __init__(self):
        super(FNet, self).__init__()
        self._res1 = ResBlock((4, 64, 128, 128, 128, 64, 4), zero_init=True)

    def forward(self, x):
        x = self._res1(x)
        return x


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self._res1 = ResBlock((4, 64, 128, 128, 128, 64, 4), zero_init=True)

    def forward(self, x):
        x = self._res1(x)
        return x
    











#%%
### train
f_net = FNet().double()
g_net = GNet().double()

target_monoarray = MonoArray2D(model_name= 'HN2D', 
                               model_params= (1, 3, 2, 4), 
                               scale_factor= 1)

input_config = InputConfig(
    num_input= (301, 301),
    rand_input= True,
    input_radius= (1, 1))

slice_config = SliceConfig(
    x_slice_id= None,
    y_slice_id= None,
    slice_ratio= (None, None),
    random_slice= True)

optim_config = OptimConfig(
    optimizer_class= optim.Adam,
    optimizer_params= {'lr': 5e-4, 'weight_decay': 1e-5},
    scheduler_class= lr_scheduler.StepLR,
    scheduler_params= {'step_size': 200, 'gamma': 0.9})

train_config = TrainConfig(
    num_epochs= 2,
    init_path= None,
    input_config= input_config,
    slice_config= slice_config,
    optim_config= optim_config,
    update_each_slice= False,
    output_step= (1, 1))

fg_config = FGLossConfig(
    loss_weights= 100,
    func_fg= nn.MSELoss(reduction='mean'))

area_config = AreaLossConfig(
    loss_weights= (1, 1),
    func_area_x= AreaLoss(farea=PolyArea_SumAbs()),
    func_area_y= AreaLoss(farea=PolyArea_SumAbs()))

symm_config = SymmLossConfig(
    loss_weights= (50, 50, 0),
    func_symm_x= SymmLoss_X(full_dim=True, fdist=nn.L1Loss(reduction='mean')),
    func_symm_y= SymmLoss_Y(full_dim=True, fdist=nn.L1Loss(reduction='mean')),
    func_symm_xy_ex= None)

arg_config = ArgLossConfig(
    loss_weights= (10, 10),
    func_arg_x= ArgLoss_X(fdist=nn.L1Loss(reduction='mean')),
    func_arg_y= ArgLoss_Y(fdist=nn.L1Loss(reduction='mean')),
    use_target= True)


train_models(f_net, g_net, target_monoarray,
             train_config, fg_config, area_config, symm_config, arg_config)















    





