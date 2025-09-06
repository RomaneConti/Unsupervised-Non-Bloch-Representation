
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
from itertools import product
from typing import Any, Union, Optional, Type, List, Dict, Tuple, Callable

from plot_helpers import*
from Area_Iter import PolyArea


# change default dtype of torch.tensor to float64
torch.set_default_dtype(torch.float64)











#%%
### helper functions
def generate_input_mesh(num_input, rand_input=False, 
                        input_radius: Tuple[float,...]=None):
    """
    Generate the input data [cos(kx), sin(kx), cos(ky), sin(ky), cos(kz), sin(kz)] for the encoder.
    Return a 4 dimensional tensor, the 1st/2nd/3rd dimension coresponds to t1(kx)/t2(ky)/t3(kz). 
    So, data[id, :, :, :] means fixing kx and slicing along ky and kz while
        data[:, id, :, :] means fixing ky and slicing along kx and kz,
        data[:, :, id, :] means fixing kz and slicing along kx and ky.

    Params:
        num_input (tuple): tuple with 3 ints, number of mesh points along t1(kx), t2(ky), and t3(kz).
        rand_input (bool): whether to add a small offset to the mesh.
        input_radius (tuple): should have form (rx, ry, rz). The mesh can be seen as 
            traversing three circles with radius rx, ry, and rz.
    """
    input_radius = input_radius or (1.,)*3
    assert len(num_input) == 3 and len(input_radius) == 3

    t_vec = [(2*torch.rand(1)/n if rand_input else 0) + torch.linspace(-1, 1, n) for n in num_input]
    t_grids = torch.meshgrid(*t_vec, indexing='ij')

    dim_mapping = {0: 'x', 1: 'y', 2: 'z'}
    k_int = {f'k{i}': torch.pi * (-1 + (t+1).remainder(2)) for i, t in enumerate(t_vec)}
    k_str = {f'k{dim_mapping.get(i, i)}': torch.pi * (-1 + (t+1).remainder(2)) for i, t in enumerate(t_vec)}

    z = torch.cat([
        torch.stack((torch.cos(torch.pi * grid) * r, torch.sin(torch.pi * grid) * r), dim=-1)
        for grid, r in zip(t_grids, input_radius)], dim=-1)

    return {**k_int, **k_str, 'z': z}





def get_xyzGBZ(GBZ: torch.Tensor):
    """
    Get βx, βy and βz, return as tensor with shape (..., 2).
    With the 1st layer Re[β] and the 2nd layer Im[β].

    Params:
        GBZ (tensor): a tensor with shape (..., 6). Usually the output of the encoder.
    """
    return tuple(GBZ[..., 2*i : 2*i+2] for i in range(3))





def get_xyzGBZ_cform(GBZ: torch.Tensor):
    """
    Get βx, βy and βz, return as their complex forms.

    Params:
        GBZ (tensor): a tensor with shape (..., 6). Usually the output of the encoder.
    """
    return tuple(GBZ[..., 2*i] + 1.j*GBZ[..., 2*i+1] for i in range(3))





def return_zero(*args, **kwargs):
    """
    Return zero for all inputs. This will be useful in those loss configs.
    """
    return torch.tensor(0.)





def my_product(*args):
    """
    Treat non-interable objects as place-holders and then do the product.
    This will be used in generating slice incides for certain dimension.
    """
    fixed_args = [elem if hasattr(elem, '__iter__') else [elem] for elem in args]
    return product(*fixed_args)












#%%
### 3D single-band model
class MonoArray3D:
    def __init__(self, model_name: str, model_params: Optional[Tuple[float,...]], scale_factor: float = 1.):
        """
        Class of the 3D single-band model.
        For simplicity, we do not need a general approach to define the 3D model.
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
    def _spectra_cform_HN3D(xGBZ_cform, yGBZ_cform, zGBZ_cform, params):
        # params should be given with form (Lx, Rx, Ly, Ry, Lz, Rz)
        assert len(params) == 6, "6 params (Lx, Rx, Ly, Ry, Lz, Rz) are needed for an HN3D model."
        return (params[0] / xGBZ_cform + params[1] * xGBZ_cform +
                params[2] / yGBZ_cform + params[3] * yGBZ_cform +
                params[4] / zGBZ_cform + params[5] * zGBZ_cform)


    def _get_spectra_from_GBZ(self, GBZ: torch.Tensor):
        """
        Get the spectra from the GBZ.
        
        Params:
            GBZ (tensor): a tensor with shape (..., 6). Usually the output of the encoder.
                The last dimension corresponds to Re/Im[βx], Re/Im[βy], Re/Im[βz] in order.
        """
        assert GBZ.size(-1) == 6, "GBZ should be an (..., 6) tensor."

        xGBZ_cform, yGBZ_cform, zGBZ_cform = get_xyzGBZ_cform(GBZ)
        registered_model = {
            'HN3D': self._spectra_cform_HN3D,
        }

        if self._model_name not in registered_model:
            raise ValueError(f"Undefined model: '{self._model_name}'.")
        E_cform = registered_model[self._model_name](xGBZ_cform, yGBZ_cform, zGBZ_cform, self._model_params)
        return torch.stack((E_cform.real, E_cform.imag), dim=-1)
        

    def __repr__(self):
        return f"MonoArray3D(name={self._model_name}, params={self._model_params})"












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
                 loss_weights: Tuple[float,...] = (1, 1, 1),
                 func_area_x: Optional[Callable] = AreaLoss(),
                 func_area_y: Optional[Callable] = AreaLoss(),
                 func_area_z: Optional[Callable] = AreaLoss()):
        """
        This class determines how to calculate the area loss.

        Params:
            loss_weights (tuple): weights of loss_area along kx and ky in order.
            func_area_x/y/z (function): determine how to calculate the area along kx/ky/kz (other 2 k's fixed).
        """
        self._total_dims = 3
        self._dim_mapping = {'x': 0, 'y': 1, 'z': 2}

        assert len(loss_weights) == self._total_dims, \
            f"loss_weights should be a tuple with {self._total_dims} float elements."
        self._loss_weights = tuple(map(float, loss_weights))
        func_area = {
            'x': func_area_x or return_zero,
            'y': func_area_y or return_zero,
            'z': func_area_z or return_zero,}
        self._func_area = {self._dim_mapping.get(k, k): v for k, v in func_area.items()}


    def _area_loss_avg(self, Spectra: torch.Tensor, dim: Union[str, int] = 'x',
                       slice_config: Optional[Dict] = None):
        assert Spectra.dim() == self._total_dims + 1 and Spectra.size(-1) == 2, \
            f"Spectra should be an (..., 2) tensor with total dims={self._total_dims + 1}."
        
        dim = self._dim_mapping.get(dim, dim)
        # assert dim in {i for i in range(self._total_dims)}, \
        #     f"dim should be a nonnegative integer smaller than {self._total_dims}."

        slice_config = slice_config or {}
        slice_config_intkey = {self._dim_mapping.get(k, k): v for k, v in slice_config.items()}

        indices = [
            slice_config_intkey.get(d, range(Spectra.size(d)))
            for d in range(self._total_dims)]
        indices[dim] = slice(None)
        remain_indices = indices[: dim] + indices[dim+1 :]
        num_slice = torch.tensor([len(elem) for elem in remain_indices]).prod()

        total_area = sum(
            self._func_area[dim](Spectra[idx])
            for idx in my_product(*indices))
        
        return self._loss_weights[dim] * total_area / num_slice

    
    
    # def _area_loss_x_avg(self, Spectra: torch.Tensor, idy = None, idz = None):
    #     assert Spectra.dim() == 4 and Spectra.size(3) == 2, "Spectra should be an (Nx, Ny, Nz, 2) tensor."
    #     idy = idy or range(Spectra.size(1))
    #     idz = idz or range(Spectra.size(2))
    #     total_area = sum(
    #         self._func_area[0](Spectra[:, yy, zz])
    #         for yy in idy for zz in idz)
    #     return self._loss_weights[0] * total_area / len(idy) / len(idz)
    

    # def _area_loss_y_avg(self, Spectra: torch.Tensor, idx = None, idz = None):
    #     assert Spectra.dim() == 4 and Spectra.size(3) == 2, "Spectra should be an (Nx, Ny, Nz, 2) tensor."
    #     idx = idx or range(Spectra.size(0))
    #     idz = idz or range(Spectra.size(2))
    #     total_area = sum(
    #         self._func_area[1](Spectra[xx, :, zz])
    #         for xx in idx for zz in idz)
    #     return self._loss_weights[1] * total_area / len(idx) / len(idz)
    

    # def _area_loss_z_avg(self, Spectra: torch.Tensor, idx = None, idy = None):
    #     assert Spectra.dim() == 4 and Spectra.size(3) == 2, "Spectra should be an (Nx, Ny, Nz, 2) tensor."
    #     idx = idx or range(Spectra.size(0))
    #     idy = idy or range(Spectra.size(1))
    #     total_area = sum(
    #         self._func_area[2](Spectra[xx, yy, :])
    #         for xx in idx for yy in idy)
    #     return self._loss_weights[2] * total_area / len(idx) / len(idy)












#%%
### symmetric loss functions
def symm_loss_exchange(GBZ: torch.Tensor, dim: Tuple = ('x', 'y'), fdist = nn.L1Loss(reduction='mean')):
    """
    A loss making |βi|=|βj|, i,j = x, y, z.
    This loss can be used when the model is symmetric for exchanging βi and βj.
    """
    dim_mapping = {'x': 0, 'y': 1, 'z': 2}
    dim = tuple(dim_mapping.get(d, d) for d in dim)
    # assert all(d in {0, 1, 2} for d in dim), "dim can only be 0/'x', 1/'y', 2/'z'."
    GBZ_cform = get_xyzGBZ_cform(GBZ)
    return fdist(GBZ_cform[dim[0]].abs(), GBZ_cform[dim[1]].abs())
    

class SymmLoss_Exchange:
    def __init__(self, dim: Tuple = ('x', 'y'), fdist = nn.L1Loss(reduction='mean')):
        self._dim = dim
        self._fdist = fdist

    def __call__(self, GBZ, dim: Tuple = ('x', 'y')):
        return symm_loss_exchange(GBZ, self._dim, self._fdist)





def symm_loss(GBZ: torch.Tensor, GBZ_conj_input: torch.Tensor,
              dim: Union[int, str] = 'x', full_dim: bool = True,
              fdist = nn.L1Loss(reduction='mean')):
    """
    A loss making the GBZ symmetric about certain dim.

    Params:
        GBZ (tensor): Usually the output of the encoder.
        GBZ_conj_input (torch.tensor): also a tensor with the same shape as GBZ.
            Obtained using the conj-input mesh (multiply the (2*dim+1)th layer by -1) for the encoder.
        dim (int, str): determine the dim that needs to be compared.
        full_dim (bool): whether to include the remaining dims when calculating the distance.
        fdist (function): a function measuring the distance between two tensors.
    """
    dim_mapping = {'x': 0, 'y': 1, 'z': 2}
    dim = dim_mapping.get(dim, dim)
    # assert dim in {0, 1, 2}, "dim can only be 0/'x', 1/'y', 2/'z'."
    assert GBZ_conj_input.shape == GBZ.shape
    mask = torch.ones(6)
    mask[2*dim+1] = -1
    GBZ_conj = GBZ * mask
    calc_dim = slice(None) if full_dim else slice(2*dim, 2*dim+2)
    return fdist(GBZ_conj[..., calc_dim], GBZ_conj_input[..., calc_dim])


class SymmLoss:
    def __init__(self, dim: Union[int, str] = 'x', full_dim: bool = True,
                 fdist = nn.L1Loss(reduction='mean')):
        self._dim = dim
        self._full_dim = full_dim
        self._fdist = fdist

    def __call__(self, GBZ, GBZ_conj_input):
        return symm_loss(GBZ, GBZ_conj_input, dim=self._dim, 
                         full_dim=self._full_dim, fdist=self._fdist)






# def symm_loss_x(GBZ: torch.Tensor, GBZ_xconj_input: torch.Tensor,
#                 full_dim: bool = False, fdist = nn.L1Loss(reduction='mean')):
#     """
#     A loss making βx symmetric about kx=0.

#     Params:
#         GBZ (tensor): a tensor with shape (..., 4). Usually the output of the encoder.
#         GBZ_conj_input (torch.tensor): also a tensor with shape (..., 4).
#             Obtained using the x-conj input mesh (multiply the 2nd layer by -1) for the encoder.
#         full_dim (bool): whether to include βy(the last 2 dims) when calculating the distance.
#         fdist (function): a function measuring the distance between two tensors.
#     """
#     # assert GBZ.size(-1) == 4, "GBZ should be a (..., 4) tensor."
#     assert GBZ_xconj_input.shape == GBZ.shape
#     GBZ_xconj = GBZ * torch.tensor([1, -1, 1, 1])
#     calc_dim = slice(None) if full_dim else slice(None, 2)
#     return fdist(GBZ_xconj[..., calc_dim], GBZ_xconj_input[..., calc_dim])

# class SymmLoss_X:
#     def __init__(self, full_dim: bool = False, fdist = nn.L1Loss(reduction='mean')):
#         self._full_dim = full_dim
#         self._fdist = fdist

#     def __call__(self, GBZ, GBZ_conj_input):
#         return symm_loss_x(GBZ, GBZ_conj_input, self._full_dim, self._fdist)
    




# def symm_loss_y(GBZ: torch.Tensor, GBZ_yconj_input: torch.Tensor,
#                 full_dim: bool = False, fdist = nn.L1Loss(reduction='mean')):
#     """
#     A loss making βy symmetric about ky=0.

#     Params:
#         GBZ (tensor): a tensor with shape (..., 4). Usually the output of the encoder.
#         GBZ_yconj_input (torch.tensor): also a tensor with shape (..., 4).
#             Obtained using the y-conj input mesh (multiply the 4nd layer by -1) for the encoder.
#         full_dim (bool): whether to include βx(the first 2 dims) when calculating the distance.
#         fdist (function): a function measuring the distance between two tensors.
#     """
#     # assert GBZ.size(-1) == 4, "GBZ should be an (..., 4) tensor."
#     assert GBZ_yconj_input.shape == GBZ.shape
#     GBZ_yconj = GBZ * torch.tensor([1, 1, 1, -1])
#     calc_dim = slice(None) if full_dim else slice(2, None)
#     return fdist(GBZ_yconj[..., calc_dim], GBZ_yconj_input[..., calc_dim])

# class SymmLoss_Y:
#     def __init__(self, full_dim: bool = False, fdist = nn.L1Loss(reduction='mean')):
#         self._full_dim = full_dim
#         self._fdist = fdist

#     def __call__(self, GBZ, GBZ_conj_input):
#         return symm_loss_y(GBZ, GBZ_conj_input, self._full_dim, self._fdist)
    



class SymmLossConfig:
    def __init__(self,
                 loss_weights: Tuple[float,...] = (50, 50, 50, 0),
                 func_symm_x: Optional[Callable] = None,
                 func_symm_y: Optional[Callable] = None,
                 func_symm_z: Optional[Callable] = None,
                 func_symm_ex: Optional[Callable] = None):
        """
        This class determines how to calculate the symm loss.

        Params:
            loss_weights (tuple): weights of loss_symm for x_sliced, y_sliced, and exchanged in order.
            func_symm_x/y/z (function): determine how to calculate the symm loss along kx/y/z.
            func_symm_xy_exchange (function): determine how to calculate the exchanged symm loss.
        """
        self._total_dims = 3
        self._dim_mapping = {'x': 0, 'y': 1, 'z': 2}

        assert len(loss_weights) == self._total_dims + 1, \
            f"loss_weights should be a tuple with {self._total_dims + 1} float elements."
        self._loss_weights = tuple(map(float, loss_weights))
        func_symm = {
            'x': func_symm_x or return_zero,
            'y': func_symm_y or return_zero,
            'z': func_symm_z or return_zero,}
        self._func_symm = {self._dim_mapping.get(k, k): v for k, v in func_symm.items()}
        self._func_symm_ex = func_symm_ex or return_zero
    

    def _symm_loss_avg(self, GBZ: torch.Tensor, GBZ_conj_input: torch.Tensor,
                       dim: Union[str, int] = 'x', slice_config: Optional[Dict] = None):
        assert GBZ.dim() == self._total_dims + 1 and GBZ.size(-1) == 2 * self._total_dims, \
            f"GBZ should be a (..., {2*self._total_dims}) tensor with total dim = {self._total_dims + 1}."
        
        dim = self._dim_mapping.get(dim, dim)
        # assert dim in {i for i in range(self._total_dims)}, \
        #     f"dim should be a nonnegative integer smaller than {self._total_dims}."

        slice_config = slice_config or {}
        slice_config_intkey = {self._dim_mapping.get(k, k): v for k, v in slice_config.items()}

        indices = [
            slice_config_intkey.get(d, range(GBZ.size(d)))
            for d in range(self._total_dims)]
        indices[dim] = slice(None)

        remain_indices = indices[: dim] + indices[dim+1 :]
        num_slice = torch.tensor([len(elem) for elem in remain_indices]).prod()

        loss = sum(
            self._func_symm[dim](GBZ[idx], GBZ_conj_input[idx])
            for idx in my_product(*indices))
        
        return self._loss_weights[dim] * loss / num_slice

    
    # def _symm_loss_x_avg(self, GBZ, GBZ_xconj_input, id1 = None, id2 = None):
    #     # assert GBZ.dim() == 3 and GBZ.size(-1) == 4, "GBZ should be an (Nx, Ny, 4) tensor."
    #     # assert GBZ.shape == GBZ_xconj_input.shape
    #     id1 = id1 or range(GBZ.size(1))
    #     id2 = id2 or range(GBZ.size(2))
    #     # assert hasattr(slice_id, '__len__')
    #     loss = sum(
    #         self._func_symm[0](GBZ[:, yy, zz], GBZ_xconj_input[:, yy, zz])
    #         for yy in id1 for zz in id2)
    #     return self._loss_weights[0] * loss / len(id1) / len(id2)
    
    
    # def _symm_loss_y_avg(self, GBZ, GBZ_yconj_input, slice_id = None):
    #     assert GBZ.dim() == 3 and GBZ.size(-1) == 4, "GBZ should be an (Nx, Ny, 4) tensor."
    #     # assert GBZ.shape == GBZ_yconj_input.shape
    #     slice_id = slice_id or range(GBZ.size(0))
    #     # assert hasattr(slice_id, '__len__')
    #     loss = sum(
    #         self._func_symm_y(GBZ[id,...], GBZ_yconj_input[id,...])
    #         for id in slice_id)
    #     return self._loss_weights[1] * loss / len(slice_id)
    

    def _symm_loss_ex(self, GBZ):
        return self._loss_weights[3] * self._func_symm_ex(GBZ)












#%%
### arg loss functions
def arg_loss(GBZ: torch.Tensor, k_lst: Optional[torch.Tensor] = None,
             dim: Union[str, int] = 'x',  fdist: Callable = nn.MSELoss(reduction='mean')):
    """
    A loss making arg[β] constant along certain dim.

    Params:
        GBZ (tensor): a tensor with shape (..., 6). Usually the output of the encoder.
        k_lst (tensor): if given, should be a tensor with Nx elements.
            We make arg[β] equals to given k's along each slice.
        dim (int, str): determine the dim that needs to be compared.
        fdist (function): function measuring the distance between arg[βx] and given kx.
    """
    dim_mapping = {'x': 0, 'y': 1, 'z': 2}
    dim = dim_mapping.get(dim, dim)
    # assert all(d in {0, 1, 2} for d in dim), "dim can only be 0/'x', 1/'y', 2/'z'."

    arg = torch.atan2(GBZ[..., 2*dim+1], GBZ[..., 2*dim])
    if k_lst is None:
        reamin_dims = tuple(i for i in range(3) if i != dim)
        return torch.mean(arg.amax(dim=reamin_dims) - arg.amin(dim=reamin_dims))
    else:
        assert k_lst.shape == (arg.size(dim),), f"k_lst should be a vectorr tensor with length {arg.size(dim)}."
        view_shape = [1] * 3
        view_shape[dim] = -1
        k_view = k_lst.view(view_shape)
        return fdist(arg, k_view.expand(arg.shape))


class ArgLoss:
    def __init__(self, dim: Union[int, str] = 'x', fdist = nn.MSELoss(reduction='mean')):
        self._dim = dim
        self._fdist = fdist

    def __call__(self, GBZ, k_lst = None):
        return arg_loss(GBZ, k_lst, dim=self._dim, fdist=self._fdist)





# def arg_loss_x(GBZ: torch.Tensor, kx_lst: Optional[torch.Tensor] = None,
#                fdist: Callable = nn.MSELoss(reduction='mean')):
#     """
#     A loss making arg[βx] constant along y-slices (fixed x, t1).

#     Params:
#         GBZ (tensor): a tensor with shape (Nx, Ny, 4). Usually the output of the encoder.
#         kx_lst (tensor): if given, should be a tensor with Nx elements.
#             We make arg[βx] equals to given kx's along each slice.
#         fdist (function): function measuring the distance between arg[βx] and given kx.
#     """
#     # assert GBZ.dim() == 3 and GBZ.size(-1) == 4, "GBZ should be an (Nx, Ny, 4) tensor."
#     argx = torch.atan2(GBZ[..., 1], GBZ[..., 0])
#     if kx_lst is None:
#         return torch.mean(argx.max(dim=1).values - argx.min(dim=1).values)
#     else:
#         assert kx_lst.shape == (argx.size(0),), f"kx_lst should be a vector tensor with length {argx.size(0)}."
#         return fdist(argx, kx_lst.unsqueeze(1).repeat(1, argx.size(1)))
#         # return torch.mean((kx.max(dim=1).values - kx_lst).abs() + (kx.min(dim=1).values - kx_lst).abs())
    
# class ArgLoss_X:
#     def __init__(self, fdist = nn.MSELoss(reduction='mean')):
#         self._fdist = fdist

#     def __call__(self, GBZ: torch.Tensor, kx_lst = None):
#         return arg_loss_x(GBZ, kx_lst, self._fdist)
    





# def arg_loss_y(GBZ: torch.Tensor, ky_lst: Optional[torch.Tensor] = None,
#                fdist: Callable = nn.MSELoss(reduction='mean')):
#     """
#     A loss making arg[βy] constant along x-slices (fixed y, t2).

#     Params:
#         GBZ (tensor): a tensor with shape (Nx, Ny, 4). Usually the output of the encoder.
#         ky_lst (tensor): if given, should be a tensor with Ny elements.
#             We make arg[βy] equals to given ky's along each slice.
#         fdist (function): function measuring the distance between arg[βy] and given ky.
#     """
#     # assert GBZ.dim() == 3 and GBZ.size(-1) == 4, "GBZ should be an (Nx, Ny, 4) tensor."
#     argy = torch.atan2(GBZ[..., 3], GBZ[..., 2])
#     if ky_lst is None:
#         return torch.mean(argy.max(dim=0).values - argy.min(dim=0).values)
#     else:
#         assert ky_lst.shape == (argy.size(1),), f"ky_lst should be a vector tensor with length {argy.size(1)}."
#         return fdist(argy, ky_lst.repeat(argy.size(0), 1))
#         # return torch.mean((ky.max(dim=0).values - ky_lst).abs() + (ky.min(dim=0).values - ky_lst).abs())
    
# class ArgLoss_Y:
#     def __init__(self, fdist = nn.MSELoss(reduction='mean')):
#         self._fdist = fdist

#     def __call__(self, GBZ: torch.Tensor, ky_lst = None):
#         return arg_loss_y(GBZ, ky_lst, self._fdist)
    




class ArgLossConfig:
    def __init__(self,
                 loss_weights: Tuple[float,...] = (1, 1, 1),
                 func_arg_x: Optional[Callable] = None,
                 func_arg_y: Optional[Callable] = None,
                 func_arg_z: Optional[Callable] = None,
                 use_target: bool = False):
        """
        This class determines how to calculate the arg loss.

        Params:
            loss_weights (tuple): weights of loss_arg for x and y direction in order.
            func_arg_x/y/z (function): determine how to calculate the arg loss for kx/y/z.
            use_target (bool): if use_target, kx,y_lst are needed in the calculation.
        """
        self._total_dims = 3
        self._dim_mapping = {'x': 0, 'y': 1, 'z': 2}

        assert len(loss_weights) == self._total_dims, \
            f"loss_weights should be a tuple with {self._total_dims} float elements."
        self._loss_weights = tuple(map(float, loss_weights))
        self._use_target = use_target

        func_arg = {
            'x': func_arg_x or return_zero,
            'y': func_arg_y or return_zero,
            'z': func_arg_z or return_zero,}
        self._func_arg = {self._dim_mapping.get(k, k): v for k, v in func_arg.items()}

    
    def _arg_loss_avg(self, GBZ, k_lst = None, dim: Union[str, int] = 'x'):
        assert GBZ.dim() == self._total_dims + 1 and GBZ.size(-1) == 2 * self._total_dims, \
            f"GBZ should be a (..., {2*self._total_dims}) tensor with total dim = {self._total_dims + 1}."
        
        dim = self._dim_mapping.get(dim, dim)
        # assert dim in {i for i in range(self._total_dims)}, \
        #     f"dim should be a nonnegative integer smaller than {self._total_dims}."

        k_lst = k_lst if self._use_target else None
        assert not self._use_target or k_lst is not None, "k_lst is needed. Or set _use_target as False to avoid this."

        return self._loss_weights[dim] * self._func_arg[dim](GBZ, k_lst)


    # def _arg_loss_x(self, GBZ, kx_lst):
    #     kx_lst = kx_lst if self._use_target else None
    #     assert not self._use_target or kx_lst is not None, "kx_lst is needed. Or set _use_target as False to avoid this."
    #     return self._loss_weights[0] * self._func_arg_x(GBZ, kx_lst)
    

    # def _arg_loss_y(self, GBZ, ky_lst):
    #     ky_lst = ky_lst if self._use_target else None
    #     assert not self._use_target or ky_lst is not None, "ky_lst is needed. Or set _use_target as False to avoid this."
    #     return self._loss_weights[1] * self._func_arg_y(GBZ, ky_lst)
        














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
    f_net_jit = torch.jit.trace(f_net, example_inputs=torch.rand(1, 6))
    g_net_jit = torch.jit.trace(g_net, example_inputs=torch.rand(1, 6))
    f_net_jit.save(f_save_path)
    g_net_jit.save(g_save_path)





def plt_training_results(encoder: nn.Module, monoarray: MonoArray3D,
                         input_radius: Tuple = (1,)*3, clim: Tuple = (0.5, 0.9)):
    """
    Plot the current results in the training process.

    Params:
        encoder (nn.Module): the encoder.
        monoarray (MonoArray3D): target 3D model.
        input_radius (tuple): params for generating the input mesh.
        clim (tuple): determine the clim of the 3D slice plot.
    """
    zin = generate_input_mesh((100, 100, 100), rand_input=True, input_radius=input_radius)['z']
    GBZ = encoder(zin)
    Spectra = monoarray._get_spectra_from_GBZ(GBZ)

    plt_GBZ_slice(GBZ, clim=clim)
    plt_GBZ_quiver_and_spectra(
        GBZ, Spectra, num_quiver=(7, 7, 7), num_spectra=(20, 20, 20))





def test_models(monoarray: MonoArray3D, load_fold: str, input_radius: Tuple = (1,)*3, clim: Tuple = (0, 2)):
    """
    Test the trained encoder and decoder.
    There should be files 'Encoder.pt' and 'Decoder.pt' under load_fold.
    """
    try:
        f_net_test = torch.jit.load(os.path.join(load_fold, 'Encoder.pt').replace('\\', '/'))
    except:
        raise Exception(f'No available file in folder {load_fold}')

    with torch.no_grad():
        plt_training_results(f_net_test, monoarray, input_radius, clim)












#%%
### train config
class InputConfig:
    def __init__(self,
                 num_input: Tuple[int, int] = (101, 101, 101),
                 rand_input: bool = True,
                 input_radius: Tuple[float, float] = (1,)*3):
        """
        This class determine the properties of the input mesh.

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





class OutputConfig:
    def __init__(self,
                 logging_step: int = 100,
                 temp_save_step: int = 300,
                 plot_kwargs: Dict = {'clim': (0, 2)}):
        """
        This class determine the output behaviors in the training process.

        Params:
            logging_step (int): determine after how many epochs will the logging work.
            temp_save_step (int): determine after how many epochs will the results be saved.
                The training results will also be plotted.
            plot_kwargs: possible inputs for function plt_training_results
        """
        self._logging_step = logging_step
        self._temp_save_step = temp_save_step
        self._plot_kwwargs = plot_kwargs





class SliceConfig:
    def __init__(self):
        pass
    
    def _create_slice_dict(self):
        return ({}, {}, {})





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
                 update_each_slice: bool = False, 
                 output_config: OutputConfig = OutputConfig()):
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
            output_config (OutputConfig): determine the output behaviors in the training process.
        """
        self._num_epochs = num_epochs
        self._init_path = init_path
        self._input_config = input_config
        self._slice_config = slice_config
        self._optim_config = optim_config
        self._update_each_slice = update_each_slice
        self._output_config = output_config
        self._validate_slice_range()


    def _validate_slice_range(self):
        pass





class LossConfig:
    def __init__(self,
                 fg_config: FGLossConfig = FGLossConfig(),
                 area_config: AreaLossConfig = AreaLossConfig(),
                 symm_config: SymmLossConfig = SymmLossConfig(),
                 arg_config: ArgLossConfig = ArgLossConfig(),):
        """
        This class determines how to calculate the losses in training.

        Params:
            fg_config (FGLossConfig): configuration of the encoder-decoder loss.
            area_config (AreaLossConfig): configuration of the area loss.
            symm_config (SymmLossConfig): configuration of the symmetric loss.
            arg_config (ArgLossConfig): configuration of the arg loss.
        """
        self._fg_config = fg_config
        self._area_config = area_config
        self._symm_config = symm_config
        self._arg_config = arg_config












#%%
### main training functions
def train_models(f_net: nn.Module, g_net: nn.Module,
                 monoarray: MonoArray3D,
                 train_config: TrainConfig = TrainConfig(),
                 loss_config: LossConfig = LossConfig()):
    """
    Train the model with given conditions.

    Params:
        f_net, g_net (nn.Module): encoder and decoder.
        monoarray (MonoArray3D): target single-band system.
        train_config (TrainConfig): the configuration parameters for training.
        loss_config (LossConfig): the configuration for how to calculate the losses.
    """
    for handler in logging.getLogger().handlers:
        handler.close()
        logging.getLogger().removeHandler(handler)

    init_models(f_net, g_net, train_config._init_path)

    input_config = train_config._input_config
    slice_config = train_config._slice_config
    optim_config = train_config._optim_config
    output_config = train_config._output_config

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
            slice_dicts = slice_config._create_slice_dict()

            if train_config._update_each_slice:
                pass
            
            losses = update_total_losses(
                f_net, g_net, monoarray, train_config, loss_config,
                f_optimizer, g_optimizer, slice_dicts)

            if f_scheduler is not None:
                f_scheduler.step()
            if g_scheduler is not None:
                g_scheduler.step()

            loss_file._write(epoch, **losses)
                
            if (epoch+1) % output_config._logging_step == 0:
                logging.info(textwrap.dedent(
                    f"""
                    Epoch [{epoch+1}/{num_epochs}]:
                        loss_fg: {losses['loss_fg']:.8g},
                        loss_area: {", ".join(f"{x:.8g}" for x in losses['loss_area'])},
                        loss_symm: {", ".join(f"{x:.8g}" for x in losses['loss_symm'])}, loss_symm_ex: {losses['loss_symm_ex']:.8g},
                        loss_arg: {", ".join(f"{x:.8g}" for x in losses['loss_arg'])},
                    """
                    ))

            if (epoch+1) % output_config._temp_save_step == 0:
                save_models(f_net, g_net, f_save_path, g_save_path)
                with torch.no_grad():
                    plt_training_results(
                        f_net, monoarray, input_config._input_radius, **output_config._plot_kwwargs)
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
            'loss_area': self._h5file.create_dataset('loss_area', (3, num_epochs)),
            'loss_symm': self._h5file.create_dataset('loss_symm', (3, num_epochs)),
            'loss_symm_ex': self._h5file.create_dataset('loss_symm_ex', (num_epochs,)),
            'loss_arg': self._h5file.create_dataset('loss_arg', (3, num_epochs)),
            'lr_f': self._h5file.create_dataset('lr_f', (num_epochs,)),
            'lr_g': self._h5file.create_dataset('lr_g', (num_epochs,)),
            }
        

    def _write(self, epoch: int, **kwargs):
        for key, value in kwargs.items():
            if key in self.datasets:
                self.datasets[key][..., epoch] = value
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




def update_slice_losses(*args, **kwargs):
    """
    Update the network for each slice along dir = 'x' or 'y'.
    """
    pass





def update_total_losses(
        f_net: nn.Module, g_net: nn.Module, monoarray: MonoArray3D,
        train_config: TrainConfig, loss_config: LossConfig,
        f_optimizer, g_optimizer, slice_dicts: Tuple[Dict,...]):
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

    zin_conj = {
        'x': zin * torch.tensor([1, -1, 1, 1, 1, 1]),
        'y': zin * torch.tensor([1, 1, 1, -1, 1, 1]),
        'z': zin * torch.tensor([1, 1, 1, 1, 1, -1])
    }
    GBZ_conj_inputs = {dim: f_net(zin_conj[dim]) for dim in zin_conj}

    loss_fg = loss_config._fg_config._fg_loss(zin, z_reconstructed)

    loss_area = torch.stack([
        loss_config._area_config._area_loss_avg(Spectra, dim=dim, slice_config=slice_dicts[i])
        for i, dim in enumerate(['x', 'y', 'z'])])
    
    loss_symm = torch.stack([
        loss_config._symm_config._symm_loss_avg(GBZ, GBZ_conj_inputs[dim], dim=dim, slice_config=slice_dicts[i])
        for i, dim in enumerate(['x', 'y', 'z'])])
    loss_symm_ex = loss_config._symm_config._symm_loss_ex(GBZ)

    loss_arg = torch.stack([
        loss_config._arg_config._arg_loss_avg(GBZ, k_lst=inputs[f'k{dim}'], dim=dim)
        for dim in ['x', 'y', 'z']])
        
    loss_total = (loss_fg + loss_area.sum() + 
                  loss_symm.sum() + loss_symm_ex +
                  loss_arg.sum())
    
    loss_total.backward()

    f_optimizer.step()
    g_optimizer.step()

    return {
        'loss_fg': loss_fg.item(),
        'loss_area': loss_area.tolist(),
        'loss_symm': loss_symm.tolist(),
        'loss_symm_ex': loss_symm_ex.item(),
        'loss_arg': loss_arg.tolist(),
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
        self._res1 = ResBlock((6, 64, 128, 128, 128, 64, 6), zero_init=True)

    def forward(self, x):
        x = self._res1(x)
        return x


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self._res1 = ResBlock((6, 64, 128, 128, 128, 64, 6), zero_init=True)

    def forward(self, x):
        x = self._res1(x)
        return x
    











#%%
### train
target_monoarray = MonoArray3D(
    model_name= 'HN3D', 
    model_params= (1, 3, 2, 4, 2, 3), 
    scale_factor= 1)

init_path = './save_weights/HN3D_1_3_2_4_2_3_x1'

input_config = InputConfig(
    num_input= (60, 60, 60),
    rand_input= True,
    input_radius= (1, 1, 1))

slice_config = SliceConfig()

optim_config = OptimConfig(
    optimizer_class= optim.Adam,
    optimizer_params= {'lr': 1e-5, 'weight_decay': 1e-5},
    scheduler_class= lr_scheduler.StepLR,
    scheduler_params= {'step_size': 500, 'gamma': 0.8})

output_config = OutputConfig(
    logging_step= 50,
    temp_save_step= 200,
    plot_kwargs= {'clim': (0.5, 0.9)})

train_config = TrainConfig(
    num_epochs= 1,
    init_path= init_path,
    input_config= input_config,
    slice_config= slice_config,
    optim_config= optim_config,
    update_each_slice= False, 
    output_config= output_config)

fg_config = FGLossConfig(
    loss_weights= 100,
    func_fg= nn.MSELoss(reduction='mean'))


area_config = AreaLossConfig(
    loss_weights= (1, 1, 1),
    func_area_x= AreaLoss(farea=PolyArea_SumAbs()),
    func_area_y= AreaLoss(farea=PolyArea_SumAbs()),
    func_area_z= AreaLoss(farea=PolyArea_SumAbs()))

symm_config = SymmLossConfig(
    loss_weights= (50, 50, 50, 0),
    func_symm_x= SymmLoss(dim='x', full_dim=True, fdist=nn.L1Loss(reduction='mean')),
    func_symm_y= SymmLoss(dim='y', full_dim=True, fdist=nn.L1Loss(reduction='mean')),
    func_symm_z= SymmLoss(dim='z', full_dim=True, fdist=nn.L1Loss(reduction='mean')),
    func_symm_ex= None)

arg_config = ArgLossConfig(
    loss_weights= (10, 10, 10),
    func_arg_x= ArgLoss(dim='x', fdist=nn.L1Loss(reduction='mean')),
    func_arg_y= ArgLoss(dim='y', fdist=nn.L1Loss(reduction='mean')),
    func_arg_z= ArgLoss(dim='z', fdist=nn.L1Loss(reduction='mean')),
    use_target= True)

loss_config = LossConfig(
    fg_config= fg_config,
    area_config= area_config,
    symm_config= symm_config,
    arg_config= arg_config)


f_net = FNet().double()
g_net = GNet().double()
train_models(f_net, g_net, target_monoarray, train_config, loss_config)