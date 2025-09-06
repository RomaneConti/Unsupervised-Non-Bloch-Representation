
# This script uses Python 3.10.11

#%%
import os
import logging
import shutil
import time

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Union, Optional, Type, List, Dict, Tuple, Callable

from plot_helpers import*
from Area_Iter import PolyArea


# change default dtype of torch.tensor to float64
torch.set_default_dtype(torch.float64)











#%%
### helper functions
def generate_input_vec(N, rand_input=False):
    """
    Generate the input data [cos(x), sin(x)] for the encoder.
    """
    if rand_input:
        t = torch.rand(1) + torch.linspace(0, 1, N).unsqueeze(1)
    else:
        t = torch.linspace(0, 1, N).unsqueeze(1)
    
    return torch.cat((torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t)), dim=1)





class SlicedArray3D:
    def __init__(self, model_name: str, model_params: Optional[Tuple[float,...]], 
                 nvec: Tuple[int, int, int] = (1, 0, 0), scale_factor: float = 1.):
        """
        Class of the sliced 3D single-band model along certain direction (p, q).
        For simplicity, we do not need a general approach to define the 3D model.
        Concrete strs will be used to label different models.

        Params:
            model_name (str): a unique tag for different models. We treat model along
                different directions as different models and encode it in the model name.
            model_params (tuple): possible hopping parameters.
                Should be given in order, or one may get the wrong model.
            nvec (tuple): a tuple with form (p, q, r), given the slice direction.
            scale_factor (float): a number that will be multiplied onto the 
                model_params. Sometimes it may be beneficial for training.
        """
        self._model_name = model_name
        self._nvec = nvec
        self._scale_factor = float(scale_factor)
        if model_params is not None:
            params = tuple(x if isinstance(x, complex) else float(x) for x in model_params)
            self._model_params = tuple(x * self._scale_factor for x in params)
            params_tag = '_' + '_'.join([f'{num:.4g}' for num in params])
        else:
            self._model_params = None
            params_tag = ''

        self._tag = f'{model_name}{params_tag}_x{self._scale_factor:.4g}'


    @staticmethod
    def _spectra_cform_Edge(GBZ_cform, params, nvec: Tuple[int, int, int], 
                            k_perp: torch.Tensor):
        # params should be given with form (t1, t2, gamma), 
        # with bulk H(kx, ky, kz) = 2t1*cos(kx+ky) + 2t2*cos(kx+kz) + 2iγ*cos(kx)
        assert len(params) == 3, "3 params (t1, t2, gamma) are needed for an Edge Theory model."
        assert k_perp.shape == (2,), "2 perpendicular wave vectors are need for a 3D sliced model."
        
        t1, t2, gamma = params
        k1, k2 = k_perp
        match f"({nvec[0]}, {nvec[1]}, {nvec[2]})":
            case '(1, 0, 0)':
                return ((t1*torch.exp(1.j*k1) + t2*torch.exp(1.j*k2)+1.j*gamma) * GBZ_cform +
                        (t1*torch.exp(-1.j*k1) + t2*torch.exp(-1.j*k2)+1.j*gamma) / GBZ_cform)
            case _:
                raise ValueError(f"Unsupported 'nvec' value: {nvec}.")
    

    def _get_spectra_from_GBZ(self, GBZ: torch.Tensor, 
                              k_perp: Tuple[torch.Tensor, torch.Tensor]):
        """
        Get the spectra from the GBZ.

        Params:
            GBZ (torch.tensor): Should be a N*2 tensor, with the first/second column Re[β]/Im[β]
            k_perp (tuple): tuple with two tensor elements, wave vectors on the perpendicular plane.
        """
        assert GBZ.size(-1) == 2, "GBZ should be an (..., 2) tensor."

        GBZ_cform = GBZ[..., 0] + 1.j*GBZ[..., 1]
        registered_model = {
            'Edge': self._spectra_cform_Edge,
        }

        if self._model_name not in registered_model:
            raise ValueError(f"Undefined model: '{self._model_name}'.")
        E_cform = registered_model[self._model_name](GBZ_cform, self._model_params, self._nvec, k_perp)
        return torch.stack((E_cform.real, E_cform.imag), dim=-1)
        

    def __repr__(self):
        return f"SlicedArray3D(name={self._model_name}, params={self._model_params}, nvec={self._nvec})"








#%%
### area loss functions
polyarea = PolyArea





def polyarea_simple(polygon):
    """
    Area of a simple polygon. S = sum_n (x[n]*y[n+1] - x[n+1]*y[n]).
    """
    assert polygon.dim() == 2 and polygon.size(1) == 2, "polygon should be an (N, 2) tensor."

    return torch.abs(torch.sum(polygon[:, 0] * (polygon[:, 1].roll(-1)) - 
                               polygon[:, 0].roll(-1) * polygon[:, 1]))





def polyarea_sumabs(polygon, ref_point = None):
    """
    Area of a simple polygon by summing up the absolute value of each term.
        S = sum_n abs(x[n]*y[n+1] - x[n+1]*y[n]).
    This is a wrong formula, but works when EObc is a straight line or several
        straight lines intersecting at a single point.
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

    def __call__(self, polygon:torch.Tensor):
        return self._farea(polygon)












#%%
### Symm oss functions
def symm_loss(GBZ, GBZ_conj_input, fdist = nn.L1Loss(reduction='mean')):
    """
    A loss making the GBZ symeetric about the real axis.
    This loss can only be used when all the hoppings are real numbers.
    The loss can be defined as the distance between F(z)* [GBZ*] and F(z*) [GBZ_conj_input].
        with F the encoder, z the input, and '*' the complex conjugation

    Params:
        GBZ (torch.tensor): usually the output of the encoder F(z).
        GBZ_conj_input (torch.tensor): also the output of the encoder, but the input 
            is taken as its conjugate F(z*).
    """
    assert GBZ.dim() == 2 and GBZ.size(1) == 2, "GBZ should be an (N, 2) tensor."
    assert GBZ_conj_input.shape == GBZ.shape, "GBZ should be an (N, 2) tensor."

    GBZ_conj = GBZ * torch.tensor([1, -1])
    return fdist(GBZ_conj, GBZ_conj_input)

class SymmLoss:
    def __init__(self, fdist = nn.L1Loss(reduction='mean')):
        self._fdist = fdist

    def __call__(self, GBZ, GBZ_conj_input):
        return symm_loss(GBZ, GBZ_conj_input, self._fdist)











#%%
### Uniform loss functions
def uniformloss_distmad(GBZ):
    """
    A loss making the predicted GBZ more evenly distributed.
    The uniformness is charactized by the mean-absolute-difference for 
        length of each segments.
    """
    assert GBZ.dim() == 2 and GBZ.size(1) == 2, "GBZ should be an (N, 2) tensor."

    dist2 = (GBZ.diff(dim=0)).pow(2).sum(dim=1)
    dist = torch.sqrt(dist2)
    return torch.mean(torch.abs(dist - dist.mean()))





def uniformloss_argmad(GBZ, ref_point = torch.zeros(1, 2)):
    """
    A loss making the predicted GBZ more evenly distributed.
    The uniformness is charactized by the mean-absolute-difference for 
        relevant angles of adjacent points.
    """
    assert GBZ.dim() == 2 and GBZ.size(1) == 2, "GBZ should be an (N, 2) tensor."
    assert ref_point.shape == (1, 2), "ref_point should be a (1, 2) tensor."

    GBZ_ref = GBZ - ref_point
    GBZ_cform = GBZ_ref[:, 0] + 1.j * GBZ_ref[:, 1]
    GBZ_div = GBZ_cform[1: ] / GBZ_cform[ :-1]
    arg_div = torch.atan2(GBZ_div.imag, GBZ_div.real)
    return torch.mean(torch.abs(arg_div - arg_div.mean()))





def uniformloss_entropy(GBZ, ref_point = torch.zeros(1, 2)):
    """
    A loss making the predicted GBZ more evenly distributed.
    The uniformness is charactized by the entropy -p*log(p) for 
        relevant angles of adjacent points.
    """
    assert GBZ.dim() == 2 and GBZ.size(1) == 2, "GBZ should be an (N, 2) tensor."
    assert ref_point.shape == (1, 2), "ref_point should be a (1, 2) tensor."

    GBZ_ref = GBZ - ref_point
    GBZ_cform = GBZ_ref[:, 0] + 1.j * GBZ_ref[:, 1]
    GBZ_div = GBZ_cform[1: ] / GBZ_cform[ :-1]
    arg_div = torch.atan2(GBZ_div.imag, GBZ_div.real)
    p = arg_div.softmax(dim=-1)
    return -torch.sum(p * torch.log(p + 1e-12))




class UniformLoss:
    def __init__(self, method: str = 'distMAD', ref_point = None):
        assert method.lower() in ['distmad', 'argmad', 'entropy'], f"Invalid method '{method}'."
        self._method = method.lower()
        self._ref_point = ref_point

    def __call__(self, GBZ):
        match self._method:
            case 'distmad':
                return uniformloss_distmad(GBZ)
            case 'argmad':
                return uniformloss_argmad(GBZ, self._ref_point)
            case 'entropy':
                return uniformloss_entropy(GBZ, self._ref_point)
            








#%%
### Loss config
class LossConfig:
    def __init__(self,
                 loss_weights: Tuple[float,...] = (100, 1, 5, 5),
                 func_fg: Callable = nn.MSELoss(reduction='mean'),
                 func_area: Callable = AreaLoss(),
                 func_symm: Optional[Callable] = None,
                 func_uniform: Optional[Callable] = None):
        """
        This class determines how to calculate the loss functions and their weights.

        Parmas:
            loss_weights (tuple): weights of loss_fg, area, symm, and uniform in order.
            func_fg (function): loss function about the encoder and decoder.
                Should accept z_in(N*2 tensor) and z_reconstructed(N*2 tensor) as inputs, return a scalar tensor.
            func_area (function): loss function about the area of the Spectra.
                Should accept monochain and GBZ(N*2 tensor) as inputs, return a scalar tensor.
            func_symm (function): loss function making the GBZ symmetric about the real axis.
                Should accept GBZ(N*2 tensor), GBZ_conj_input(N*2 tensor) as inputs, return a scalar tensor.
            func_uniform(function): loss function making the GBZ more evenly distributed.
                Should accept GBZ(N*2 tensor) as the input, return a scalar tensor.
        """
        assert len(loss_weights) == 4, "loss_weights should be a tuple with 4 float elements."
        
        self._loss_weights = tuple(map(float, loss_weights))
        self._func_fg = func_fg
        self._func_area = func_area
        self._func_symm = func_symm
        self._func_uniform = func_uniform


    def _fg_loss(self, z_in, z_reconstructed):
        return self._loss_weights[0] * self._func_fg(z_in, z_reconstructed)
    

    def _area_loss(self, Spectra):
        return self._loss_weights[1] * self._func_area(Spectra)
    

    def _symm_loss(self, GBZ, GBZ_conj_input):
        if self._func_symm is not None:
            return self._loss_weights[2] * self._func_symm(GBZ, GBZ_conj_input)
        return torch.tensor(0.)
    

    def _uniform_loss(self, GBZ):
        if self._func_uniform is not None:
            return self._loss_weights[3] * self._func_uniform(GBZ)
        return torch.tensor(0.)
   












#%%
### Configuration classes used for training and test
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
                 num_input: int = 301,
                 rand_input: bool = True,
                 optim_config: OptimConfig = OptimConfig(),
                 output_step: Tuple[int, int] = (50, 200)):
        """
        This class determines the hyper parameters in training.

        Params:
            num_epochs (int): total number of epochs of training.
            init_path (str): path for initializing model weights.
            num_input (int): number of sampling points of the input tensor.
            rand_input (bool): whether to add an offset to the input tensor.
            optim_config (OptimConfig): determine the optimizer and scheduler.
            output_step (tuple): a tuple with 2 ints, the 1st/2nd one is the output step for logging/plot.
        """
        self._num_epochs = num_epochs
        self._init_path = init_path
        self._num_input = num_input
        self._rand_input = rand_input
        self._optim_config = optim_config
        self._output_step = output_step

     






#%%
### assistant training and test functions
def plt_training_results(encoder: nn.Module, slicedarray: SlicedArray3D,
                         k_perp: torch.Tensor, num_input: int):
    """
    Plot the current results in the training process.

    Params:
        encoder (nn.Module): the encoder.
        slicedarray (SlicedArray3D): target 3D model.
        k_perp (tensor): together with the slicedarray, determine how to get the spectra.
        num_input: determine how many input points will be chosen for the encoder.
    """
    zin = generate_input_vec(num_input, rand_input=True)
    GBZ = encoder(zin)
    Spectra = slicedarray._get_spectra_from_GBZ(GBZ, k_perp)
    plt_GBZ_and_Spectra(GBZ, Spectra)





def test_models(slicedarray: SlicedArray3D, k_perp: torch.Tensor,
                load_fold: str, N: int = 301, rand_input: bool = True,
                loss_config: LossConfig = LossConfig()):
    """
    Test the trained encoder and decoder.
    There should be files 'Encoder.pt' and 'Decoder.pt' under load_fold.
    """
    if not slicedarray._tag in load_fold:
        logging.basicConfig(format='%(message)s', level=logging.WARNING)
        logger = logging.getLogger()
        logger.warning(f"Warning: '{load_fold}' seems not match with the given monochain" + 
                       f"\n{slicedarray}.")

    try:
        f_net_test = torch.jit.load(os.path.join(load_fold, 'Encoder.pt').replace('\\', '/'))
        g_net_test = torch.jit.load(os.path.join(load_fold, 'Decoder.pt').replace('\\', '/'))
    except:
        raise Exception(f'No available file in folder {load_fold}')
    
    z_test = generate_input_vec(N, rand_input=rand_input)
    GBZ_test = f_net_test(z_test)
    z_test_reconstructed = g_net_test(GBZ_test)
    Spectra_test = slicedarray._get_spectra_from_GBZ(GBZ_test, k_perp)

    loss_fg = loss_config._fg_loss(z_test, z_test_reconstructed)
    loss_area = loss_config._area_loss(Spectra_test)
    conj_input = z_test * torch.tensor([1, -1])
    GBZ_test_conj_input = f_net_test(conj_input)
    loss_symm = loss_config._symm_loss(GBZ_test, GBZ_test_conj_input)
    loss_uniform = loss_config._uniform_loss(GBZ_test)
    with torch.no_grad():
        plt_GBZ_and_Spectra(GBZ_test, Spectra_test)

    print(f'Test monochain={slicedarray},\n'
          f'k_perp/pi={", ".join(f"{i:.3f}" for i in k_perp)}\n'+
          f'    loss_fg: {loss_fg.item()}, ' +
          f'    loss_symm: {loss_symm.item()}\n' +
          f'    loss_area: {loss_area.item()}, ' +
          f'    loss_uniform: {loss_uniform.item()}')  
    








#%%
### main training function
def train_models(f_net: nn.Module, g_net: nn.Module,
                 slicedarray: SlicedArray3D, k_perp: torch.Tensor,
                 train_config: TrainConfig = TrainConfig(),
                 loss_config: LossConfig = LossConfig()):
    """
    Train the model with given conditions.

    Params:
        f_net, g_net (nn.Module): encoder and decoder.
        monochain: target single-band system.
        train_config (TrainConfig): the configuration parameters for training.
        loss_config (LossConfig): the configuration of the loss calculations.
    """
    for handler in logging.getLogger().handlers:
        handler.close()
        logging.getLogger().removeHandler(handler)

    init_models(f_net, g_net, train_config._init_path)

    temp_paths = init_temp_save_path(
        temp_save_fold= './temp_save_weights',
        slicedarray=slicedarray, k_perp=k_perp)
    f_save_path = temp_paths['f_save_path']
    g_save_path = temp_paths['g_save_path']

    f_optimizer = train_config._optim_config._create_optimizer(f_net.parameters())
    g_optimizer = train_config._optim_config._create_optimizer(g_net.parameters())

    f_scheduler = train_config._optim_config._create_scheduler(f_optimizer)
    g_scheduler = train_config._optim_config._create_scheduler(g_optimizer)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(temp_paths['logging_path'], mode='a'),
                                  logging.StreamHandler()])

    num_epochs = train_config._num_epochs
    loss_path = temp_paths['loss_path']

    loss_file = H5LossFile(loss_path, num_epochs)
    try:
        for epoch in range(num_epochs):
            losses = update_total_losses(
                f_net, g_net, slicedarray, k_perp,
                train_config._num_input, train_config._rand_input,
                loss_config, f_optimizer, g_optimizer
            )

            if f_scheduler is not None:
                f_scheduler.step()
            if g_scheduler is not None:
                g_scheduler.step()

            loss_file._write(epoch, **losses)

            if (epoch+1) % train_config._output_step[0] == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}]:"+ '\n' + 
                             f"    loss_fg: {losses['loss_fg']:.8g}, " +
                             f"    loss_symm: {losses['loss_symm']:.8g}" + '\n' +
                             f"    loss_area: {losses['loss_area']:.8g}, " +
                             f"    loss_uniform: {losses['loss_uniform']:.8g}")

            if (epoch+1) % train_config._output_step[1] == 0:
                save_models(f_net, g_net, f_save_path, g_save_path)
                with torch.no_grad():
                    plt_training_results(f_net, slicedarray, k_perp, 301)
    finally:
        loss_file._close()

    save_models(f_net, g_net, f_save_path, g_save_path)
    plt_record_loss(loss_path)    

    for handler in logging.getLogger().handlers:
        handler.close()
        logging.getLogger().removeHandler(handler)





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
        




def init_temp_save_path(temp_save_fold: str, slicedarray: SlicedArray3D, k_perp: torch.Tensor):
    """
    Clear all the files under temp_save_fold/model_tag/k_perp=xxxpi
        and return the path for saving the encoder and decoder.

    Params:
        temp_save_fold (str): target fold to create the tag-named fold.
        slicedarray (SlicedArray3D): target model, determine the model_tag.
        k_perp (tensor): target k_perp, the bottom fold to save the data. It contains
            all files for the model, including saved weights, loss data, logging...
    """
    model_tag = f"{slicedarray._tag}_" + "_".join(f"{n}_{val}" for n, val in zip("pqr", slicedarray._nvec))
    k_tag = f"k_perp=" + "_".join(f"{k/torch.pi:.2f}pi" for k in k_perp)

    model_fold = os.path.join(temp_save_fold, model_tag).replace('\\', '/')
    temp_save_path = os.path.join(model_fold, k_tag).replace('\\', '/')
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    for root, dirs, files in os.walk(temp_save_path):
        for file in files:
            os.remove(os.path.join(root, file).replace('\\', '/'))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir).replace('\\', '/'))    

    f_save_path = os.path.join(temp_save_path, 'Encoder.pt').replace('\\', '/')
    g_save_path = os.path.join(temp_save_path, 'Decoder.pt').replace('\\', '/')
    loss_path = os.path.join(temp_save_path, 'loss_data.h5').replace('\\', '/')
    logging_path = os.path.join(temp_save_path, 'training_log.log').replace('\\', '/')

    return {'f_save_path': f_save_path, 
            'g_save_path': g_save_path,
            'loss_path': loss_path,
            'logging_path': logging_path}
        




def save_models(f_net: nn.Module, g_net: nn.Module, f_save_path, g_save_path):
    """
    Save the encoder and decoder to the given path.
    """
    f_net_jit = torch.jit.trace(f_net, example_inputs=torch.rand(1, 2))
    g_net_jit = torch.jit.trace(g_net, example_inputs=torch.rand(1, 2))
    f_net_jit.save(f_save_path)
    g_net_jit.save(g_save_path)






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
            'loss_area': self._h5file.create_dataset('loss_area', (num_epochs,)),
            'loss_symm': self._h5file.create_dataset('loss_symm', (num_epochs,)),
            'loss_uniform': self._h5file.create_dataset('loss_uniform', (num_epochs,)),
            'lr_f': self._h5file.create_dataset('lr_f', (num_epochs,)),
            'lr_g': self._h5file.create_dataset('lr_g', (num_epochs,)),
            }
        

    def _write(self, epoch: int, **kwargs):
        for key, value in kwargs.items():
            if key in self.datasets:
                self.datasets[key][epoch] = value
            else:
                raise KeyError(f"Dataset '{key}' not found in H5LossFile.")

    
    def _close(self):
        self._h5file.close()





def update_total_losses(
        f_net: nn.Module, g_net: nn.Module, 
        slicedarray: SlicedArray3D, k_perp: torch.Tensor,
        num_input: int, rand_input: bool,
        loss_config: LossConfig, f_optimizer, g_optimizer):
    """
    Update the network for the total losses.
    """
    f_optimizer.zero_grad()
    g_optimizer.zero_grad()

    zin = generate_input_vec(num_input, rand_input)
    GBZ = f_net(zin)
    z_reconstructed = g_net(GBZ)
    Spectra = slicedarray._get_spectra_from_GBZ(GBZ, k_perp)

    zin_conj = zin * torch.tensor([1, -1])
    GBZ_conj_input = f_net(zin_conj)

    loss_fg = loss_config._fg_loss(zin, z_reconstructed)
    loss_area = loss_config._area_loss(Spectra)
    loss_symm = loss_config._symm_loss(GBZ, GBZ_conj_input)
    loss_uniform = loss_config._uniform_loss(GBZ)
            
    loss_total = loss_fg + loss_area + loss_symm + loss_uniform
    
    loss_total.backward()
    f_optimizer.step()
    g_optimizer.step()

    return {
        'loss_fg': loss_fg.item(),
        'loss_area': loss_area.item(),
        'loss_symm': loss_symm.item(),
        'loss_uniform': loss_uniform.item(),
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
        self.res1 = ResBlock((2, 64, 128, 128, 128, 64, 2), zero_init=True)

    def forward(self, x):
        x = self.res1(x)
        return x


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.res1 = ResBlock((2, 64, 128, 128, 128, 64, 2), zero_init=True)

    def forward(self, x):
        x = self.res1(x)
        return x
    








#%%
### train
target_model = SlicedArray3D(
    model_name= 'Edge',
    model_params= (2, 4, 1),
    nvec= (1, 0, 0),
    scale_factor=1)

optim_config = OptimConfig(
    optimizer_class= optim.Adam,
    optimizer_params= {'lr': 1e-3},
    scheduler_class= lr_scheduler.StepLR,
    scheduler_params= {'step_size': 100, 'gamma': 0.9})

train_config = TrainConfig(
    num_epochs= 1,
    init_path= None,
    num_input= 301,
    rand_input= True,
    optim_config= optim_config,
    output_step= (1, 200))

loss_config = LossConfig(
    loss_weights= (500, 1, 50, 50),
    func_fg= nn.MSELoss(reduction='mean'),
    func_area= AreaLoss(),
    func_symm= SymmLoss(fdist=nn.L1Loss(reduction='mean')),
    func_uniform= UniformLoss(method='DistMAD'))

# k_perp = torch.pi * torch.tensor(0.5)
k_perp_lst = torch.linspace(0, 2, 21) * torch.pi

k1_lst = torch.linspace(-1, 1, 3) * torch.pi
k2_lst = torch.linspace(-1, 1, 3) * torch.pi
k1, k2 = torch.meshgrid(k1_lst, k2_lst, indexing='ij')
k_perp_lst = torch.stack([k1.ravel(), k2.ravel()], dim=1)

for root, dirs, files in os.walk('./temp_save_weights/'):
    for file in files:
        os.remove(os.path.join(root, file).replace('\\', '/'))
    for dir in dirs:
        shutil.rmtree(os.path.join(root, dir).replace('\\', '/'))    

for id_perp in range(k_perp_lst.size(0)):
    f_net = FNet().double()
    g_net = GNet().double()
    train_models(f_net, g_net, target_model, k_perp_lst[id_perp],
                 train_config, loss_config)







# %%
