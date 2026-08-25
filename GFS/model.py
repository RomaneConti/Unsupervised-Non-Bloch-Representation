
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


torch.set_default_dtype(torch.float64)











#%%
### helper functions
polyarea = PolyArea





def polyarea_simple(polygon):
    """
    Area of a simple polygon. S = sum_n (x[n]*y[n+1] - x[n+1]*y[n]).
    """
    assert polygon.dim() == 2 and polygon.size(1) == 2, "polygon should be an (N, 2) tensor."

    return torch.abs(torch.sum(polygon[:, 0] * (polygon[:, 1].roll(-1)) - 
                               polygon[:, 0].roll(-1) * polygon[:, 1]))





def generate_input_vec(N, rand_input=False):
    """
    Generate the input data [cos(x), sin(x)] for the encoder.
    """
    if rand_input:
        t = torch.rand(1) + torch.linspace(0, 1, N).unsqueeze(1)
    else:
        t = torch.linspace(0, 1, N).unsqueeze(1)
    
    return torch.cat((torch.cos(2*torch.pi*t), torch.sin(2*torch.pi*t)), dim=1)










class GFSModel:
    def __init__(self, tx, ty, txy, u, E0: torch.Tensor,imag_branch: str = 'lower'):
        self._tx = tx
        self._ty = ty
        self._txy = txy
        self._u = u

        assert E0.shape == (1, 2), "E0 should be a (1, 2) tensor."
        self._E0 = E0

        if imag_branch.lower()not in ("lower", "upper"):
            raise ValueError(
                f"imag_branch must be 'lower' or 'upper', got {imag_branch!r}."
            )
        self._imag_branch = imag_branch


    def get_rhoy_from_betax(self, betax: torch.Tensor):
        """
        Get the rhoy from the betax.
        rhoy can be obtained by solving the transfer matrix eigen-equation
            A rhoy^2 + B rhoy + C = 0, with 
                A = ty*βx + txy*βx^2, 
                B = tx*(1 + βx^2) + βx*(u-E0), 
                C = ty*βx + txy.
        Params:
            imag_branch : {"lower", "upper"}
                select the root with the smaller/larger imaginary part.
        """
        assert betax.dim() == 2 and betax.size(1) == 2, "betax should be an (N, 2) tensor."

        betax_cform = betax[:, 0] + 1.j * betax[:, 1]
        E0_cform = self._E0[0, 0] + 1.j * self._E0[0, 1]
        A = 2 * (self._ty * betax_cform + self._txy * betax_cform.pow(2))
        B = -self._tx + E0_cform * betax_cform - self._u * betax_cform - self._tx * betax_cform.pow(2)
        D = torch.sqrt(B.pow(2) - 2 * A * (self._ty * betax_cform + self._txy))
        rhoy1 = (B + D) / A
        rhoy2 = (B - D) / A

        if self._imag_branch.lower() == "lower":
            rhoy = torch.where(rhoy1.imag <= rhoy2.imag, rhoy1, rhoy2)
        else:
            rhoy = torch.where(rhoy1.imag >= rhoy2.imag, rhoy1, rhoy2)

        return torch.stack((rhoy.real, rhoy.imag), dim=1)
            



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
### Loss functions
def area_loss(loop: torch.Tensor, farea: Callable = polyarea):
    """
    Area loss using given method.
    """
    assert loop.dim() == 2 and loop.size(1) == 2, "loop should be an (N, 2) tensor."
    return farea(loop)

class AreaLoss:
    def __init__(self, farea = polyarea):
        self._farea = farea

    def __call__(self, loop: torch.Tensor):
        return area_loss(loop, self._farea)







def uniformloss_distmad(loop: torch.Tensor):
    """
    A loss making the predicted loop more evenly distributed.
    The uniformness is charactized by the mean-absolute-difference for 
        length of each segments.
    """
    assert loop.dim() == 2 and loop.size(1) == 2, "loop should be an (N, 2) tensor."

    dist2 = (loop.diff(dim=0)).pow(2).sum(dim=1)
    dist = torch.sqrt(dist2)
    return torch.mean(torch.abs(dist - dist.mean()))





def uniformloss_argmad(loop: torch.Tensor, ref_point = torch.zeros(1, 2)):
    """
    A loss making the predicted loop more evenly distributed.
    The uniformness is charactized by the mean-absolute-difference for 
        relevant angles of adjacent points.
    """
    assert loop.dim() == 2 and loop.size(1) == 2, "loop should be an (N, 2) tensor."
    assert ref_point.shape == (1, 2), "ref_point should be a (1, 2) tensor."

    loop_ref = loop - ref_point
    loop_cform = loop_ref[:, 0] + 1.j * loop_ref[:, 1]
    loop_div = loop_cform[1: ] / loop_cform[ :-1]
    arg_div = torch.atan2(loop_div.imag, loop_div.real)
    return torch.mean(torch.abs(arg_div - arg_div.mean()))





def uniformloss_entropy(loop: torch.Tensor, ref_point = torch.zeros(1, 2)):
    """
    A loss making the predicted loop more evenly distributed.
    The uniformness is charactized by the entropy -p*log(p) for 
        relevant angles of adjacent points.
    """
    assert loop.dim() == 2 and loop.size(1) == 2, "loop should be an (N, 2) tensor."
    assert ref_point.shape == (1, 2), "ref_point should be a (1, 2) tensor."

    loop_ref = loop - ref_point
    loop_cform = loop_ref[:, 0] + 1.j * loop_ref[:, 1]
    loop_div = loop_cform[1: ] / loop_cform[ :-1]
    arg_div = torch.atan2(loop_div.imag, loop_div.real)
    p = arg_div.softmax(dim=-1)
    return -torch.sum(p * torch.log(p + 1e-12))




class UniformLoss:
    def __init__(self, method: str = 'distMAD', ref_point = None):
        assert method.lower() in ['distmad', 'argmad', 'entropy'], f"Invalid method '{method}'."
        self._method = method.lower()
        self._ref_point = ref_point

    def __call__(self, loop: torch.Tensor):
        match self._method:
            case 'distmad':
                return uniformloss_distmad(loop)
            case 'argmad':
                return uniformloss_argmad(loop, self._ref_point)
            case 'entropy':
                return uniformloss_entropy(loop, self._ref_point)











#%%
### Configuration classes used for training and test
class TrainConfig:
    def __init__(self,
                 num_epochs: int = 3000,
                 init_path: Optional[str] = None,
                 num_input: int = 301,
                 rand_input: bool = True,
                 optimizer_class: Type[optim.Optimizer] = optim.Adam,
                 optimizer_params: Optional[Dict] = {'lr': 1e-3},
                 scheduler_class: Optional[Type[lr_scheduler._LRScheduler]] = lr_scheduler.StepLR,
                 scheduler_params: Optional[Dict] = {'step_size': 100, 'gamma': 0.9}):
        """
        This class determines the hyper parameters in training.

        Params:
            num_epochs (int): total number of epochs of training.
            init_path (str): path for initializing model weights.
            num_input (int): number of sampling points of the input tensor.
            rand_input (bool): whether to add an offset to the input tensor.
            optimizer_class (Type): optimizer class, this parameter cannot be None.
            optimizer_params (dict): parameters for the optimizer.
                Default optimizer is set as Adam with lr = 1e-3.
            scheduler_class (Type): learning rate scheduler class.
            scheduler_params (dict): parameters for the scheduler.
                Default schefuler is set as StepLR with step 100 and gamma 0.9.
        """
        self._num_epochs = num_epochs
        self._init_path = init_path
        self._num_input = num_input
        self._rand_input = rand_input

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
        




class LossConfig:
    def __init__(self,
                 loss_weights: Tuple[float,...] = (100, 1, 5),
                 func_fg: Callable = nn.MSELoss(reduction='mean'),
                 func_area: Callable = AreaLoss(),
                 func_uniform: Optional[Callable] = None):
        """
        This class determines how to calculate the loss functions and their weights.

        Parmas:
            loss_weights (tuple): weights of loss_fg, area, symm, and uniform in order.
            func_fg (function): loss function about the encoder and decoder.
            func_area (function): loss function about the area of the Spectra.
            func_uniform(function): loss function making the predicted loop more evenly distributed.
        """
        assert len(loss_weights) == 3, "loss_weights should be a tuple with 3 float elements."

        self._loss_weights = tuple(map(float, loss_weights))
        self._func_fg = func_fg
        self._func_area = func_area
        self._func_uniform = func_uniform

        self._validate_inputs()


    def _validate_inputs(self):
        z_in = torch.randn(10, 2)
        z_reconstructed = torch.randn(10, 2)
        loop = torch.randn(10, 2)

        try:
            x1 = self._fg_loss(z_in, z_reconstructed)
            x2 = self._area_loss(loop)
            x3 = self._uniform_loss(loop)
            assert x1.shape == torch.Size([]), "fg_loss should return a scalar tensor."
            assert x2.shape == torch.Size([]), "area_loss should output a scalar tensor."
            assert x3.shape == torch.Size([]), "uniform_loss should output a scalar tensor."
        except Exception as e:
            raise ValueError(f"Some given functions failed with default inputs: {e}")
        

    def _fg_loss(self, z_in, z_reconstructed):
        return self._loss_weights[0] * self._func_fg(z_in, z_reconstructed)
    

    def _area_loss(self, loop):
        return self._loss_weights[1] * self._func_area(loop)
    

    def _uniform_loss(self, loop):
        if self._func_uniform is not None:
            return self._loss_weights[2] * self._func_uniform(loop)
        return torch.tensor(0.)
    
        










#%%
### training and test functions
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
    f_net_jit = torch.jit.trace(f_net, example_inputs=torch.rand(1, 2))
    g_net_jit = torch.jit.trace(g_net, example_inputs=torch.rand(1, 2))
    f_net_jit.save(f_save_path)
    g_net_jit.save(g_save_path)





def test_models(gfs_model: GFSModel,
                load_fold: str, 
                N: int = 301, rand_input: bool = True,
                ref_point_lst=None,
                loss_config: LossConfig = LossConfig()):
    """
    Test the trained encoder and decoder.
    There should be files 'Encoder.pt' and 'Decoder.pt' under load_fold.
    """

    try:
        f_net_test = torch.jit.load(os.path.join(load_fold, 'Encoder.pt').replace('\\', '/'))
        g_net_test = torch.jit.load(os.path.join(load_fold, 'Decoder.pt').replace('\\', '/'))
    except:
        raise Exception(f'No available file in folder {load_fold}')
    
    z_test = generate_input_vec(N, rand_input=rand_input)
    GFS_test = f_net_test(z_test)
    rhoy_test = gfs_model.get_rhoy_from_betax(GFS_test)
    z_test_reconstructed = g_net_test(GFS_test)

    loss_fg = loss_config._fg_loss(z_test, z_test_reconstructed)
    loss_area = loss_config._area_loss(rhoy_test)
    loss_uniform = loss_config._uniform_loss(GFS_test)
    with torch.no_grad():
        plt_GFS(GFS_test, rhoy_test, ref_point_lst=None, Spectra_major_step=(2, 2))

    print(f'Test model={gfs_model}:'+ '\n' + 
          f'    loss_fg: {loss_fg.item()}, ' +
          f'    loss_area: {loss_area.item()}, ' +
          f'    loss_uniform: {loss_uniform.item()}')





def train_models(f_net: nn.Module, g_net: nn.Module,
                 gfs_model: GFSModel,
                 save_tag: str = 'default',
                 train_config: TrainConfig = TrainConfig(),
                 loss_config: LossConfig = LossConfig()):
    """
    Train the model with given conditions.

    Params:
        f_net, g_net (nn.Module): encoder and decoder.
        gfs_model (GFSModel): target system.
        train_config (TrainConfig): the configuration parameters for training.
        loss_config (LossConfig): the configuration of the loss calculations.
    """
    for handler in logging.getLogger().handlers:
        handler.close()
        logging.getLogger().removeHandler(handler)

    init_models(f_net, g_net, train_config._init_path)

    temp_save_fold = f'./temp_save_weights/'
    for root, dirs, files in os.walk(temp_save_fold):
        for file in files:
            os.remove(os.path.join(root, file).replace('\\', '/'))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir).replace('\\', '/'))

    temp_save_path = os.path.join(temp_save_fold, save_tag).replace('\\', '/')
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    f_save_path = os.path.join(temp_save_path, 'Encoder.pt').replace('\\', '/')
    g_save_path = os.path.join(temp_save_path, 'Decoder.pt').replace('\\', '/')

    f_optimizer = train_config._create_optimizer(f_net.parameters())
    g_optimizer = train_config._create_optimizer(g_net.parameters())

    f_scheduler = train_config._create_scheduler(f_optimizer)
    g_scheduler = train_config._create_scheduler(g_optimizer)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(temp_save_path, 'training_log.log').replace('\\', '/'), mode='a'),
                                  logging.StreamHandler()])
    
    num_epochs = train_config._num_epochs
    loss_path = os.path.join(temp_save_path, 'loss_data.h5').replace('\\', '/')
    with h5py.File(loss_path, 'w') as f:
        dset_loss_fg = f.create_dataset('loss_fg', (num_epochs,))
        dset_loss_area = f.create_dataset('loss_area', (num_epochs,))
        dset_loss_uniform = f.create_dataset('loss_uniform', (num_epochs,))
        dset_lr_f = f.create_dataset('lr_f', (num_epochs,))
        dset_lr_g = f.create_dataset('lr_g', (num_epochs,))

        for epoch in range(num_epochs):

            f_optimizer.zero_grad()
            g_optimizer.zero_grad()

            zin = generate_input_vec(train_config._num_input, train_config._rand_input)

            GFS_pred = f_net(zin)
            z_reconstructed = g_net(GFS_pred)

            loss_fg = loss_config._fg_loss(zin, z_reconstructed)

            rhoy_pred = gfs_model.get_rhoy_from_betax(GFS_pred)
            loss_area = loss_config._area_loss(rhoy_pred)
            
            loss_uniform = loss_config._uniform_loss(GFS_pred)
            
            loss = loss_fg + loss_area + loss_uniform

            loss.backward()
            f_optimizer.step()
            g_optimizer.step()

            if f_scheduler is not None:
                f_scheduler.step()
            if g_scheduler is not None:
                g_scheduler.step()

            dset_loss_fg[epoch] = loss_fg.item()
            dset_loss_area[epoch] = loss_area.item()
            dset_loss_uniform[epoch] = loss_uniform.item()
            dset_lr_f[epoch] = f_optimizer.param_groups[0]['lr']
            dset_lr_g[epoch] = g_optimizer.param_groups[0]['lr']

            if (epoch+1) % 50 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}]:'+ '\n' + 
                             f'    loss_fg: {loss_fg.item()}, ' +
                             f'    loss_area: {loss_area.item()}, ' +
                             f'    loss_uniform: {loss_uniform.item()}')

            if (epoch+1) % 200 == 0:
                save_models(f_net, g_net, f_save_path, g_save_path)

                with torch.no_grad():
                    plt_GFS(GFS_pred, rhoy_pred, Spectra_major_step=(2, 2)) 

    plt_record_loss(loss_path)        
    save_models(f_net, g_net, f_save_path, g_save_path)
    with torch.no_grad():
        plt_GFS(GFS_pred, rhoy_pred, Spectra_major_step=(2, 2)) 

    for handler in logging.getLogger().handlers:
        handler.close()
        logging.getLogger().removeHandler(handler)











#%%
### concrete models, hyper-parameters, and train
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
    

f_net = FNet().double()
g_net = GNet().double()

imag_branch = 'upper'
target_model = GFSModel(
    tx=1.0, ty=1.0, txy=1.j/2, u=-1.j,
    E0=torch.tensor([[0.819, -1.108]]), imag_branch=imag_branch
)

parent_init = './save_weights/L_1_R_1'

train_config = TrainConfig(
    num_epochs= 4,
    init_path= None,
    num_input= 301,
    rand_input= True,
    optimizer_class= optim.Adam,
    optimizer_params= {'lr': 1e-5},
    scheduler_class= lr_scheduler.StepLR,
    scheduler_params= {'step_size': 100, 'gamma': 0.9})

loss_config = LossConfig(
    loss_weights= (500, 1, 0),
    func_fg= nn.MSELoss(reduction='mean'),
    func_area= AreaLoss(farea=PolyArea),
    func_uniform= UniformLoss(method='DistMAD'))


train_models(
    f_net, g_net,
    target_model, save_tag= 'GFS_upper',
    train_config=train_config, loss_config=loss_config,
)





test_models(target_model, 
            load_fold='./temp_save_weights/GFS_upper', 
            N=301, rand_input=True, loss_config=loss_config,
            ref_point_lst=None)
# %%
