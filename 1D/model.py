
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





class MonoChain:
    def __init__(self, LCoupleCoef: Tuple[float,...], RCoupleCoef: Tuple[float,...], scale_factor: float = 1.):
        """
        Class of the single band chain.

        Params:
            LCoupleCoef (tuple): couple coeffcients towards left, corresponding to t_{-1}, t_{-2},... in order.
            RCoupleCoef (tuple): couple coeffcients towards right, corresponding to t_{1}, t_{2},... in order.
                Note that we have omitted the onsite term t_{0} since it is just a shift for single band models.

        """
        self._scale_factor = float(scale_factor)
        LCoef = tuple(x if isinstance(x, complex) else float(x) for x in LCoupleCoef)
        RCoef = tuple(x if isinstance(x, complex) else float(x) for x in RCoupleCoef)
        self._LCoupleCoef = tuple(x * self._scale_factor for x in LCoef)
        self._RCoupleCoef = tuple(x * self._scale_factor for x in RCoef)
        
        self._tag = 'L_' + '_'.join([f'{num:.4g}' for num in LCoef]) + \
            '_R_' + '_'.join([f'{num:.4g}' for num in RCoef]) + \
            f'_x{self._scale_factor:.4g}'


    def _get_spectra_from_GBZ(self, GBZ: torch.Tensor):
        """
        Get the spectra from the GBZ.
        Given a beta, the spectra is obtained by E = sum_p R(p)*beta^p + sum_q L(q)/beta^q.
        Note that we have omitted the constant term here.

        Params:
            GBZ (torch.tensor): Should be a N*2 tensor, with the first/second column Re[β]/Im[β] 
        """
        assert GBZ.dim() == 2 and GBZ.size(1) == 2, "GBZ should be an (N, 2) tensor."

        beta_cform = (GBZ[:, 0] + 1.j * GBZ[:, 1]).unsqueeze(1)

        R_pow = torch.arange(1, len(self._RCoupleCoef)+1)
        L_pow = torch.arange(1, len(self._LCoupleCoef)+1)
        E_cform = torch.sum(torch.tensor(self._RCoupleCoef) * beta_cform.pow(R_pow), dim=1)
        E_cform += torch.sum(torch.tensor(self._LCoupleCoef) / beta_cform.pow(L_pow), dim=1)

        return torch.stack((E_cform.real, E_cform.imag), dim=1)
    

    def __repr__(self):
        return f"MonoChain(LCoupleCoef={self._LCoupleCoef}, RCoupleCoef={self._RCoupleCoef})"
    










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
def area_loss(monochain: MonoChain, GBZ: torch.Tensor, farea: Callable = polyarea):
    """
    Area loss using given method.
    """
    assert GBZ.dim() == 2 and GBZ.size(1) == 2, "GBZ should be an (N, 2) tensor."

    Spectra = monochain._get_spectra_from_GBZ(GBZ)
    return farea(Spectra)

class AreaLoss:
    def __init__(self, farea = polyarea):
        self._farea = farea

    def __call__(self, monochain: MonoChain, GBZ: torch.Tensor):
        return area_loss(monochain, GBZ, self._farea)





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

        self._validate_inputs()


    def _validate_inputs(self):
        z_in = torch.randn(10, 2)
        z_reconstructed = torch.randn(10, 2)
        monochain = MonoChain((1,), (1, 1))  # Replace with an actual monochain if needed
        GBZ = torch.randn(10, 2)
        GBZ_conj_input = torch.randn(10, 2)

        try:
            x1 = self._fg_loss(z_in, z_reconstructed)
            x2 = self._area_loss(monochain, GBZ)
            x3 = self._symm_loss(GBZ, GBZ_conj_input)
            x4 = self._uniform_loss(GBZ)
            assert x1.shape == torch.Size([]), "fg_loss should return a scalar tensor."
            assert x2.shape == torch.Size([]), "area_loss should output a scalar tensor."
            assert x3.shape == torch.Size([]), "symm_loss should output a scalar tensor."
            assert x4.shape == torch.Size([]), "uniform_loss should output a scalar tensor."
        except Exception as e:
            raise ValueError(f"Some given functions failed with default inputs: {e}")
        

    def _fg_loss(self, z_in, z_reconstructed):
        return self._loss_weights[0] * self._func_fg(z_in, z_reconstructed)
    

    def _area_loss(self, monochain, GBZ):
        return self._loss_weights[1] * self._func_area(monochain, GBZ)
    

    def _symm_loss(self, GBZ, GBZ_conj_input):
        if self._func_symm is not None:
            return self._loss_weights[2] * self._func_symm(GBZ, GBZ_conj_input)
        return torch.tensor(0.)
    

    def _uniform_loss(self, GBZ):
        if self._func_uniform is not None:
            return self._loss_weights[3] * self._func_uniform(GBZ)
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





def test_models(monochain: MonoChain,
                load_fold: str, 
                N: int = 301, rand_input: bool = True,
                loss_config: LossConfig = LossConfig()):
    """
    Test the trained encoder and decoder.
    There should be files 'Encoder.pt' and 'Decoder.pt' under load_fold.
    """
    if not monochain._tag in load_fold:
        logging.basicConfig(format='%(message)s', level=logging.WARNING)
        logger = logging.getLogger()
        logger.warning(f"Warning: '{load_fold}' seems not match with the given monochain" + 
                       f"\n{monochain}.")

    try:
        f_net_test = torch.jit.load(os.path.join(load_fold, 'Encoder.pt').replace('\\', '/'))
        g_net_test = torch.jit.load(os.path.join(load_fold, 'Decoder.pt').replace('\\', '/'))
    except:
        raise Exception(f'No available file in folder {load_fold}')
    
    z_test = generate_input_vec(N, rand_input=rand_input)
    GBZ_test = f_net_test(z_test)
    z_test_reconstructed = g_net_test(GBZ_test)

    loss_fg = loss_config._fg_loss(z_test, z_test_reconstructed)
    loss_area = loss_config._area_loss(monochain, GBZ_test)
    conj_input = z_test * torch.tensor([1, -1])
    GBZ_test_conj_input = f_net_test(conj_input)
    loss_symm = loss_config._symm_loss(GBZ_test, GBZ_test_conj_input)
    loss_uniform = loss_config._uniform_loss(GBZ_test)
    with torch.no_grad():
        plt_GBZ_and_Spectra(GBZ_test, monochain._get_spectra_from_GBZ(GBZ_test))

    print(f'Test monochain={monochain}:'+ '\n' + 
          f'    loss_fg: {loss_fg.item()}, ' +
          f'    loss_symm: {loss_symm.item()}' + '\n' +
          f'    loss_area: {loss_area.item()}, ' +
          f'    loss_uniform: {loss_uniform.item()}')





def train_models(f_net: nn.Module, g_net: nn.Module,
                 monochain: MonoChain,
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

    temp_save_fold = f'./temp_save_weights/'
    for root, dirs, files in os.walk(temp_save_fold):
        for file in files:
            os.remove(os.path.join(root, file).replace('\\', '/'))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir).replace('\\', '/'))

    temp_save_tag = monochain._tag
    temp_save_path = os.path.join(temp_save_fold, temp_save_tag).replace('\\', '/')
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
        dset_loss_symm = f.create_dataset('loss_symm', (num_epochs,))
        dset_loss_uniform = f.create_dataset('loss_uniform', (num_epochs,))
        dset_lr_f = f.create_dataset('lr_f', (num_epochs,))
        dset_lr_g = f.create_dataset('lr_g', (num_epochs,))

        for epoch in range(num_epochs):

            f_optimizer.zero_grad()
            g_optimizer.zero_grad()

            zin = generate_input_vec(train_config._num_input, train_config._rand_input)

            GBZ_pred = f_net(zin)
            z_reconstructed = g_net(GBZ_pred)

            loss_fg = loss_config._fg_loss(zin, z_reconstructed)
            loss_area = loss_config._area_loss(monochain, GBZ_pred)

            conj_input = zin * torch.tensor([1, -1])
            GBZ_pred_conj_input = f_net(conj_input)
            loss_symm = loss_config._symm_loss(GBZ_pred, GBZ_pred_conj_input)
            
            loss_uniform = loss_config._uniform_loss(GBZ_pred)
            
            loss = loss_fg + loss_area + loss_symm + loss_uniform
    
            loss.backward()
            f_optimizer.step()
            g_optimizer.step()

            if f_scheduler is not None:
                f_scheduler.step()
            if g_scheduler is not None:
                g_scheduler.step()

            dset_loss_fg[epoch] = loss_fg.item()
            dset_loss_area[epoch] = loss_area.item()
            dset_loss_symm[epoch] = loss_symm.item()
            dset_loss_uniform[epoch] = loss_uniform.item()
            dset_lr_f[epoch] = f_optimizer.param_groups[0]['lr']
            dset_lr_g[epoch] = g_optimizer.param_groups[0]['lr']

            if (epoch+1) % 50 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}]:'+ '\n' + 
                             f'    loss_fg: {loss_fg.item()}, ' +
                             f'    loss_symm: {loss_symm.item()}' + '\n' +
                             f'    loss_area: {loss_area.item()}, ' +
                             f'    loss_uniform: {loss_uniform.item()}')

            if (epoch+1) % 200 == 0:
                save_models(f_net, g_net, f_save_path, g_save_path)

                with torch.no_grad():
                    plt_GBZ_and_Spectra(GBZ_pred, monochain._get_spectra_from_GBZ(GBZ_pred),
                                        Spectra_major_step=(2, 2))

    plt_record_loss(loss_path)        
    save_models(f_net, g_net, f_save_path, g_save_path)

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
LCoupleCoef = (1, 1+1.j)
RCoupleCoef = (1, 1-1.j)
target_monochain = MonoChain(LCoupleCoef, RCoupleCoef, scale_factor=1)

parent_monochain = MonoChain((1,), (1, 1), scale_factor=1)
parent_init = './save_weights/' + parent_monochain._tag

train_config = TrainConfig(
    num_epochs= 2,
    init_path= None,
    num_input= 301,
    rand_input= True,
    optimizer_class= optim.Adam,
    optimizer_params= {'lr': 1e-3},
    scheduler_class= lr_scheduler.StepLR,
    scheduler_params= {'step_size': 100, 'gamma': 0.9})

loss_config = LossConfig(
    loss_weights= (500, 1, 50, 50),
    func_fg= nn.MSELoss(reduction='mean'),
    func_area= AreaLoss(farea=PolyArea),
    func_symm= SymmLoss(fdist=nn.L1Loss(reduction='mean')),
    func_uniform= UniformLoss(method='DistMAD'))

# loss_config = LossConfig(
#     loss_weights= (500, 1, 0, 50),
#     func_fg= nn.MSELoss(reduction='mean'),
#     func_area= AreaLoss(farea=PolyArea),
#     func_symm= None,
#     func_uniform= UniformLoss(method='DistMAD'))


train_models(f_net, g_net, target_monochain, train_config, loss_config)








