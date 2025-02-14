import json

from config.config     import *
from src.TCP.tcp_utils import *
from gaussian_processes.Spatial_GP_repo import utils as GP_utils
from src.Win_side.generate_electrode_info import generate_electrode_info
'''
This utils should have all the functions used in the main.py script but also the ones related 
to the modeling that dont do the heavy lifting of the GP training, inference and utility estimation
(those come from GP_utils)


counting triggers -> tcp_utils.py

pick best image   -> main_utils.py
'''
############### ELECTRODE INFO FUNCTIONS ####################
def upload_electrode_info(electrode_info_path, print_info = True, testmode = False):

    '''
    Uploads the electrode_info_file from the given path set in the config file.
    '''
    if testmode:
        print(f'uploadi_electrode_data in TEST mode, values generated from config.py')
        electrode_info = generate_electrode_info(testmode)

    with open(electrode_info_path, 'r') as f:
        electrode_info = json.load(f)

    for key, value in electrode_info.items():
        if type(value) == float:
            print(f'   {key}: {value:.2f}')
        else:
            print(f'   {key}: {value:.0f}')

    return electrode_info

def theta_from_electrode_info( electrode_info ):
    '''
    Generates the theta dictionary of hyperparameters from the electrode_info dictionary:

    Rescales hyperparameters gave in pixels to [-1,1] 
    Transforms them into tensors.
    '''
    
    beta_px   = torch.tensor(electrode_info['RF_size'])
    rho_px    = torch.tensor(electrode_info['localker_smoothness'])
    eps_0x_px = torch.tensor(electrode_info['RF_x_center'])
    eps_0y_px = torch.tensor(electrode_info['RF_y_center'])

    sigma_0 = torch.tensor(electrode_info['acosker_sigma_0']) 
    Amp = torch.tensor(electrode_info['localker_amplitude'])

    # region _____ Rescale hyperparameters in pixels to [-1,1] ______
    # Pixel quantities go from 0 to nat_img_px_nb-1, I bring it to [-1,1] multiplying by 2 and shifting
    eps_0x = ( eps_0x_px / (nat_img_px_nb - 1))*2 - 1
    eps_0y = ( eps_0y_px / (nat_img_px_nb - 1))*2 - 1
    beta   = beta_px / (nat_img_px_nb-1)*2 - 1
    rho    = rho_px  / (nat_img_px_nb-1)*2 - 1
    # endregion

    logbetaexpr = -2*torch.log(2*beta)
    logrhoexpr  = -torch.log(2*rho*rho)

    theta = {'sigma_0'  : torch.tensor(sigma_0),     'Amp': torch.tensor(Amp),   # Hypermarameters are expected to be 0-dimensional tensors
            'eps_0x'    : torch.tensor(eps_0x) ,     'eps_0y'   : torch.tensor(eps_0y), 
            '-2log2beta': torch.tensor(logbetaexpr), '-log2rho2': torch.tensor(logrhoexpr) }
    

    return theta

def model_from_electrode_info( electrode_info ):
    '''
    Sets up the start_model with the hyperparameters initialized in electrode_info
    '''

    # region _____ Set hyp amd params ______
    theta = theta_from_electrode_info( electrode_info )

    A        = torch.tensor(A_init)
    logA     = torch.log(A)
    lambda0  = torch.tensor(lambda0_init)

    hyperparams_tuple = GP_utils.gen_hyp_tuple( theta, freeze_list=['Amp'], display_hyper=True )
    f_params          = GP_utils.set_f_params( logA, lambda0 )
    # endregion

    # region _____ Genenerate idxs for starting dataset of images to be shown______
    _, _, idx_tuple  = GP_utils.get_idx_for_training_testing_validation( 
            X=[], R=[], ntrain=ntrain_init, ntilde=ntilde_init, ntest_lk=0)

    xtilde_idx, in_use_idx, remaining_idx, test_lk_idx = idx_tuple


    fit_parameters = {'ntilde':    ntilde_init,
                    'maxiter':     maxiter_init,
                    'nMstep':      nMstep_init,
                    'nEstep':      nEstep_init,
                    'nFparamstep': nFparamstep_init,
                    'kernfun':     GP_utils.kernfun,
                    'cellid':      cellid_init,
                    'n_px_side':   n_px_side_init,
                    'in_use_idx':  in_use_idx,     # Used idx for generating xtilde, referred to the whole X dataset
                    'xtilde_idx':  xtilde_idx,     # Used idx for generating the complete set, referred to the whole X dataset
                    'start_idx':   in_use_idx,     # Indexes used to generate the initial training set
                    'lr_Mstep':      lr_Mstep_init, 
                    'lr_Fparamstep': lr_Fparamstep_init
    }

    init_model = {
            'fit_parameters':    fit_parameters,
            'xtilde':            xtilde,
            'hyperparams_tuple': hyperparams_tuple,     # Contains also the upper and lower bounds for the hyperparameters
            'f_params':          f_params,
            # 'm':                 torch.zeros( (ntilde) )
            # 'm': torch.ones( (ntilde) )
            #'V': dont initialize V if you want it to be initialized as K_tilde and projected _exactly_ as K_tilde_b for stabilisation
        }

    init_model['hyperparams_tuple'] = (theta, init_model['hyperparams_tuple'][1], init_model['hyperparams_tuple'][2])

    # region _____ Hyperparameters choice plotted on the STA ______

    initial_STA_fig, ax = plt.subplots(1, 1, figsize=(5,5)) 

    # Eps_0 : Center of the receptive field
    center_idxs = torch.tensor([(n_px_side-1)/2, (n_px_side-1)/2])
    eps_idxs    = torch.tensor( [
        center_idxs[0]*(1+eps_0x), 
        center_idxs[1]*(1+eps_0y)
        ])

    # Beta : Width of the receptive field - Implemented by the "alpha_local" part of the C covariance matrix
    ycord, xcord = torch.meshgrid( torch.linspace(-1, 1, n_px_side), torch.linspace(-1, 1, n_px_side), indexing='ij') # a grid of 108x108 points between -1 and 1
    xcord = xcord.flatten()
    ycord = ycord.flatten()
    logalpha    = -torch.exp( theta['-2log2beta']     )*((xcord - eps_0x)**2+(ycord - eps_0y)**2  )
    alpha_local =  torch.exp(logalpha)    # aplha_local in the paper

    # Levels of the contour plot for distances [1sigma, 2sigma, 3sigma]
    # (x**2 + y**2) = n*sigma -> alpha_local = exp( - (n*sigma)^2 / (2*sigma^2) )
    levels = torch.tensor( [np.exp(-4.5), np.exp(-2), np.exp(-1/2) ])
    ax.contour( alpha_local.reshape(n_px_side,n_px_side).cpu(), levels=levels.cpu(), colors='red', alpha=0.5)
    ax.scatter( eps_idxs[0].cpu(),eps_idxs[1].cpu(), color='white', s=30, marker="o", label='Initial center guess', )
    # ax.imshow( alpha_local.reshape(n_px_side,n_px_side).cpu(),)
    ax.imshow(STA, origin='lower')
    ax.legend(loc='upper right')
    initial_STA_fig.suptitle(f'STA of cell: {cellid} - Initial hyperparameters')

    # endregion

def update_model(new_spike_count, current_img_id, current_model, print_lock):
    '''
    Fits a new model adding a new image - spike count pair to current_model

    Updates the GP variational (m,V) and likelihood (A, lambda0) parameters. 
    Does not update the hyperparameters (kernel parameters) which are fixed.
    
    Returns:
      updated_model: dict - The updated model parameters
    
    Args:
        new_spike_count (int): The number of spikes received after the image was displayed
        current_img_id (int):  The ID of the image that was displayed
        current_model (dict):  The current model parameters 
        print_lock (threading.Lock): The lock for printing to the console, since this functon is called by a thread
    '''
    start_time = time.time()
    with print_lock: print(f"\n...Fit Thread: Starting fit using {new_spike_count} spikes...")
    new_spike_count = torch.tensor(new_spike_count, device=DEVICE)
    # with print_lock:
        # print(f"\n...Thread: New_spike_count is on device: {new_spike_count.device}")

    


    time.sleep(1)
    with print_lock: print(f"\n...Fit Thread: Fit finished in {time.time()-start_time:.2f}s,")
    return updated_model

def temp_gen_hyp_tuple(theta, freeze_list, display_hyper=True):
    '''
    This should be in GP_utils.py, we put it here to not change the submodule.

    (Better) Alternative to generate_theta.

    Generates the hyperparameters tuple (theta, theta_lower_lims, theta_higher_lims) 

    Sets the requires_grad attribute of hyp to True except for the ones in freeze_list

    Args:
        theta: dictionary of hyperparameters
    Returns:
        tuple of hyperparameters
    
    '''
    upp_lim =  torch.tensor(1.)
    low_lim = -torch.tensor(1.)
    theta_lower_lims  = {'sigma_0': 0           , 'eps_0x':low_lim, 'eps_0y':low_lim, '-2log2beta': -float('inf'), '-log2rho2':-float('inf'), 'Amp': 0. }
    theta_higher_lims = {'sigma_0': float('inf'), 'eps_0x':upp_lim,  'eps_0y':upp_lim,  '-2log2beta':  float('inf'), '-log2rho2': float('inf'), 'Amp': float('inf') }

    # Set the gradient of the hyperparemters to be updateable 
    for key, value in theta.items():
    # to exclude a single hyperparemeters from the optimization ( to exclude them all just set nMstep=0)
        if key in freeze_list:
            continue
        theta[key] = value.requires_grad_()
    if display_hyper: 
        print(f'{key} is {value.cpu().item():.4f}')

    return ( theta, theta_lower_lims, theta_higher_lims )