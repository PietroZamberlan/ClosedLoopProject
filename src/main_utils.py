import json

from config.config     import *
from src.TCP.tcp_utils import *
import utils as GPutils # ( importing utils from the Gaussian-Processes repo folder )

'''
This utils should have all the functions used in the main.py script but also the ones related 
to the modeling that dont do the heavy lifting of the GP training, inference and utility estimation
(those come from GPutils)


counting triggers -> tcp_utils.py

pick best image   -> main_utils.py
'''
############### ELECTRODE INFO FUNCTIONS ####################
def upload_electrode_info(electrode_info_path, print_info = True):

    '''
    Uploads the electrode_info_file from the given path set in the config file.
    '''

    with open(electrode_info_path, 'r') as f:
        electrode_info = json.load(f)

    for key, value in electrode_info.items():
        if type(value) == float:
            print(f'   {key}: {value:.2f}')
        else:
            print(f'   {key}: {value:.0f}')

    return electrode_info

def theta_from_electrode_info( electrode_info ):
    beta = torch.tensor(electrode_info['RF_size'])
    rho  = torch.tensor(electrode_info['localker_smoothness'])

    logbetaexpr = -2*torch.log(2*beta)
    logrhoexpr  = -torch.log(2*rho*rho)

    sigma_0 = torch.tensor(electrode_info['acosker_sigma_0']) 

    Amp = torch.tensor(electrode_info['localker_amplitude'])

    eps_0x = torch.tensor(electrode_info['RF_x_center'])
    eps_0y = torch.tensor(electrode_info['RF_y_center'])

    theta = {'sigma_0'  : sigma_0,     'Amp': Amp,   # Hypermarameters are expected to be 0-dimensional tensors
            'eps_0x'    : eps_0x ,     'eps_0y'   : eps_0y, 
            '-2log2beta': logbetaexpr, '-log2rho2': logrhoexpr }

    return theta

def model_from_electrode_info( electrode_info ):

    '''
    Sets up the start_model with the hyperparameters initialized in electrode_info
    '''

    # region _____ Set hyp amd params ______
    theta = theta_from_electrode_info( electrode_info )

    A        = torch.tensor(0.01)
    logA     = torch.log(A)
    lambda0  = torch.tensor(1.)

    hyperparams_tuple = GPutils.gen_hyp_tuple( theta, freeze_list=['Amp'], display_hyper=True )
    f_params          = GPutils.set_f_params( logA, lambda0 )
    # endregion

    # region _____ Upload dataset with the recorded responses and generate idxs ______
    _, _, idx_tuple  = GPutils.get_idx_for_training_testing_validation( 
            X=[], R=[], ntrain=ntrain_init, ntilde=ntilde_init, ntest_lk=0)

    xtilde_idx, in_use_idx, remaining_idx, test_lk_idx = idx_tuple


    fit_parameters = {'ntilde':    ntilde_init,
                    'maxiter':     maxiter_init,
                    'nMstep':      nMstep_init,
                    'nEstep':      nEstep_init,
                    'nFparamstep': nFparamstep_init,
                    'kernfun':     GPutils.kernfun,
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
