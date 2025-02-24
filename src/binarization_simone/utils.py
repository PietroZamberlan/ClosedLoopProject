import torch
from torch import nn
# Plotting
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.animation as animation


import numpy as np
import pickle
import os
import glob
import re
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import csv
from colorama import Fore, Style
#import params
import math
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import dendrogram
from scipy.ndimage import convolve

######################################################################## 
def init_weights(model):
    """
    weight initialization, especially for the linear layer + bias 
    """
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(-4.93) #-5. bias should be initialized ~ average neural activity... 
                                 #(remember that the value passes through the sofplus function)
            
    
    # Conv initialization shouldn't be needed   
    # the default is already kaiming 
            
    #if isinstance(m, nn.Conv3d):
    #    torch.nn.init.uniform_(m.weight, a = -np.sqrt(1.55/nn.init._calculate_fan_in_and_fan_out(m.weight)[0]),
    #                           b = np.sqrt(1.55/nn.init._calculate_fan_in_and_fan_out(m.weight)[0]))
        #nn.init._calculate_fan_in_and_fan_out()
        #torch.nn.init.kaiming_uniform_(m.weight)
        

######################################################################## 
def clean_gpu_ram(model, delete_model = False):
    """
    Utility function to free gpu's RAM with PyTorch
    """
    
    if delete_model == True:
        del model
        
    torch.cuda.empty_cache()
    
    
######################################################################## 
def get_n_params(model):
    """
    Utility function to count model's parameters 
    
    """
    
    params_count=0
    for params in list(model.parameters()):
        
        nn=1
        for s in list(params.size()):
            nn = nn*s
            
        params_count += nn
        
    return params_count


######################################################################## 
def get_values_matching_keys(y, weight, device = "cpu"):
    """ 
    Utility function to weight Poisson loss
    
    in case we would like to further weight the loss function
    this utility function allows to match the output with the weight 
    to make a loss function that depends on that 
    """
    
    y = y.detach().cpu().type(torch.int8)
    
    return torch.Tensor([weight[key] for key in y]).to(device)




def plot_animation(imagelist, interval = 100, plot = True):
    """
    Utility function to plot animated gradients and export them as .gif
    
        - interval is related to the frame rate, 1000/interval = frame rate (seconds)
    """
    
    
    if plot==False: 
        plt.ioff()
        
    
    #fig = plt.figure() # make figure
    fig, ax = plt.subplots()

    # make axesimage object
    # the vmin and vmax here are very important to get the color map correct
    im = ax.imshow(imagelist[0], vmin = imagelist.min(), vmax = imagelist.max(), cmap = "coolwarm")
    cbar = fig.colorbar(im)
   
    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        ax.set_title("Gradients at t = " + str(j))
        # return the artists set
        return [im]

    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(len(imagelist)), 
                                  interval=interval, blit = False)
    return ani, fig


######################################################################## 
def batch_combinations(inp, comb_order = 3, prod = True, device = "cpu"):
    """
    Utility function to get neurons cross-moments, 
    i.e. higher-order correlations through nonlinearity
    """
    
    c = torch.combinations(torch.arange(inp.size(1)), r= comb_order, with_replacement = True)
    x = inp[:,None].expand(-1,len(c),-1)
    idx = c[None].expand(len(x), -1, -1).to(device)
    x_out = x.gather(dim=2, index=idx)
    
    if prod == True: 
        x_out = torch.prod(x_out, dim = 2)
    
    return x_out




#################

def gaussian2D(shape, amp, x0, y0, sigma_x, sigma_y, angle,):
    if sigma_x == 0:
        sigma_x = 0.001
    
    if sigma_y == 0:
        sigma_y = 0.001
    shape = (int(shape[0]),int(shape[1]))
    x=np.linspace(0,shape[1],shape[1])
    y=np.linspace(0,shape[0],shape[0])
    X,Y = np.meshgrid(x,y)
    
    theta = 3.14*angle/180
    a = (math.cos(theta)**2)/(2*sigma_x**2) + (math.sin(theta)**2)/(2*sigma_y**2)
    b = -(math.sin(2*theta))/(4*sigma_x**2) + (math.sin(2*theta))/(4*sigma_y**2)
    c = (math.sin(theta)**2)/(2*sigma_x**2) + (math.cos(theta)**2)/(2*sigma_y**2)
    
    return amp*np.exp( - (a*np.power((X-x0),2) + 2*b*np.multiply((X-x0),(Y-y0))+ c*np.power((Y-y0),2)))

def gaussian2D_flat(x, amp, x0, y0, rx, ry, rot):
    return gaussian2D(x, amp, x0, y0, rx, ry, rot).flatten()

def reduced_gaussian2D(x, amp, sigma_x, sigma_y, angle,):
    
    shape = (int(x[0]),int(x[1]))
    x0 = int(x[2])
    y0 = int(x[3])
    
    x=np.linspace(0,shape[1],shape[1])
    y=np.linspace(0,shape[0],shape[0])
    X,Y = np.meshgrid(x,y)
    
    theta = 3.14*angle/180
    a = (math.cos(theta)**2)/(2*sigma_x**2) + (math.sin(theta)**2)/(2*sigma_y**2)
    b = -(math.sin(2*theta))/(4*sigma_x**2) + (math.sin(2*theta))/(4*sigma_y**2)
    c = (math.sin(theta)**2)/(2*sigma_x**2) + (math.cos(theta)**2)/(2*sigma_y**2)
    
    return amp*np.exp( - (a*np.power((X-x0),2) + 2*b*np.multiply((X-x0),(Y-y0))+ c*np.power((Y-y0),2)))

def reduced_gaussian2D_flat(x, amp, rx, ry, rot):
    return reduced_gaussian2D(x, amp, rx, ry, rot).flatten()

def gaussian_ellipse(amp, x0, y0, sigma_x, sigma_y, angle, ratio = math.sqrt(2)):

    level = amp*0.5
    
    theta = 3.14*angle/180
    a = (math.cos(theta)**2)/(2*sigma_x**2) + (math.sin(theta)**2)/(2*sigma_y**2)
    b = -(math.sin(2*theta))/(4*sigma_x**2) + (math.sin(2*theta))/(4*sigma_y**2)
    c = (math.sin(theta)**2)/(2*sigma_x**2) + (math.cos(theta)**2)/(2*sigma_y**2)
    


    lim = math.sqrt((c*math.log(level/amp))/(b**2-c*a))
    xmin = x0 - lim
    xmax = x0 + lim
    
    X  = np.linspace(xmin,xmax,1000)
    Ym = y0 + (-2*b*(X-x0)-np.sqrt(4*b**2*(X-x0)**2 - 4*c*(a*(X-x0)**2+math.log(level/amp))))/2*c
    Yp = y0 + (-2*b*(X-x0)+np.sqrt(4*b**2*(X-x0)**2 - 4*c*(a*(X-x0)**2+math.log(level/amp))))/2*c

    return np.append(X,X), np.append(Ym,Yp)

####  Gabriel's analysis  ####

def gabriel_preprocessing(sta_3D, nb_frames = 15, kernel_lenght = 2, tresholding_factor = 2):
    
    data = sta_3D[-nb_frames:,:,:]
    
    #smoothing along time    
    kernel = np.ones(kernel_lenght)[:,None,None]/kernel_lenght
    data = convolve(data, kernel, mode='nearest')
    
    ## Take variance
    data = data.var(0)
    data -= np.median(data)
    data /= np.max(np.abs(data))
    spatial_sta = data.copy()
    
    ## Thresholding
    tresholding_factor = 2
    k_gauss = 1.5 # 1.5 mad ~ 1 std for gaussian noise
    mad = np.median(np.abs(data-np.median(data)))
    data[data<tresholding_factor*k_gauss*mad] = 0
    
    return data, spatial_sta

def gabriel_temporal_sta(sta_3D, gaussian_params):
    
    shape = (sta_3D.shape[1], sta_3D.shape[2])
    smoothing_kernel  = gaussian2D(shape, *gaussian_params)
    smoothing_kernel /= np.sum(smoothing_kernel)
    
    # Find max in space
    smoothed_sta = convolve(sta_3D.var(0), smoothing_kernel, mode='nearest')
    x_max, y_max = np.unravel_index(np.argmax(smoothed_sta), shape=shape)
    # Gaussian weighting kernel

    gaussian_kernel = gaussian2D(shape, gaussian_params[0], x_max, y_max, *gaussian_params[3:])
    gaussian_kernel = gaussian_kernel/np.sum(gaussian_kernel)
    # Weighted temporal trace
    return np.mean(gaussian_kernel[None,:,:]*sta_3D, (1,2))
    
def fit_gaussian(sta_spatial):
    
    center = np.unravel_index(np.argmax(sta_spatial, axis=None), sta_spatial.shape)
    guess  = [np.max(sta_spatial), center[1], center[0], 1, 1, 0]    
    
    xdata  = sta_spatial.shape
    ydata  = sta_spatial.flatten()
    
    ellispe_params_bounds = ((-2, 0, 0, 0.1, 0.1, 0),(2, sta_spatial.shape[0], sta_spatial.shape[0], sta_spatial.shape[0], sta_spatial.shape[0], 180))

    return curve_fit(gaussian2D_flat, xdata, ydata, p0=guess, bounds=ellispe_params_bounds)

def analyse_sta_gab(sta, cell_id):
    sta_3D = sta.copy()
    fitting_data, spatial_sta = gabriel_preprocessing(sta_3D)
    try:
        ellipse_params, cov = fit_gaussian(fitting_data)
    except:
        print(f'Error Could not fit ellipse {cell_id}')
        plt.imshow(fitting_data)
        plt.show(block=False)
        return {'Spatial':sta_spatial, 'Temporal':sta_temporal, 'EllipseCoor':[0, 0, 0, 0.001, 0.001, 0], 'Cell_delay' : np.nan}
    temporal_sta = gabriel_temporal_sta(sta_3D, ellipse_params)
    
    return {'Spatial':spatial_sta, 'Temporal':temporal_sta, 'EllipseCoor':ellipse_params, 'Cell_delay' : np.nan}
    
####  Matias sta analysis ####

def smooth_sta(sta, alpha, max_time_window=15):
    pading_size = 1
    paded_sta       = np.pad(sta, pading_size)    ### change from a zeros matrice to a padded one. Countour of rf is not 0 now but the sta value itself
    receptive_field = np.zeros(sta.shape)
    for x in range(sta.shape[1]):
        for y in range(sta.shape[2]):
            receptive_field[:,x,y] = paded_sta[1:-1,x+pading_size,y+pading_size]*alpha + (1-alpha)*paded_sta[1:-1,x+pading_size-1:x+pading_size+2,y+pading_size-1:y+pading_size+2].sum(axis=(1,2)) 
    
    best    = np.unravel_index(np.argmax(np.abs(receptive_field[-max_time_window:,:,:])), receptive_field.shape)
    best_t = best[0] + max(sta.shape[0]-max_time_window,0)
    
    return receptive_field, (best_t,best[1],best[2]), receptive_field[best]


def get_cell_shift(sta):
    sta_3D = sta.copy()
    smooth_sta_1_3D, best_1,max_val1 = smooth_sta(sta_3D, alpha=0.5)
    smooth_sta_2_3D, best_2 ,max_val2 = smooth_sta(sta_3D, alpha=0.8)
    
    if abs(max_val1) > abs(max_val2):
        return best_1
    else:
        return best_2

def preprocess_fitting_matias(spatial, treshold=0.1):
    
    sta_spa = spatial.copy()
    sta_treshold = np.max(np.abs(spatial))*treshold
    sta_spa[np.abs(sta_spa)<sta_treshold] = 0
    return sta_spa

def matias_temporal_spatial_sta(sta_3D):
    if np.max(np.abs(sta_3D)) == 0:
        print(f'Cell {cell_id} : Could not find sta')
        return 'Error detected : 3D sta empty', 'Error detected : 3D sta empty'
    
    (best_t, best_x, best_y)  = get_cell_shift(sta_3D)
    sta_temporal  = sta_3D[:,best_x,best_y]
    sta_spatial   = sta_3D[best_t,:,:]
    sta_spatial  /= np.max(np.abs(sta_spatial)) 
    
    return sta_temporal, sta_spatial, (best_t, best_x, best_y)


def double_gaussian_fit(spatial):
    
    center = np.unravel_index(np.argmax(np.abs(spatial), axis=None), spatial.shape)
    ydata = spatial.flatten()
    
    #First fit without center variability
    first_guess = [spatial[center[0],center[1]], 1, 1, 0]   
    xdata = [spatial.shape[0],spatial.shape[1],center[1],center[0]]
    
    ellispe_params_bounds = ((-2, 0.1, 0.1, 0),(2, spatial.shape[0], spatial.shape[0], 180))

    opt, cov = curve_fit(reduced_gaussian2D_flat, xdata, ydata, p0=first_guess, bounds=ellispe_params_bounds)

    #Second fit with center variability
    xdata = spatial.shape
    second_guess = [opt[0], center[1], center[0], opt[1], opt[2], opt[3]]

    ellispe_params_bounds = ((-2, 0, 0, 0.1, 0.1, 0),(2, spatial.shape[0], spatial.shape[0], spatial.shape[0], spatial.shape[0], 180))
    return curve_fit(gaussian2D_flat, xdata, ydata, p0=second_guess, bounds=ellispe_params_bounds)
    
def analyse_sta_matias(sta, cell_id):
    sta_3D = sta.copy()
    
    sta_temporal, sta_spatial, best = matias_temporal_spatial_sta(sta_3D)
    fitting_data = preprocess_fitting_matias(sta_spatial)
    try :
        ellipse_params,cov = double_gaussian_fit(fitting_data)
    except:
        print(f'Error Could not fit ellipse {cell_id}')
        plt.imshow(fitting_data)
        plt.show(block=False)
        return {'Spatial':sta_spatial, 'Temporal':sta_temporal, 'EllipseCoor':[0, 0, 0, 0.001, 0.001, 0], 'Cell_delay' : best[0]}
    return {'Spatial':sta_spatial, 'Temporal':sta_temporal, 'EllipseCoor':ellipse_params, 'Cell_delay' : best[0]}

### Guilhem sta analysis ### (mixed between both)

def analyse_sta(sta, cell_id):
    sta_3D = sta.copy()
    fitting_data, spatial_sta = gabriel_preprocessing(sta_3D, tresholding_factor=1)
    try:
        ellipse_params, cov = double_gaussian_fit(fitting_data)
        temporal_sta = gabriel_temporal_sta(sta_3D, ellipse_params)
        best_t  = np.argmax(np.abs(temporal_sta[-15:]))
        best_t += max(sta.shape[0]-15,0)

        spatial_sta = sta_3D[best_t]

        return {'Spatial':spatial_sta, 'Temporal':temporal_sta, 'EllipseCoor':ellipse_params, 'Cell_delay' : best_t}

    except:
        print(f'Error Could not fit ellipse {cell_id}')
        plt.imshow(fitting_data)
        plt.show(block=False)
        return {'Spatial':spatial_sta, 'Temporal':np.zeros(40), 'EllipseCoor':[0, 0, 0, 0.001, 0.001, 0], 'Cell_delay':np.nan}
    
    
def plot_sta(ax, spatial_sta, ellipse_params, level_factor=0.4):
    #magnified_ellipse_params=(np.array(ellipse_params)*[gaussian_factor, 1,1,gaussian_factor,gaussian_factor,1])
    gaussian = gaussian2D(spatial_sta.shape,*ellipse_params)
    ax.imshow(spatial_sta)
    if ellipse_params[0] != 0:
        ax.contour(np.abs(gaussian),levels = [level_factor*np.max(np.abs(gaussian))], colors='w',linestyles = 'solid', alpha = 0.8)
    return ax
