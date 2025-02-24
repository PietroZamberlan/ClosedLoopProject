from scipy.special import i0  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image
import matplotlib.animation as animation
import os
import glob



def rm_arrays_imgs(PATH):
    """
    remove saved numpy arrays and images in specified PATH
    
    """
    
    files = glob.glob(PATH + 'numpy/*')
    for f in files:
        os.remove(f)

    files = glob.glob(PATH + 'imgs/*')
    for f in files:
        os.remove(f)


#####
def plot_animation(imagelist, interval = 25):
    """
    To plot animated stimuli
    interval is related to the frame rate, 1000/interval = frame rate (seconds)
    """
    
    fig = plt.figure() # make figure

    # make axesimage object
    # the vmin and vmax here are very important to get the color map correct
    im = plt.imshow(imagelist[0], cmap= "gray", vmin=0, vmax=255)

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        # return the artists set
        return [im]

    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(len(imagelist)), 
                                  interval=interval, blit = True)
    return ani
    


def plot_von_mises(mu, kappa):
    
    x = np.linspace(-np.pi, np.pi, num=501)
    y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa)) 

    theta = np.linspace(-np.pi, np.pi, num=50, endpoint=False)
    radii = np.exp(kappa*np.cos(theta-mu))/(2*np.pi*i0(kappa)) 

    # PLOT
    plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)

    # Bin width?
    width = (2*np.pi) / 50

    # Angles increase clockwise from North
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1);

    bars = ax.bar(x=theta, height = radii, width=width, bottom=0)

    # Plot Line
    line = ax.plot(x, y, linewidth=2, color='firebrick', zorder=3 ) 

    # 'Trick': This will display Zero as a circle. Fitted Von-Mises function will lie along zero.
    ax.set_ylim(-0.5, 1.5);

    ax.set_rgrids(np.arange(0, 1.6, 0.5), angle=60, weight= 'bold',
                 labels=np.arange(0,1.6,0.5));
    
    

def plot(imgs, orig_img, with_orig=True, row_title=None, **imshow_kwargs):
    """
    Plot sequence of transformations 
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize = (20,7))
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()