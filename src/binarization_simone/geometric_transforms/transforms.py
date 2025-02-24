LEN_RAND_VECS = 500


import numpy as np
from geometric_transforms.random_motion import ornstein_uhlenbeck
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T
import cv2


def get_perspectives(width: int, height: int, distortion_scale: float, N = LEN_RAND_VECS):
    
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            N (int) : successive N positions of the four corners (ornstein-ulhenbeck evolution)

        Returns:
            List of List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
        """
        
        half_height = height // 2
        half_width = width // 2
        
        
        t = np.arange(0, N, 1)
        # simulate a path of the OU process on a given grid t, starting with x_0 
        ou_1 = ornstein_uhlenbeck(0., t, 0.95, 0., 0.05)   
        ou_2 = ornstein_uhlenbeck(0., t, 0.95, 0., 0.05)
        ou_3 = ornstein_uhlenbeck(0., t, 0.95, 0., 0.05)  
        ou_4 = ornstein_uhlenbeck(0., t, 0.95, 0., 0.05)
        ou_5 = ornstein_uhlenbeck(0., t, 0.95, 0., 0.05)  
        ou_6 = ornstein_uhlenbeck(0., t, 0.95, 0., 0.05)
        ou_7 = ornstein_uhlenbeck(0., t, 0.95, 0., 0.05) 
        ou_8 = ornstein_uhlenbeck(0., t, 0.95, 0., 0.05)
        
        
        max_1 = int(distortion_scale * half_width)
        max_2 = int(distortion_scale * half_height)
        
        ou_1_norm = np.array((ou_1 - ou_1.min())/(ou_1.max() - ou_1.min())*max_1).astype(np.uint8)
        ou_2_norm = np.array((ou_2 - ou_2.min())/(ou_2.max() - ou_2.min())*max_2).astype(np.uint8)
        ou_3_norm = np.array((ou_3 - ou_3.min())/(ou_3.max() - ou_3.min())*max_1).astype(np.uint8)
        ou_4_norm = np.array((ou_4 - ou_4.min())/(ou_4.max() - ou_4.min())*max_2).astype(np.uint8)
        
        ou_5_norm = np.array((ou_5 - ou_5.min())/(ou_5.max() - ou_5.min())*max_1).astype(np.uint8)
        ou_6_norm = np.array((ou_6 - ou_6.min())/(ou_6.max() - ou_6.min())*max_2).astype(np.uint8)
        ou_7_norm = np.array((ou_7 - ou_7.min())/(ou_7.max() - ou_7.min())*max_1 ).astype(np.uint8)
        ou_8_norm = np.array((ou_8 - ou_8.min())/(ou_8.max() - ou_8.min())*max_2 ).astype(np.uint8)
 
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        
        points_ = []
        points_.append(startpoints)
        
        for i in range(len(ou_1_norm)):
        
            topleft = [ou_1_norm[i] + 1, ou_2_norm[i] + 1]
            topright = [width - ou_3_norm[i] -1, ou_4_norm[i] + 1]
            botright = [width - ou_5_norm[i] -1, height - ou_6_norm[i] -1]
            botleft = [ou_7_norm[i] + 1,  height - ou_8_norm[i] -1]
        
            endpoints = [topleft, topright, botright, botleft]
            points_.append(endpoints)
        
        return points_
    
    
def interpolate_perspectives(startpoints, endpoints, n_frames = 20):
    """
    Interpolate previously generated perspectives to make the animation smoother. 
    
    """
    
   
    interpolatedpoints = np.zeros((8, n_frames), dtype = np.int32)

    for idx in range(len(startpoints)):

        start_p = startpoints[idx]
        end_p = endpoints[idx]

        x_start, y_start = start_p
        x_end, y_end = end_p 

        x_int = np.linspace(x_start, x_end, num = n_frames, dtype = np.int32)
        y_int = np.linspace(y_start, y_end, num = n_frames, dtype = np.int32)

        pos_int = np.stack((x_int, y_int))
        interpolatedpoints[idx*2:idx*2+2,:] = pos_int

    list_int_points = []
    for i in range(interpolatedpoints.shape[1]):

        topleft = [interpolatedpoints[0,i], interpolatedpoints[1,i]]
        topright = [interpolatedpoints[2,i], interpolatedpoints[3,i]]
        botright = [interpolatedpoints[4,i], interpolatedpoints[5,i]]
        botleft = [interpolatedpoints[6,i], interpolatedpoints[7,i]]

        points_ = [topleft, topright, botright, botleft]
        list_int_points.append(points_)
        
    return list_int_points



#############################################################
def interpolate_affine_params(x_trans, y_trans, rot_angles, scales, shears, n_frames = 20):
    """
    Interpolate parameters of the previously generated affine transforms, to make the animation smoother. 
    Slightly different from the previous function 'interpolate_perspectives', I thought it would have been easier to do it 
    directly on the parameters for the affine one. 
    
    """
    
    x_trans_int = np.zeros((x_trans.shape[0]*n_frames))
    y_trans_int = np.zeros((x_trans.shape[0]*n_frames))
    rot_angles_int = np.zeros((x_trans.shape[0]*n_frames))
    scales_int = np.zeros((x_trans.shape[0]*n_frames))
    shears_int = np.zeros((x_trans.shape[0]*n_frames))
    
    
    for i in tqdm(range(x_trans.shape[0]-2)): 
        
        x_start, x_end = x_trans[i], x_trans[i+1]
        y_start, y_end = y_trans[i], y_trans[i+1]
        rot_angle_start, rot_angle_end = rot_angles[i], rot_angles[i+1]
        scale_start, scale_end = scales[i], scales[i+1]
        shear_start, shear_end = shears[i], shears[i+1]
        
        x_int = np.linspace(x_start, x_end, num = n_frames)
        y_int = np.linspace(y_start, y_end, num = n_frames)
        rot_angle_int = np.linspace(rot_angle_start, rot_angle_end, num = n_frames)
        scale_int = np.linspace(scale_start, scale_end, num = n_frames)
        shear_int = np.linspace(shear_start, shear_end, num = n_frames)
        
        x_trans_int[i*n_frames:(i+1)*n_frames] = x_int
        y_trans_int[i*n_frames:(i+1)*n_frames]= y_int
        rot_angles_int[i*n_frames:(i+1)*n_frames] = rot_angle_int
        scales_int[i*n_frames:(i+1)*n_frames] = scale_int
        shears_int[i*n_frames:(i+1)*n_frames] = shear_int
    
    
    #select non-zero elements
    x_trans_int = x_trans_int[:-2*n_frames]
    y_trans_int = y_trans_int[:-2*n_frames]
    rot_angles_int = rot_angles_int[:-2*n_frames]
    scales_int = scales_int[:-2*n_frames]
    shears_int = shears_int[:-2*n_frames]
    
    return x_trans_int, y_trans_int, rot_angles_int, scales_int, shears_int


###################
def generate_affine(texture, background, angle, translate, scale, shear, resize_to, colors = False):
    """
    Generate affine transformed texture on top of a background, takes care of different dimensions and resize to a specified one. 
    
    -----
    Inputs: 
        - path_texture (str): path of the foreground texture (stimulus)
        - path_background (str): path of the background image 
        - angle (float): rotation angle for the affine transform (torchvision.functional.affine)
        - translate (List[int]) : translation on x and y axes, list w/ [x_pos, y_pos] (torchvision.functional.affine)
        - scale (float): scaling parameter, affine transform (torchvision.functional.affine)
        - shear (float): shearing (deformation) parameter, affine transform (torchvision.functional.affine)
        - resize_to (Tuple[int]): resize output to specified x, y dimensions
        - colors (bool): whether to have a colored image or grayscale
        
    
    Outputs: 
        - stimulus (numpy array[int]): final image with foreground (texture) and background as a numpy array
    """
    
    img_texture = texture.resize((resize_to[1], resize_to[0]))
    aff_img = torch.tensor(np.asarray(T.functional.affine(img_texture, angle = angle, translate= translate, scale=scale, shear = shear)))
    
    back_img = torch.tensor(np.asarray(background.resize((aff_img.shape[1], aff_img.shape[0])))[:,:,0])
    back_img[aff_img > 0 ] = aff_img[aff_img > 0]
    
    # to rewrite
    if colors == True:
        stimulus = back_img.numpy()
        
    else:
        stimulus = back_img.numpy()#[:,:,0]
    
    return stimulus



##########################################################################################################
############################# Transformation parameters recovering #######################################
##########################################################################################################

def find_homography(src_points, dst_points, method=cv2.USAC_MAGSAC, ransac_reproj_threshold=2.0):
    """
    Compute the homography matrix using the given source and destination points.

    Args:
        src_points (ndarray): The source points, specified as a 2D array of shape (N, 1, 2).
        dst_points (ndarray): The destination points, specified as a 2D array of shape (N, 1, 2).
        method (int, optional): The method used for homography estimation. Defaults to cv2.USAC_MAGSAC.
        ransac_reproj_threshold (float, optional): The maximum allowed reprojection error for inliers.
                                                   Defaults to 2.0.

    Returns:
        homography (ndarray): The computed homography matrix.
        mask (ndarray): The mask indicating the inlier points.

    """
    homography, mask = cv2.findHomography(src_points, dst_points, method, ransac_reproj_threshold,
                                          maxIters=50000, confidence=0.99)
    return homography, mask


def compute_homography(frames):
    """
    Compute the homography matrices between consecutive frames.

    Args:
        frames (List[ndarray]): The list of grayscale frames.

    Returns:
        homography_matrices (List[ndarray]): The list of computed homography matrices.

    """
    homography_matrices = []

    for i in range(len(frames) - 1):
        prev_frame_gray = frames[i]
        curr_frame_gray = frames[i + 1]

        # Convert grayscale images to 8-bit depth
        prev_frame_gray = cv2.convertScaleAbs(prev_frame_gray)
        curr_frame_gray = cv2.convertScaleAbs(curr_frame_gray)

        # Convert grayscale images to color images
        prev_frame_color = cv2.cvtColor(prev_frame_gray, cv2.COLOR_GRAY2BGR)
        curr_frame_color = cv2.cvtColor(curr_frame_gray, cv2.COLOR_GRAY2BGR)

        # Create ORB detector
        orb = cv2.ORB_create(1000000)

        # Find keypoints and compute descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(prev_frame_color, None)
        keypoints2, descriptors2 = orb.detectAndCompute(curr_frame_color, None)

        # Match keypoints between frames
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)

        # Extract matched keypoints
        src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography
        homography, _ = find_homography(src_points, dst_points)

        homography_matrices.append(homography)

    return homography_matrices