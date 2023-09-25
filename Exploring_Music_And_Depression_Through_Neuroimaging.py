# %% [markdown]
# ### Team A Code
# 
# Amina Asghar <br>
# Nasri Binsaleh <br>
# Onintsoa Ramananandroniaina
# 
# Reference code: https://carpentries-incubator.github.io/SDC-BIDS-fMRI/aio/index.html

# %%
# laod required packages
import sys                          # for installing package
import os                           # for os related functions
import numpy as np                  # for dealing with numpy array
import pandas as pd                 # for dealing with dataframe
import matplotlib.pyplot as plt     # for plotting
%matplotlib inline
import nibabel as nib               # for dealing with fMRI data
from numpy.linalg import inv        # for inv()
%matplotlib inline
from scipy import stats             # for stats function e.g. to find p-value
import nilearn
from nilearn import plotting        # for plotting brain map
import math                         # for math function, checking NaN 
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# # Inspecting data

# %% [markdown]
# ## Warning
# 
# ##### Not all subjects/sessions/runs have the same scanning parameters.
# ##### LOCATION:/sub-control08/func/sub-control08_task-nonmusic_run-5_bold.nii.gz
# ##### REASON: The most common set of dimensions is: 80,80,50,105 (voxels), This file has the dimensions: 80,80,50,71 (voxels).
# 
# ##### LOCATION: /sub-mdd05/func/sub-mdd05_task-nonmusic_run-1_bold.nii.gz
# ##### REASON: The most common set of dimensions is: 80,80,50,105 (voxels), This file has the dimensions: 80,80,50,89 (voxels).

# %%
# importing the functional MRI data file
fmri_file = 'Dataset/dataset/sub-control01/func/sub-control01_task-music_run-1_bold.nii.gz'

# %%
# check the functional MRI data shape, voxel size, and units 
f_img = nib.load(fmri_file)
print(f_img.shape)
print(f_img.header.get_zooms())
print(f_img.header.get_xyzt_units())

# %%
# getting fMRI data
f_img_data = f_img.get_fdata()
print(f_img_data.shape)

# %%
# extract the time series of the middle voxel
mid_vox_ts = f_img_data[39, 39, 24, :]
print("Voxel timeseries shape: %s" % (mid_vox_ts.shape,))

# %%
# time-by-time fluctuations in signal intensity by 
# plotting the time series directly
plt.figure(figsize=(20, 5))
plt.plot(mid_vox_ts, 'o-', ms=4)
plt.xlim(-1, mid_vox_ts.size)
#plt.ylim(38000, 55000)
plt.ylabel('Signal intensity', fontsize=15)
plt.xlabel('Time (volumes)', fontsize=15)
plt.show()

# %% [markdown]
# ## Constructing GLM

# %%
# get signal for a voxel
voxel_signal = f_img_data[39, 39, 24, :]
# get the events
events_df = pd.read_csv('Dataset/dataset/sub-control01/func/sub-control01_task-music_run-1_events.tsv', 
                        sep='\t')
events_df

# %%
## record the response onset in arrays
response_onsets = events_df[events_df['trial_type'] == 
                            'negative_music']
response_onsets = response_onsets.append(events_df[events_df['trial_type'] == 
                                                   'positive_music'])
onsets = response_onsets['onset'].to_numpy()
onsets = onsets.astype(int)
onsets

# %%
## make a range of onsets
onset_range = []
for onset in onsets:
    onset_range = onset_range + list(range(onset,onset+32))

# %%
## convert the onset array to a proper predictor 
## (with the same shape as the fMRI signal)
# length of experiment is 105 volume with TR = 3s 
predictor = np.zeros(105*3)   
predictor[onset_range] = 1  # set the predictor at the indices to 1
print("Shape of predictor: %s" % (predictor.shape,))
print("\nContents of predictor array:\n%r" % predictor.T)

# %%
## visualizing the predictor
plt.figure(figsize=(25, 5))
#plt.plot(predictor_congruent, marker='o')
plt.plot(predictor, c='tab:blue')
plt.xlabel('Time points (seconds)', fontsize=20)
plt.ylabel('Activity (arbitrary units)', fontsize=20)
#plt.xlim(0, 300)
#plt.ylim(-.5, 1.5)
plt.title('Stimulus predictor', fontsize=25)
plt.grid()
plt.show()

# %%
## since the voxel in fMRI was measured in volume with TR of 3 seconds,
## we have to down sampling the predictor to fit the same unit of time.
from scipy.interpolate import interp1d

original_scale = np.arange(0, 105*3, 1) # from 0 to 105*3 seconds
print("Original scale has %i datapoints (0-315, in seconds)" % 
      original_scale.size)
resampler = interp1d(original_scale, predictor)
desired_scale = np.arange(0, 105*3, 3)
print("Desired scale has %i datapoints (0, 3, 6, ... 105, in volumes)" % 
      desired_scale.size)
predictor_ds = resampler(desired_scale)
print("Downsampled predictor has %i datapoints (in volumes)" % 
      predictor_ds.size)
print(predictor_ds)

# %%
## inspecting the predictor and the actual signal before 
## moving on to linear regression
plt.figure(figsize=(25, 10))
plt.plot(voxel_signal)
plt.plot(predictor_ds + voxel_signal.mean(), lw=3)
plt.xlabel('Time (in volumes)', fontsize=20)
plt.ylabel('Activity (A.U.)', fontsize=20)
plt.legend(['Voxel timeseries', 'Predictor'], fontsize=15, 
           loc='upper right', frameon=False)
plt.title("Signal and music stimulus predictor", fontsize=25)
plt.grid()
plt.show()

# %%
# Regression
if predictor_ds.ndim == 1:  # This adds a singleton dimension, 
                            # such that you can call np.hstack on it
    predictor_ds = predictor_ds[:, np.newaxis]

icept = np.ones((predictor_ds.size, 1))
X_simple = np.hstack((icept, predictor_ds))
betas_simple = inv(X_simple.T @ X_simple) @ X_simple.T @ voxel_signal
y_hat = X_simple[:, 0] * betas_simple[0] + X_simple[:, 1] * betas_simple[1]
print(betas_simple)
numerator = np.sum((voxel_signal - y_hat) ** 2) 
denominator = np.sum((voxel_signal - np.mean(voxel_signal)) ** 2)
r_squared = 1 - numerator / denominator
print('The R² value is: %.3f' % r_squared)

# %%
# function for plotting predicted signal vs. actual signal
def plot_signal_and_predicted_signal(y, X, x_lim=None, y_lim=None):
    """ Plots a signal and its GLM prediction. """
    des = np.hstack((np.ones((y.size, 1)), X))
    betas_simple = np.linalg.lstsq(des, y, rcond=None)[0]
    plt.figure(figsize=(15, 5))
    plt.plot(y)
    plt.plot(des @ betas_simple, lw=2)
    plt.xlabel('Time (in volumes)', fontsize=15)
    plt.ylabel('Activity (A.U.)', fontsize=15)

#     if x_lim is not None:
#         plt.xlim(x_lim)

#     if y_lim is not None:
#         plt.ylim(y_lim)

    plt.legend(['True signal', 'Predicted signal'], 
               loc='upper right', 
               fontsize=15)
    plt.title("Signal and predicted signal", fontsize=25)
    plt.grid()
    plt.show()


# %%
# plotting predicted signal vs. actual signal
plot_signal_and_predicted_signal(voxel_signal, predictor_ds)

# %%
# Now, model with BOLD response function
# install nilearn package
# import sys
# !{sys.executable} -m pip install nilearn

# use glover_hrf from nilearn
from nilearn.glm.first_level.hemodynamic_models import glover_hrf

# creating the canonical HFR 
TR = 3
osf = 3
length_hrf = 31 # sec # have to shape it according to our data 
                # (let's say to match the ITI of the experiment)
canonical_hrf = glover_hrf(tr=TR, oversampling=osf, 
                           time_length=length_hrf, 
                           onset=0)
canonical_hrf /= canonical_hrf.max()
print("Size of canonical hrf variable: %i" % canonical_hrf.size)

# %%
# visualizing the canonical HRF
t = np.arange(0, canonical_hrf.size)
plt.figure(figsize=(12, 5))
plt.plot(t,canonical_hrf)
plt.xlabel('Time (seconds) after stimulus onset', fontsize=15)
plt.ylabel('Activity (A.U.)', fontsize=15)
plt.title('Double gamma HRF', fontsize=25)
plt.grid()
plt.show()

# %%
# convolve the predictor with the HRF
predictor_conv = np.convolve(predictor.squeeze(), canonical_hrf)
print("The shape of the convolved predictor after convolution: %s" % 
      (predictor_conv.shape,))

# After convolution, we also neem to "trim" off 
# some excess values from the convolved signal 
predictor_conv = predictor_conv[:predictor.size]
print("After trimming, the shape is: %s" % 
      (predictor_conv.shape,))

# And we have to add a new axis again to go from shape (N,) to (N, 1), 
# which is important for stacking the intercept later
predictor_conv = predictor_conv[:, np.newaxis]
print("Shape after adding the new axis: %s" % 
      (predictor_conv.shape,))

# %%
# visualizing the predictor before and after the convolution
# congruent 
plt.figure(figsize=(25, 5))
plt.plot(predictor)
plt.plot(predictor_conv)
#plt.xlim(-1, 800)
plt.title("Predictor before and after convolution", fontsize=25)
plt.xlabel("Time (seconds)", fontsize=20)
plt.ylabel("Activity (A.U.)", fontsize=20)
plt.legend(['Before', 'After'], loc='upper right', fontsize=15)
plt.grid()
plt.show()

# %%
# scale the convolved predictor back to the scale of volume
original_scale = np.arange(0, 105*3, 1) # from 0 to 105*3 seconds
resampler = interp1d(original_scale, np.squeeze(predictor_conv))
desired_scale = np.arange(0, 105*3, 3)
predictor_conv_ds = resampler(desired_scale)
# x_lim, y_lim = (0, 400), (990, 1020)
plt.figure(figsize=(15, 5))
plt.plot(predictor_conv_ds + voxel_signal.mean())
plt.plot(voxel_signal)
plt.grid()
plt.title('Downsampled/convolved predictor + signal', fontsize=20)
plt.ylabel('Activity (A.U.)', fontsize=15)
plt.xlabel('Time (seconds)', fontsize=15)
plt.legend(['Predictor', 'Signal'])
# plt.xlim(x_lim)
plt.show()

# %%
# Now, fitting convolved predictor in GLM
if predictor_conv_ds.ndim == 1:
    # Add back a singleton axis (which was removed before downsampling)
    # otherwise stacking will give an error
    predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
    
intercept = np.ones((predictor_conv_ds.size, 1))
X_conv = np.hstack((intercept, predictor_conv_ds))
betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(voxel_signal)
plt.plot(X_conv @ betas_conv, lw=2)
# plt.xlim(x_lim)
plt.ylabel("Activity (A.U.)", fontsize=20)
plt.title("Model fit with convolved regressor", fontsize=25)
plt.legend(['True signal', 'Predicted signal'], fontsize=12, 
           loc='upper right')
plt.grid()

# %%
# Evaluate the model
from numpy.linalg import lstsq 
# numpy implementation of OLS, because we're lazy
y_hat_conv = X_conv @ betas_conv
y_hat_orig = X_simple @ lstsq(X_simple, voxel_signal, rcond=None)[0]
MSE_conv = ((y_hat_conv - voxel_signal) ** 2).mean()
MSE_orig = ((y_hat_orig - voxel_signal) ** 2).mean()
print("MSE of model with convolution is %.3f while the MSE of the model without convolution is %.3f." %
 (MSE_conv, MSE_orig))
R2_conv = 1 - (np.sum((voxel_signal - y_hat_conv) ** 2) / 
               np.sum((voxel_signal - voxel_signal.mean()) ** 2))
R2_orig = 1 - (np.sum((voxel_signal - y_hat_orig) ** 2) / 
               np.sum((voxel_signal - voxel_signal.mean()) ** 2))
print("R-squared of model with convolution is %.5f and without convolution it is %.5f." %
 (R2_conv, R2_orig))

# %% [markdown]
# # T-maps Music

# %% [markdown]
# ## Sub-control01 Run-1

# %%
# importing the functional MRI data file
fmri_file = 'Dataset/dataset/sub-control01/func/sub-control01_task-music_run-1_bold.nii.gz'
f_img = nib.load(fmri_file)
# getting fMRI data
f_img_data = f_img.get_fdata()

# %%
def design_variance(X, which_predictor=1): # X : Numpy Array of shape (N, P) 
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0

    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

# %%
# get a statistical map (a for loop to compute t-value of every voxel)
t_map = f_img_data[:, :, :, 0]  # to store t-values of each voxel (3D map)
for i in range(f_img_data.shape[0]):
    for j in range(f_img_data.shape[1]):
        for k in range(f_img_data.shape[2]):
            voxel_signal = f_img_data[i, j, k, :]
            # start the regression
            if predictor_conv_ds.ndim == 1:
                # Add back a singleton axis 
                #(which was removed before downsampling)
                # otherwise stacking will give an error
                predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
            # linear regression    
            intercept = np.ones((predictor_conv_ds.size, 1))
            X_conv = np.hstack((intercept, predictor_conv_ds))
            betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
            # get design variance
            design_variance_predictor = design_variance(X_conv, 
                                                        which_predictor=1)  
            # get degree of freedom and noise terms
            y_hat = X_conv @ betas_conv
            N = voxel_signal.size
            P = X_conv.shape[1]
            df = (N - P)  # degree of freedom
            sigma_hat = np.sum((voxel_signal - y_hat) ** 2) / df # noise
            #get t-value
            t = betas_conv[1] / np.sqrt(sigma_hat * 
                                        design_variance_predictor)
            if math.isnan(t):
                t = 0
            # store the t-value in t-map
            t_map[i, j, k] = round(t, 3)         

# %%
t_map

# %%
## convert t-map to NifTi image

# load the data
func = nib.load(fmri_file)

# to save this 3D (ndarry) numpy use this
ni_img = nib.Nifti1Image(t_map, func.affine)

# %%
# Get a cortical mesh
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

# %%
print(fsaverage['description'])

# %%
# Sample the 3D data around each node of the mesh
from nilearn import surface
texture = surface.vol_to_surf(ni_img, fsaverage.pial_right)

# %%
# plot the result 
from nilearn import plotting

# fig = plotting.plot_surf_stat_map(
#     fsaverage.infl_right, texture, hemi='right',
#     title='Surface right hemisphere', colorbar=True,
#     threshold=0,  engine='plotly', bg_map=fsaverage.sulc_right,
#     engine='plotly'
# )

fig = plotting.plot_surf_stat_map(fsaverage.pial_left, texture, 
                                  cmap = 'PRGn', engine='plotly', 
                                  threshold="90%", title='Left hemisphere')
fig.show()

# %%
fig = plotting.plot_surf_stat_map(fsaverage.pial_right, texture, 
                                  cmap = 'PRGn', engine='plotly', 
                                  threshold="90%", title='Right hemisphere')
fig.show()

# %% [markdown]
# ## Sub-control02 Run-3

# %%
# importing the functional MRI data file
fmri_file = 'Dataset/dataset/sub-control02/func/sub-control02_task-music_run-3_bold.nii.gz'

# getting fMRI data
f_img_data = f_img.get_fdata()

# %%
def design_variance(X, which_predictor=1): # X : Numpy Array of shape (N, P) 
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0

    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

# %%
# get a statistical map (a for loop to compute t-value of every voxel)
t_map = f_img_data[:, :, :, 0]  # to store t-values of each voxel (3D map)
for i in range(f_img_data.shape[0]):
    for j in range(f_img_data.shape[1]):
        for k in range(f_img_data.shape[2]):
            voxel_signal = f_img_data[i, j, k, :]
            # start the regression
            if predictor_conv_ds.ndim == 1:
                # Add back a singleton axis (which was removed before downsampling)
                # otherwise stacking will give an error
                predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
            # linear regression    
            intercept = np.ones((predictor_conv_ds.size, 1))
            X_conv = np.hstack((intercept, predictor_conv_ds))
            betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
            # get design variance
            design_variance_predictor = design_variance(X_conv, 
                                                        which_predictor=1)  
            # get degree of freedom and noise terms
            y_hat = X_conv @ betas_conv
            N = voxel_signal.size
            P = X_conv.shape[1]
            df = (N - P)  # degree of freedom
            sigma_hat = np.sum((voxel_signal - y_hat) ** 2) / df # noise
            #get t-value
            t = betas_conv[1] / np.sqrt(sigma_hat * design_variance_predictor)
            if math.isnan(t):
                t = 0
            # store the t-value in t-map
            t_map[i, j, k] = round(t, 3)         

# %%
## convert t-map to NifTi image

# load the data
func = nib.load(fmri_file)

# to save this 3D (ndarry) numpy use this
ni_img = nib.Nifti1Image(t_map, func.affine)

# %%
# Get a cortical mesh
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

# %%
# Sample the 3D data around each node of the mesh
from nilearn import surface
texture = surface.vol_to_surf(ni_img, fsaverage.pial_right)

# %%
# plot the result 
from nilearn import plotting

fig = plotting.plot_surf_stat_map(fsaverage.pial_left, texture, cmap = 'PRGn', engine='plotly')
fig.show()

# %%
fig = plotting.plot_surf_stat_map(fsaverage.pial_right, texture, cmap = 'PRGn', engine='plotly', threshold=0)
fig.show()

# %% [markdown]
# ## Sub-mdd01 Run-1

# %%
# importing the functional MRI data file
fmri_file = 'Dataset/dataset/sub-mdd01/func/sub-mdd01_task-music_run-1_bold.nii.gz'
f_img = nib.load(fmri_file)
# getting fMRI data
f_img_data = f_img.get_fdata()

# %%
def design_variance(X, which_predictor=1): # X : Numpy Array of shape (N, P) 
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0

    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

# %%
# get a statistical map (a for loop to compute t-value of every voxel)
t_map = f_img_data[:, :, :, 0]  # to store t-values of each voxel (3D map)
for i in range(f_img_data.shape[0]):
    for j in range(f_img_data.shape[1]):
        for k in range(f_img_data.shape[2]):
            voxel_signal = f_img_data[i, j, k, :]
            # start the regression
            if predictor_conv_ds.ndim == 1:
                # Add back a singleton axis (which was removed before downsampling)
                # otherwise stacking will give an error
                predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
            # linear regression    
            intercept = np.ones((predictor_conv_ds.size, 1))
            X_conv = np.hstack((intercept, predictor_conv_ds))
            betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
            design_variance_predictor = design_variance(X_conv, which_predictor=1)  # get design variance
            # get degree of freedom and noise terms
            y_hat = X_conv @ betas_conv
            N = voxel_signal.size
            P = X_conv.shape[1]
            df = (N - P)  # degree of freedom
            sigma_hat = np.sum((voxel_signal - y_hat) ** 2) / df # noise
            #get t-value
            t = betas_conv[1] / np.sqrt(sigma_hat * design_variance_predictor)
            if math.isnan(t):
                t = 0
            # store the t-value in t-map
            t_map[i, j, k] = round(t, 3)         

# %%
## convert t-map to NifTi image

# load the data
func = nib.load(fmri_file)

# to save this 3D (ndarry) numpy use this
ni_img = nib.Nifti1Image(t_map, func.affine)

# %%
# Get a cortical mesh
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

# %%
# Sample the 3D data around each node of the mesh
from nilearn import surface
texture = surface.vol_to_surf(ni_img, fsaverage.pial_right)

# %%
# plot the result 
from nilearn import plotting

fig = plotting.plot_surf_stat_map(fsaverage.pial_left, texture, cmap = 'PRGn', engine='plotly', threshold="90%", title='Left hemisphere')
fig.show()

# %%
fig = plotting.plot_surf_stat_map(fsaverage.pial_right, texture, cmap = 'PRGn', engine='plotly', threshold="90%", title='Right hemisphere')
fig.show()

# %% [markdown]
# ## Sub-mdd03 Run-1

# %%
# importing the functional MRI data file
fmri_file = 'Dataset/dataset/sub-mdd03/func/sub-mdd03_task-music_run-1_bold.nii.gz'

# getting fMRI data
f_img_data = f_img.get_fdata()

# %%
def design_variance(X, which_predictor=1): # X : Numpy Array of shape (N, P) 
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0

    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

# %%
# get a statistical map (a for loop to compute t-value of every voxel)
t_map = f_img_data[:, :, :, 0]  # to store t-values of each voxel (3D map)
for i in range(f_img_data.shape[0]):
    for j in range(f_img_data.shape[1]):
        for k in range(f_img_data.shape[2]):
            voxel_signal = f_img_data[i, j, k, :]
            # start the regression
            if predictor_conv_ds.ndim == 1:
                # Add back a singleton axis (which was removed before downsampling)
                # otherwise stacking will give an error
                predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
            # linear regression    
            intercept = np.ones((predictor_conv_ds.size, 1))
            X_conv = np.hstack((intercept, predictor_conv_ds))
            betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
            design_variance_predictor = design_variance(X_conv, which_predictor=1)  # get design variance
            # get degree of freedom and noise terms
            y_hat = X_conv @ betas_conv
            N = voxel_signal.size
            P = X_conv.shape[1]
            df = (N - P)  # degree of freedom
            sigma_hat = np.sum((voxel_signal - y_hat) ** 2) / df # noise
            #get t-value
            t = betas_conv[1] / np.sqrt(sigma_hat * design_variance_predictor)
            if math.isnan(t):
                t = 0
            # store the t-value in t-map
            t_map[i, j, k] = round(t, 3)         

# %%
## convert t-map to NifTi image

# load the data
func = nib.load(fmri_file)

# to save this 3D (ndarry) numpy use this
ni_img = nib.Nifti1Image(t_map, func.affine)

# %%
# Get a cortical mesh
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

# %%
# Sample the 3D data around each node of the mesh
from nilearn import surface
texture = surface.vol_to_surf(ni_img, fsaverage.pial_right)

# %%
# plot the result 
from nilearn import plotting

fig = plotting.plot_surf_stat_map(fsaverage.pial_left, texture, cmap = 'PRGn', engine='plotly')
fig.show()

# %%
fig = plotting.plot_surf_stat_map(fsaverage.pial_right, texture, cmap = 'PRGn', engine='plotly', threshold="95%", title='Right hemisphere')
fig.show()

# %% [markdown]
# ## Sub-control16 Run-1

# %%
# importing the functional MRI data file
fmri_file = 'Dataset/dataset/sub-control16/func/sub-control16_task-music_run-3_bold.nii.gz'

# getting fMRI data
f_img_data = f_img.get_fdata()

# %%
def design_variance(X, which_predictor=1): # X : Numpy Array of shape (N, P) 
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0

    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

# %%
# get a statistical map (a for loop to compute t-value of every voxel)
t_map = f_img_data[:, :, :, 0]  # to store t-values of each voxel (3D map)
for i in range(f_img_data.shape[0]):
    for j in range(f_img_data.shape[1]):
        for k in range(f_img_data.shape[2]):
            voxel_signal = f_img_data[i, j, k, :]
            # start the regression
            if predictor_conv_ds.ndim == 1:
                # Add back a singleton axis (which was removed before downsampling)
                # otherwise stacking will give an error
                predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
            # linear regression    
            intercept = np.ones((predictor_conv_ds.size, 1))
            X_conv = np.hstack((intercept, predictor_conv_ds))
            betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
            design_variance_predictor = design_variance(X_conv, which_predictor=1)  # get design variance
            # get degree of freedom and noise terms
            y_hat = X_conv @ betas_conv
            N = voxel_signal.size
            P = X_conv.shape[1]
            df = (N - P)  # degree of freedom
            sigma_hat = np.sum((voxel_signal - y_hat) ** 2) / df # noise
            #get t-value
            t = betas_conv[1] / np.sqrt(sigma_hat * design_variance_predictor)
            if math.isnan(t):
                t = 0
            # store the t-value in t-map
            t_map[i, j, k] = round(t, 3)         

# %%
## convert t-map to NifTi image

# load the data
func = nib.load(fmri_file)

# to save this 3D (ndarry) numpy use this
ni_img = nib.Nifti1Image(t_map, func.affine)

# %%
# Get a cortical mesh
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

# %%
# Sample the 3D data around each node of the mesh
from nilearn import surface
texture = surface.vol_to_surf(ni_img, fsaverage.pial_right)

# %%
# plot the result 
from nilearn import plotting

fig = plotting.plot_surf_stat_map(fsaverage.pial_left, texture, cmap = 'PRGn', engine='plotly')
fig.show()

# %%
fig = plotting.plot_surf_stat_map(fsaverage.pial_right, texture, cmap = 'PRGn', engine='plotly', threshold=0)
fig.show()

# %% [markdown]
# ## Sub-mdd14 Run-1

# %%
# importing the functional MRI data file
fmri_file = 'Dataset/dataset/sub-mdd14/func/sub-mdd14_task-music_run-3_bold.nii.gz'

# getting fMRI data
f_img_data = f_img.get_fdata()

# %%
def design_variance(X, which_predictor=1): # X : Numpy Array of shape (N, P) 
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0

    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

# %%
# get a statistical map (a for loop to compute t-value of every voxel)
t_map = f_img_data[:, :, :, 0]  # to store t-values of each voxel (3D map)
for i in range(f_img_data.shape[0]):
    for j in range(f_img_data.shape[1]):
        for k in range(f_img_data.shape[2]):
            voxel_signal = f_img_data[i, j, k, :]
            # start the regression
            if predictor_conv_ds.ndim == 1:
                # Add back a singleton axis (which was removed before downsampling)
                # otherwise stacking will give an error
                predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
            # linear regression    
            intercept = np.ones((predictor_conv_ds.size, 1))
            X_conv = np.hstack((intercept, predictor_conv_ds))
            betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
            design_variance_predictor = design_variance(X_conv, which_predictor=1)  # get design variance
            # get degree of freedom and noise terms
            y_hat = X_conv @ betas_conv
            N = voxel_signal.size
            P = X_conv.shape[1]
            df = (N - P)  # degree of freedom
            sigma_hat = np.sum((voxel_signal - y_hat) ** 2) / df # noise
            #get t-value
            t = betas_conv[1] / np.sqrt(sigma_hat * design_variance_predictor)
            if math.isnan(t):
                t = 0
            # store the t-value in t-map
            t_map[i, j, k] = round(t, 3)         

# %%
## convert t-map to NifTi image

# load the data
func = nib.load(fmri_file)

# to save this 3D (ndarry) numpy use this
ni_img = nib.Nifti1Image(t_map, func.affine)

# %%
# Get a cortical mesh
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

# %%
# Sample the 3D data around each node of the mesh
from nilearn import surface
texture = surface.vol_to_surf(ni_img, fsaverage.pial_right)

# %%
# plot the result 
from nilearn import plotting

fig = plotting.plot_surf_stat_map(fsaverage.pial_left, texture, cmap = 'PRGn', engine='plotly', threshold="95%", title='Left hemisphere')
fig.show()

# %%
fig = plotting.plot_surf_stat_map(fsaverage.pial_right, texture, cmap = 'PRGn', engine='plotly', threshold="95%", title='Right hemisphere')
fig.show()

# %% [markdown]
# # T-maps Non-Music

# %% [markdown]
# ## Sub-control01

# %%
# importing the functional MRI data file
fmri_file = 'Dataset/dataset/sub-control01/func/sub-control01_task-nonmusic_run-4_bold.nii.gz'
f_img = nib.load(fmri_file)
# getting fMRI data
f_img_data = f_img.get_fdata()

# %%
def design_variance(X, which_predictor=1): # X : Numpy Array of shape (N, P) 
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0

    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

# %%
# get a statistical map (a for loop to compute t-value of every voxel)
t_map = f_img_data[:, :, :, 0]  # to store t-values of each voxel (3D map)
for i in range(f_img_data.shape[0]):
    for j in range(f_img_data.shape[1]):
        for k in range(f_img_data.shape[2]):
            voxel_signal = f_img_data[i, j, k, :]
            # start the regression
            if predictor_conv_ds.ndim == 1:
                # Add back a singleton axis (which was removed before downsampling)
                # otherwise stacking will give an error
                predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
            # linear regression    
            intercept = np.ones((predictor_conv_ds.size, 1))
            X_conv = np.hstack((intercept, predictor_conv_ds))
            betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
            design_variance_predictor = design_variance(X_conv, which_predictor=1)  # get design variance
            # get degree of freedom and noise terms
            y_hat = X_conv @ betas_conv
            N = voxel_signal.size
            P = X_conv.shape[1]
            df = (N - P)  # degree of freedom
            sigma_hat = np.sum((voxel_signal - y_hat) ** 2) / df # noise
            #get t-value
            t = betas_conv[1] / np.sqrt(sigma_hat * design_variance_predictor)
            if math.isnan(t):
                t = 0
            # store the t-value in t-map
            t_map[i, j, k] = round(t, 3)         

# %%
## convert t-map to NifTi image

# load the data
func = nib.load(fmri_file)

# to save this 3D (ndarry) numpy use this
ni_img = nib.Nifti1Image(t_map, func.affine)

# %%
# Get a cortical mesh
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

# %%
# Sample the 3D data around each node of the mesh
from nilearn import surface
texture = surface.vol_to_surf(ni_img, fsaverage.pial_right)

# %%
# plot the result 
from nilearn import plotting

# fig = plotting.plot_surf_stat_map(
#     fsaverage.infl_right, texture, hemi='right',
#     title='Surface right hemisphere', colorbar=True,
#     threshold=0, bg_map=fsaverage.sulc_right,
#     engine='plotly'
# )

fig = plotting.plot_surf_stat_map(fsaverage.pial_left, texture, 
                                  cmap = 'PRGn', threshold="90%", 
                                  engine='plotly', title='Left hemisphere')
fig.show()

# %%
fig = plotting.plot_surf_stat_map(fsaverage.pial_right, texture, 
                                  cmap = 'PRGn', threshold="90%", 
                                  engine='plotly', title='Right hemisphere')
fig.show()

# %% [markdown]
# ## Sub-mdd01

# %%
# importing the functional MRI data file
fmri_file = 'Dataset/dataset/sub-mdd01/func/sub-mdd01_task-nonmusic_run-4_bold.nii.gz'
f_img = nib.load(fmri_file)
# getting fMRI data
f_img_data = f_img.get_fdata()

# %%
def design_variance(X, which_predictor=1): # X : Numpy Array of shape (N, P) 
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0

    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

# %%
# get a statistical map (a for loop to compute t-value of every voxel)
t_map = f_img_data[:, :, :, 0]  # to store t-values of each voxel (3D map)
for i in range(f_img_data.shape[0]):
    for j in range(f_img_data.shape[1]):
        for k in range(f_img_data.shape[2]):
            voxel_signal = f_img_data[i, j, k, :]
            # start the regression
            if predictor_conv_ds.ndim == 1:
                # Add back a singleton axis (which was removed before downsampling)
                # otherwise stacking will give an error
                predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
            # linear regression    
            intercept = np.ones((predictor_conv_ds.size, 1))
            X_conv = np.hstack((intercept, predictor_conv_ds))
            betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
            design_variance_predictor = design_variance(X_conv, which_predictor=1)  # get design variance
            # get degree of freedom and noise terms
            y_hat = X_conv @ betas_conv
            N = voxel_signal.size
            P = X_conv.shape[1]
            df = (N - P)  # degree of freedom
            sigma_hat = np.sum((voxel_signal - y_hat) ** 2) / df # noise
            #get t-value
            t = betas_conv[1] / np.sqrt(sigma_hat * design_variance_predictor)
            if math.isnan(t):
                t = 0
            # store the t-value in t-map
            t_map[i, j, k] = round(t, 3)         

# %%
## convert t-map to NifTi image

# load the data
func = nib.load(fmri_file)

# to save this 3D (ndarry) numpy use this
ni_img = nib.Nifti1Image(t_map, func.affine)

# %%
# Get a cortical mesh
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

# %%
# Sample the 3D data around each node of the mesh
from nilearn import surface
texture = surface.vol_to_surf(ni_img, fsaverage.pial_right)

# %%
# plot the result 
from nilearn import plotting

# fig = plotting.plot_surf_stat_map(
#     fsaverage.infl_right, texture, hemi='right',
#     title='Surface right hemisphere', colorbar=True,
#     threshold=0, bg_map=fsaverage.sulc_right,
#     engine='plotly'
# )

fig = plotting.plot_surf_stat_map(fsaverage.pial_left, texture, cmap = 'PRGn', engine='plotly', threshold="90%", title='Left hemisphere')
fig.show()

# %%
fig = plotting.plot_surf_stat_map(fsaverage.pial_right, texture, cmap = 'PRGn', engine='plotly', threshold="90%", title='Right hemisphere')
fig.show()

# %%


# %% [markdown]
# # Improved GLM

# %%
# importing the functional MRI data file
fmri_file = 'Dataset\\dataset\\derivatives\\sub-control01\\func\\sub-control01_task-music_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

# %%
# check the functional MRI data shape, voxel size, and units 
f_img = nib.load(fmri_file)
print(f_img.shape)
print(f_img.header.get_zooms())
print(f_img.header.get_xyzt_units())

# %%
# getting fMRI data
f_img_data = f_img.get_fdata()
print(f_img_data.shape)

# %%
# extract the time series of the middle voxel
mid_vox_ts = f_img_data[39, 39, 24, :]
print("Voxel timeseries shape: %s" % (mid_vox_ts.shape,))

# %%
# time-by-time fluctuations in signal intensity by 
# plotting the time series directly
plt.figure(figsize=(20, 5))
plt.plot(mid_vox_ts, 'o-', ms=4)
plt.xlim(-1, mid_vox_ts.size)
#plt.ylim(38000, 55000)
plt.ylabel('Signal intensity', fontsize=15)
plt.xlabel('Time (volumes)', fontsize=15)
plt.show()

# %%
# get the events
events_df = pd.read_csv('Dataset/dataset/sub-control01/func/sub-control01_task-music_run-1_events.tsv', sep='\t')
events_df

# %%
# get signal for a voxel
voxel_signal = f_img_data[39, 39, 24, :]
plt.figure(figsize=(20, 5))
plt.plot(voxel_signal)
plt.xlabel('Time points (volumes)', fontsize=20)
plt.ylabel('Activity (arbitrary units)', fontsize=20)
plt.title('Voxel signal', fontsize=25)
plt.grid()
plt.show()

# %%
# Deal with mean-shift (transform time-series data from non-stationary to stationary)
temp_df = pd.DataFrame(voxel_signal)
# diff() function take the difference of the data point and its previous point
voxel_signal_stationary = temp_df[0].diff()  

# the first data point is recorded to 'nan' after transformation 
# (because it does have anything before it to subtract with)
# impute that first value with the mean
voxel_signal_stationary = voxel_signal_stationary.fillna(voxel_signal_stationary.mean())
temp_voxel_signal = voxel_signal_stationary.to_numpy()

# %%
# try plotting the transformed
voxel_signal = temp_voxel_signal
plt.figure(figsize=(25, 5))
plt.plot(voxel_signal)
plt.xlabel('Time points (volumes)', fontsize=20)
plt.ylabel('Activity (arbitrary units)', fontsize=20)
# plt.xlim(x_lim)
# plt.ylim(y_lim)
plt.title('Stationary Voxel signal', fontsize=25)
plt.grid()
plt.show()

# %%
## record the response onset in arrays
response_onsets = events_df[events_df['trial_type'] == 'negative_music']
response_onsets = response_onsets.append(events_df[events_df['trial_type'] == 
                                                   'positive_music'])
onsets = response_onsets['onset'].to_numpy()
onsets = onsets.astype(int)
onsets

# %%
## make a range of onsets
onset_range = []
for onset in onsets:
    onset_point = onset
    for i in range(3):
        onset_range = onset_range + list(range(onset_point,onset_point+9))
        onset_point = onset_point + 10

# %%
## convert the onset array to a proper predictor 
# (with the same shape as the fMRI signal)
# for congruent condition
predictor = np.zeros(105*3)   # length of experiment is 105 volume with TR = 3s 
predictor[onset_range] = 1  # set the predictor at the indices to 1
print("Shape of predictor: %s" % (predictor.shape,))
print("\nContents of predictor array:\n%r" % predictor.T)

# %%
## visualizing the predictor
plt.figure(figsize=(25, 5))
#plt.plot(predictor_congruent, marker='o')
plt.plot(predictor, c='tab:blue')
plt.xlabel('Time points (seconds)', fontsize=20)
plt.ylabel('Activity (arbitrary units)', fontsize=20)
#plt.xlim(0, 300)
#plt.ylim(-.5, 1.5)
plt.title('Stimulus predictor', fontsize=25)
plt.grid()
plt.show()

# %%
## since the voxel in fMRI was measured in volume with TR of 3 seconds, 
## we have to down sampling the predictor to fit the same unit of time.
from scipy.interpolate import interp1d

original_scale = np.arange(0, 105*3, 1) # from 0 to 105*3 seconds
print("Original scale has %i datapoints (0-315, in seconds)" % 
      original_scale.size)
resampler = interp1d(original_scale, predictor)
desired_scale = np.arange(0, 105*3, 3)
print("Desired scale has %i datapoints (0, 3, 6, ... 105, in volumes)" % 
      desired_scale.size)
predictor_ds = resampler(desired_scale)
print("Downsampled predictor has %i datapoints (in volumes)" % 
      predictor_ds.size)
print(predictor_ds)

# %%
## inspecting the predictor and the actual signal before moving on to linear regression
plt.figure(figsize=(25, 10))
plt.plot(voxel_signal)
plt.plot(predictor_ds + voxel_signal.mean(), lw=3)
plt.xlabel('Time (in volumes)', fontsize=20)
plt.ylabel('Activity (A.U.)', fontsize=20)
plt.legend(['Voxel timeseries', 'Predictor'], fontsize=15, 
           loc='upper right', frameon=False)
plt.title("Signal and music stimulus predictor", fontsize=25)
plt.grid()
plt.show()

# %%
# Regression
if predictor_ds.ndim == 1: # This adds a singleton dimension, such that you can call np.hstack on it
    predictor_ds = predictor_ds[:, np.newaxis]

icept = np.ones((predictor_ds.size, 1))
X_simple = np.hstack((icept, predictor_ds))
betas_simple = inv(X_simple.T @ X_simple) @ X_simple.T @ voxel_signal
y_hat = X_simple[:, 0] * betas_simple[0] + X_simple[:, 1] * betas_simple[1]
print(betas_simple)
numerator = np.sum((voxel_signal - y_hat) ** 2) 
denominator = np.sum((voxel_signal - np.mean(voxel_signal)) ** 2)
r_squared = 1 - numerator / denominator
print('The R² value is: %.3f' % r_squared)

# %%
# function for plotting predicted signal vs. actual signal
def plot_signal_and_predicted_signal(y, X, x_lim=None, y_lim=None):
    """ Plots a signal and its GLM prediction. """
    des = np.hstack((np.ones((y.size, 1)), X))
    betas_simple = np.linalg.lstsq(des, y, rcond=None)[0]
    plt.figure(figsize=(15, 5))
    plt.plot(y)
    plt.plot(des @ betas_simple, lw=2)
    plt.xlabel('Time (in volumes)', fontsize=15)
    plt.ylabel('Activity (A.U.)', fontsize=15)

    plt.legend(['True signal', 'Predicted signal'], 
               loc='upper right', fontsize=15)
    plt.title("Signal and predicted signal", fontsize=25)
    plt.grid()
    plt.show()


# %%
# plotting predicted signal vs. actual signal
plot_signal_and_predicted_signal(voxel_signal, predictor_ds)

# %%
# Now, model with BOLD response function
# install nilearn package
# import sys
# !{sys.executable} -m pip install nilearn

# use glover_hrf from nilearn
from nilearn.glm.first_level.hemodynamic_models import glover_hrf

# creating the canonical HFR 
TR = 3
osf = 3
length_hrf = 10 # sec # have to shape it according to our data 
# (let's say to match the ITI of the experiment)
canonical_hrf = glover_hrf(tr=TR, oversampling=osf, 
                           time_length=length_hrf, onset=0)
canonical_hrf /= canonical_hrf.max()
print("Size of canonical hrf variable: %i" % canonical_hrf.size)

# %%
# visualizing the canonical HRF
t = np.arange(0, canonical_hrf.size)
plt.figure(figsize=(12, 5))
plt.plot(t,canonical_hrf)
plt.xlabel('Time (seconds) after stimulus onset', fontsize=15)
plt.ylabel('Activity (A.U.)', fontsize=15)
plt.title('Double gamma HRF', fontsize=25)
plt.grid()
plt.show()

# %%
# convolve the predictor with the HRF
predictor_conv = np.convolve(predictor.squeeze(), canonical_hrf)
print("The shape of the convolved predictor after convolution: %s" % (predictor_conv.shape,))

# After convolution, we also neem to "trim" off some excess 
# values from the convolved signal 
predictor_conv = predictor_conv[:predictor.size]
print("After trimming, the shape is: %s" % (predictor_conv.shape,))

# And we have to add a new axis again to go from shape (N,) to (N, 1), 
# which is important for stacking the intercept later
predictor_conv = predictor_conv[:, np.newaxis]
print("Shape after adding the new axis: %s" % (predictor_conv.shape,))

# %%
# visualizing the predictor before and after the convolution
# congruent 
plt.figure(figsize=(25, 5))
plt.plot(predictor)
plt.plot(predictor_conv)
#plt.xlim(-1, 800)
plt.title("Predictor before and after convolution", fontsize=25)
plt.xlabel("Time (seconds)", fontsize=20)
plt.ylabel("Activity (A.U.)", fontsize=20)
plt.legend(['Before', 'After'], loc='upper right', fontsize=15)
plt.grid()
plt.show()

# %%
# scale the convolved predictor back to the scale of volume
original_scale = np.arange(0, 105*3, 1) # from 0 to 105*3 seconds
resampler = interp1d(original_scale, np.squeeze(predictor_conv))
desired_scale = np.arange(0, 105*3, 3)
predictor_conv_ds = resampler(desired_scale)
# x_lim, y_lim = (0, 400), (990, 1020)
plt.figure(figsize=(15, 5))
plt.plot(predictor_conv_ds + voxel_signal.mean())
plt.plot(voxel_signal)
plt.grid()
plt.title('Downsampled/convolved predictor + signal', fontsize=20)
plt.ylabel('Activity (A.U.)', fontsize=15)
plt.xlabel('Time (seconds)', fontsize=15)
plt.legend(['Predictor', 'Signal'])
# plt.xlim(x_lim)
plt.show()

# %%
# Now, fitting convolved predictor in GLM
if predictor_conv_ds.ndim == 1:
    # Add back a singleton axis (which was removed before downsampling)
    # otherwise stacking will give an error
    predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
    
intercept = np.ones((predictor_conv_ds.size, 1))
X_conv = np.hstack((intercept, predictor_conv_ds))
betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(voxel_signal)
plt.plot(X_conv @ betas_conv, lw=2)
# plt.xlim(x_lim)
plt.ylabel("Activity (A.U.)", fontsize=20)
plt.title("Model fit with convolved regressor", fontsize=25)
plt.legend(['True signal', 'Predicted signal'], fontsize=12, 
           loc='upper right')
plt.grid()

# %%
# Evaluate the model
from numpy.linalg import lstsq # numpy implementation of OLS, because we're lazy
y_hat_conv = X_conv @ betas_conv
y_hat_orig = X_simple @ lstsq(X_simple, voxel_signal, rcond=None)[0]
MSE_conv = ((y_hat_conv - voxel_signal) ** 2).mean()
MSE_orig = ((y_hat_orig - voxel_signal) ** 2).mean()
print("MSE of model with convolution is %.3f while the MSE of the model without convolution is %.3f." %
 (MSE_conv, MSE_orig))
R2_conv = 1 - (np.sum((voxel_signal - y_hat_conv) ** 2) / np.sum((voxel_signal - voxel_signal.mean()) ** 2))
R2_orig = 1 - (np.sum((voxel_signal - y_hat_orig) ** 2) / np.sum((voxel_signal - voxel_signal.mean()) ** 2))
print("R-squared of model with convolution is %.5f and without convolution it is %.5f." %
 (R2_conv, R2_orig))

# %% [markdown]
# #### As can be seen here, the R2 is much better than the previous GLM

# %% [markdown]
# ### fmriprep preproc Data

# %%
# importing the functional MRI data file
fmri_file = 'Dataset\\dataset\\derivatives\\sub-control01\\func\\sub-control01_task-music_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
f_img = nib.load(fmri_file)
# getting fMRI data
f_img_data = f_img.get_fdata()

# %%
def design_variance(X, which_predictor=1): # X : Numpy Array of shape (N, P) 
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0

    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

# %%
# get a statistical map (a for loop to compute t-value of every voxel)
t_map = f_img_data[:, :, :, 0]  # to store t-values of each voxel (3D map)
for i in range(f_img_data.shape[0]):
    for j in range(f_img_data.shape[1]):
        for k in range(f_img_data.shape[2]):
            voxel_signal = f_img_data[i, j, k, :]
            # start the regression
            if predictor_conv_ds.ndim == 1:
                # Add back a singleton axis (which was removed before downsampling)
                # otherwise stacking will give an error
                predictor_conv_ds = predictor_conv_ds[:, np.newaxis]
            # linear regression    
            intercept = np.ones((predictor_conv_ds.size, 1))
            X_conv = np.hstack((intercept, predictor_conv_ds))
            betas_conv = inv(X_conv.T @ X_conv) @ X_conv.T @ voxel_signal
            design_variance_predictor = design_variance(X_conv, which_predictor=1)  # get design variance
            # get degree of freedom and noise terms
            y_hat = X_conv @ betas_conv
            N = voxel_signal.size
            P = X_conv.shape[1]
            df = (N - P)  # degree of freedom
            sigma_hat = np.sum((voxel_signal - y_hat) ** 2) / df # noise
            #get t-value
            t = betas_conv[1] / np.sqrt(sigma_hat * design_variance_predictor)
            if math.isnan(t):
                t = 0
            # store the t-value in t-map
            t_map[i, j, k] = round(t, 3)         

# %%
t_map

# %%
## convert t-map to NifTi image

# load the data
func = nib.load(fmri_file)

# to save this 3D (ndarry) numpy use this
ni_img = nib.Nifti1Image(t_map, func.affine)

# %%
from nilearn import plotting

plotting.plot_glass_brain(ni_img, threshold=0.5)

# %%
# Get a cortical mesh
from nilearn import datasets
fsaverage = datasets.fetch_surf_fsaverage()

# %%
# Sample the 3D data around each node of the mesh
from nilearn import surface
texture = surface.vol_to_surf(ni_img, fsaverage.pial_right)

# %%
# plot the result 
from nilearn import plotting

fig = plotting.plot_surf_stat_map(fsaverage.pial_left, texture, 
                                  cmap = 'PRGn', threshold="90%", 
                                  engine='plotly',  title='Left hemisphere')
fig.show()

# %%
fig = plotting.plot_surf_stat_map(fsaverage.pial_right, texture, 
                                  cmap = 'PRGn', threshold="90%", 
                                  engine='plotly',  title='Right hemisphere')
fig.show()

# %%


# %% [markdown]
# # Functional Connectivity Analysis

# %%
# install pybids module
# !{sys.executable} -m pip install pybids
import bids

# %%
layout = bids.BIDSLayout('Dataset\dataset\derivatives', 
                         config=['bids','derivatives'])

# %%
layout.get_subjects()

# %%
layout.get_tasks()

# %% [markdown]
# # Cleaning Confounders

# %%
from nilearn import image as nimg
from nilearn import plotting as nplot

# %%
# Setting up Motion Estimates
sub = 'control01'
fmriprep_dir = 'Dataset\dataset\derivatives'
layout = bids.BIDSLayout(fmriprep_dir,validate=False,
                        config=['bids','derivatives'])

# %%
func_files = layout.get(subject=sub,
                        datatype='func', task='music',
                        desc='preproc',
                        space='MNI152NLin2009cAsym',
                        extension='nii.gz',                       
                        return_type='file')

mask_files = layout.get(subject=sub,
                        datatype='func', task='music',
                        desc='brain',
                        suffix='mask',
                        space='MNI152NLin2009cAsym',
                        extension="nii.gz",
                        return_type='file')

confound_files = layout.get(subject=sub,
                            datatype='func', task='music',
                            desc='confounds',
                            extension="tsv",
                            return_type='file')

# %%
func_file = func_files[0]
mask_file = mask_files[0]
confound_file = confound_files[0]

# %%
# read in the confounds.tsv file
confound_df = pd.read_csv(confound_file, delimiter='\t')
confound_df.head()

# %%
*The Yeo 2011 Pre-processing schema*
Confound regressors
6 motion parameters (trans_x, trans_y, trans_z, rot_x, rot_y, rot_z)
Global signal (global_signal)
Cerebral spinal fluid signal (csf)
White matter signal (white_matter)
This is a total of 9 base confound regressor variables. Finally we add temporal derivatives of each of these signals as well (1 temporal derivative for each), the result is 18 confound regressors.

# %%
# Setting up Confound variables for regression
# Computing temporal derivatives for confound variables
# Select confounds
confound_vars = ['trans_x','trans_y','trans_z', 'rot_x','rot_y','rot_z',
                 'global_signal', 'csf', 'white_matter']

# %%
# pick the derivatives for our confound_vars
# Get derivative column names
derivative_columns = ['{}_derivative1'.format(c) for c
                     in confound_vars]

print(derivative_columns)

# %%
# join two lists together
final_confounds = confound_vars + derivative_columns
print(final_confounds)

# %%
confound_df = confound_df[final_confounds]
confound_df.head()

# %%
# Dummy TR Drop
# load in our data and check the shape
raw_func_img = nimg.load_img(func_file)
raw_func_img.shape

# %%
# drop first 4 timepoints
func_img = raw_func_img.slicer[:,:,:,4:]
func_img.shape

# %%
#Drop confound dummy TRs
drop_confound_df = confound_df.loc[4:]
print(drop_confound_df.shape) #number of rows should match that of the functional image
# drop_confound_df.head()

# %%
# Applying confound regression -- clean our data of our selected confound variables
confounds_matrix = drop_confound_df.values

#Confirm matrix size is correct
confounds_matrix.shape

# %%
# Set some constants
high_pass= 0.009
low_pass = 0.08
t_r = 3

# Clean data
clean_img = nimg.clean_img(func_img, confounds=confounds_matrix, 
                           detrend=True, standardize=True,
                           low_pass=low_pass, high_pass=high_pass, t_r=t_r)

# %% [markdown]
# # Applying Parcellations to Data

# %%
# Retrieving the Atlas
# using a set of parcellation from Yeo et al. 2011. 
from nilearn import datasets
parcel_dir = 'resources/rois/'
atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011(parcel_dir)

# %%
atlas_yeo_2011.keys()

# %%
# checking Yeo Atlas
#Define where to slice the image
cut_coords = (8, -4, 9)
#Show a colorbar
colorbar=True
#Color scheme to show when viewing image
cmap='Paired'

#Plot all parcellation schemas referred to by atlas_yeo_2011
nplot.plot_roi(atlas_yeo_2011['thin_7'], cut_coords=cut_coords, 
               colorbar=colorbar, cmap=cmap, title='thin_7')
nplot.plot_roi(atlas_yeo_2011['thin_17'], cut_coords=cut_coords, 
               colorbar=colorbar, cmap=cmap, title='thin_17')
nplot.plot_roi(atlas_yeo_2011['thick_7'], cut_coords=cut_coords, 
               colorbar=colorbar, cmap=cmap, title='thick_7')
nplot.plot_roi(atlas_yeo_2011['thick_17'], cut_coords=cut_coords, 
               colorbar=colorbar, cmap=cmap, title='thick_17')

# %%
thick_7 variation which includes the following networks:
- Visual
- Somatosensory
- Dorsal Attention
- Ventral Attention
- Limbic
- Frontoparietal
- Default

The parcel areas labelled with 0 are background voxels not 
associated with a particular network

# %%
# get thick_7 atlas
atlas_yeo = atlas_yeo_2011['thick_7']

# %%
# Spatial Separation of Network
from nilearn.regions import connected_label_regions
region_labels = connected_label_regions(atlas_yeo)
nplot.plot_roi(region_labels, cut_coords=(-20,-10,0,10,20,30,40,50,60,70),
               display_mode='z',colorbar=True,cmap='Paired',
               title='Relabeled Yeo Atlas')

# %%
# Resampling the Atlas
# store the separated version of the atlas into a NIFTI file to work with it later
region_labels.to_filename('resources/rois/yeo_2011/Yeo_JNeurophysiol11_MNI152/relabeled_yeo_atlas.nii.gz')

# %%
func_img = nib.load(func_file)

# %%
# Print dimensions of functional image and atlas image

print("Size of functional image:", func_img.shape)
print("Size of atlas image:", region_labels.shape)

resampled_yeo = nimg.resample_to_img(region_labels, func_img, 
                                     interpolation = 'nearest')

# %%
# what the resampled atlas looks like overlayed on a slice of our NifTI file
# Note that we're pulling a random timepoint from the fMRI data
nplot.plot_roi(resampled_yeo, func_img.slicer[:, :, :, 54])

# %%
for i in range(50):
    a = "a == " + str(i)
    # print("ROI: " + a)
    # Make a mask for ROI
    roi_mask = nimg.math_img(a, a=resampled_yeo)  
    # Visualize ROI
    nplot.plot_roi(roi_mask, title = a)

# %% [markdown]
# # Functional Connectivity Analysis

# %%
# How can we estimate brain functional connectivity 
# patterns from data?
from nilearn import image as nimg
from nilearn import plotting as nplot
import numpy as np
import pandas as pd
from bids import BIDSLayout

# %%
# Use PyBIDS to parse BIDS data structure
layout = BIDSLayout(fmriprep_dir, config=['bids','derivatives'])

# %%
# Get musical data (preprocessed, mask, and confounds file)
func_files = layout.get(subject=sub,
                        datatype='func', task='music',
                        desc='preproc',
                        space='MNI152NLin2009cAsym',
                        extension='nii.gz',
                        return_type='file')

mask_files = layout.get(subject=sub,
                        datatype='func', task='music',
                        desc='brain',
                        suffix="mask",
                        space='MNI152NLin2009cAsym',
                        extension='nii.gz',
                        return_type='file')

confound_files = layout.get(subject=sub,
                            datatype='func',
                            task='music',
                            desc='confounds',
                            extension='tsv',
                            return_type='file')

# %%
#Load separated parcellation
parcel_file = 'resources/rois/yeo_2011/Yeo_JNeurophysiol11_MNI152/relabeled_yeo_atlas.nii.gz'
yeo_7 = nimg.load_img(parcel_file)

# %%
# import a package from nilearn, called input_data which 
# allows us to pull data using the parcellation file,
# and at the same time applying data cleaning
from nilearn import input_data
masker = input_data.NiftiLabelsMasker(labels_img=yeo_7,
                                      standardize=True,
                                      memory='nilearn_cache',
                                      verbose=1,
                                      detrend=True,
                                      low_pass = 0.08,
                                      high_pass = 0.009,
                                      t_r=3)

# %%
# Pull the first subject's data
func_file = func_files[0]
mask_file = mask_files[0]
confound_file = confound_files[0]

# %%
# Make confounds matrix
def extract_confounds(confound_tsv, confounds, dt=True):
    '''
    Arguments:
        confound_tsv    Full path to confounds.tsv
        confounds       A list of confounder variables to extract
        dt              Compute temporal derivatives [default = True]
        
    Outputs:
        confound_mat                    
    '''
    
    if dt:    
        dt_names = ['{}_derivative1'.format(c) for c in confounds]
        confounds = confounds + dt_names
    
    #Load in data using Pandas then extract relevant columns
    confound_df = pd.read_csv(confound_tsv,delimiter='\t') 
    confound_df = confound_df[confounds]
    
 
    #Convert into a matrix of values (timepoints)x(variable)
    confound_mat = confound_df.values 
    
    #Return confound matrix
    return confound_mat

# %%
# Drop Dummy TRs that are to be excluded from our cleaning, 
# parcellation, and averaging step
# Load functional image
tr_drop = 4
func_img = nimg.load_img(func_file)

# Remove the first 4 TRs
func_img = func_img.slicer[:,:,:,tr_drop:]

# Use the above function to pull out a confound matrix
confounds = extract_confounds(confound_file,
                              ['trans_x','trans_y','trans_z',
                               'rot_x','rot_y','rot_z',
                               'global_signal',
                               'white_matter','csf'])
# Drop the first 4 rows of the confounds matrix
confounds = confounds[tr_drop:,:] 

# %%
use the masker to perform our:
- Confounds cleaning
- Parcellation
- Averaging within a parcel

# %%
# Using the masker, apply cleaning, parcellation 
# and extraction to functional data
cleaned_and_averaged_time_series = masker.fit_transform(func_img, 
                                                        confounds)
cleaned_and_averaged_time_series.shape

# %%
# check which ROIs are kept
print(masker.labels_)
print("Number of labels", len(masker.labels_))

# %%
# fills in the regions that were removed with 0 values 
# (for ease of use when working with multiple subjects)
# first, identify all ROIs from the original atlas
# Get the label numbers from the atlas
atlas_labels = np.unique(yeo_7.get_fdata().astype(int))

# Get number of labels that we have
NUM_LABELS = len(atlas_labels)
print(NUM_LABELS)

# %%
Now we’re going to create an array that contains:
- A number of rows matching the number of timepoints
- A number of columns matching the total number of regions

# %%
# Remember fMRI images are of size (x,y,z,t)
# where t is the number of timepoints
num_timepoints = func_img.shape[3]

# Create an array of zeros that has the correct size
final_signal = np.zeros((num_timepoints, NUM_LABELS))

# Get regions that are kept
regions_kept = np.array(masker.labels_)

# Fill columns matching labels with signal values
final_signal[:, regions_kept] = cleaned_and_averaged_time_series

print(final_signal.shape)

# %%
# keep track of regions that was not removed my masker 
valid_regions_signal = final_signal[:, regions_kept]
print(valid_regions_signal.shape)

# %%
np.array_equal(
    valid_regions_signal,
    cleaned_and_averaged_time_series)

# %% [markdown]
# In fMRI imaging, connectivity typically refers to the 
# correlation of the timeseries of 2 ROIs. 
# Therefore we can calculate a full connectivity matrix by 
# computing the correlation between all pairs of ROIs in our 
# parcellation scheme.

# %%
# Calculating Connectivity
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')

# %%
# calculate the full correlation matrix for our parcellated data
full_correlation_matrix = correlation_measure.fit_transform([final_signal])
full_correlation_matrix.shape

# %% [markdown]
# The result is a matrix which has:
# - A number of rows matching the number of ROIs in our parcellation atlas
# - A number of columns, that also matches the number of ROIs in our parcellation atlas
# 
# read this correlation matrix as follows:
# Suppose we wanted to know the correlation between ROI 30 and ROI 40.
# Then Row 30, Column 40 gives us this correlation.

# %%
full_correlation_matrix[0, 43, 45]

# %% [markdown]
# # Music

# %%
# now apply it to every subject
# First we're going to create some empty lists to store all our data in
pooled_subjects = []
ctrl_subjects = []
mdd_subjects = []

# Which confound variables should we use?
confound_variables = ['trans_x','trans_y','trans_z',
                               'rot_x','rot_y','rot_z',
                               'global_signal',
                               'white_matter','csf']
# get the list of subjects
subjects = layout.get_subjects()
for sub in subjects:

    #Get the functional file for the subject (MNI space)
    func_files = layout.get(subject=sub,
                           datatype='func', task='music',
                           desc='preproc',
                           extension="nii.gz",
                           return_type='file')
    
    #Get the confounds file for the subject (MNI space)
    confound_files = layout.get(subject=sub, datatype='func',
                             task='music',
                             desc='confounds',
                             extension='tsv',
                             return_type='file')
    
    for i in range(len(func_files)):
        #Load the functional file in
        func_img = nimg.load_img(func_files[i])
    
        #Drop the first 4 TRs
        func_img = func_img.slicer[:,:,:,tr_drop:]
    
        #Extract the confound variables using the function
        confounds = extract_confounds(confound_files[i],
                                      confound_variables)
    
        #Drop the first 4 rows from the confound matrix
        #Which rows and columns should we keep?
        confounds = confounds[tr_drop:,:]
    
        #Apply the parcellation + cleaning to our data
        #What function of masker is used to clean and average data?
        time_series = masker.fit_transform(func_img, confounds)
    
        # fill the drop ROIs
        # Remember fMRI images are of size (x,y,z,t)
        # where t is the number of timepoints
        num_timepoints = func_img.shape[3]

        # Create an array of zeros that has the correct size
        final_signal = np.zeros((num_timepoints, NUM_LABELS))

        # Get regions that are kept
        regions_kept = np.array(masker.labels_)

        # Fill columns matching labels with signal values
        final_signal[:, regions_kept] = time_series
    
        #This collects a list of all subjects
        pooled_subjects.append(final_signal)
    
        #If the subject ID starts with a "control" then they are control
        if sub.startswith('control'):
            ctrl_subjects.append(final_signal)
        #If the subject ID starts with a "mdd" then they are mdd
        if sub.startswith('mdd'):
            mdd_subjects.append(final_signal)

# %%
# calculate the full correlation matrix for all data 
# (correlation_measure works on the list as well)
ctrl_correlation_matrices = correlation_measure.fit_transform(ctrl_subjects)
mdd_correlation_matrices = correlation_measure.fit_transform(mdd_subjects)

# %% [markdown]
# ## Visualizing Correlation Matrices and Group Differences

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
sns.heatmap(ctrl_correlation_matrices[0], cmap='RdBu_r')

# %% [markdown]
# to make it more apparent:
# - Taken the absolute value of our correlations so that the 0’s are the darkest color
# - Used a different color scheme
# 
# The dark line is basically the ROIs that were drop (dropped because it contains no voxels)

# %%
sns.heatmap(np.abs(ctrl_correlation_matrices[0]), cmap='viridis').set_title('Correlation Matrix for ND group, Music Task')

# %%
sns.heatmap(np.abs(mdd_correlation_matrices[0]), cmap='viridis').set_title('Correlation Matrix for MDD group, Music Task')

# %%
print(ctrl_correlation_matrices.shape)

# %%
# pull out just the correlation values between ROI 44 and 41 across all our subjects
ctrl_roi_vec = ctrl_correlation_matrices[:,44,41]
mdd_roi_vec = mdd_correlation_matrices[:,44,41]

# %%
# arrange this data into a table
#Create control dataframe
ctrl_df = pd.DataFrame(data={'AC_ACC_corr':ctrl_roi_vec, 
                             'group':'control'})
ctrl_df.head()

# %%
# Create the mdd dataframe
mdd_df = pd.DataFrame(data={'AC_ACC_corr':mdd_roi_vec, 
                            'group' : 'mdd'})
mdd_df.head()

# %%
# For visualization stack the two tables together
#Stack the two dataframes together
df_music = pd.concat([ctrl_df,mdd_df], ignore_index=True)

# Show some random samples from dataframe
# df.sample(n=7)

df_music

# %%
# Visualize results

# Create a figure canvas of equal width and height
plot = plt.figure(figsize=(5,5))
                  
# Create a box plot, with the x-axis as group
# the y-axis as the correlation value
ax = sns.boxplot(x='group',y='AC_ACC_corr',
                 data=df_music,palette='Set2')

# Create a "swarmplot" as well
ax = sns.swarmplot(x='group',y='AC_ACC_corr',
                   data=df_music,color='0.25')

# Set the title and labels of the figure
ax.set_title('AC - ACC Intra-network Connectivity, Musical Task')
ax.set_ylabel(r'Intra-network connectivity $\mu_\rho$')

plt.show()

# %%
# pull out just the correlation values between 
# ROI 44 and 49 across all our subjects
ctrl_roi_vec = ctrl_correlation_matrices[:,44,49]
mdd_roi_vec = mdd_correlation_matrices[:,44,49]

# %%
# arrange this data into a table
#Create control dataframe
ctrl_df = pd.DataFrame(data={'AC_ACC_corr':ctrl_roi_vec, 
                             'group':'control'})
ctrl_df.head()

# %%
# Create the mdd dataframe
mdd_df = pd.DataFrame(data={'AC_ACC_corr':mdd_roi_vec, 
                            'group' : 'mdd'})
mdd_df.head()

# %%
# For visualization stack the two tables together
#Stack the two dataframes together
df_music = pd.concat([ctrl_df,mdd_df], ignore_index=True)

# Show some random samples from dataframe
# df.sample(n=7)

df_music

# %%
# Visualize results

# Create a figure canvas of equal width and height
plot = plt.figure(figsize=(5,5))
                  
# Create a box plot, with the x-axis as group
# the y-axis as the correlation value
ax = sns.boxplot(x='group',y='AC_ACC_corr',
                 data=df_music,palette='Set2')

# Create a "swarmplot" as well
ax = sns.swarmplot(x='group',y='AC_ACC_corr',
                   data=df_music,color='0.25')

# Set the title and labels of the figure
ax.set_title('AC - Left ACC Intra-network Connectivity, Musical Task')
ax.set_ylabel(r'Intra-network connectivity $\mu_\rho$')

plt.show()

# %% [markdown]
# # Non-Music

# %%
# now apply it to every subject for non-musical task
# First we're going to create some empty lists to store all our data in
pooled_subjects = []
ctrl_subjects = []
mdd_subjects = []

# Which confound variables should we use?
confound_variables = ['trans_x','trans_y','trans_z',
                               'rot_x','rot_y','rot_z',
                               'global_signal',
                               'white_matter','csf']
# get the list of subjects
subjects = layout.get_subjects()
for sub in subjects:

    #Get the functional file for the subject (MNI space)
    func_files = layout.get(subject=sub,
                           datatype='func', task='nonmusic',
                           desc='preproc',
                           extension="nii.gz",
                           return_type='file')
    
    #Get the confounds file for the subject (MNI space)
    confound_files = layout.get(subject=sub, datatype='func',
                             task='nonmusic',
                             desc='confounds',
                             extension='tsv',
                             return_type='file')
    
    for i in range(len(func_files)):
        #Load the functional file in
        func_img = nimg.load_img(func_files[i])
    
        #Drop the first 4 TRs
        func_img = func_img.slicer[:,:,:,tr_drop:]
    
        #Extract the confound variables using the function
        confounds = extract_confounds(confound_files[i],
                                      confound_variables)
    
        #Drop the first 4 rows from the confound matrix
        #Which rows and columns should we keep?
        confounds = confounds[tr_drop:,:]
    
        #Apply the parcellation + cleaning to our data
        #What function of masker is used to clean and average data?
        time_series = masker.fit_transform(func_img, confounds)
    
        # fill the drop ROIs
        # Remember fMRI images are of size (x,y,z,t)
        # where t is the number of timepoints
        num_timepoints = func_img.shape[3]

        # Create an array of zeros that has the correct size
        final_signal = np.zeros((num_timepoints, NUM_LABELS))

        # Get regions that are kept
        regions_kept = np.array(masker.labels_)

        # Fill columns matching labels with signal values
        final_signal[:, regions_kept] = time_series
    
        #This collects a list of all subjects
        pooled_subjects.append(final_signal)
    
        #If the subject ID starts with a "control" then they are control
        if sub.startswith('control'):
            ctrl_subjects.append(final_signal)
        #If the subject ID starts with a "mdd" then they are mdd
        if sub.startswith('mdd'):
            mdd_subjects.append(final_signal)

# %%
# calculate the full correlation matrix for all data 
# (correlation_measure works on the list as well)
ctrl_correlation_matrices = correlation_measure.fit_transform(ctrl_subjects)
mdd_correlation_matrices = correlation_measure.fit_transform(mdd_subjects)

# %% [markdown]
# ## Visualizing Correlation Matrices and Group Differences

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
sns.heatmap(ctrl_correlation_matrices[0], cmap='RdBu_r')

# %% [markdown]
# to make it more apparent:
# - Taken the absolute value of our correlations so that the 0’s are the darkest color
# - Used a different color scheme
# 
# The dark line is basically the ROIs that were drop (dropped because it contains no voxels)

# %%
sns.heatmap(np.abs(ctrl_correlation_matrices[0]), cmap='viridis').set_title('Correlation Matrix for ND group, Non-Music Task')

# %%
sns.heatmap(np.abs(mdd_correlation_matrices[0]), cmap='viridis').set_title('Correlation Matrix for MDD group, Non-Music Task')

# %%
print(ctrl_correlation_matrices.shape)

# %%
# pull out just the correlation values between 
# ROI 44 and 41 across all our subjects
ctrl_roi_vec = ctrl_correlation_matrices[:,44,41]
mdd_roi_vec = mdd_correlation_matrices[:,44,41]

# %%
# arrange this data into a table
#Create control dataframe
ctrl_df = pd.DataFrame(data={'AC_ACC_corr':ctrl_roi_vec, 'group':'control'})
ctrl_df.head()

# %%
# Create the mdd dataframe
mdd_df = pd.DataFrame(data={'AC_ACC_corr':mdd_roi_vec, 'group' : 'mdd'})
mdd_df.head()

# %%
# For visualization stack the two tables together
#Stack the two dataframes together
df_nonmusic = pd.concat([ctrl_df,mdd_df], ignore_index=True)

# Show some random samples from dataframe
# df.sample(n=7)

df_nonmusic

# %%
# Visualize results

# Create a figure canvas of equal width and height
plot = plt.figure(figsize=(5,5))
                  
# Create a box plot, with the x-axis as group
# the y-axis as the correlation value
ax = sns.boxplot(x='group',y='AC_ACC_corr',data=df_nonmusic,palette='Set2')

# Create a "swarmplot" as well
ax = sns.swarmplot(x='group',y='AC_ACC_corr',data=df_nonmusic,color='0.25')

# Set the title and labels of the figure
ax.set_title('AC - Right ACC Intra-network Connectivity, Non-Musical Task')
ax.set_ylabel(r'Intra-network connectivity $\mu_\rho$')

plt.show()

# %%
# pull out just the correlation values between ROI 44 and 49 across all our subjects
ctrl_roi_vec = ctrl_correlation_matrices[:,44,49]
mdd_roi_vec = mdd_correlation_matrices[:,44,49]

# %%
# arrange this data into a table
#Create control dataframe
ctrl_df = pd.DataFrame(data={'AC_ACC_corr':ctrl_roi_vec, 'group':'control'})
ctrl_df.head()

# %%
# Create the mdd dataframe
mdd_df = pd.DataFrame(data={'AC_ACC_corr':mdd_roi_vec, 'group' : 'mdd'})
mdd_df.head()

# %%
# For visualization stack the two tables together
#Stack the two dataframes together
df_music = pd.concat([ctrl_df,mdd_df], ignore_index=True)

# Show some random samples from dataframe
# df.sample(n=7)

df_music

# %%
# Visualize results

# Create a figure canvas of equal width and height
plot = plt.figure(figsize=(5,5))
                  
# Create a box plot, with the x-axis as group
# the y-axis as the correlation value
ax = sns.boxplot(x='group',y='AC_ACC_corr',
                 data=df_nonmusic,palette='Set2')

# Create a "swarmplot" as well
ax = sns.swarmplot(x='group',y='AC_ACC_corr',
                   data=df_nonmusic,color='0.25')

# Set the title and labels of the figure
ax.set_title('AC - Left ACC Intra-network Connectivity, Musical Task')
ax.set_ylabel(r'Intra-network connectivity $\mu_\rho$')

plt.show()

# %%
from nilearn import plotting
coords = plotting.find_parcellation_cut_coords(yeo_7)

# %%
# to match the dimension of the matrix, get coordinates
coords = np.append(coords, [[0, 0, 0]], axis = 0) 

# %%
coords.shape

# %%
coords

# %%
plotting.plot_connectome(ctrl_correlation_matrices[0], coords,
                         edge_threshold="99%", 
                         title='Correlation, Non-Musical')

# %%
view = plotting.view_connectome(ctrl_correlation_matrices[0], coords,
                               edge_threshold="95%", 
                                title='Correlation, Non-Musical')
# In a Jupyter notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view


