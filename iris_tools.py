import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.io import readsav
import glob
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import warnings
from scipy.optimize import least_squares
from scipy.linalg import svd

def gaussian(x,a0,a1,a2,c):
    """Function to define a Gaussian profile over range x

    Parameters
    ----------
    x  : array-like
        The range over which the gaussian is calculated
    a0 : float
        The height of the Gaussian peak
    a1 : float
        The offset of the Gaussian core
    a2 : float
        The standard deviation of the Gaussian, the width
    c  : float
        Vertical offset of the gaussian continuum

    Returns
    -------
    y : array-like
        The corresponding intensities for a Gaussian defined over x
    """
    z = (x-a1)/a2
    y = a0*np.exp(- z**2 /2.) + c
    return y

def find_nearest(array,value):
    """Determines the index of the closest value in an array to a sepecified other value

    Parameters
    ----------
    array : array-like
        An array of int/float values
    value : int,float
        A value that we will check for in the array

    Returns
    -------
    idx : int
        The index of the input array where the closest value is found
    """
    idx = (np.abs(array-value)).argmin()
    return idx

class IRIS_FUV_spec:
    
    dark_uncertainty = 3.1
    planck_h = 6.626196e-27
    c_angstrom = 2.998e18
    c_kms = 2.998e5
    
    IRIS_FUV_linelist = {
        'C II' : [1334.5323,1335.7077], #1335.6627
        'C I'  : [1352.751,1352.988,1354.288,1355.844,1357.134,1357.659,1357.857,1358.188],
        'Cl I' : [1351.656],
        'O I'  : [1355.5977],
        'O IV' : [1338.615,1343.514,1397.198,1399.799,1401.158,1404.799],
        'Si II': [1346.873,1348.543,1350.057,1350.520,1350.658,1352.635,1353.718],
        'Si IV': [1393.755,1402.770],
        'S I'  : [1388.436,1389.154,1392.589,1396.113,1401.514],
        'S IV' : [1404.826,1406.043],
        'Ca II': [1342.554,1342.554],
        'Fe II': [1353.0218,1354.0131,1354.7434,1390.3162,1392.1480,1392.817,1399.9605,1401.7766,1403.1002,1403.2537,1404.1191,1405.6081,1405.7986],
        'H 2'  : [1333.475,1333.797,1338.565,1342.257,1344.033,1353.365,1393.451,1393.961,1396.221,1398.954,1403.982],
        'Fe XII':[1349.400],
        'Ni II': [1335.203,1348.333,1393.330,1399.026],
        'Fe XXI':[1354.064]}
    
    params = {
    'text.latex.preamble': '\\usepackage{gensymb}',
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 24, # fontsize for x and y labels (was 10)
    'axes.titlesize': 28,
    'font.size': 16, # was 10
    'legend.fontsize': 16, # was 10
    'xtick.labelsize': 32,
    'ytick.labelsize': 36,
    'text.usetex': True,
    'figure.figsize': [15, 7],
    'font.family': 'serif'
    }
    plt.rcParams.update(params)

    
    def __init__(self,IRIS_im_file):
        """Initialize the IRIS_FUV_spec class by passing the location and name of the IRIS level 3 fits file.
        Parameters:
        -----------
        self : class
            The base class
        IRIS_im_file : str
            The location and name of the IRIS level 3 file. Requires the 'im' version, not 'sp'
        """
        self.file = IRIS_im_file
    
    def read_IRIS_file(self):
        """Helper function to read IRIS file and initialize the 
        wavelength_array, 
        spectral_array, 
        and timestamp_array class attributes
        """
        file = fits.open(self.file)
        self.fits_header = file[0].header
        self.wavelength_array = file[1].data.astype('float32')[7:-10]
        self.spectral_array = file[0].data.astype('float32')[:,7:-10,:,:]
        timestamp_array = (np.datetime64(file[0].header["STARTOBS"]) + 
                           (1000. * file[2].data).astype('timedelta64[ms]'))
        if len(timestamp_array.shape) == 1:
            timestamp_array = timestamp_array.reshape(len(timestamp_array),1)
        self.timestamp_array = timestamp_array
        file.close()
    
    def calibrate_spectrum(self, iresp_file, l2_file_location, bin_factor = 1,radiometric = False):
        """Calibrates spectral_array, either to photons per second, or cgs units.
        Parameters:
        -----------
        self : class
            The class
        iresp_file : str
            Path to the iresp file for these observations
        l2_file_location : str
            Path to the IRIS level 2 files, needed for exposure times, which don't carry over into the level 3 file for some ducking reason.
        bin_factor : int
            Binning factor for final calibrated spectrum. Default 1 (unbinned)
        radiometric : bool
            Switch to determine whether flux calibration is carried out, or only photon. Default False
        """
        
        #### Check if the read_IRIS_file routine has already been run ####
        
        if not hasattr(self,'wavelength_array'):
            self.read_IRIS_file()
        
        iresp_file = readsav(iresp_file)
        self.iresp = iresp_file[list(iresp_file)[0]]
        
        #### Get exposure times for each time step
        
        l2_list = sorted(glob.glob(l2_file_location + 'iris_l2*.fits'))
        
        if self.spectral_array.shape[-1] == 1:
            temp_l2 = fits.open(l2_list[0])
            t_exp_FUV = temp_l2[-2].data[:,3]
            self.original_binning_factor = int(temp_l2[0].header["SUMSPTRF"])
            temp_l2.close()
            t_exp_FUV = t_exp_FUV.reshape(len(t_exp_FUV),1)
            
        else:
            t_exp_FUV = np.zeros((self.spectral_array.shape[0],self.spectral_array.shape[-1]))
                
            for time in range(self.spectral_array.shape[0]):
                temp_l2 = fits.open(l2_list[time])
                t_exp_FUV[time,:] = temp_l2[-2].data[:,3]
                self.original_binning_factor = int(temp_l2[0].header["SUMSPTRF"])
                temp_l2.close()
                    
        self.full_binning_factor = bin_factor * self.original_binning_factor

        self.exposure_time_array = t_exp_FUV
        self.dn2phot_FUV = self.iresp.DN2PHOT_SG[0][0]
        
        if bin_factor != 1:
            new_length = self.spectral_array.shape[2]%bin_factor
            self.spectral_array = self.spectral_array[:,:,:-new_length,:]

            rs_sa = self.spectral_array.reshape(self.spectral_array.shape[0],
                                                self.spectral_array.shape[1],
                                                int(self.spectral_array.shape[2]/bin_factor),
                                                bin_factor,
                                                self.spectral_array.shape[3])
            self.spectral_array = np.sum(rs_sa,axis = 3)
        
        if not radiometric:
        
            photons_per_second_data = np.empty(self.spectral_array.shape,dtype = 'float32')
            photons_per_second_error = np.empty(self.spectral_array.shape,dtype = 'float32')

            #### Photon calibration ####

            for raster in range(self.spectral_array.shape[-1]):
                for time in range(self.spectral_array.shape[0]):
                    photons_per_second_data[time,:,:,raster] = self.spectral_array[time,:,:,raster] * self.dn2phot_FUV / t_exp_FUV[time,raster]
                    photons_per_second_error[time,:,:,raster] = np.sqrt((self.dark_uncertainty*self.dn2phot_FUV)**2 * 
                                                                          self.spectral_array[time,:,:,raster] * self.dn2phot_FUV) / t_exp_FUV[time,raster]
            #### Perform Spatial binning along slit ####
            
            self.photons_per_second_data = photons_per_second_data
            self.photons_per_second_error = photons_per_second_error
            self.spectral_array = None

        elif radiometric:
        
            #### We're now photon calibrated -- the below will apply full radiometric calibration for all ####
            #### you sick freaks out there who like erg. Eugh. You disgust me. ####
            #### It's me. I like erg. ####
            iresp_window = (self.iresp.LAMBDA[0] <= self.wavelength_array[-1]/10) & (self.iresp.LAMBDA[0] >= self.wavelength_array[0]/10)
            photon_energy = self.planck_h * self.c_angstrom / self.wavelength_array
            a_eff = self.iresp.AREA_SG[0][0,iresp_window]
            a_eff_lambda = self.iresp.LAMBDA[0][iresp_window]
            a_eff_interp = interp1d(a_eff_lambda,a_eff,fill_value = 'extrapolate')(self.wavelength_array/10.)

            pix_xy = self.full_binning_factor * np.pi/(180.*3600.*6.)
            pix_lambda = self.wavelength_array[1] - self.wavelength_array[0]
            w_slit = np.pi / (180.*3600.*3.)

            flux_cal_wvl_product = (photon_energy * self.dn2phot_FUV)/(a_eff_interp * pix_xy * pix_lambda * w_slit)

            calibrated_data = np.zeros(self.spectral_array.shape,dtype = 'float32')
            calibrated_error = np.zeros(self.spectral_array.shape,dtype = 'float32')

            #### And performing the radiometric calibration

            for time in range(self.spectral_array.shape[0]):
                for y in range(self.spectral_array.shape[2]):
                    for x in range(self.spectral_array.shape[3]):
                        calibrated_data[time,:,y,x] = self.spectral_array[time,:,y,x] * self.dn2phot_FUV * flux_cal_wvl_product / t_exp_FUV[time,x]
                        calibrated_error[time,:,y,x] = (flux_cal_wvl_product * 
                                                        np.sqrt((self.dark_uncertainty*self.dn2phot_FUV)**2 * 
                                                                self.spectral_array[time,:,y,x] * self.dn2phot_FUV)) / t_exp_FUV[time,x]
                        
            self.calibrated_data = calibrated_data
            self.calibrated_error = calibrated_error
            self.spectral_array = None
    

    def plot_spectrum(self,indices = None):
        """Function to plot a selected slice of the calibrated IRIS spectrum with blend lines overlaid
        Parameters:
        -----------
        indices : None, or array-like
            Indices of the array of the format [t,y,x] to plot. 
            Default None will plot the slice with the largest flux.
        """
        #### Distinguish between photon vs radiometric calibration
        #### Case 1: Photon calibration, no indices given
        if hasattr(self,'photons_per_second_data'):
            if indices == None:
                collapsed_data = np.sum(np.nan_to_num(self.photons_per_second_data),axis = 1)
                max_index = np.where(collapsed_data == np.amax(collapsed_data))
                slice_to_plot = self.photons_per_second_data[max_index[0][0],:,max_index[1][0],max_index[2][0]]
                title = str(self.timestamp_array[max_index[0][0],max_index[-1]][0]) + ', Max Photons/s Slice: ' + str(max_index[0][0]) + "," + str(max_index[1][0]) + "," + str(max_index[2][0])
            else:
                slice_to_plot = self.photons_per_second_data[indices[0],:,indices[1],indices[2]]
                title = str(self.timestamp_array[indices[0],indices[-1]]) + ",Photons/s Slice: " + str(indices[0]) + ',' + str(indices[1]) + ',' + str(indices[2])
        elif hasattr(self,'calibrated_data'):
            if indices == None:
                collapsed_data = np.sum(np.nan_to_num(self.calibrated_data),axis = 1)
                max_index = np.where(collapsed_data == np.amax(collapsed_data))
                slice_to_plot = self.calibrated_data[max_index[0][0],:,max_index[1][0],max_index[2][0]]
                title = str(self.timestamp_array[max_index[0][0],max_index[-1]][0]) + ', Max Calibrated Slice: ' + str(max_index[0][0]) + "," + str(max_index[1][0]) + "," + str(max_index[2][0])
            else:
                slice_to_plot = self.calibrated_data[indices[0],:,indices[1],indices[2]]
                title = str(self.timestamp_array[indices[0],indices[-1]]) + ",Calibrated Slice: " + str(indices[0]) + ',' + str(indices[1]) + ',' + str(indices[2])
        else:
            warnings.warn("Warning, no calibrated data found. Proceeding using DN.")
            if not hasattr(self,'wavelength_array'):
                self.read_IRIS_file()
            if indices == None:
                collapsed_data = np.sum(np.nan_to_num(self.spectral_array),axis = 1)
                max_index = np.where(collapsed_data == np.amax(collapsed_data))
                slice_to_plot = self.spectral_array[max_index[0][0],:,max_index[1][0],max_index[2][0]]
                title = str(self.timestamp_array[max_index[0][0],max_index[-1]][0]) + ', Max DN Slice: ' + str(max_index[0][0]) + "," + str(max_index[1][0]) + "," + str(max_index[2][0])
            else:
                slice_to_plot = self.spectral_array[indices[0],:,indices[1],indices[2]]
                title = str(self.timestamp_array[indices[0],indices[-1]]) + ", DN Slice: " + str(indices[0]) + ',' + str(indices[1]) + ',' + str(indices[2])
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.wavelength_array,
                slice_to_plot,
                drawstyle = 'steps-mid',
                linewidth = 3,
                label = 'Data')
        
        window_length = 19
        
        ax.plot(self.wavelength_array,
                savgol_filter(slice_to_plot,window_length,3),
                linewidth = 3,
                color = 'C1',
                label = 'Smoothed',
                linestyle = '-.')
        counter = 2
        for key in self.IRIS_FUV_linelist.keys():
            blends = np.array(self.IRIS_FUV_linelist[key])
            blends = blends[(blends > self.wavelength_array[0] - 0.2) & 
                            (blends < self.wavelength_array[-1] + 0.2)]
            if counter <= 9:
                style = '-'
            else:
                style = '--'
            if len(blends) != 0:
                ax.axvline(blends[0],
                           color = 'C'+str(counter),
                           linewidth = 2, 
                           label = str(key),
                           linestyle = style)
                for line in blends:
                    ax.axvline(line,
                               color = 'C'+str(counter),
                               linewidth = 2,
                               linestyle = style)
            counter += 1
        ax.legend(loc = 'center left',bbox_to_anchor = (1,0.5),fancybox = True,shadow = True,ncol = 2)
        ax.set_title(title,pad = 20)
        plt.show()
    
    def fit_slice(self,indices = None):
        """Fits a dynamically-generated model of the data at "indices". 
            The model is a sum of Gaussian components, with a number of Gaussian components 
            equal to the number of spectral lines contained within the wavelength range, 
            plus additional red and blue wing components for the strong C II and Si IV doublets.
        
        Parameters:
        -----------
        indices : None, or array-like 
            Indices of the array of the format [t,lambda_min,lambda_max,y,x] to create a model for. 
            Default None will create a model for the slice with the largest flux.
        mcmc : bool
            Switches between least-squares fitting and MCMC fitting. 
            Least-squares is faster, and may produce better results, 
            but the errors are not well constrained. MCMC is frequently
            problematic, and significantly slower, but the errors are better.
        """
        #### Helper information -- let's us easily flag our blends
        primary_line_ions = ['C II','Si IV','Fe XXI']
        blend_line_ions = ['C I','Cl I','O I',
                           'O IV','Si II','S I',
                           'S IV','Ca II','Fe II',
                           'H 2','Fe XII','Ni II']
        
        #### Turns out certain blends have definite shifts. Pain in my ass much?
        #### Flares, man. Doesn't help that Fe XXI gets quick.
        blend_allowed_shift = 75. #km/s
        CSi_allowed_shift = 150. #km/s
        Fe_allowed_shift = 400. #km/s
        
        #### Yeah...
        blend_allowed_width = 0.08
        CSi_allowed_width = 0.15
        Fe_allowed_width = 1.5 / (2. * np.sqrt(2. * np.log(2.)))

        #### Split off the slice to fit and the error array for that slice.
        #### Case 0: Data wasn't calibrated, raise an error telling the user to calibrate data
        #### Case 1: Photon calibrated: split off from photons
        #### Case 2: Radiometric calibrated: split off from calibrated
        if not (hasattr(self,'photons_per_second_data') or hasattr(self,'calibrated_data')):
            raise KeyError("You need to run calibrate_spectrum() before constructing a model.")
        elif hasattr(self,'photons_per_second_data'):
            if indices == None:
                collapsed_data = np.sum(np.nan_to_num(self.photons_per_second_data),axis = 1)
                max_index = np.where(collapsed_data == np.amax(collapsed_data))
                slice_to_fit = self.photons_per_second_data[max_index[0][0],:,max_index[1][0],max_index[2][0]]
                error_slice = self.photons_per_second_error[max_index[0][0],:,max_index[1][0],max_index[2][0]]
                wavelength_range_to_fit = self.wavelength_array
            elif len(indices) == 3:
                slice_to_fit = self.photons_per_second_data[indices[0],:,indices[1],indices[2]]
                error_slice = self.photons_per_second_error[indices[0],:,indices[1],indices[2]]
                wavelength_range_to_fit = self.wavelength_array
            else:
                wavelength_range_to_fit = self.wavelength_array[(self.wavelength_array >= indices[1]) & 
                                                                (self.wavelength_array <= indices[2])]
                slice_to_fit = self.photons_per_second_data[indices[0],
                                                            :,
                                                            indices[3],
                                                            indices[4]]
                slice_to_fit = slice_to_fit[(self.wavelength_array >= indices[1]) & 
                                            (self.wavelength_array <= indices[2])]
                error_slice = self.photons_per_second_error[indices[0],
                                                            :,
                                                            indices[3],
                                                            indices[4]]
                error_slice = error_slice[(self.wavelength_array >= indices[1]) & 
                                          (self.wavelength_array <= indices[2])]
        elif hasattr(self,'calibrated_data'):
            if indices == None:
                collapsed_data = np.sum(np.nan_to_num(self.calibrated_data),axis = 1)
                max_index = np.where(collapsed_data == np.amax(collapsed_data))
                slice_to_fit = self.calibrated_data[max_index[0][0],:,max_index[1][0],max_index[2][0]]
                error_slice = self.calibrated_error[max_index[0][0],:,max_index[1][0],max_index[2][0]]
                wavelength_range_to_fit = self.wavelength_array
            elif len(indices) == 3:
                slice_to_fit = self.calibrated_data[indices[0],:,indices[1],indices[2]]
                error_slice = self.calibrated_error[indices[0],:,indices[1],indices[2]]
                wavelength_range_to_fit = self.wavelength_array
            else:
                wavelength_range_to_fit = self.wavelength_array[(self.wavelength_array >= indices[1]) & 
                                                                (self.wavelength_array <= indices[2])]
                slice_to_fit = self.calibrated_data[indices[0],
                                                    :,
                                                    indices[3],
                                                    indices[4]]
                slice_to_fit = slice_to_fit[(self.wavelength_array >= indices[1]) & 
                                            (self.wavelength_array <= indices[2])]
                error_slice = self.calibrated_error[indices[0],
                                                    :,
                                                    indices[3],
                                                    indices[4]]
                error_slice = error_slice[(self.wavelength_array >= indices[1]) & 
                                          (self.wavelength_array <= indices[2])]

        slice_to_fit = np.nan_to_num(slice_to_fit)
        slice_to_fit[slice_to_fit == 0] = np.median(slice_to_fit)
        error_slice = np.nan_to_num(error_slice)
        error_slice[error_slice == 0] = np.median(error_slice)
        #### Now we've got slice_to_fit, error_slice, and wavelength_range_to_fit, 
        #### which may or may not be truncated.
        #### Next up is to construct four arrays:
        #### Array 1: Model Inputs. A list of all blends, of form I, C, W, I, C, W
        #### Array 2: Low Bounds on Model Inputs, of form I_lo, C_lo, W_lo, I_lo, C_lo, W_lo
        #### Array 3: High Bounds on Model Inputs, of form I_hi, C_hi, W_hi, I_hi, C_hi, W_hi
        #### Array 4: Array of keys. Of Form "LINE ID" I, "LINE ID" C, "LINE ID" W, etc.
        #### Note that if your wavelength window contains C II or Si IV,
        #### the arrays created will have initial guesses for a blue and red wing for each of these.
        
        list_of_parameters = []
        parameter_lower_bounds = []
        parameter_upper_bounds = []
        key_array = []
        for key in self.IRIS_FUV_linelist.keys():
            potential_blends = np.array(self.IRIS_FUV_linelist[key])
            blends = potential_blends[(potential_blends > wavelength_range_to_fit[0] - 0.2) & 
                                      (potential_blends < wavelength_range_to_fit[-1] + 0.2)]
            for i in range(len(blends)):
                if key in primary_line_ions:
                    if (key == 'C II') or (key == 'Si IV'):
                        allowed_offset = CSi_allowed_shift/self.c_kms * blends[i]
                        
                        blue_limit_index = find_nearest(wavelength_range_to_fit,blends[i] - allowed_offset)
                        red_limit_index = find_nearest(wavelength_range_to_fit,blends[i] + allowed_offset)
                        
                        peak_index = np.where(slice_to_fit == slice_to_fit[blue_limit_index:red_limit_index].max())[0][0]
                        
                        #### Line Core ####
                        
                        list_of_parameters = (list_of_parameters + 
                                              [np.amax(slice_to_fit[blue_limit_index:red_limit_index]),
                                               wavelength_range_to_fit[peak_index],
                                               0.1])
                        key_array = key_array + [key + " " + str(blends[i])[:6] + " I",
                                                 key + " " + str(blends[i])[:6] + " C", 
                                                 key + " " + str(blends[i])[:6] + " W"]
                        parameter_lower_bounds = (parameter_lower_bounds + 
                                                  [np.amax(slice_to_fit[blue_limit_index:red_limit_index])/2.,
                                                   wavelength_range_to_fit[peak_index] - allowed_offset/2,
                                                   1e-3])
                        parameter_upper_bounds = (parameter_upper_bounds + 
                                                  [np.inf,
                                                   wavelength_range_to_fit[peak_index] + allowed_offset/2., 
                                                   CSi_allowed_width])
                        
                        #### Potential Red-Wing Enhancement ####

                        list_of_parameters = (list_of_parameters + 
                                              [np.amax(slice_to_fit[blue_limit_index:red_limit_index])/4.,
                                               wavelength_range_to_fit[peak_index] + 0.35,
                                               0.15])
                        key_array = key_array + [key + " " + str(blends[i])[:6] + " RW I", 
                                                 key + " " + str(blends[i])[:6] + " RW C",
                                                 key + " " + str(blends[i])[:6] + " RW W"]
                        parameter_lower_bounds = (parameter_lower_bounds + 
                                                  [0,
                                                   wavelength_range_to_fit[peak_index] + 0.1,
                                                   1e-3])
                        parameter_upper_bounds = (parameter_upper_bounds + 
                                                  [np.amax(slice_to_fit[blue_limit_index:red_limit_index]) * 0.5,
                                                   wavelength_range_to_fit[peak_index] + 0.35 + allowed_offset/2,
                                                   1.2*CSi_allowed_width])
                        
                        #### Potential Blue-Wing Enhancement ####

                        list_of_parameters = (list_of_parameters + 
                                              [np.amax(slice_to_fit[blue_limit_index:red_limit_index])/4.,
                                               wavelength_range_to_fit[peak_index] - 0.35,
                                               0.15])
                        key_array = key_array + [key + " " + str(blends[i])[:6] + " BW I",
                                                 key + " " + str(blends[i])[:6] + " BW C",
                                                 key + " " + str(blends[i])[:6] + " BW W"]
                        parameter_lower_bounds = (parameter_lower_bounds + 
                                                  [0,
                                                   wavelength_range_to_fit[peak_index] - 0.35 - allowed_offset/2,
                                                   1e-3])
                        parameter_upper_bounds = (parameter_upper_bounds + 
                                                  [np.amax(slice_to_fit[blue_limit_index:red_limit_index]) * 0.5,
                                                   wavelength_range_to_fit[peak_index] - 0.1,
                                                   CSi_allowed_width])
                    elif key == 'Fe XXI':
                        allowed_offset = Fe_allowed_shift/self.c_kms * blends[i]
                        central_index = np.where(np.abs(wavelength_range_to_fit - blends[i]) == 
                                                 np.abs(wavelength_range_to_fit - blends[i]).min())[0]
                        high_idx = np.where(np.abs(wavelength_range_to_fit - blends[i] - allowed_offset) == 
                                           np.abs(wavelength_range_to_fit - blends[i] - allowed_offset).min())[0][0]
                        low_idx = np.where(np.abs(wavelength_range_to_fit - blends[i] + allowed_offset) == 
                                            np.abs(wavelength_range_to_fit - blends[i] + allowed_offset).min())[0][0]

                        list_of_parameters = list_of_parameters + [0,
                            #np.mean(error_slice[low_idx:high_idx]),
                                                                   blends[i],0.5]
                        key_array = key_array + [key + " " + str(blends[i])[:6] + " I", 
                                                 key + " " + str(blends[i])[:6] + " C",
                                                 key + " " + str(blends[i])[:6] + " W"]
                        parameter_lower_bounds = parameter_lower_bounds + [-2*np.mean(error_slice[low_idx:high_idx]),
                                                                           blends[i] - allowed_offset,0.1]
                        parameter_upper_bounds = parameter_upper_bounds + [slice_to_fit[low_idx:high_idx].max()*2,blends[i] + allowed_offset,Fe_allowed_width]
                                            
                elif key in blend_line_ions:
                    allowed_offset = blend_allowed_shift/self.c_kms * blends[i]
                    central_index = np.where(np.abs(wavelength_range_to_fit - blends[i]) == 
                                             np.abs(wavelength_range_to_fit - blends[i]).min())[0]
                    high_idx = np.where(np.abs(wavelength_range_to_fit - blends[i] - allowed_offset) == 
                                       np.abs(wavelength_range_to_fit - blends[i] - allowed_offset).min())[0][0]
                    low_idx = np.where(np.abs(wavelength_range_to_fit - blends[i] + allowed_offset) == 
                                        np.abs(wavelength_range_to_fit - blends[i] + allowed_offset).min())[0][0]
                    
                    list_of_parameters = list_of_parameters + [error_slice[low_idx:high_idx].max(),
                                                               blends[i],
                                                               0.04]
                    key_array = key_array + [key + " " + str(blends[i])[:6] + " I", 
                                             key + " " + str(blends[i])[:6] + " C",
                                             key + " " + str(blends[i])[:6] + " W"]
                    parameter_lower_bounds = parameter_lower_bounds + [-2*np.mean(error_slice[low_idx:high_idx]),
                                                                       blends[i] - allowed_offset,
                                                                       0]
                    parameter_upper_bounds = parameter_upper_bounds + [3*slice_to_fit[low_idx:high_idx].max(),
                                                                       blends[i] + allowed_offset,
                                                                       blend_allowed_width]
                    
        def model_function(list_of_params,xdata):
            """Model function for use in scipy least squares fitting.
            Parameters
            ----------
            xdata : array-like
                The array of wavelengths to form a model over
            list_of_params : array-like
                List of parameters to use in formation of the model. It is of variable length.
                This list should be of the form [I0,C0,W0,I1,C1,W1, etc...]
            """
            blended_model = np.zeros(len(xdata))
            for i in range(int(len(list_of_params)/3)):
                blended_model = blended_model + gaussian(xdata,
                                                         list_of_params[3*i],
                                                         list_of_params[3*i + 1],
                                                         list_of_params[3*i + 2],
                                                         0)
            return blended_model
        
        def error_function(list_of_params,x,y):
            return (model_function(list_of_params,x) - y)
        
        def chisq(fit_result,waves,data,error):
            chisq = (((error_function(fit_result,waves,data)**2) /  
                      error**2).sum() / 
                     (len(data) - len(fit_result)))
            return chisq
        
        list_of_parameters = np.array(list_of_parameters)
        parameter_lower_bounds = np.array(parameter_lower_bounds)
        parameter_upper_bounds = np.array(parameter_upper_bounds)
            
        fit_result = least_squares(error_function,
                                   x0 = np.array(list_of_parameters),
                                   bounds = (parameter_lower_bounds,parameter_upper_bounds),
                                   args = (wavelength_range_to_fit,slice_to_fit),
                                   jac = '3-point',tr_solver = 'lsmr',verbose = 2)

        _,s,VT = svd(fit_result.jac,full_matrices = False)
        threshold = np.finfo(float).eps * max(fit_result.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov_init = np.dot(VT.T / s**2,VT)

        chisq_val = chisq(fit_result.x,wavelength_range_to_fit,slice_to_fit,error_slice)

        pcov = pcov_init * chisq_val

        return fit_result.x,pcov,chisq_val,key_array
