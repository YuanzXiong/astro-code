import os
import numpy as np
import astropy
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Gaussian2DKernel
from reproject import reproject_interp
import aplpy
import astropy.units as u
from numpy import nan_to_num
from astropy.coordinates import SkyCoord
import pandas as pd
from statsmodels.formula.api import ols


def get_constants():
    constants = {
        'ra': 284.25,                       # RA in degrees
        'dec': 1.66,                        # DEC in degrees
        'radius_arcmin': 8.0,               # Radius in arcmin
        'wavelengths': [70, 160, 250, 350, 500],  # Microns
        'target_resolution': 36.9,          # arcsec
        'native_beamsize_list': [5.8, 11.4, 18.1, 25.2, 36.3],  # arcsec
        'pixel_size_list': [3.2, 3.2, 6.0, 9.72, 14.0],  # Pixel scale in arcsec
        'h': 6.626 * 10**(-34),             # Planck's constant in J*s
        'k': 1.380649 * 10**(-23),          # Boltzmann constant in J/K
        'c': 2.998 * 10**8,                 # Speed of light in m/s
        'mu': 2.8,                          # Mean molecular weight
        'm_H': 1.674 * 10**(-24),           # Mass of hydrogen atom in g
        'kappa_1000': 0.1,                  # in cm²/g
        'kappa_230': 0.09,                  # in cm²/g
        'h_k': 6.626 * 10**(-34) / (1.380649 * 10**(-23)),  # h/k
        'm_H2': 2 * 1.674 * 10**(-24),      # Mass of hydrogen molecule in g
        'M_sun': 1.988e33,                  # Solar mass in g
        'tunit': 'MJy/sr',                  # Target unit
        'pixel_size': 8.888888888888889e-4, # Pixel size in degrees
        'distance_pc': 2370,                # Distance to the molecular cloud in parsecs
        'num_files': 2,                     # Number of files to process
        
        # New parameters
        'new_directory': "D:\\astro",       # Directory for processing
        'sourcename': "Field_35",           # Source name
        'instrument': "Herschel",           # Instrument name
        'target_folder': "./SEDresult/Field_35"  # Target folder path
    }
    return constants


def process_fits_files(new_directory, sourcename, instrument, wavelengths, output_filelist_name):
    # Define the target folder path
    target_folder = f"./SEDresult/{sourcename}"
    
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Initialize the file path list
    globals()[output_filelist_name] = [] 
    
    # Build the file paths
    for wavelength in wavelengths:
        file_path = f"./data/{sourcename}_select/{sourcename}_{instrument}_{wavelength}micron.fits"
        filelist.append(file_path) 
    
    # Print the file paths
    for path in filelist:
        print(path)
    
    # Change to the new working directory
    os.chdir(new_directory)
    print("New working directory:", os.getcwd())
    
    # Open the first file and display its information
    with fits.open(filelist[0]) as hdul:
        hdul.info()

def convert_units_for_files(filelist, num_files, tunit, pixel_size, target_folder, output_filelist_name):
    """
        Purpose:
        Convert the units for the first `num_files` FITS files in the filelist,
        replacing the original files with the converted files and keeping the other files unchanged.

        Input:
        filelist            [list]   : List of FITS file paths to convert.
        num_files           [int]    : Number of files to process.
        tunit               [str]    : The target unit for conversion (e.g., 'MJy/sr').
        pixel_size          [float]  : Pixel size in degrees.
        target_folder       [str]    : Folder to save the converted FITS files.
        output_filelist_name [str]   : The name of the global variable to store the converted file paths.
        """
    
    # Use globals() to dynamically create a global variable and initialize it as an empty list
    globals()[output_filelist_name] = []

    # Define a conversion function: Convert Jy/pixel to MJy/sr
    def convert_jy_per_pixel_to_Mjy_per_sr(value, pixel_size):
        # Convert Jy/pixel to MJy/pixel
        value = value * u.Jy / u.pixel
        value = value.to(u.MJy / u.pixel)

        # Calculate the solid angle of each pixel (in sr)
        pixel_angle = pixel_size * u.deg
        pixel_solid_angle = (pixel_angle.to(u.rad))**2

        # Convert from MJy/pixel to MJy/sr
        result = value / pixel_solid_angle

        return result.value

    # Define a function to process each file
    def conv_u(hdul, tunit, ucfile, pixel):
        data = hdul['image'].data
        header = hdul['image'].header

        # Convert data units
        data_converted = convert_jy_per_pixel_to_Mjy_per_sr(data, pixel)

        # Convert error data units
        error_data = hdul['error'].data
        error_data_converted = convert_jy_per_pixel_to_Mjy_per_sr(error_data, pixel)

        # Update the BUNIT in the header
        header['BUNIT'] = tunit

        # Create a new HDU list
        hdu_new = fits.HDUList([fits.PrimaryHDU(data=data_converted, header=header), 
                                fits.ImageHDU(data=error_data_converted, name='stDev')])

        # Define the new file name
        base_filename, ext = os.path.splitext(os.path.basename(ucfile))
        new_filename = base_filename + '_unit_convert.fits'
        new_file_path = os.path.join(target_folder, new_filename)

        # Write the new FITS file
        hdu_new.writeto(new_file_path, overwrite=True)

        # Return the path to the converted file
        return new_file_path

    # Loop through the file list, processing the first num_files files
    for i, file in enumerate(filelist):
        if i < num_files:
            # Perform unit conversion
            hdu = fits.open(file)
            ucfile = file  # Assume ucfile is the same as the current file
            converted_file = conv_u(hdu, tunit, ucfile, pixel_size)
            globals()[output_filelist_name].append(converted_file)  # Add converted file to the global list
        else:
            # Keep the unchanged files
            globals()[output_filelist_name].append(file)

    # Print the final file list to ensure it includes the converted files and unchanged files
    print(f"Updated file list ({output_filelist_name}):", globals()[output_filelist_name])


def crop_and_plot_fits(input_filelist_name, output_filelist_name, ra, dec, radius_arcmin, target_folder):
    """
        Purpose:
        Crop FITS files and plot the images.

        Input:
        input_filelist_name [list[str]] : List of input FITS file paths.
        output_filelist_name [str]      : The name of the global variable to store the file paths.
        ra                   [float]    : Right ascension of the target object (degrees).
        dec                  [float]    : Declination of the target object (degrees).
        radius_arcmin        [float]    : Radius of the cropping area (arcminutes).
        target_folder        [str]      : Folder to save the cropped FITS files.
        """
    # Define an empty list using a global variable to store cropped file paths
    globals()[output_filelist_name] = []  

    for fits_file in input_filelist_name:
        with fits.open(fits_file) as hdul:
            data_list = [hdul[i].data for i in range(len(hdul))]
            header_list = [hdul[i].header for i in range(len(hdul))]

            hdu_list_new = fits.HDUList()

            for i, (data, header) in enumerate(zip(data_list, header_list)):
                if data is None or data.size == 0:
                    print(f"No data in HDU {i} of {fits_file}. Skipping this HDU.")
                    continue

                w = WCS(header)
                coord = SkyCoord(ra, dec, unit='deg', frame='icrs')
                pix_x, pix_y = w.world_to_pixel(coord)

                radius_pixel = radius_arcmin * (60 / 3600) / np.abs(w.wcs.cdelt[0])
                x_min, x_max = np.clip([pix_x - radius_pixel, pix_x + radius_pixel], 0, data.shape[1]).astype(int)
                y_min, y_max = np.clip([pix_y - radius_pixel, pix_y + radius_pixel], 0, data.shape[0]).astype(int)

                cropped_data = data[y_min:y_max, x_min:x_max]
                final_cropped_data = np.full((y_max - y_min, x_max - x_min), np.nan, dtype=np.float32)
                final_cropped_data[:cropped_data.shape[0], :cropped_data.shape[1]] = cropped_data

                hdu_new = fits.ImageHDU(data=final_cropped_data, header=header)
                hdu_list_new.append(hdu_new)

            base_filename, ext = os.path.splitext(os.path.basename(fits_file))
            new_filename = f"{base_filename}_cut.fits"
            new_file_path = os.path.join(target_folder, new_filename)

            hdu_list_new.writeto(new_file_path, overwrite=True)
            print(f"Cropped FITS file saved as '{new_file_path}'")
            
            # Add the path of the cropped file to the global list
            globals()[output_filelist_name].append(new_file_path)  # Add path to output_filelist_name

    print(f"List of cropped files: {globals()[output_filelist_name]}")
    
    # Create a new figure
    fig = plt.figure(figsize=(9, 9))

    # Iterate through the file list and display each image
    for i, file in enumerate(globals()[output_filelist_name]):
        with fits.open(file) as hdul:
            data = hdul[0].data

            # Compute the 5th and 95th percentiles of the data
            lower_percentile = np.nanpercentile(data, 5)
            upper_percentile = np.nanpercentile(data, 99.5)

            # Plot with aplpy
            ax = aplpy.FITSFigure(file, hdu=0, figure=fig, subplot=(3, 2, i + 1))
            ax.show_colorscale(cmap='gist_yarg', vmin=lower_percentile, vmax=upper_percentile)  # Set color limits
            ax.add_colorbar()
            ax.colorbar.set_axis_label_text(r'$Flux\,\,\, (MJy/sr)$')  # Colorbar axis label
            ax.tick_labels.set_font(size='small')  # Set tick label font size
            ax.axis_labels.set_font(size='small')   # Set axis label font size
            ax.colorbar.set_location('top')          # Set colorbar location
            ax.colorbar.set_width(0.15)              # Set colorbar width
            ax.colorbar.show()                       # Show colorbar

    # Adjust the layout and display the images
    plt.tight_layout()
    plt.show()


def process_fits_files_and_convolve_reproject(input_filelist_name, output_filelist_name, target_folder, target_resolution, native_beamsize, pixel_size):
    """
        Purpose:
        Process FITS files, perform convolution and reprojection, and save the results.

        Input:
        input_filelist_name  [list[str]] : List of paths to the FITS files to be processed.
        output_filelist_name [str]       : The name of the global variable to store the file paths.
        target_folder        [str]       : The folder where the processed files will be saved.
        target_resolution    [float]     : The target resolution for convolution.
        native_beamsize      [array]     : The array of native beam sizes.
        pixel_size           [array]     : The array of pixel sizes.
        """
    globals()[output_filelist_name] = []
    
    def convolve_data(data, target_resolution, native_beamsize, pixel_size):
        """Convolve the data with a Gaussian kernel."""
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a NumPy array.")
        
        FWHM_TO_SIGMA = 1. / np.sqrt(8 * np.log(2))

        kernel_size1 = ((target_resolution * FWHM_TO_SIGMA) ** 2 - (native_beamsize * FWHM_TO_SIGMA) ** 2) ** 0.5
        pixel_n1 = kernel_size1 / pixel_size

        if pixel_n1 <= 0:
            raise ValueError("Calculated pixel_n1 must be greater than zero.")

        gauss_kernel = Gaussian2DKernel(pixel_n1)
        masked_data = np.ma.masked_invalid(data)  # Mask invalid values (NaN)
        data_convolve = convolve(masked_data, gauss_kernel, normalize_kernel=True, boundary='fill', fill_value=0)
        data_convolve_filled = np.nan_to_num(data_convolve, nan=0.0)
        return data_convolve_filled

    def reproject_data(hdu, header_target):
        """Reproject the data to the target coordinate system."""
        data_reproject = reproject_interp(hdu, header_target, return_footprint=False)
        return data_reproject

    # Get the header information from the target file
    with fits.open(input_filelist_name[4]) as hdul_t:
        header_t = hdul_t[1].header

    # Process each FITS file
    for i, file in enumerate(input_filelist_name):
        with fits.open(file) as hdul:
            if i < 2:
                data = hdul[0].data
                stDev = hdul[1].data
                header = hdul[0].header
            else:
                data = hdul[1].data
                stDev = hdul[2].data
                header = hdul[1].header

            # Ensure that native_beamsize and pixel_size are lists
            if isinstance(native_beamsize, list) and isinstance(pixel_size_list, list):
                # Convolve the data
                data_convolved = convolve_data(data, target_resolution, native_beamsize_list[i], pixel_size_list[i])
                stDev_convolved = convolve_data(stDev, target_resolution, native_beamsize_list[i], pixel_size_list[i])
            else:
                raise ValueError("native_beamsize and pixel_size must be lists.")

            # Create the convolved HDU
            hdu_convolved = fits.HDUList([
                fits.PrimaryHDU(data_convolved, header=header),
                fits.ImageHDU(stDev_convolved, name='stDev', header=header)
            ])

            # Reproject the data
            data_reprojected = reproject_data(hdu_convolved[0], header_t)
            stDev_reprojected = reproject_data(hdu_convolved[1], header_t)
            
            # Create the reprojected HDU
            hdu_new = fits.HDUList([
                fits.PrimaryHDU(data=data_reprojected, header=header_t),
                fits.ImageHDU(data=stDev_reprojected, name='stDev', header=header_t)
            ])

            # Save the new FITS file
            base_filename, ext = os.path.splitext(os.path.basename(file))
            new_filename = base_filename + '_conv_reproj.fits'
            new_file_path = os.path.join(target_folder, new_filename)
            globals()[output_filelist_name].append(new_file_path)
            hdu_new.writeto(new_file_path, overwrite=True)
        
    # Print file paths and information
    print(globals()[output_filelist_name])
    hdulist70 = fits.open(globals()[output_filelist_name][0])
    hdulist500 = fits.open(globals()[output_filelist_name][4])
    print(hdulist70.info())
    print(hdulist500.info())


def SED(lamda, T, N):
    """
        Purpose:
        Calculate the Spectral Energy Distribution (SED) at a given wavelength and temperature.

        Input:
        lamda  [array]   : Wavelength in microns, an array of wavelength values.
        T      [float]   : Temperature in Kelvin.
        N      [float]   : Optical depth.

        Returns:
        array   : The calculated Spectral Energy Distribution (SED) in units of MJy/sr.
        """
    beta = 1.8
    nu = c / (lamda * 1e-6)  # Convert wavelength to frequency (Hz)
    kappa = kappa_1000 * (nu / (1000 * 1e9)) ** beta  # Absorption coefficient
    return 1e20 * 2 * h * nu**3 / c**2 * (1-np.exp(-kappa * N)) / (np.exp(h_k * nu/T)-1)


def fit_SED_all(input_filelist_name, output_filelist_name, wavelengths, c, kappa_1000, h, h_k):
    """
        Purpose:
        Fit the Spectral Energy Distribution (SED) for each pixel in a list of FITS files.

        Input:
        input_filelist_name  [list[str]]: List of FITS image file paths
        output_filelist_name [str]      : Name of the output global variable to store the result
        wavelengths          [array]    : Wavelengths in microns corresponding to each data point
        c                    [float]    : Speed of light in cm/s
        kappa_1000           [float]    : Absorption coefficient at 1000 microns
        h                    [float]    : Planck's constant in erg·s
        h_k                  [float]    : Planck's constant divided by the Boltzmann constant (h / k_B)
        """    
    data = [fits.getdata(image_file, ext=0) for image_file in input_filelist_name]
    data = np.stack(data, axis=0)
    stDev_data = [fits.getdata(image_file, ext=1) for image_file in input_filelist_name]
    stDev_data = np.stack(stDev_data, axis=0)
    min_length = min(data.shape[1], data.shape[2])
    globals()[output_filelist_name] = np.zeros((data.shape[1], data.shape[2], 2))

    for i in range(min_length):
        for j in range(min_length):
            pixel_values = data[:, j, i]
            pixel_values = np.where(pixel_values == 0, np.nan, pixel_values) 
            pixel_values[np.isinf(pixel_values)] = np.nan
            stDev = stDev_data[:, j, i]        
            if np.any(np.isnan(pixel_values)) or np.any(np.isnan(stDev)):
                continue
            stDev_nonzero = nan_to_num(stDev, nan=np.nanmin(stDev))
            popt, pcov = curve_fit(SED, wavelengths, pixel_values, p0=[10, 10], bounds=[[0.1, 0.001], [50, 1000]], sigma=stDev_nonzero, maxfev=2000)
            param_map[j, i] = popt


  def fit_single_pixel(input_filelist_name, output_filelist_name, wavelengths, c, kappa_1000, h, h_k, pixel_coords):
    """
        Purpose:
        Check the fitting at a single point.

        Input:
        input_filelist_name  [list[str]]: List of FITS image file paths
        output_filelist_name [str]      : Name of the output global variable to store the result
        wavelengths          [array]    : Wavelengths in microns corresponding to each data point
        c                    [float]    : Speed of light in cm/s
        kappa_1000           [float]    : Absorption coefficient at 1000 microns
        h                    [float]    : Planck's constant in erg·s
        h_k                  [float]    : Planck's constant divided by the Boltzmann constant (h / k_B)
        """    
    # Initialize the global variable for storing the result
    globals()[output_filelist_name] = []
    
    # Read FITS image data
    data = [fits.getdata(image_file, ext=0) for image_file in input_filelist_name]
    data = np.stack(data, axis=0)
    
    # Read standard deviation data
    stDev_data = [fits.getdata(image_file, ext=1) for image_file in input_filelist_name]
    stDev_data = np.stack(stDev_data, axis=0)

    # Extract the pixel values at the given coordinates (x, y)
    x, y = pixel_coords
    pixel_values = data[:, x, y]
    stDev = stDev_data[:, x, y]
    
    # Handle NaN and Inf values in the data
    pixel_values = np.where(pixel_values == 0, np.nan, pixel_values)
    pixel_values[np.isinf(pixel_values)] = np.nan
    stDev = np.nan_to_num(stDev, nan=np.nanmin(stDev))

    # Ensure that wavelengths is a numpy array
    wavelengths = np.array(wavelengths, dtype=np.float64)
    
    try:
        # Use curve_fit to perform curve fitting
        popt, pcov = curve_fit(lambda lamda, T, N: SED(lamda, T, N), 
                               wavelengths, pixel_values, p0=[10, 10], 
                               bounds=[[0.1, 0.001], [50, 1000]], sigma=stDev, maxfev=2000)
        
        # Extract the fitted temperature and column density
        T_fit, N_fit = popt
        
        # Generate a finer wavelength grid to plot the fitted SED curve
        fine_wavelengths = np.linspace(wavelengths.min(), wavelengths.max(), 500)
        fitted_sed = SED(fine_wavelengths, T_fit, N_fit)

        # Plot the original data and the fitted SED curve
        plt.figure(figsize=(8, 6))
        plt.plot(fine_wavelengths, fitted_sed, label=f'Fitted SED (T={T_fit:.2f} K, N={N_fit:.2e} cm^-2)', color='green')
        plt.scatter(wavelengths, pixel_values, label='Data', color='blue')
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('SED (MJy/sr)')
        plt.title('Fitting the Spectral Energy Distribution (SED)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        # Print error message if fitting fails
        print(f"Error fitting pixel at (40, 40): {e}")
        print(f"Pixel Values: {pixel_values}")


def log_and_save_fits(param_map):
    """
        Purpose:
        Log-transform the N_H2 array and save the result along with the temperature (T) array to FITS files.

        Input:
        param_map     [numpy array 3D] : The parameter map, where the third dimension contains [T, N_H2] values
        target_folder [string]         : The folder where the FITS files will be saved
        sourcename    [string]         : The name of the target source
        instrument    [string]         : The instrument used for the observations
        mu            [float]          : The mean molecular weight
        m_H           [float]          : The mass of hydrogen in grams
        """

    # Extract N_H2 and T data
    N_H2 = param_map[:, :, 1] / (mu * m_H)
    T = param_map[:, :, 0]
    hdul = fits.open(f"{target_folder}/{sourcename}_{instrument}_500micron_conv_reproj_cut.fits")
    header = hdul[0].header
    hdul.close()

    # Apply log transformation to N_H2, setting 0 values to NaN
    log_N_H2 = np.empty_like(N_H2)
    log_N_H2[:] = np.nan  # Initialize with NaN
    
    non_zero_indices = np.where(N_H2 != 0)
    log_N_H2[non_zero_indices] = np.log10(N_H2[non_zero_indices])
    
    header['BUNIT'] = 'cm^-2'  # Set the unit for N_H2
    hdu = fits.PrimaryHDU(N_H2, header=header)
    NH2_path = f"{target_folder}/{sourcename}_{instrument}_NH2.fits"
    hdu.writeto(NH2_path, overwrite=True)
    
    # Save log(N_H2) to a FITS file
    header['BUNIT'] = 'cm^-2'  # Set the unit for N_H2
    hdu = fits.PrimaryHDU(log_N_H2, header=header)
    logNH2_path = f"{target_folder}/{sourcename}_{instrument}_logNH2.fits"
    hdu.writeto(logNH2_path, overwrite=True)
    
    # Handle temperature data, set 0 values to NaN
    Temp = T.copy()
    Temp[T == 0] = np.nan
    
    # Save temperature (T) to a FITS file
    header['BUNIT'] = 'K'  # Set the unit for temperature
    hdu = fits.PrimaryHDU(Temp, header=header)
    temp_path = f"{target_folder}/{sourcename}_{instrument}_T.fits"
    hdu.writeto(temp_path, overwrite=True)

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 10))
    
    # Display the NH2 map
    ax = aplpy.FITSFigure(NH2_path, hdu=0, figure=fig, subplot=(2, 2, 1))
    ax.show_colorscale(cmap='gist_yarg')
    ax.add_colorbar()  # Add colorbar
    ax.colorbar.set_location('top')  # Set colorbar position
    ax.colorbar.set_axis_label_text(r'$N_{H_{2}}\,\,\, (cm^{-2})$')  # Set colorbar label
    ax.colorbar.set_axis_label_font(size='x-large')  # Set colorbar label font size
    ax.tick_labels.set_font(size='small')  # Set tick label font size
    ax.axis_labels.set_font(size='small')  # Set axis label font size

    # Display the log(NH2) map
    ax = aplpy.FITSFigure(logNH2_path, hdu=0, figure=fig, subplot=(2, 2, 2))
    ax.show_colorscale(cmap='gist_yarg')
    ax.add_colorbar()  # Add colorbar
    ax.colorbar.set_location('top')  # Set colorbar position
    ax.colorbar.set_axis_label_text(r'$logN_{H_{2}}\,\,\, (cm^{-2})$')  # Set colorbar label
    ax.colorbar.set_axis_label_font(size='x-large')  # Set colorbar label font size
    ax.tick_labels.set_font(size='small')  # Set tick label font size
    ax.axis_labels.set_font(size='small')  # Set axis label font size

    # Display the temperature map
    ax2 = aplpy.FITSFigure(temp_path, hdu=0, figure=fig, subplot=(2, 2, 3))
    ax2.show_colorscale(cmap='gist_yarg')
    ax2.add_colorbar()  # Add colorbar
    ax2.colorbar.set_location('top')  # Set colorbar position
    ax2.colorbar.set_axis_label_text(r'$T\,\,\, (K)$')  # Set colorbar label
    ax2.colorbar.set_axis_label_font(size='x-large')  # Set colorbar label font size
    ax2.tick_labels.set_font(size='small')  # Set tick label font size
    ax2.axis_labels.set_font(size='small')  # Set axis label font size

    # Adjust the layout
    plt.subplots_adjust(hspace=0.5)  # Increase vertical spacing between subplots
    plt.tight_layout(pad=3.0)  # Increase padding in the figure
    plt.show()


def calculate_cloud_mass(pixel_size_arcsec, distance_pc):
    """
        Purpose:
        Calculate the total mass of a molecular cloud using the column density data from a FITS file.
        The mass is computed based on the pixel size, distance to the cloud, and hydrogen mass.
        
        Input:
        pixel_size_arcsec  [float]   : The size of each pixel in arcseconds
        distance_pc        [float]   : The distance to the cloud in parsecs
        """
    # Convert arcseconds to radians
    fits_file = target_folder + f"/{sourcename}_{instrument}_NH2.fits"  # Replace with your FITS file path
    with fits.open(fits_file) as hdul:
        N_H2_array = hdul[0].data  # Read data from the first extension

    pixel_size_rad = pixel_size_arcsec * (np.pi / (180 * 3600))  # From arcseconds to radians
    
    # Calculate the linear size of each pixel at the given distance (cm)
    distance_cm = distance_pc * 3.086e18  # 1 pc = 3.086e18 cm
    pixel_size_cm = distance_cm * pixel_size_rad  # cm

    # Area of each pixel
    area_per_pixel = pixel_size_cm ** 2  # cm²

    # Calculate total number of hydrogen molecules
    N_tot = np.sum(N_H2_array)  # Sum up the column density of all pixels
    total_mass = N_tot * area_per_pixel * m_H2  # Total mass (g)

    # Convert to solar masses
    mass_in_solar_masses = total_mass / M_sun
    print(f"Total mass: {mass_in_solar_masses:.2f} M☉")
