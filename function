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


def convert_jy_per_pixel_to_Mjy_per_sr(
                                       value,
                                       pixel_size
                                       ):
    """
        Purpose:
        Convert flux density from Jy/pixel to MJy/sr.

        Input:
        value        [float or Quantity]: The flux density in Jy/pixel.
        pixel_size   [float]: The size of the pixel in degrees.

        Return:
        result       [float]: The flux density converted to MJy/sr.
        """

    # Convert the input value from Jy/pixel to MJy/pixel
    value = value * u.Jy / u.pixel
    value = value.to(u.MJy / u.pixel)

    # Calculate the solid angle of the pixel (in steradians)
    pixel_angle = pixel_size * u.deg
    pixel_solid_angle = (pixel_angle.to(u.rad)) ** 2

    # Convert the value from MJy/pixel to MJy/sr
    result = value / pixel_solid_angle

    return result.value


    
def conv_u(
           hdul,
           tunit,
           ucfile,
           pixel,
           target_folder,
           filelist_uc
           ):
    """
        Purpose:
        Convert the data units in a FITS file and save the modified file.

        Input:
        hdul         [HDUList]: The HDU list of the input FITS file.
        tunit        [str]: The new unit to set in the header.
        ucfile       [str]: The original filename of the FITS file (for generating new filename).
        pixel        [float]: The pixel size in degrees for unit conversion.
        target_folder[str]: The folder to save the converted FITS file.
        filelist_uc  [list]: List to store the paths of the converted files.
        """

    # Retrieve data and header information from the HDU list
    data = hdul['image'].data
    header = hdul['image'].header

    # Convert data units from Jy/pixel to MJy/sr
    data_converted = convert_jy_per_pixel_to_Mjy_per_sr(data, pixel)

    # Convert error data units from Jy/pixel to MJy/sr
    error_data = hdul['error'].data
    error_data_converted = convert_jy_per_pixel_to_Mjy_per_sr(error_data, pixel)

    # Update the header with the new unit
    header['BUNIT'] = tunit

    # Create a new HDU list with the converted data
    hdu_new = fits.HDUList([
        fits.PrimaryHDU(data=data_converted, header=header),
        fits.ImageHDU(data=error_data_converted, name='stDev')
    ])

    # Generate the new filename and path
    base_filename, _ = os.path.splitext(os.path.basename(ucfile))
    new_filename = f"{base_filename}_unit_convert.fits"
    new_file_path = os.path.join(target_folder, new_filename)

    # Append the new file path to the list and write the new FITS file
    filelist_uc.append(new_file_path)
    hdu_new.writeto(new_file_path, overwrite=True)

    
# Constants
FWHM_TO_SIGMA = 1. / np.sqrt(8 * np.log(2))
c = 3.0e10  # Speed of light in cm/s
h = 6.626e-27  # Planck's constant in erg*s
h_k = h / 1.38e-16  # Planck's constant divided by Boltzmann constant
kappa_1000 = 0.1  # Example value for opacity at 1000 microns

def convolve_data(
                  data,
                  target_resolution,
                  native_beamsize,
                  pixel_size
                  ):
    """
        Purpose:
        Convolve input data with a Gaussian kernel.

        Input：
        data               (np.ndarray): Input data to be convolved.
        target_resolution  (float): Target resolution in arcseconds.
        native_beamsize    (float): Native beam size in arcseconds.
        pixel_size         (float): Pixel size in arcseconds.

        Returns:
        np.ndarray: Convolved data with NaN values replaced by 0.
        """
   # Check the input data type
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    
    # Calculate the size of the convolution kernel
    kernel_size = np.sqrt((target_resolution * FWHM_TO_SIGMA) ** 2 - (native_beamsize * FWHM_TO_SIGMA) ** 2)
    pixel_n = kernel_size / pixel_size
    
    # Ensure the convolution kernel size is reasonable
    if pixel_n <= 0:
        raise ValueError("Calculated pixel_n must be greater than zero.")

    # Create a Gaussian convolution kernel
    gauss_kernel = Gaussian2DKernel(pixel_n)

    # Handle NaN values using a masked array
    masked_data = np.ma.masked_invalid(data)  # Mask NaN values

    # Perform convolution operation using the masked array
    convolved_data = convolve(masked_data, gauss_kernel, normalize_kernel=True, boundary='fill', fill_value=0)

    # Replace NaN values with 0
    convolved_data_filled = np.nan_to_num(convolved_data, nan=0.0)
    
    return convolved_data_filled


def reproject_data(
                   hdu,
                   header_target
                   ):
    """
        Purpose:
        Reproject input data to a target header.

        Parameters:
        hdu: Input data in HDU format.
        header_target: Target header to reproject to.

        Returns:
        np.ndarray: Reprojected data.
        """
    data_reprojected = reproject_interp(hdu, header_target, return_footprint=False)
    
    return data_reprojected


def SED(
        lamda,
        T,
        N
        ):
    """
        Purpose:
        Calculate the spectral energy distribution (SED) at a given wavelength and temperature.

        Parameters:
        lamda (float array): Wavelength in microns.
        T     (float): Temperature in Kelvin.
        N     (float): Optical depth or column density.

        Returns:
        sed   (float array): The calculated SED in units of MJy/sr.
        """
    beta = 1.8  # Exponent for opacity
    nu = c / (lamda * 1e-6)  # Frequency in Hz
    kappa = kappa_1000 * (nu / (1000 * 1e9)) ** beta  # Opacity
    sed = (1e20 * 2 * h * nu**3 / c**2) * (1 - np.exp(-kappa * N)) / (np.exp(h_k * nu / T) - 1)  # SED in MJy/sr
    
    return sed
