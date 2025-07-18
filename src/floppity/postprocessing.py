import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.interpolate as sci

def vrot(v_array, wvl, spectrum_array, eps=0.6, nr=10, ntheta=100, dif=0.0):
    """
    Apply rotational broadening to multiple spectra.

    Parameters
    ----------
    v_array : array_like, shape (n_spectra,)
        Projected rotational velocities (km/s) for each spectrum.
    wvl : array_like, shape (n_wvl,)
        Wavelength grid.
    spectrum_array : array_like, shape (n_spectra, n_wvl)
        Input spectra to be rotationally broadened.
    eps : float, optional
        Limb darkening coefficient (default: 0.6).
    nr : int, optional
        Number of radial bins (default: 10).
    ntheta : int, optional
        Azimuthal bins in outer annulus (default: 100).
    dif : float, optional
        Differential rotation coefficient (default: 0.0).

    Returns
    -------
    broadened_array : array_like, shape (n_spectra, n_wvl)
        Rotationally broadened spectra.
    """
    broadened_array = np.zeros_like(spectrum_array)

    for i, (v, spectrum) in enumerate(zip(v_array, spectrum_array)):
        ns = np.zeros_like(spectrum)
        tarea = 0.0
        dr = 1.0 / nr

        for j in range(nr):
            r = dr / 2.0 + j * dr
            nphi = int(ntheta * r)
            area = ((r + dr/2.0)**2 - (r - dr/2.0)**2) / nphi * (1.0 - eps + eps * np.cos(np.arcsin(r)))

            for k in range(nphi):
                th = np.pi / nphi + k * 2.0 * np.pi / nphi

                if dif != 0:
                    vl = v * r * np.sin(th) * (1.0 - dif/2.0 - dif/2.0 * np.cos(2.0 * np.arccos(r * np.cos(th))))
                else:
                    vl = v * r * np.sin(th)

                shifted = wvl + wvl * vl / 2.9979e5  # Doppler shift
                ns += area * np.interp(shifted, wvl, spectrum)
                tarea += area

        broadened_array[i] = ns / tarea

    return broadened_array

def RV(v_array, wvl, flux_array, edgeHandling='firstlast', fillValue=None):
    """
    Doppler shift multiple spectra given an array of velocities.

    Parameters
    ----------
    v_array : array_like, shape (n_spectra,)
        Doppler velocities in km/s for each spectrum.
    wvl : array_like, shape (n_wvl,)
        Shared input wavelength grid.
    flux_array : array_like, shape (n_spectra, n_wvl)
        Flux values for each spectrum.
    edgeHandling : str, optional
        'firstlast' (default) or 'fillValue'.
    fillValue : float, optional
        Value to use if edgeHandling='fillValue'.

    Returns
    -------
    nflux_array : array_like, shape (n_spectra, n_wvl)
        Doppler-shifted fluxes, resampled onto the original wavelength grid.
    """
    cvel = 299_792.458  # speed of light in km/s
    n_spectra, n_wvl = flux_array.shape
    nflux_array = np.empty_like(flux_array)

    for i in range(n_spectra):
        v = v_array[i]
        flux = flux_array[i]

        # Shifted wavelength
        wlprime = wvl * (1.0 + v / cvel)

        # Set interpolation fill value
        fv = np.nan if edgeHandling != "fillValue" else fillValue

        # Interpolate shifted flux back to original wavelength grid
        nflux = sci.interp1d(wlprime, flux, bounds_error=False, fill_value=fv)(wvl)

        if edgeHandling == "firstlast":
            nin = ~np.isnan(nflux)
            if not nin[0]:
                fvindex = np.argmax(nin)
                nflux[:fvindex] = nflux[fvindex]
            if not nin[-1]:
                lvindex = -np.argmax(nin[::-1]) - 1
                nflux[lvindex + 1:] = nflux[lvindex]

        nflux_array[i] = nflux

    return nflux_array

def offset(offset_array, wvl, flux_array):
    """
    Apply an additive offset to each spectrum.

    Parameters
    ----------
    offset_array : array_like, shape (n_spectra,)
        Offsets to apply to each spectrum.
    wvl : array_like, shape (n_wvl,)
        Wavelength array (not used in calculation, included for consistency).
    flux_array : array_like, shape (n_spectra, n_wvl)
        Input flux spectra.

    Returns
    -------
    shifted_flux : array_like, shape (n_spectra, n_wvl)
        Flux after applying offsets.
    """
    offset_array = np.asarray(offset_array).reshape(-1, 1)  # shape: (n_spectra, 1)
    return flux_array + offset_array

def scaling(scaling_array, wvl, flux_array):
    """
    Apply a multiplicative scaling to each spectrum.

    Parameters
    ----------
    scaling_array : array_like, shape (n_spectra,)
        Scaling factors to apply to each spectrum.
    wvl : array_like, shape (n_wvl,)
        Wavelength array (not used in calculation, included for consistency).
    flux_array : array_like, shape (n_spectra, n_wvl)
        Input flux spectra.

    Returns
    -------
    scaled_flux : array_like, shape (n_spectra, n_wvl)
        Flux after applying scaling.
    """
    scaling_array = np.asarray(scaling_array).reshape(-1, 1)  # shape: (n_spectra, 1)
    return flux_array * scaling_array

def wvl_offset(offset_array, wvl, flux_array, edgeHandling='firstlast', fillValue=None):
    """
    Apply a wavelength offset to each spectrum, shifting the fluxes but keeping the original wavelength axis.

    Parameters
    ----------
    offset_array : array_like, shape (n_spectra,)
        Wavelength offsets to apply to each spectrum.
    wvl : array_like, shape (n_wvl,)
        Original wavelength grid.
    flux_array : array_like, shape (n_spectra, n_wvl)
        Input flux spectra.
    edgeHandling : str, optional
        'firstlast' (default) or 'fillValue'.
    fillValue : float, optional
        Value to use if edgeHandling='fillValue'.

    Returns
    -------
    shifted_flux : array_like, shape (n_spectra, n_wvl)
        Flux after applying wavelength offsets.
    """
    n_spectra, n_wvl = flux_array.shape
    shifted_flux = np.empty_like(flux_array)

    for i in range(n_spectra):
        offset = offset_array[i]
        flux = flux_array[i]

        # Shifted wavelength
        wlprime = wvl + offset

        # Set interpolation fill value
        fv = np.nan if edgeHandling != "fillValue" else fillValue

        # Interpolate shifted flux back to original wavelength grid
        shifted = sci.interp1d(wlprime, flux, bounds_error=False, fill_value=fv)(wvl)

        if edgeHandling == "firstlast":
            nin = ~np.isnan(shifted)
            if not nin[0]:
                fvindex = np.argmax(nin)
                shifted[:fvindex] = shifted[fvindex]
            if not nin[-1]:
                lvindex = -np.argmax(nin[::-1]) - 1
                shifted[lvindex + 1:] = shifted[lvindex]

        shifted_flux[i] = shifted

    return shifted_flux

def instrumental_broadening(broadening, wvl, flux_array, **kwargs):

    """
    Apply instrumental broadening to the spectrum using a Gaussian kernel.

    Parameters
    ----------
    broadening : float
        Full width at half maximum (FWHM) of the Gaussian kernel in wavelength units.
    wvl : array_like, shape (n_wvl,)
        Wavelength grid.
    flux_array : array_like, shape (n_spectra, n_wvl)
        Input flux spectra.
    kwargs : dict, optional
        Additional arguments for customization.

    Returns
    -------
    broadened_flux : array_like, shape (n_spectra, n_wvl)
        Flux after applying instrumental broadening.
    """

    # Convert FWHM to standard deviation for Gaussian kernel
    sigma = broadening / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Apply Gaussian smoothing to each spectrum
    broadened_flux = np.array([gaussian_filter1d(flux, sigma / np.mean(np.diff(wvl))) for flux in flux_array])

    return broadened_flux

def add_bb(T, A, wvl, flux):
    """
    Add a blackbody contribution to a spectrum.

    Parameters
    ----------
    T : float
        Temperature of the blackbody in Kelvin.
    A : float
        Scaling factor for the blackbody contribution.
    wvl : array_like, shape (n_wvl,)
        Wavelength grid in microns.
    flux : array_like, shape (n_wvl,)
        Input flux spectrum in Jansky.

    Returns
    -------
    modified_flux : array_like, shape (n_wvl,)
        Flux spectrum with the blackbody contribution added.
    """
    # Constants
    h = 6.62607015e-34  # Planck's constant (Joule second)
    c = 2.99792458e8    # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (Joule/Kelvin)

    # Convert wavelength from microns to meters
    wvl_m = wvl * 1e-6

    # Planck's law for blackbody radiation
    bb_flux = (2.0 * h * c**2) / (wvl_m**5) / (np.exp((h * c) / (wvl_m * k * T)) - 1.0)

    # Convert blackbody flux to Jansky (1 Jansky = 1e-26 W/m^2/Hz)
    bb_flux_jy = bb_flux * 1e26 * (wvl_m**2 / c)

    # Scale the blackbody flux and add to the input flux
    modified_flux = flux + A * bb_flux_jy

    return modified_flux

def conv_non_uniform_R(R, model_flux, model_wl,  obs_wl):
    """
    From brewster v2 (Fei Wang)

    Convolve a model spectrum with a wavelength-dependent resolving power 
    onto the observed wavelength grid ???

    Parameters:
    - model_flux: 1D array of model flux values.
    - model_wl: 1D array of model wl values.
    - obs_wl: 1D array of observed wl values.
    - R: 1D array of resolving power values (for the obs_wl grid.)

    Returns:
    - convolved_flux: 1D array of convolved flux values on the obs_wl grid.
    """
    # create the array for the convolved flux
    convolved_flux = np.zeros_like(obs_wl)

    for i, wl_center in enumerate(obs_wl): 
        
        # compute FWHM and sigma for each wl
        # print('wl_center', wl_center)
        # print('R[i]', R[i])
        
        fwhm = wl_center / R[i]
        # print('fwhm', fwhm)
        sigma = fwhm / 2.355


        # compute the Gaussian kernel for the current wl
       
        gaussian_kernel = np.exp(-((model_wl-wl_center) ** 2) / (2 * sigma **2))
        #print('gaussian_kernel before normalisation', gaussian_kernel)

        # normalisation
        gaussian_kernel /= np.sum(gaussian_kernel)
        # print('gaussian_kernel after normalisation', gaussian_kernel)



        # apply the kernel to the flux
        convolved_flux[i] = np.sum(model_flux * gaussian_kernel)
    
    return convolved_flux