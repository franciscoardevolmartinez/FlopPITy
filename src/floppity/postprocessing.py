import numpy as np
import numpy.typing as npt
import scipy.interpolate as sci
from scipy.ndimage import gaussian_filter1d

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

def add_bb(temperature_array, wvl, flux_array):

    """
    Add a blackbody component to each spectrum.

    Parameters
    ----------
    temperature_array : array_like, shape (n_spectra,)
        Temperatures (in Kelvin) for the blackbody component to add to each spectrum.
    wvl : array_like, shape (n_wvl,)
        Wavelength grid in nanometers.
    flux_array : array_like, shape (n_spectra, n_wvl)
        Input flux spectra.

    Returns
    -------
    modified_flux : array_like, shape (n_spectra, n_wvl)
        Flux after adding the blackbody component.
    """
    h = 6.626e-34  # Planck's constant (Joule seconds)
    c = 3.0e8      # Speed of light (meters per second)
    k = 1.381e-23  # Boltzmann constant (Joule per Kelvin)

    # Convert wavelength from nanometers to meters
    wvl_m = wvl * 1e-9

    # Initialize the modified flux array
    modified_flux = np.empty_like(flux_array)

    for i, T in enumerate(temperature_array):
        # Calculate blackbody radiation using Planck's law
        bb_flux = (2 * h * c**2 / wvl_m**5) / (np.exp(h * c / (wvl_m * k * T)) - 1)

        # Add the blackbody component to the original flux
        modified_flux[i] = flux_array[i] + bb_flux

    return modified_flux

def convolve_ppre(resel, wvl_obs, flux_array, wvl_mod):
    """
    python translation from brewster

    Convolve a model spectrum to the resolution of an observed spectrum.

    Parameters
    ----------
    obspec : array_like, shape (2, Nobs)
        Observed spectrum (wavelength grid in row 0, flux in row 1).
        Only the wavelength grid is used.
    modspec : array_like, shape (2, Nmod)
        Model spectrum (wavelength in row 0, flux in row 1).
    resel : float
        Pixels per resolution element (e.g. 2.0 for Nyquist sampled).

    Returns
    -------
    Fratio_int : ndarray, shape (Nobs,)
        Model spectrum convolved and sampled onto observed grid.
    """
    #as a start both obs and model have same wvl grid, generalize in future?
    wlobs = wvl_obs
    wlmod = wvl_mod
    Fmod  = flux_array

    Fratio_int = np.zeros(len(wvl_obs))

    # local wavelength spacing (delta λ)
    delta = np.zeros(len(wvl_obs))
    delta[1:-1] = 0.5 * (np.abs(wlobs[1:-1] - wlobs[0:-2]) +
                         np.abs(wlobs[2:]   - wlobs[1:-1]))
    delta[0] = delta[1]
    delta[-1] = delta[-2]

    # loop over observed wavelengths
    for i in range(len(wvl_obs)):
        sigma = delta[i] * resel / 2.355  # Gaussian sigma
        gauss = np.exp(-(wlmod - wlobs[i])**2 / (2 * sigma**2))
        gauss /= gauss.sum()
        Fratio_int[i] = np.sum(gauss * Fmod)

    return Fratio_int

def convolve(spread, wvl, flux_array, type='resel', **kwargs):
    if type=='resel':
        new_flux = convolve_resel(wvl, flux_array, spread, **kwargs)
    elif type=='R':
        new_flux = convolve_R( flux_array, wvl, spread, **kwargs)
    else:
        raise Exception(f'Non-existent convolution type: {type}.')

def convolve_resel(wl, flux, resel, w=4.0):
    """
    Convolve a spectrum with a Gaussian of variable width.
    Model and obs share the same wavelength grid (wl),
    but wavelength spacing is non-uniform.

    Parameters
    ----------
    wl : (N,) array
        Wavelength grid (same for obs & model), in microns or nm.
    flux : (N,) array
        Model flux on the same grid.
    resel : float
        Pixels per resolution element (e.g. 2.2).
    w : float
        Half-width of Gaussian window in units of sigma (default 4).

    Returns
    -------
    flux_conv : (N,) array
        Model flux convolved to the resolution.
    """
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    N = wl.size
    flux_conv = np.empty(N)

    # local Δλ (edge-extended)
    delta = np.empty(N)
    delta[1:-1] = 0.5 * (np.abs(wl[1:-1] - wl[:-2]) + np.abs(wl[2:] - wl[1:-1]))
    delta[0] = delta[1]
    delta[-1] = delta[-2]

    # loop over each pixel
    for i in range(N):
        sigma = delta[i] * resel / 2.355
        left = wl[i] - w * sigma
        right = wl[i] + w * sigma

        a = np.searchsorted(wl, left, side="left")
        b = np.searchsorted(wl, right, side="right")

        dx = wl[a:b] - wl[i]
        weights = np.exp(-0.5 * (dx / sigma) ** 2)

        s = weights.sum()
        if s > 0:
            flux_conv[i] = np.dot(weights, flux[a:b]) / s
        else:
            flux_conv[i] = np.nan

    return flux_conv

def convolve_dont(resel, wvl, flux_array):

    """
    Convolve a model spectrum to the desired resolution, 
    assuming constant resel and same wavelength grid as observation.

    Parameters
    ----------
    modspec : array_like, shape (2, N)
        Model spectrum [wavelength, flux].
    resel : float
        Pixels per resolution element.

    Returns
    -------
    flux_conv : ndarray, shape (N,)
        Convolved flux on the same wavelength grid.
    """

    # sigma in pixels: FWHM = resel → sigma = resel / 2.355
    sigma_pix = resel / 2.355

    flux_conv = gaussian_filter1d(flux_array, sigma=sigma_pix, mode="nearest")

    return flux_conv

def convolve_R(model_flux, model_wl, R, obs_wl):

    """
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

def gaussian_weights_running(sigmas: npt.NDArray, truncate: float = 4.0) -> npt.NDArray:
    """
    From pRT
    Compute 1D Gaussian convolution kernels for an array of standard deviations.

    Based on scipy.ndimage gaussian_filter1d and _gaussian_kernel1d.

    Args:
        sigmas:
            Standard deviations for Gaussian kernel.
        truncate:
            Truncate the filter at this many standard deviations.

    Returns:

    """
    # Make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * np.max(sigmas) + 0.5)

    x = np.arange(-radius, radius + 1)
    sd = np.tile(sigmas, (x.size, 1)).T

    phi_x = np.exp(-0.5 / sd ** 2 * x ** 2)

    return np.transpose(phi_x.T / phi_x.sum(axis=1))


@staticmethod
def _convolve_running(input_wavelengths, input_spectrum, convolve_resolving_power, input_resolving_power=None,
                        **kwargs):
    """
    From pRT
    Convolve a spectrum to a new resolving power.
    The spectrum is convolved using Gaussian filters with a standard deviation
        std_dev = R_in(lambda) / R_new(lambda) * input_wavelengths_bins.
    Both the input resolving power and output resolving power can vary with wavelength.
    The input resolving power is given by:
        lambda / Delta_lambda
    where lambda is the center of a wavelength bin and Delta_lambda is the difference between the edges of the bin.

    The weights of the convolution are stored in a (N, M) matrix, with N being the size of the input, and M the size
    of the convolution kernels.
    To speed-up calculations, a matrix A of shape (N, M) is built from the inputs such as:
        A[i, :] = s[i - M/2], s[i - M/2 + 1], ..., s[i - M/2 + M],
    with s the input spectrum.
    The definition of the convolution C of s by constant weights with wavelength is:
        C[i] = sum_{j=0}^{j=M-1} s[i - M/2 + j] * weights[j].
    Thus, the convolution of s by weights at index i is:
        C[i] = sum_{j=0}^{j=M-1} A[i, j] * weights[i, j].

    Args:
        input_wavelengths: (cm) wavelengths of the input spectrum
        input_spectrum: input spectrum
        convolve_resolving_power: resolving power of output spectrum
        input_resolving_power: if not None, skip its calculation using input_wavelengths

    Returns:
        convolved_spectrum: the convolved spectrum at the new resolving power
    """
    _ = kwargs  # kwargs is not used in this function, this is intended

    if input_resolving_power is None:
        input_resolving_power = SpectralModel.compute_bins_resolving_power(input_wavelengths)

    sigma_lsf_gauss_filter = input_resolving_power / convolve_resolving_power / (2 * np.sqrt(2 * np.log(2)))
    weights = gaussian_weights_running(sigma_lsf_gauss_filter)

    input_length = weights.shape[1]
    central_index = int(input_length / 2)

    # Create a matrix
    input_matrix = np.moveaxis(
        np.array([np.roll(input_spectrum, i - central_index, axis=-1) for i in range(input_length)]),
        0,
        -1
    )

    convolved_spectrum = np.sum(input_matrix * weights, axis=-1)
    n_dims_non_wavelength = len(input_spectrum.shape[:-1])

    # Replace non-valid convolved values by non-convolved values (inaccurate but better than 'reflect' or 0 padding)
    for i in range(input_length):
        if i - central_index < 0:
            ind = np.arange(0, central_index - i, dtype=int)

            for j in range(n_dims_non_wavelength):
                ind = np.expand_dims(ind, axis=0)

            np.put_along_axis(
                convolved_spectrum,
                ind,
                np.take_along_axis(
                    input_spectrum,
                    ind,
                    axis=-1
                ),
                axis=-1
            )
        elif i - central_index > 0:
            ind = -np.arange(1, i - central_index + 1, dtype=int)

            for j in range(n_dims_non_wavelength):
                ind = np.expand_dims(ind, axis=0)

            np.put_along_axis(
                convolved_spectrum,
                ind,
                np.take_along_axis(
                    input_spectrum,
                    ind,
                    axis=-1
                ),
                axis=-1
            )

    return convolved_spectrum
