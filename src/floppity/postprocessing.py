import numpy as np
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