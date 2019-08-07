import numpy as np

import astropy.table as t
import astropy.io.fits as fits
from astropy import (units as u, constants as c,
                     coordinates as coords, table as t,
                     wcs, cosmology as cosmo)

import os, sys

from glob import glob

from matplotlib import gridspec, colors
import matplotlib.ticker

import copy

DAP_MPL_versions = {'MPL-8': '2.3.0'}
DRP_MPL_versions = {'MPL-8': 'v2_5_3'}  
                
SAS_BASE_DIR = os.environ['SAS_BASE_DIR']

CosmoModel = cosmo.WMAP9

# =====
# units
# =====

# Maggy (SDSS flux unit)
Mgy = u.def_unit(
    s='Mgy', represents=3631. * u.Jy,
    doc='SDSS flux unit', prefixes=True)

spec_unit = 1e-17 * u.erg / u.s / u.cm**2. / u.AA
l_unit = u.AA
bandpass_sol_l_unit = u.def_unit(
    s='bandpass_solLum', format={'latex': r'\overbar{\mathcal{L}_{\odot}}'},
    prefixes=False)
m_to_l_unit = u.def_unit(
    s='mass_to_light', represents=u.Msun / bandpass_sol_l_unit,
    format={'latex': r'\frac{M_{\odot}}{\overbar{\mathcal{L}_{\odot}}}'},
    prefixes=False)

u.add_enabled_units([bandpass_sol_l_unit, m_to_l_unit, Mgy])

fits_formats = ['fits', 'fits.gz']

def read_datacube(fname):
    hdu = fits.open(fname)
    return hdu

def shuffle_table(tab):
    a = np.arange(len(tab))
    np.random.shuffle(a)
    tab = tab[a]
    return tab

def mask_from_maskbits(a, b=[10]):
    '''
    transform a maskbit array into a cube or map mask
    '''

    # assume 32-bit maskbits
    a = a.astype('uint32')
    n = 32
    m = np.bitwise_or.reduce(
        ((a[..., None] & (1 << np.arange(n))) > 0)[..., b], axis=-1)

    return m

class MaNGA_DNE_Error(Exception):
    '''
    generic file access does-not-exist
    '''

class IFU_DNE_Error(MaNGA_DNE_Error):
    '''
    generic IFU does-not-exist error, for both DRP & DAP
    '''

    def __init__(self, plate, ifu, fullpath, type_of_data='unspecified'):
        self.plate = plate
        self.ifu = ifu
        self.fullpath = fullpath
        self.type_of_data = type_of_data

    def __str__(self):
        return '{} product for {}-{} DNE in given location {}'.format(
            self.type_of_data, self.plate, self.ifu, self.fullpath)


class DRP_IFU_DNE_Error(IFU_DNE_Error):
    '''
    DRP IFU does not exist
    '''
    type_of_data = 'DRP'


class DAP_IFU_DNE_Error(IFU_DNE_Error):
    '''
    DAP IFU does not exist
    '''
    type_of_data = 'DAP'

def load_drpall(mpl_v, index=None, drptype='manga'):
    if drptype.lower() == 'manga':
        hdu = 1
    elif drptype.lower() == 'mastar':
        hdu = 2
    fname = os.path.join(
        SAS_BASE_DIR, 'mangawork/manga/spectro/redux', DRP_MPL_versions[mpl_v],
        'drpall-{}.fits'.format(DRP_MPL_versions[mpl_v]))
    tab = t.Table.read(fname, hdu=hdu)

    if index is not None:
        tab.add_index(index)

    return tab

def load_dapall(mpl_v, index=None):
    drpver, dapver = DRP_MPL_versions[mpl_v], DAP_MPL_versions[mpl_v]
    fname = os.path.join(
        SAS_BASE_DIR, 'mangawork/manga/spectro/analysis', drpver, dapver,
        'dapall-{}-{}.fits'.format(drpver, dapver))
    tab = t.Table.read(fname)

    if index is not None:
        tab.add_index(index)

    return tab

def load_fits_prefer_uncompress(fnamebase, possible_extensions, type_of_data='unspecified'):
    '''load FITS file with a base name, preferring one extensions over a lits of others
    
    load a FITS file with a base name, but attempt several file extensions in order
    
    Parameters
    ----------
    fnamebase : str
        file name base
        (for "/path/to/file.fits" or "/path/to/file.fits.gz", use "/path/to/file")
    possible_extensions : list of str
        list of possible file extensions
        (for "/path/to/file.fits" or "/path/to/file.fits.gz", use [".fits", "fits.gz"])

    Raises
    ------
    IFU_DNE_Error
        when no valid file found
    '''

    # loop through preferred file formats
    for ff in possible_extensions:
        fname = f'{fnamebase}.{ff}'

        # continue if file not present with current extension
        if not os.path.isfile(fname):
            continue
        # load the first valid file found
        else:
            hdulist = fits.open(fname)
            break
    # if you reach the end of a list without finding a valid file, raise an error
    else:
        raise IFU_DNE_Error(plate, ifu, fname, type_of_data=type_of_data)

    return hdulist


def load_drp_logcube(plate, ifu, mpl_v, file_exts=fits_formats):
    drpver = DRP_MPL_versions[mpl_v]
    plate, ifu = str(plate), str(ifu)

    fnamebase = os.path.join(
            SAS_BASE_DIR, 'mangawork/manga/spectro/redux/', drpver, plate, 'stack/',
            f'manga-{plate}-{ifu}-LOGCUBE')
    hdulist = load_fits_prefer_uncompress(fnamebase, file_exts, 'DRP:LOGCUBE')

    return hdulist

def get_gal_bpfluxes(plate, ifu, mpl_v, bs, th=5.):
    hdulist = load_drp_logcube(plate, ifu, mpl_v)
    snr = np.median(
        hdulist['FLUX'].data * np.sqrt(hdulist['IVAR'].data), axis=0)
    fluxes = np.column_stack(
        [hdulist['{}IMG'.format(band.upper())].data[(snr >= th)]
        for band in bs])
    hdulist.close()
    return fluxes

def get_drp_hdrval(plate, ifu, mpl_v, k):
    hdulist = load_drp_logcube(plate, ifu, mpl_v)
    val = hdulist[0].header[k]
    hdulist.close()
    return val

def hdu_data_extract(hdulist, names):
    return [hdulist[n].data for n in names]

def load_dap_maps(plate, ifu, mpl_v, kind, file_exts=fits_formats):
    drpver, dapver = DRP_MPL_versions[mpl_v], DAP_MPL_versions[mpl_v]
    plate, ifu = str(plate), str(ifu)
    fnamebase = os.path.join(
        SAS_BASE_DIR, 'mangawork/manga/spectro/analysis',
        f'{drpver}/{dapver}/{kind}/{plate}/{ifu}', f'manga-{plate}-{ifu}-MAPS-{kind}')
    
    hdulist = load_fits_prefer_uncompress(fnamebase, file_exts, 'DAP:MAPS')

    return hdulist

def make_key2channel(hdu, axis=0, start=1, channel_key_start='C'):
    '''
    make a channel-to-key dictionary
    '''
    # how many elements? determined by what axis we're partitioning along
    n = hdu.data.shape[axis]

    hdr = hdu.header
    # figure out how to left-pad keys with zeros
    ndigits = len(str(n))
    # the actual array channels must start at zero,
    # so begin by constructing them
    arr_channels = list(range(n))
    # then make the actual header keys by
    # adding `start` argument to each of arr_channels
    hdr_keys = list(map(lambda i: '{}{}'.format(
        channel_key_start, str(i + start).rjust(ndigits, '0')),
        range(n)))
    # now marry the array channels numbers and the channel names
    # in a dict
    key2channel = dict(zip(map(lambda k: hdr[k.strip(' ')], hdr_keys), arr_channels))
    return key2channel
