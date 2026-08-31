# Fixed parameters for this instrument

# TODO: asymmetric plate scale?

import astropy.units as u
from astropy.table import QTable

slit_w_range = [0.1, 10.0] * u.arcsec
slit_h = 60.0 * u.arcsec  # slit height (long direction)

# Limits on 2nd slit mode parameter
slitmodes = {
    "SET": [0.1, 10.0],
    "LOSS": [0.1, 90.0],
    "RES": [252, 8500],  ### Trial and error limits that mitigate solver errors
    "SNR": [10, 100],
    "AUTO": [95],
}  # AUTO is a special case of SNR


telescope_D = 508.0 * u.cm  # Diameter
Obscuration = 0.3  # As a ratio of M1

moffat_beta = 4.765
moffat_theta_factor = (
    0.5 / (2 ** (1.0 / moffat_beta) - 1.0) ** 0.5
)  # theta = factor*seeingFWHM

# Paths to data for this instrument

# Leave CSVdir=None to use the default data in the repo
CSVdir = None  #'/home/developer/Software/ETC/CSV/'
default_waveunit = u.nm  # assume units for all CSV files

skybackground_file = (
    "sky_spectra_palomar-grey.csv"  # Measured 2026-07;  flux units are /Angstrom
)

throughputFile_atm = (
    "atm-extinction-Palomar.csv"  # dimensionless T  (Flux/Flux_above_atmosphere)
)
throughputFile_telescope = "throughput-Palomar-200inch.csv"
throughputFile_slicer = "throughput_slicersides_temp.csv"

# Channel-wise fixed parameters for this instrument; all channel keys must match

channels = ("U", "G", "R", "I")  # use tuple not list to allow function caching

chanConfig = QTable([channels], names=["channel"])
chanConfig.add_index("channel")  # Allows us to specify rows by channel

# chanConfig['channelRange']=[[305.1,443.5], [415.7,593.1], [561.9,793.4], [751.2,1040.5]] * u.nm
chanConfig["channelRange"] = [
    [3051.0, 4435.0],
    [4157.0, 5931.0],
    [5619.0, 7934.0],
    [7512.0, 10405.0],
] * u.AA

# Width of detector (px) in the dispersion direction
chanConfig["Npix_dispers"] = (4114, 4114, 4114, 4114)
chanConfig["dLambda"] = [
    (cr[1] - cr[0]) / npd
    for (cr, npd) in zip(chanConfig["channelRange"], chanConfig["Npix_dispers"])
]

chanConfig["platescale"] = (0.185, 0.185, 0.185, 0.185) * u.arcsec / u.pix

chanConfig["darkcurrent"] = (2.0, 2.0, 2.0, 2.0) * u.count / u.pix / u.hr

chanConfig["readnoise"] = (3.1, 7.3, 3.5, 4.7) * u.count / u.pix  # rms e-

# LSFFile={}  # Wait for data

# FWHM ~ sigma*2.35
# FWHM in pixels: (U, G, R, I) = (2.89, 2.67, 2.94, 2.94)
chanConfig["LSFsigma_px"] = [1.23, 1.14, 1.25, 1.25]  # Measured by C. Fremling, 2026
chanConfig["LSFsigma"] = chanConfig["LSFsigma_px"] * chanConfig["dLambda"]

# Colors for plotting
chanConfig["channelColor"] = ("blue", "green", "red", "magenta")

# Spectrograph Throughput here includes CCD QE
chanConfig["throughputFile_spectrograph"] = (
    "throughput-spectrograph-20260206-U.csv",
    "throughput-spectrograph-20260206-G.csv",
    "throughput-spectrograph-20260206-R.csv",
    "throughput-spectrograph-20260206-I.csv",
)

# Make standalone dicts from the columns in the data table
chanConfigi = chanConfig.loc  # lets us use the index column
for col in chanConfig.colnames:
    globals()[col] = {ch: chanConfigi[ch][col] for ch in channels}
