import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.utils.data import get_pkg_data_filename
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


"""
This code generates the cutout images of target galaxies for the raw Swift/UVOT filter data.
The cutouts generated are for the counts, the counts error and the exposure for the target galaxy.
"""

#define paths
path_to_txt_file = '/Users/Nikhil/code/Newtext/' #Path to the text file that contains all MaNGA Ids for the specific filter
path_to_MaNGA_drpall = '/Users/Nikhil/Data/MaNGAPipe3D/Newmanga/' #Path to the MaNGA DRPALL file
path_to_MaNGA_hybrid_Maps = '/Volumes/Nikhil/MPL-7_Files/HYB10-GAU-MILESHC/' #Path to MaNGA HYB10 MAPS file
path_to_Swift_raw_data = '/Volumes/Nikhil/SWIFT/' #Path to a folder that contains raw Swift counts, error and exposure maps
path_to_store_output_files = '/Volumes/Nikhil/SWIFT/New_cutout/' #Path to store the cutout images

with open(path_to_txt_file+'W1_ID.txt') as f:
   Line = [line.rstrip('\n') for line in open(path_to_txt_file+'W1_ID.txt')]

for id in range (0,np.shape(Line)[0]):
    #path to MaNGA drpall file
    drpall = fits.open(path_to_MaNGA_drpall+'drpall-v2_3_1.fits')
    tbdata = drpall[1].data
    ind = np.where(tbdata['mangaid'] == Line[id])
    objectra = tbdata['objra'][ind][0]
    objectdec = tbdata['objdec'][ind][0]
    redshift = tbdata['nsa_z'][ind][0]
    plate = tbdata['plate'][ind][0]
    ifu = tbdata['ifudsgn'][ind][0]

    hdu = fits.open(path_to_MaNGA_hybrid_Maps+str(plate)+'/'+str(ifu)+'/manga-'+str(plate)+'-'+str(ifu)+'-MAPS-HYB10-GAU-MILESHC.fits.gz')
    GFLUX = hdu['EMLINE_GFLUX'].data
    S_cutout = (int(np.shape(GFLUX)[1]/2)+40,int(np.shape(GFLUX)[1]/2)+40)
    hdu_counts = fits.open(get_pkg_data_filename(path_to_Swift_raw_data+'CNTS/m'+str(Line[id])+'-w1d-cts.fits'))[0]
    hdu_exp = fits.open(get_pkg_data_filename(path_to_Swift_raw_data+'EXP/m'+str(Line[id])+'-w1d-ex-dc.fits'))[0]
    hdu_err = fits.open(get_pkg_data_filename(path_to_Swift_raw_data+'ERR/m'+str(Line[id])+'-w1d-cts-err.fits'))[0]
    wcs = WCS(hdu_counts.header)
    P1,P2 = wcs.wcs_world2pix((objectra),(objectdec),0)
    cutout_counts = Cutout2D(hdu_counts.data, position=(P1,P2), size=S_cutout, wcs=wcs)
    cutout_err = Cutout2D(hdu_err.data, position=(P1,P2), size=S_cutout, wcs=wcs)
    cutout_exp = Cutout2D(hdu_exp.data, position=(P1,P2), size=S_cutout, wcs=wcs)
    
    
    hdu0 = fits.PrimaryHDU(cutout_counts.data)
    hdu0.header.update(cutout_counts.wcs.to_header())
    hdu1 = fits.ImageHDU(cutout_err.data)
    hdu2 = fits.ImageHDU(cutout_exp.data)
    new_hdul = fits.HDUList([hdu0,hdu1,hdu2])
    new_hdul.writeto(path_to_store_output_files+'UVW1_cutout_'+str(Line[id])+'.fits')

