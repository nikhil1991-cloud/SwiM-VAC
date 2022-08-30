from matplotlib import pyplot as plt
from astropy.io import fits
import scipy.stats as st
import math
from scipy import signal
from astropy import wcs
import sys
from scipy import interpolate
import numpy as np
from numpy import inf


"""
This code uses the SDSS r band images to calculate SDSS aperture corrections to the
Swift/UVOT uvw1,uvw2,uvm2 integrated magnitudes.
"""

#define paths
path_to_txt_file = '/Users/Nikhil/code/Newtext/' #Path to the text file that contains all MaNGA Ids for the SwiM Galaxies
path_to_MaNGA_drpall = '/Users/Nikhil/Data/MaNGAPipe3D/Newmanga/' #Path to the MaNGA DRPALL file
path_to_NSA = '/Users/nikhil/Data/MaNGAPipe3D/Newmanga/' #Path to the NSA ALL file
path_to_SDSS_images = '/Volumes/Nikhil/MPL-7_Files/SDSS/' #Path to SDSS r band images
path_to_store_AP_corr_table = '/Volumes/Nikhil/SWIFT/New_cat/' #Path to store Aperture corrections in fits format

#Define psfs sigma of Swift & SDSS
SwftSigmaw2 = 2.92/2.355
SwftSigmam2 = 2.45/2.355
SwftSigmaw1 = 2.37/2.355
SigmaSloan = 1.4/2.355

#define sigma for gaussian convolution kernels
sigma_kernel_w2 = np.sqrt((SwftSigmaw2)**2 - SigmaSloan**2)
sigma_kernel_m2 = np.sqrt((SwftSigmam2)**2 - SigmaSloan**2)
sigma_kernel_w1 = np.sqrt((SwftSigmaw1)**2 - SigmaSloan**2)

#Define convolution kernel for PSF matching from SDSS to uvw2
def gkern(sdss_scale,target_sigma,SigmaSloan):
    SigmaSloan = (1.4/2.355)
    lsln = math.ceil(4*SigmaSloan)
    sln = np.sqrt((target_sigma)**2 - SigmaSloan**2)
    x_sln1 = np.arange(0,lsln+1,sdss_scale)
    x_sln = np.zeros(np.shape(x_sln1)[0]+np.shape(x_sln1)[0]-1)
    x_sln[10:21] = x_sln1
    x_sln[0:10] = np.flip(x_sln1[1:21])*(-1)
    XS,YS = np.meshgrid(x_sln,x_sln)
    KS = np.exp(-(XS ** 2 + YS ** 2) / (2 * sln ** 2))
    return KS/np.sum(KS)

with open(path_to_txt_file+'ALL_ID.txt') as f: #ALL_ID.txt is a file that contains MaNGA IDs of all SwiM galaxies
   Line = [line.rstrip('\n') for line in open(path_to_txt_file+'ALL_ID.txt')]

APcorr = np.zeros((3,np.shape(Line)[0]))
Rmag_nsa = np.zeros(np.shape(Line)[0])
Rmag_cal = np.zeros(np.shape(Line)[0])

for q in range (0,np.shape(Line)[0]):
    drpall = fits.open(path_to_MaNGA_drpall+'drpall-v2_3_1.fits')
    tbdata = drpall[1].data
    ind = np.where(tbdata['mangaid'] == Line[q])
    objectra = tbdata['objra'][ind][0]
    objectdec = tbdata['objdec'][ind][0]
    redshift = tbdata['nsa_z'][ind][0]
    plate = tbdata['plate'][ind][0]
    ifu = tbdata['ifudsgn'][ind][0]
    sloan = tbdata['nsa_iauname'][ind][0]
    pa = tbdata['nsa_elpetro_phi'][ind][0]
    axs = tbdata['nsa_elpetro_ba'][ind][0]
    ub = tbdata['nsa_elpetro_flux'][ind][0,4]
    nsa_all = drpall = fits.open(path_to_NSA+'nsa_v1_0_1.fits')
    tbdata1 = nsa_all[1].data
    ind1 = np.where(tbdata1['IAUNAME'] == sloan)
    uq = tbdata1['PETRO_FLUX'][ind1][0,4]
    Relp = tbdata1['PETRO_THETA'][ind1][0]
    
    
    hdu = fits.open(path_to_SDSS_images+str(sloan)+'-r.fits')
    r_band = hdu[0].data
    wsln =wcs.WCS(hdu[0].header,naxis=2)
    pix1,pix2 = wsln.wcs_world2pix(objectra,objectdec,1)
    LZ = r_band.shape[0]
    Rx = [0]* (LZ)
    i=0
    while i<(LZ):
          Rx[i]=i
          i=i+1
                
    Ry = [Rx,]*(LZ)
    Ry.reverse()
    X = np.array(Ry)
    Y = np.transpose(Ry)

        
    rho = pa*(math.pi/180)
    thet = math.pi/2
        
    Yr = X*(np.cos(rho)) + Y*(np.sin(rho))
    Xr = (-1)*X*(np.sin(rho)) + Y*(np.cos(rho))
    pix2r = pix1*(np.cos(rho)) + pix2*(np.sin(rho))
    pix1r = (-1)*pix1*(np.sin(rho))+pix2*(np.cos(rho))
        
                        
    li = [0] *(LZ)
    Li = [li,]*(LZ)
    Dist = np.array(Li)
    Dellip = np.array(Li)
    i=0
    while i<(LZ) :
          j=0
          while j<(LZ):
                Dist[i][j] = math.sqrt((X[i][j]-pix1)**2 + (Y[i][j]-pix2)**2)
                Dellip[i][j] = math.sqrt((Xr[i][j]-pix1r)**2 + ((Yr[i][j]-pix2r)**2)/(axs**2))
                j=j+1
          i=i+1


    B1= np.ones(r_band.shape)
    B2= np.ones(r_band.shape)
    B3= np.ones(r_band.shape)
    B4= np.ones(r_band.shape)
    BE= np.ones(r_band.shape)
    B5 = np.zeros(r_band.shape)
    i=0
    while i < (LZ):
          j = 0
          while j < (LZ):
              if Dellip[i,j] < 0.5*(Relp/0.396):
                 B1[i,j] = 0
              j=j+1
          i=i+1
    i=0
    while i < (LZ):
          j = 0
          while j < (LZ):
              if Dellip[i,j] > 0.5*(Relp/0.396) and Dellip[i,j] < (Relp/0.396):
                 B2[i,j] = 0
              j=j+1
          i=i+1
    i=0
    while i < (LZ):
          j = 0
          while j < (LZ):
              if Dellip[i,j] > (Relp/0.396) and Dellip[i,j] < 1.5*(Relp/0.396):
                 B3[i,j] = 0
              j=j+1
          i=i+1

    i=0
    while i < (LZ):
        j = 0
        while j < (LZ):
            if Dellip[i,j] > 1.5*(Relp/0.396) and Dellip[i,j] < 2*(Relp/0.396):
                B4[i,j] = 0
            j=j+1
        i=i+1
    i=0
    while i < (LZ):
          j = 0
          while j < (LZ):
            if Dellip[i,j] < 2*(Relp/0.396):
                BE[i,j] = 0
            j=j+1
          i=i+1

    masked_r_band1 =np.ma.array(r_band,mask=B1)
    masked_r_band2 =np.ma.array(r_band,mask=B2)
    masked_r_band3 =np.ma.array(r_band,mask=B3)
    masked_r_band4 =np.ma.array(r_band,mask=B4)
    
    sum_r_band1 = np.sum(np.ma.array(r_band,mask=B1))
    sum_r_band2 = np.sum(np.ma.array(r_band,mask=B2))
    sum_r_band3 = np.sum(np.ma.array(r_band,mask=B3))
    sum_r_band4 = np.sum(np.ma.array(r_band,mask=B4))
    N_pix1 = len(np.where(B1 == 0)[0])
    N_pix2 = len(np.where(B2 == 0)[0])
    N_pix3 = len(np.where(B3 == 0)[0])
    N_pix4 = len(np.where(B4 == 0)[0])

    i=0
    while i < (LZ):
        j = 0
        while j < (LZ):
                if Dellip[i,j] < 0.5*(Relp/0.396):
                    B5[i,j] = sum_r_band1/N_pix1
                j=j+1
        i=i+1
    i=0
    while i < (LZ):
        j = 0
        while j < (LZ):
                if Dellip[i,j] > 0.5*(Relp/0.396) and Dellip[i,j] < (Relp/0.396):
                    B5[i,j] = sum_r_band2/N_pix2
                j=j+1
        i=i+1
    i=0
    while i < (LZ):
        j = 0
        while j < (LZ):
                if Dellip[i,j] > (Relp/0.396) and Dellip[i,j] < 1.5*(Relp/0.396):
                    B5[i,j] = sum_r_band3/N_pix3
                j=j+1
        i=i+1
    i=0
    while i < (LZ):
        j = 0
        while j < (LZ):
                if Dellip[i,j] > 1.5*(Relp/0.396) and Dellip[i,j] < 2*(Relp/0.396):
                    B5[i,j] = sum_r_band4/N_pix4
                j=j+1
        i=i+1
        
    Masked_isop = np.ma.array(B5,mask=BE)
    Gaussian_w2 = gkern(0.396,SwftSigmaw2,SigmaSloan)
    Gaussian_m2 = gkern(0.396,SwftSigmam2,SigmaSloan)
    Gaussian_w1 = gkern(0.396,SwftSigmaw1,SigmaSloan)
    #uvw2
    W2_Convolved_r_isop = signal.convolve2d(B5,Gaussian_w2,boundary='symm', mode='same')
    Masked_W2_Convolved_r_isop = np.ma.array(W2_Convolved_r_isop,mask=BE)
    #uvm2
    M2_Convolved_r_isop = signal.convolve2d(B5,Gaussian_m2,boundary='symm', mode='same')
    Masked_M2_Convolved_r_isop = np.ma.array(M2_Convolved_r_isop,mask=BE)
    #uvw1
    W1_Convolved_r_isop = signal.convolve2d(B5,Gaussian_w1,boundary='symm', mode='same')
    Masked_W1_Convolved_r_isop = np.ma.array(W1_Convolved_r_isop,mask=BE)
    
    APcorr[0,q] = np.sum(Masked_isop)/np.sum(Masked_W2_Convolved_r_isop)
    APcorr[1,q] = np.sum(Masked_isop)/np.sum(Masked_M2_Convolved_r_isop)
    APcorr[2,q] = np.sum(Masked_isop)/np.sum(Masked_W1_Convolved_r_isop)
    
    Rmag_nsa[q] = 22.5 - 2.5*np.log10(ub)
    Rmag_cal[q] = 22.5 - 2.5*np.log10(np.sum(Masked_isop))


col1 = Line
col2 = APcorr
col3 = Rmag_nsa
col4 = Rmag_cal

c1 = fits.Column(name='MangaID',format='10A', array=col1)
c2 = fits.Column(name='ApertureCorrect(w2,m2,w1)',format='3E', array=col2.T)
c3 = fits.Column(name='r-band Mag nsa',format='E', array=col3)
c4 = fits.Column(name='r-band Mag cal',format='E', array=col4)
hdu = fits.BinTableHDU.from_columns([c1,c2,c3,c4])
hdu.writeto(path_to_store_AP_corr_table+'Aperture_Correction.fits')
