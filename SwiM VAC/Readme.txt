Files needed for each galaxy:

-SWIFT/UVOT
 -Swift counts.fits
 -Swift exposure.fits
 -Swift error.fits

-MaNGA
 - plate-ifu-MAPS-HYB10-GAU-MILESHC.fits
 - plate-ifu-LOGCUBE-HYB10-GAU-MILESHC.fits
 - plate-ifu-LOGCUBE.fits
 - DRP_ALL (dapall-v2_4_3-2.2.1.fits)

-SDSS
 - nsa_name-u.fits
 - nsa_name-g.fits
 - nsa_name-r.fits
 - nsa_name_i.fits
 - nsa_name-z.fits
 - NSA_ALL (nsa_v1_0_1.fits)

-Source Extractor Parameter files (These can be used for all galaxies. Make sure to place them in a folder where the output of Cutout.py is stored)
 -default-r.sex
 -default-wise.sex
 -default.conv
 -default.nnw
 -default.param
 -default.sex

-Text files
 -W2_ID.txt (Text file that contains MaNGA IDs of all galaxies with uvw2 observations)
 -W1_ID.txt (Text file that contains MaNGA IDs of all galaxies with uvw1 observations)
 -M2_ID.txt (Text file that contains MaNGA IDs of all galaxies with uvm2 observations)
 -ALL_ID.txt (Text file that contains MaNGA IDs of all galaxies)
 -W2W1M2TOT.txt (Text file that contains MaNGA IDs of all galaxies with uvw2, uvw1 & uvm2 observations)
 -W2W1TOT.txt (Text file that contains MaNGA IDs of all galaxies with only uvw2 & uvw1 observations)



Step 1: Run the codes Cutout.py for each Swift/UVOT filter to generate the cutout images for the counts, exposure and counts error maps for the target galaxy. Then use the code Sextractor.py to generate the segmented maps for all the sources in the Swift filed of view. The code Sextractor.py uses cutout images generated by Cutout.py. 

Step 2: The NSA catalog has integrated magnitudes which are calculated based on SDSS r-band elliptical apertures. While calculating the Swift integrated magnitudes, we must take into account the r-band aperture corrections. Use the code NSA_AC.py (Aperture corrections) to generate a fits file that has aperture correction for all galaxies. While using Sextractor.py make sure that the folder where you store cutout images (generated from Cutout.py) contains the Source Extractor parameter files listed above.

Step 3: Use the code Flux.py to generate the sky-subtracted Swift flux maps. This code will use the aperture corrections fits file and the cutout images of Swift filters (generated using Cutout.py). This code will generate the Swift sky-subtracted flux maps which will have the following information in the header of the first HDU:
-'SKYC' = sky_counts for the specific Swift NUV filter
-'ESKYC' = 1 sigma error in sky counts 
-'NUV(Mag) APC' = Integrated aperture corrected SWIFT elpetro flux in magnitudes 
-'ENUV(Mag)' = 1 sigma error in SWIFT elpetro flux (in magnitudes)
The 'NUV(Mag) APC' and 'ENUV(Mag)' are Integrated aperture corrected SWIFT elpetro fluxes (in magnitudes) and 1 sigma errors which can used to generate the SwiM ALL file (SwiM_all_v3.fits/Data_model: https://data.sdss.org/datamodel/files/MANGA_SWIM/SWIM_VER/SwiM_all_v3.html).

Step 4: Once the Steps 1 - 3 are done, use the code SwiM_Maps.py to generate the SwiM MAPs files. All the files required for this code are generated in steps 1 - 4.
