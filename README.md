# MODIS-MISR-Multi-Layered-Clouds
Involves 1D Plane-Parallel First Principle Radiative Transfer techniques to rectify CO2-slicing errors in CTP/emissivity for a higher-cloud in a 2-layered cloud using MISR low-level CTH (as low-level truth) and Reanalysis profiles.

This uses a ANN based model that is trained with multi-spectral information and low-cloud heights. In practice, the model uses MODIS multi-spectral information and MISR low cloud CTH to predict high-cloud height and emissivity for 2-layered scenes. This improves the accuracy of MODIS height and emissivity retreivals for multi-layered clouds, which currently uses a 1-layered approximation and is hence, unreliable.
