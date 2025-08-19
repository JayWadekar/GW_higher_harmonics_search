# Data products from the `IAS-HM` O3 search

This repo has data products form the IAS GW search with higher harmonics in [arXiv:2312.06631](https://arxiv.org/abs/2312.06631).
 
For running the pipeline on raw GW strain data from LVK, 
use the codes and tutorials in our pipeline repository: [gwIAS-HM](https://github.com/JayWadekar/gwIAS-HM) 

Please feel free to email us at jayw@ias.edu if there are any issues or you need
more files or data than currently uploaded here.

The first folder (New_Events_arXiv_2312.06631) contains a simple notebook in the folder to show how to read the data and make corner plots. The notebook uses the following data products which can be downloaded from [google drive](https://drive.google.com/drive/folders/1YkuIo-yIJhIOSX3B0zRGlSJzwIT5D61U?usp=sharing):
1. PE samples for the new candidate events in [arXiv:2312.06631](https://arxiv.org/abs/2312.06631) using [cogwheel](https://github.com/jroulet/cogwheel).  
2. The candidates catalog (containing above and below threshold candidates),
which contains details such as t_gps.
3. Results of our pipeline on injection samples were discussed in our companion paper [arXiv:2501.17939](https://arxiv.org/abs/2501.17939). They are provided in [Zenodo](https://zenodo.org/records/16887064). We have used the publicly provided O3 injection samples by the LIGO-Virgo_Kagra collaboration.

The second folder (Pipeline_modules_arXiv_2405.1740) contains jupyter notebooks for two specific modules described in [arXiv:2405.17400](https://arxiv.org/abs/2405.17400):
1. Collecting triggers based on the coherent marginalized score
2. Using the band eraser
