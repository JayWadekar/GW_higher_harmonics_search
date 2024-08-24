# Data products from the IAS-HM O3 search

This repo has data products form the IAS GW search with higher harmonics in[arXiv:2312.06631](https://arxiv.org/abs/2312.06631).
 
For running the pipeline on raw GW strain data from LVK, 
use the codes and tutorials in [gwIAS-HM](https://github.com/JayWadekar/gwIAS-HM) 

Please feel free to email me at jayw@ias.edu if there are any issues or you need
more files or data than currently uploaded here. Different folders contain:

1. The PE samples for the new events described in [arXiv:2312.06631](https://arxiv.org/abs/2312.06631) 
and the candidates catalog (containing above and below threshold candidates),
which contains details such as t_gps, can be downloaded from [google drive](https://drive.google.com/drive/folders/1YkuIo-yIJhIOSX3B0zRGlSJzwIT5D61U?usp=sharing)

We also provide a basic notebook in the folder to show how to read the data and make corner plots
using [cogwheel](https://github.com/jroulet/cogwheel).

2. The jupyter notebooks for two specific modules described in [arXiv:2405.17400](https://arxiv.org/abs/2405.17400):
    - Collecting triggers based on the coherent marginalized score
    - Using the band eraser
