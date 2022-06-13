# Bioinformatics_Project

This repository contains the main functions used in the project to recognize active and inactive cis regulatory regions in the human genome.
In particular on the cell line A549 and GM12878, obtained from FANTOM5 and ENCODE project.

[![Project](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cfXL4QpyIu-LjG6_aDH7kcJLTltTgDRk) The colab notebook containing the project

[![Test](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KSjc3EgSVWA2qVbrdOBDlr92gRj_vVdm) The colab notebook containing the code to run the tests
```bash
---------- coverage: platform linux, python 3.7.13-final-0 -----------
Name                                                    Stmts   Miss  Cover
---------------------------------------------------------------------------
BioinformaticsProject/DataDownload.py                      18      0   100%
BioinformaticsProject/DataPreProcessing.py                105     17    84%
BioinformaticsProject/DataVisualization.py                 12     12     0%
BioinformaticsProject/GenomeFunction.py                    15      0   100%
BioinformaticsProject/LabelsBinarization.py                11      0   100%
BioinformaticsProject/NNModels.py                         108     23    79%
BioinformaticsProject/test/test_DataDownload.py            11      0   100%
BioinformaticsProject/test/test_DataPreProcessing.py       78      0   100%
BioinformaticsProject/test/test_GenomeFunction.py          20      0   100%
BioinformaticsProject/test/test_LabelsBinarization.py      24      0   100%
BioinformaticsProject/test/test_NNModels.py                32      2    94%
---------------------------------------------------------------------------
TOTAL                                                     434     54    88%


======================== 18 passed in 97.36s (0:01:37) =========================

----------- coverage: platform linux, python 3.6.6-final-0 -----------
Name                                                     Stmts   Miss  Cover
----------------------------------------------------------------------------
BioinformaticsProject/v_test/DataVisualization.py           12      1    92%
BioinformaticsProject/v_test/test_DataVisualization.py      14      0   100%
----------------------------------------------------------------------------
TOTAL                                                       26      1    96%


============================== 2 passed in 3.99s ===============================

```
