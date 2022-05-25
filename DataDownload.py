from epigenomic_dataset import load_epigenomes
import pandas as pd
import numpy as np
from tqdm import tqdm


def download_cell_lines(cell_lines : list):

  window_sizes = [2**x for x in range(6, 11)]

  original_epig = {}
  original_labels = {}

  for cell_line in cell_lines:
    print("---",cell_line)
    for window_size in tqdm(window_sizes):
      promoters_epigenomes, promoters_labels = load_epigenomes(
                                                cell_line = cell_line,
                                                dataset = "fantom",
                                                region = "promoters",
                                                window_size = window_size
                                                )
      enhancers_epigenomes, enhancers_labels = load_epigenomes(
                                                cell_line = cell_line,
                                                dataset = "fantom",
                                                region = "enhancers",
                                                window_size = window_size
                                                )
      original_epig[cell_line + '_promoters_' + str(window_size)] = promoters_epigenomes
      original_labels[cell_line + '_promoters_' + str(window_size)] = promoters_labels
      original_epig[cell_line + '_enhancers_' + str(window_size)] = enhancers_epigenomes
      original_labels[cell_line + '_enhancers_' + str(window_size)] = enhancers_labels
  return original_epig, original_labels

