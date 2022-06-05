from sklearn.manifold import TSNE as STSNE
from sklearn.decomposition import PCA
from tsnecuda import TSNE as CTSNE
from multiprocessing import cpu_count

def pca(x:np.ndarray, n_components:int=2)->np.ndarray:
  """
  This function applies the PCA over the passed data input
  Params:
    -x: the data that will be reduced
    -n_components: the number of components
  Return:
    The reduced data
  """
  return PCA(n_components=n_components, random_state=42).fit_transform(x)
    
def cannylab_tsne(x:np.ndarray, perplexity:int, dimensionality_threshold:int=50):
  """
  This function applies the TSNE dimensionality reduction over the passed data input
  Params:
    -x: the data that will be reduced
    -perplexity: which perplexity apply over the data
    -dimensionality_threshold: the number dimensions over which is applied a first PCA
  Return:
    The reduced data
  """
  if x.shape[1] > dimensionality_threshold:
      x = pca(x, n_components=dimensionality_threshold)
  return CTSNE(perplexity=perplexity, random_seed=42).fit_transform(x)
