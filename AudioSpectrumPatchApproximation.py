"""
ASPA - AudioSpectrumPatchApproximation.py
	Time-frequency 2D sparse code dictionary learning and approximation in Python

https://github.com/bregmanstudio/Audio2DSparseApprox 

Assumes python distribution with numply / scipy installed

Michael A. Casey, Bregman Media Labs 2014-2015, Dartmouth College
See attached license file (Apache 2.0)

Dependencies:
  sklearn - http://scikit-learn.org/stable/
  skimage - http://scikit-image.org/docs/dev/api/skimage.html
  bregman - https://github.com/bregmanstudio/BregmanToolkit
  scikits.audiolab - https://pypi.python.org/pypi/scikits.audiolab/

Example: (see AudioSpectrumPatchApproximation.ipynb)
"""


import os, sys, glob
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import ProjectedGradientNMF
from skimage.filter import gabor_kernel
import bregman.suite as br 
import pdb

class SparseApproxSpectrum(object):
    """class for 2D patch analysis of audio files
    initialization:
    	patch_size=(12,12) - size of time-frequency 2D patches in spectrogram units (freq,time)
    	max_samples=1000000 - if num audio patches exceeds this threshold, randomly sample spectrum
    """
    def __init__(self, patch_size=(12,12), max_samples=1000000, **kwargs):		
        self.omp = OrthogonalMatchingPursuit()
        self.patch_size = patch_size
        self.max_samples = max_samples
        self.D = None
        self.data = None
        self.components = None
        self.zscore=False
        self.log_amplitude=False

    def _extract_data_patches(self, X, zscore, log_amplitude):
    	"utility method for converting spectrogram data to 2D patches "
        self.zscore=zscore
        self.log_amplitude=log_amplitude
        self.X = X
        if self.log_amplitude:
            X = np.log(1+X)
        data = extract_patches_2d(X, self.patch_size)
        data = data.reshape(data.shape[0], -1)
        if len(data)>self.max_samples:
            data = np.random.permutation(data)[:self.max_samples]
        print data.shape
        if self.zscore:
            self.mn = np.mean(data, axis=0) 
            self.std = np.std(data, axis=0)
            data -= self.mn
            data /= self.std
        self.data = data

    def make_gabor_field(self, X, zscore=True, log_amplitude=True, thetas=range(4), 
    		sigmas=(1,3), frequencies=(0.05, 0.25)) :
        """Given a spectrogram, prepare 2D patches and Gabor filter bank kernels
        inputs:
           X - spectrogram data (frequency x time)
           zscore - whether to zscore the ensemble of 2D patches [True]
           log_amplitude - whether to apply log(1+X) scaling of spectrogram data [True]
           thetas - list of 2D Gabor filter orientations in units of pi/4. [range(4)]
           sigmas - list of 2D Gabor filter standard deviations in oriented direction [(1,3)]
           frequencies - list of 2D Gabor filter frequencies [(0.05,0.25)]
        outputs:
           self.data - 2D patches of input spectrogram
           self.D.components_ - Gabor dictionary of thetas x sigmas x frequencies atoms
        """
        self._extract_data_patches(X, zscore, log_amplitude)
        self.n_components = len(thetas)*len(sigmas)*len(frequencies)
        self.thetas = thetas
        self.sigmas = sigmas
        self.frequencies = frequencies
        a,b = self.patch_size
        self.kernels = []
        for theta in thetas:
            theta = theta / 4. * np.pi
            for sigma in sigmas:
                for frequency in frequencies:
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    c,d = kernel.shape
                    if c<=a:
                        z = np.zeros(self.patch_size)
                        z[(a/2-c/2):(a/2-c/2+c),(b/2-d/2):(b/2-d/2+d)] = kernel
                    else:
                        z = kernel[(c/2-a/2):(c/2-a/2+a),(d/2-b/2):(d/2-b/2+b)]
                    self.kernels.append(z.flatten())
        class Bunch:
            def __init__(self, **kwds):
                self.__dict__.update(kwds)
        self.D = Bunch(components_ = np.vstack(self.kernels))

    def extract_codes(self, X, n_components=16, zscore=True, log_amplitude=True):
    	"""Given a spectrogram, learn a dictionary of 2D patch atoms from spectrogram data
        inputs:
            X - spectrogram data (frequency x time)
    	    n_components - how many components to extract [16]
            zscore - whether to zscore the ensemble of 2D patches [True]
            log_amplitude - whether to apply log(1+X) scaling of spectrogram data [True]
        outputs:
            self.data - 2D patches of input spectrogram
            self.D.components_ - dictionary of learned 2D atoms for sparse coding
        """
        self._extract_data_patches(X, zscore, log_amplitude)
        self.n_components = n_components
        self.dico = MiniBatchDictionaryLearning(n_components=self.n_components, alpha=1, n_iter=500)
        print "Dictionary learning from data..."
        self.D = self.dico.fit(self.data)

    def plot_codes(self, cbar=False, show_axis=False, **kwargs):
        "plot the learned or generated 2D sparse code dictionary"
        N = int(np.ceil(np.sqrt(self.n_components)))
        kwargs.setdefault('cmap', plt.cm.gray_r)
        kwargs.setdefault('origin','bottom')
        kwargs.setdefault('interpolation','nearest')
        for i, comp in enumerate(self.D.components_):
            plt.subplot(N, N, i+1)
            plt.imshow(comp.reshape(self.patch_size), **kwargs)
            if cbar:
                plt.colorbar()
            if not show_axis:
                plt.axis('off')
            plt.xticks(())
            plt.yticks(())
            plt.title('%d'%(i))
        plt.suptitle('Dictionary of Spectrum Patches\n', fontsize=14)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    def extract_audio_dir_codes(self, dir_expr='/home/mkc/exp/FMRI/stimuli/Wav6sRamp/*.wav',**kwargs):
    	"apply dictionary learning to entire directory of audio files (requires LOTS of RAM)"
        flist=glob.glob(dir_expr)
        self.X = np.vstack([br.feature_scale(br.LogFrequencySpectrum(f, nbpo=24, nhop=1024).X,normalize=1).T for f in flist]).T
        self.D = extract_codes(self.X, **kwargs)
        self.plot_codes(**kwargs)

    def _get_approximation_coefs(self,data, components):
    	"""utility function to fit dictionary components to data
    	inputs:
    		data - spectrogram data (frqeuency x time)
    	  components - the dictionary components to fit to the data
        """
        w = np.array([self.omp.fit(components.T, d.T).coef_ for d in data])
        return w

    def reconstruct_spectrum(self, w=None, randomize=False):
    	"""reconstruct by fitting current 2D dictionary to self.data 
        returns:
            self.X_hat - spectral reconstruction of self.data
        """
        data = self.data
        components = self.D.components_
        if w is None:
            self.w = self._get_approximation_coefs(data, components)
            w = self.w
        if randomize:
            components = np.random.permutation(components)
        recon = np.dot(w, components)
        if self.zscore:
            recon = recon * self.std
            recon = recon + self.mn
        recon = recon.reshape(-1, *self.patch_size)
        self.X_hat = reconstruct_from_patches_2d(recon, self.X.shape)
        if self.log_amplitude:
            self.X_hat = np.exp(self.X_hat) - 1.0 # invert log transform

    def reconstruct_individual_spectra(self, w=None, randomize=False, plotting=False, rectify=True, **kwargs):
    	"""fit each dictionary component to self.data
        inputs:
            w - per-component reconstruction weights [None=calculate weights]
            randomize - randomly permute components after getting weights [False]
            plotting - whether to subplot individual spectrum reconstructions [True]
            rectify- remove negative ("dark energy") from individual reconstructions [True]
        returns:
            self.X_hat_l - list of indvidual spectrum reconstructions per dictionary atom
        """
        self.reconstruct_spectrum(w,randomize)
        w, components = self.w, self.D.components_
        self.X_hat_l = []
        for i in range(len(self.w.T)):
	    	r=np.array((np.matrix(w)[:,i]*np.matrix(components)[i,:])).reshape(-1,*self.patch_size)
        	X_hat = reconstruct_from_patches_2d(r, self.X.shape)
                if self.log_amplitude:
                    X_hat = np.exp(X_hat) - 1.0
                if rectify: # half wave rectification
                    X_hat[X_hat<0] = 0
                self.X_hat_l.append(X_hat)
        if plotting:
            self.plot_individual_spectra(**kwargs)

    def plot_individual_spectra(self, **kwargs):
        "plot individual spectrum reconstructions for self.X_hat_l"
        if self.X_hat_l is None: return
        plt.figure()
        rn = np.ceil(self.n_components**0.5)
        for k in range(self.n_components):
            plt.subplot(rn,rn,k+1)
            br.feature_plot(self.X_hat_l[k], nofig=1, **kwargs)
            plt.title('%d'%(k))
        plt.suptitle('Component Reconstructions\n', fontsize=14)

class NMFSpectrum(SparseApproxSpectrum):
    """Sparse dictionary learning from non-negative 2D spectrogram patches 
    initialization:
    	patch_size=(12,12) - size of time-frequency 2D patches in spectrogram units (freq,time)
    	max_samples=1000000 - if num audio patches exceeds this threshold, randomly sample spectrum
    """
    def __init__(self, **kwargs):
        SparseApproxSpectrum.__init__(self,**kwargs)

    def extract_codes(self, X, n_components=16, log_amplitude=True, **kwargs):
    	"""Given a spectrogram, learn a dictionary of 2D patch atoms from spectrogram data
        inputs:
           X - spectrogram data (frequency x time)
           n_components - how many components to extract [16]
           log_amplitude - weather to apply log amplitude scaling log(1+X)
        outputs:
           self.data - 2D patches of input spectrogram
           self.D.components_ - dictionary of 2D NMF components
        """
        zscore=False
        self._extract_data_patches(X, zscore, log_amplitude)
        self.n_components=n_components
        kwargs.setdefault('sparseness','components')
        kwargs.setdefault('init','nndsvd')
        kwargs.setdefault('beta',0.5)
        print "NMF..."
        self.model = ProjectedGradientNMF(n_components=self.n_components, **kwargs)
        self.model.fit(self.data)        
        self.D = self.model

    def reconstruct_spectrum(self, w=None, randomize=False):
    	"reconstruct by fitting current NMF 2D dictionary to self.data"
        if w is None:
            self.w = self.model.transform(self.data)
            w = self.w
        return SparseApproxSpectrum.reconstruct_spectrum(self, w=w, randomize=randomize)
