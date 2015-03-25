# ASPA - AudioSpectrumPatchApproximation.py
# 	Time-frequency 2D sparse code dictionary learning and approximation in Python
#
# https://github.com/bregmanstudio/Audio2DSparseApprox 
#
# Assumes python distribution with numply / scipy installed
#
# (c) 2014-2015 Michael A. Casey, Bregman Media Labs, Dartmouth College
# See attached license file (Apache)
#
# Dependencies:
#   sklearn - http://scikit-learn.org/stable/
#   skimage - http://scikit-image.org/docs/dev/api/skimage.html
# 	bregman - https://github.com/bregmanstudio/BregmanToolkit
#   scikits.audiolab - https://pypi.python.org/pypi/scikits.audiolab/
# 
# Example: (see AudioSpectrumPatchApproximation.ipynb)
# 
#

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
    # class for 2D patch analysis of audio files
    def __init__(self, n_components=16, patch_size=(12,12), max_samples=1000000, **kwargs):		
        # initialization:
        #	n_components=16 - how many components to extract
        #	patch_size=(12,12) - size of time-frequency 2D patches in spectrogram units (freq,time)
        #	max_samples=1000000 - if num audio patches exceeds this threshold, randomly sample spectrum
        self.omp = OrthogonalMatchingPursuit()
        self.n_components = n_components
        self.patch_size = patch_size
        self.max_samples = max_samples
        self.D = None
        self.data = None
        self.components = None
        self.standardize=False

    def _extract_data_patches(self, X):
    	# utility method for converting spectrogram data to 2D patches 
        self.X = X
        data = extract_patches_2d(X, self.patch_size)
        data = data.reshape(data.shape[0], -1)
        if len(data)>self.max_samples:
            data = np.random.permutation(data)[:self.max_samples]
        print data.shape
        if self.standardize:
            self.mn = np.mean(data, axis=0) 
            self.std = np.std(data, axis=0)
            data -= self.mn
            data /= self.std
        self.data = data

    def make_gabor_field(self, X, standardize=False, thetas=range(4), 
    		sigmas=(1,3), frequencies=(0.05, 0.25)) :
        # Given a spectrogram, prepare 2D patches and Gabor filter bank kernels
        # inputs:
        #    X - spectrogram data (frequency x time)
        #    standardize - whether to standardize the ensemble of 2D patches
        #    thetas - list of 2D Gabor filter orientations in units of pi/4.
        #    sigmas - list of 2D Gabor filter standard deviations in oriented direction
        #    frequencies - list of 2D Gabor filter frequencies
        # outputs:
        #    self.data - 2D patches of input spectrogram
        #    self.D.components_ - Gabor dictionary of thetas x sigmas x frequencies atoms
        self.standardize=standardize
        self._extract_data_patches(X)
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

    def extract_codes(self, X, standardize=False):
    	# Given a spectrogram, learn a dictionary of 2D patch atoms from spectrogram data
        # inputs:
        #    X - spectrogram data (frequency x time)
        #    standardize - whether to standardize the ensemble of 2D patches
        # outputs:
        #    self.data - 2D patches of input spectrogram
        #    self.D.components_ - dictionary of learned 2D atoms for sparse coding
        self.standardize=standardize
        self._extract_data_patches(X)
        self.dico = MiniBatchDictionaryLearning(n_components=self.n_components, alpha=1, n_iter=500)
        print "Dictionary learning from data..."
        self.D = self.dico.fit(self.data)
        return self

    def plot_codes(self, cbar=False, **kwargs):
        # plot the learned or generated 2D sparse code dictionary
        N = int(np.ceil(np.sqrt(self.n_components)))
        kwargs.setdefault('cmap', plt.cm.gray_r)
        kwargs.setdefault('origin','bottom')
        kwargs.setdefault('interpolation','nearest')
        for i, comp in enumerate(self.D.components_):
            plt.subplot(N, N, i + 1)
            plt.imshow(comp.reshape(self.patch_size), **kwargs)
            if cbar:
                plt.colorbar()
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('Dictionary of spectrum patches\n', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    def extract_audio_dir_codes(self, dir_expr='/home/mkc/exp/FMRI/stimuli/Wav6sRamp/*.wav',**kwargs):
    	# apply dictionary learning to entire directory of audio files (requires LOTS of RAM)
        flist=glob.glob(dir_expr)
        self.X = np.vstack([br.feature_scale(br.LogFrequencySpectrum(f, nbpo=24, nhop=1024).X,normalize=1).T for f in flist]).T
        self.D = extract_codes(self.X, **kwargs)
        self.plot_codes(**kwargs)
        return self

    def _get_approximation_coefs(self,data, components):
    	# utility function to fit dictionary components to data
    	# inputs:
    	#	data - spectrogram data (frqeuency x time)
    	#   components - the dictionary components to fit to the data
        w = np.array([self.omp.fit(components.T, d.T).coef_ for d in data])
        return w

    def reconstruct_spectrum(self, w=None, randomize=False):
    	# reconstruct by fitting current 2D dictionary to self.data 
        data = self.data
        components = self.D.components_
        if w is None:
            self.w = self._get_approximation_coefs(data, components)
            w = self.w
        if randomize:
            components = np.random.permutation(components)
        recon = np.dot(w, components).reshape(-1, *self.patch_size)
        self.X_hat = reconstruct_from_patches_2d(recon, self.X.shape)
        return self

    def reconstruct_individual_spectra(self, w=None, randomize=False, plotting=False, **kwargs):
    	# fit each dictionary component to self.data
        self.reconstruct_spectrum(w,randomize)
        w, components = self.w, self.D.components_
        self.X_hat_l = []
        for i in range(len(self.w.T)):
	    	r=np.array((np.matrix(w)[:,i]*np.matrix(components)[i,:])).reshape(-1,*self.patch_size)
        	self.X_hat_l.append(reconstruct_from_patches_2d(r, self.X.shape))
        if plotting:
            plt.figure()
            for k in range(self.n_components):
                plt.subplot(self.n_components**0.5,self.n_components**0.5,k+1)
                X_hat = self.X_hat_l[k]
                X_hat[X_hat<0] = 0
                br.feature_plot(X_hat,nofig=1,**kwargs)
        return self

class NMFSpectrum(SparseApproxSpectrum):
	# Sparse dictionary learning from non-negative 2D spectrogram patches 
    def __init__(self, **kwargs):
        SparseApproxSpectrum.__init__(self,**kwargs)

    def extract_codes(self, X, **kwargs):
    	# Given a spectrogram, learn a dictionary of 2D patch atoms from spectrogram data
        # inputs:
        #    X - spectrogram data (frequency x time)
        # outputs:
        #    self.data - 2D patches of input spectrogram
        #    self.D.components_ - dictionary of 2D NMF components
        self.standardize=False
        self._extract_data_patches(X)
        kwargs.setdefault('sparseness','components')
        kwargs.setdefault('init','nndsvd')
        kwargs.setdefault('beta',0.5)
        print "NMF..."
        self.model = ProjectedGradientNMF(n_components=self.n_components, **kwargs)
        self.model.fit(self.data)        
        self.D = self.model
        return self

    def reconstruct_spectrum(self, w=None, randomize=False):
    	# reconstruct by fitting current NMF 2D dictionary to self.data 
        if w is None:
            self.w = self.model.transform(self.data)
            w = self.w
        return SparseApproxSpectrum.reconstruct_spectrum(self, w=w, randomize=randomize)
