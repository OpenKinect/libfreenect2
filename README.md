This is a patch that adds a depth packet processor using the phase unwrapping 
algorithm described in the paper "Efficient Phase Unwrapping using Kernel
Density Estimation", ECCV 2016, Felix Järemo Lawin, Per-Erik Forssen and 
Hannes Ovren, see http://www.cvl.isy.liu.se/research/datasets/kinect2-dataset/. 

# Invocation

To use the OpenCL implementation of the new method, run Protonect with:

  ./bin/Protonect clkde

To instead use the CUDA implementation, run Protonect as:

  ./bin/Protonect cudakde

# Parameters

Parameters for the KDE method are found in depth_packet_processor.cpp.
They are related to variables in the paper as follows:

* kde_sigma_sqr: the scale of the kernel in the KDE, h in eq (13).
* unwrapping_likelihood_scale: scale parameter for the unwrapping likelihood, s_1^2 in eq (15).
* phase_confidence_scale: scale parameter for the phase likelihood, s_2^2 in eq (23)
* kde_threshold: threshold on the KDE output in eq (25), defines the inlier/outlier rate trade-off
* kde_neigborhood_size: spatial support of the KDE, i.e. N(x) in eq
  (11). This parameter defines a filter size of (2*kde_neigborhood_size+1 x 2*kde_neigborhood_size+1)
* num_hyps: number of phase unwrapping hypotheses considered by the
  KDE in each pixel. I.e. the size of the set I in eq (11).

# Feedback

For feedback contact Felix Järemo-Lawin <felix.jaremo-lawin@liu.se>
