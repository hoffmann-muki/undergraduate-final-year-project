This is the dataset underlying the results presented in the PLoS Computational Biology article
"Using Matrix and Tensor Factorizations for the Single-Trial Analysis of Population Spike Trains"
by Arno Onken, Jian K. Liu, Palipahana P. Chamanthi R. Karunasekara, Ioannis Delis, Tim Gollisch, Stefano Panzeri.

This dataset contains the visual stimuli that were used for the retina recordings and also the spike trains and the
receptive fields of the recorded retinal ganglion cells.

In 'Stimuli', each file stores an image / movie frame in raw 'uint8' format.
In Matlab, the pixel matrices of the natural images can be read with the Matlab command
pm = fread(fopen('filename.raw'),[256,256],'uint8');
and the pixel matrices of the movie frames and gratings can be read with the Matlab command
pm = fread(fopen('filename.raw'),[360,360],'uint8');

In NeuralData each file stores data in a Matlab '.mat'-file. The files are named:

NaturalImages1.mat - Recording 1 with natural image stimuli
NaturalImages2.mat - Recording 2 with natural image stimuli
Gratings.mat       - Recording with flashed gratings
Movie1Exp1.mat     - Recording 1 with movie 1
Movie2Exp2.mat     - Recording 2 with movie 1
Movie2Exp1.mat     - Recording 1 with movie 2
Movie1Exp2.mat     - Recording 2 with movie 2
Shifts1.mat        - Recording 1 with shifted natural images
Shifts2.mat        - Recording 2 with shifted natural images


Folder tree of the data:

 '- NeuralData

  '- Spikes
     Each Matlab file stores a variable "Spikes" with spike times in milliseconds after stimulus onset.
     For the recordings with movie stimuli, "Spikes" is a cell array with (#neurons x #trials) spike trains.
     For the natural images and gratings stimuli, "Spikes" is a cell vector of length (#stimuli). Each element in
     the vector is a cell array with (#neurons x #trials) spike trains.
     For the shifted natural images, "Spikes" is a cell array of size (#shifts x #images). Again, each element in
     the array is a cell array with (#neurons x #trials) spike trains.

  '- ReceptiveFields
     Each Matlab file stores a cell array 'ReceptiveFields' that contains the receptive fields of all cells in image
     pixel coordinates. The receptive field of cell i can be plotted with the Matlab command
     plot(ReceptiveFields{i}(1,:),ReceptiveFields{i}(2,:))

 '- Stimuli

  '- NaturalImages
     Contains the 60 natural images.

  '- Movie1
     Contains all frames of movie 1.

  '- Movie2
     Contains all frames of movie 2.

  '- Gratings
     Contains the 60 gratings.

