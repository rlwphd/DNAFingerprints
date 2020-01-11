# DNAFingerprints
Scripts for analyzing DNA fragments in a lossy style format using machine learning

This code has been developed for the analysis of bacterial DNA (both genomic and plasmidic) when broken up into different segment lengths to be analyzed in a lossy format in which the total A, T, G, and C content of the segment is known but not the order of the segment.
This code was developed at Brigham Young Univeristy under the direction of Dr. William Pitt and Dr. Mark Clement (http://csl.cs.byu.edu/) while working on a joint project with Dr. Pprashant Nagpal from the University of Colorado Boulder (https://www.colorado.edu/lab/nagpal/).
Dr. Nagpal developed a device which can read DNA bases in segments and print out the total number of A's, T's, G's, and C's found in the segment.

For example: a 10 base DNA segment of AAATTTGGGC and a 10 base DNA segment of CAAAGGGTTT would both print out 3-A, 3-T, 3-G, 1-C.

The purpose of this project was to determine whether or not you could identify bacterial species and known antibiotic resistances using DNA segment data in the lossy format from Dr. Nagpal's device.
The project is still ongoing and the code is still under development.

The current working code is called CompleteCode. This code can be downloaded along with the Bacteria folder. The only thing that needs to be updated is the paths for where you want files stored to. If you open up the file, you need to adjust the path to the Bacteria folder, the path to the folder to store the simulated BOC reads (program does not create a folder), and the path to the folder to store the prediction arrays and confusion matrix arrays (program does not create a folder).

To plot the confustion matrices for the data, you will need to download the Confusion_matrix file and adjust the path to the folder where you stored the confusion matrix arrays. You can also cahnge the error rate and number of reads to plot your desired confusion matrix for the specified parameters.

The file G_Plotting_functions plots the original FBC spectra for the genomes and the bias subtracted spectra for the genomes. In addition to downloading this file, you will also need to download the two text files, Split color array 10mer data.txt and Split_data_range.txt. You will also need to change the location to the BOC files in order to run this code.
