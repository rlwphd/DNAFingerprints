# DNAFingerprints
Scripts for analyzing DNA fragments in a lossy style format using machine learning

This code has been developed for the analysis of bacterial DNA (both genomic and plasmidic) when broken up into different segment lengths to be analyzed in a lossy format in which the total A, T, G, and C content of the segment is known but not the order of the segment.
This code was developed at Brigham Young Univeristy under the direction of Dr. William Pitt and Dr. Mark Clement (http://csl.cs.byu.edu/) while working on a joint project with Dr. Pprashant Nagpal from the University of Colorado Boulder (https://www.colorado.edu/lab/nagpal/).
Dr. Nagpal developed a device which can read DNA bases in segments and print out the total number of A's, T's, G's, and C's found in the segment.

For example: a 10 base DNA segment of AAATTTGGGC and a 10 base DNA segment of CAAAGGGTTT would both print out 3-A, 3-T, 3-G, 1-C.

The purpose of this project was to determine whether or not you could identify bacterial species and known antibiotic resistances using DNA segment data in the lossy format from Dr. Nagpal's device.
The project is still ongoing and the code is still under development.
The code is currently split into two folders.
The initial code written at the beginning of the concept and the current working code.
The initial code is kept in order to ensure that the main ideas from the beginning are preserved and may be removed at a future time.
