**********************************************
The German Traffic Sign Recognition Benchmark
**********************************************

This archive contains the test set of the 
"German Traffic Sign Recognition Benchmark". 

This is the official test set that is used during the final 
competition session at IJCNN 2011. After the conference, 
it will be published on our website. 

If you want to publish results based on GTSRB, 
this data set is to be used for final testing purposes
only. Please use the GTSRB (final) training set for training
(It is the joint dataset of the online competition's 
training and test set).


**********************************************
Archive content
**********************************************
This archive contains the following structure:

There is one directory that contains all test images (12,630 images) 
and one text file with annotations (GT-final_test.test.csv)


**********************************************
Image format and naming
**********************************************
The images are PPM images (RGB color). Files are numbered in ascending order:

   00000.ppm to 12629.ppm

Images are in random order, ie. there is no track structure or class information
as in the training set.

**********************************************
Annotation format
**********************************************

The annotations are stored in CSV format (field separator
is ";" (semicolon) ). The annotations contain meta information 
about the image and the class id.


In detail, the annotations provide the following fields:

Filename        - Image file the following information applies to
Width, Height   - Dimensions of the image
Roi.x1,Roi.y1,
Roi.x2,Roi.y2   - Location of the sign within the image
		  (Images contain a border around the actual sign
                  of 10 percent of the sign size, at least 5 pixel)

**********************************************
Further information
**********************************************
For more information on the competition procedures and to obtain the test set, 
please visit the competition website at

	http://benchmark.ini.rub.de

If you have any questions, do not hesitate to contact us 
    
	tsr-benchmark@ini.rub.de


**********************************************
Institut für Neuroinformatik
Real-time computer vision research group

Ruhr-Universität Bochum
Germany
**********************************************