# ALR

This is the code for Accelerated Higher Order Logistic Regression.

For more details -- Refer to:
ALRn: accelerated higher-order logistic regression
Zaidi, N.A., Webb, G.I., Carman, M.J. et al. Mach Learn (2016). doi:10.1007/s10994-016-5574-8

An Example command line of the code:

java ALR.BVDcrossvalx -t /Users/nayyar/WData/datasets_DM/nursery.arff -i 2 -x 2 -W ALR.wdAnJE -- -S "A1JE" -P "dCCBN" -I "Flat" -O "GD"

BVDcrossvalx does 'i' rounds of 'x' fold cross validation and calls ALR classifier.

ALR takes in following arguments:

-S A1JE, A2JE, A3JE, A4JE, A5JE

This specifies to use linear, quadratic, cubic, quartic or quintic features

-P MAP, dCCBN, wCCBN

This specifies to use either generative MAP learning or LR or ALR

-I Flat, Indexed, IndexedBig, BitMap

This depends on the data and its dimensions and implements the model effectively
