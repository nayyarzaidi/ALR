# ALR

This is the code for Accelerated Higher Order Logistic Regression.

For more details -- Refer to:
```
@article{Zaidi2016-ALR,
  author    = {Zaidi, Nayyar A. and Webb, Geoffrey I. and Carman, Mark J. and Petitjean, Francois},
  title     = {$ALR^n$: Accelerated Higher-order Logistic Regression},
  journal   = {Machine Learning},
  volume    = {104},
  number    = {2},
  pages     = {181--194},
  year      = {2016}
}
```
Note the paper is available [here](http://francois-petitjean.com/Research/ALR.pdf)

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
