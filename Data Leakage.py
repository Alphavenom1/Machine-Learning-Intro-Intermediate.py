# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex7 import *
print("Setup Complete")

#Step 1: The Data Science of Shoelaces
#it depends,if the amount of leather wasn t  set at the begining we wouldn't have access to datato make predictions ,if it was set at the begining , and we used less this will cause leakage

#Step 2: Return of the Shoelaces
#it depends on whether they order shoelaces first or leather first ,f they order shoelaces first, you won't know how much leather they've ordered when you predict their shoelace needs. If leather was ordered first, then you'll have that number available when you place your shoelace order

#Step 4: Preventing Infections
#This poses a risk of both target leakage and train-test contamination ,target leakage will occur if a patient's outcome contributes in the infection rate of his surgeon,
#train-test contamination occures if we  calculate  using all the surgeries of a surgeon, including  the test-setones.

#Step 5: Housing Prices
#Which of the features is most likely to be a source of leakage?
potential_leakage_feature = 2 #Average sales price of homes in the same neighborhood
