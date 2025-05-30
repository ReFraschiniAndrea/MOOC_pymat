import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from manim_slides import Slide
from W2Anim import *

config.renderer='cairo'
config.background_color = WHITE
config.pixel_width=960
config.pixel_height=720

LABELS_SIZE=0.75

class W2Theory_slides(Slide):
    def construct(self):
        self.wait_time_between_slides = 0.05
        self.skip_reversing = True
        # SLIDE 01:  ===========================================================
        # AXES WITH FIRE INDEX AND TEMPERATURE ICONS APPEAR
        # DATASET POINTS AND REGRESSION LINE APPEAR
        self.next_slide(
            notes=
            '''In this project we have learnt how to perform linear regression,
            to extract valuable information about the relationship among
            variables from a set of data points. [CLICK]
            '''
        )
        # SLIDE 02:  ===========================================================
        # RESIDUAL LINES ARE DRAWN
        # SQUARED ERRORS SUM FORMULA APPEARS
        self.next_slide(
            notes=
            '''First, we saw that the line that best fits a set of data points
            can be defined as the one that minimizes the sum of the squared
            residuals,  [CLICK]
            '''
        )
        # SLIDE 03:  ===========================================================
        # LINEAR REGRESSION COEFFICIENTS FORMULA REPLACES 'E'
        self.next_slide(
            notes=
            '''... and that the coefficients of this line can be found
            through appropriate calculations.
            We then saw how to implement these calculations in Python and MATLAB.
            [CLICK]
            '''
        )
        # SLIDE 04:  ===========================================================
        # SUM TERMS ARE EXTRACTED FROM THE FORMULA AND GO TO TOP
        # CODE WINDOWS APPEAR WITH THE FIRST SUMS
        self.next_slide(
            notes=
            '''To do so efficiently, we first precomputed the terms that appear
            multiple times in the expressions, and we used vectorized
            computations, which allow us to operate directly on arrays without
            explicit loops.  [CLICK]
            '''
        )
        # SLIDE 05:  ===========================================================
        # '.^' AND '.*' HIGHLIGHTED IN MATLAB CODE
        self.next_slide(
            notes=
            '''While the logic is essentially the same in both languages, a key
            syntactic difference is that in MATLAB we need to use a dot before
            operators like * and ^ to indicate element-wise operations. [CLICK]
            '''
        )
        # SLIDE 05:  ===========================================================
        # '*' AND '**' HIGHLIGHTED IN PYTHON CODE
        self.next_slide(
            notes=
            '''In Python with NumPy, on the other hand, element-wise behavior is
            the default when working with arrays. [CLICK]
            '''
        )
        # SLIDE 06:  ===========================================================
        # 'np.sum' AND 'sum' HIGLIGHTED IN THE TWO CODES
        self.next_slide(
            notes=
            '''Finally, for computing these summations, in Python we used the
            sum function from the NumPy module, while in MATLAB we used the
            built-in sum function. [CLICK]
            '''
        )
        # SLIDE 07:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''We then implemented a function that takes as input two arrays
            containing the x- and y-coordinates of the available data points,
            and computes the coefficients m and q of the regression line.
            [CLICK]
            '''
        )
        # SLIDE 08:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''To define functions, the syntax differs slightly between the two
            languages: in Python, we use the keyword def to start the function
            definition and return results using the return statement. [CLICK]
            '''
        )
        # SLIDE 09:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''In MATLAB, on the other hand, functions are introduced with the
            keyword function, and the output variables are listed directly in
            the function header. [CLICK]
            '''
        )
        # SLIDE 10:  ===========================================================
        # HEAD TABLE DATASET APPEARS
        self.next_slide(
            notes=
            '''Using the functions we implemented on a dataset that includes
            detailed features related to a set of forest fires recorded in
            Algeria,  [CLICK]
            '''
        )
        # SLIDE 11:  ===========================================================
        # PLOT OF LINEAR REGRESSION (TEMP. VS FWI) APPEARS
        self.next_slide(
            notes=
            '''we obtained a regression line linking temperature to fire risk.
            This allowed us to gain a quantitative insight into how an increase
            in temperature raises the likelihood of wildfires. [CLICK]
            '''
        )
        # SLIDE 12:  ===========================================================
        # PLOT OF LINEAR REGRESSION (HUMIDITY. VS FWI) REPLACES FIRST ONE
        self.next_slide(
            notes=
            '''Similarly, we saw how higher humidity can mitigate this risk,
            reducing the overall fire danger.
            But what happens if we consider both temperature and humidity? [CLICK]
            '''
        )
        # SLIDE 13:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''So far, we've focused on "simple" linear regression, where we
            have a single independent variable, x [CLICK]
            '''
        )
        # SLIDE 14:  ===========================================================
        #
        self.next_slide(
            notes=
            '''But in many cases, we have multiple independent variables—let's
            call them x1, x2, all the way up to xp. In this situation, the
            regression model includes p coefficients,
            '''
        )
        # SLIDE 15:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''m1 through mp, one for each independent variable [CLICK]
            '''
        )
        # SLIDE 16:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Just like in simple linear regression, in multiple linear
            regression, we calculate yi hat, which is the model's prediction for
            the i-th data point. [CLICK]
            '''
        )
        # SLIDE 17:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''And again, the “best” fitting model is the one that minimizes the
            sum of the squares of the residuals, just as we defined before.
            Solving this minimization problem involves solving a linear system
            of equations. [CLICK]

            In the supplementary material, you can explore this technique
            further, in its extended form known as "multiple linear regression".
            '''
        )
        # SLIDE 18:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''But a key question that we didn't address is the following: how
            well the regression line is fitting the available data? In other
            words, how effectively does the regression line explain the
            relationship between x and y? [CLICK]
            '''
        )
        # SLIDE 01:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''It is clear that the data on the left are better fitted by a line
            compared to the data on the right. But how can we quantify this
            intuition?  You can find the answer in the supplementary material.
            And now, it's your turn. [CLICK]
            '''
        )