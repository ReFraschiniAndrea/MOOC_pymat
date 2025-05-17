import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from manim_slides import Slide
from W2Anim import *

class W2Theory_slides(Slide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''To what extent do environmental variables, such as temperature
            and humidity, influence the risk of wildfires? This might be the
            policymaker's question about wildfires, which are a serious
            environmental and economic concern. Therefore, understanding their
            causes is crucial for prevention and mitigation. [CLICK]
            '''
        )
        # SLIDE 02:  ===========================================================
        # AXIS GRAPH WITH NEGATIVE CORRELATION APPEARS
        self.next_slide(
            notes=
            '''For example, we might expect that, the higher the humidity in the
            air, the lower the fire risk on that day. [CLICK]
            '''
        )
        # SLIDE 03:  ===========================================================
        # AXIS GRAPH WITH POSITIVE CORRELATION APPEARS
        self.next_slide(
            notes=
            '''Similarly, we expect that high temperature increases the risk of
            fire. [CLICK]
            '''
        )
        # SLIDE 04:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''To answer this question quantitatively, we use real-world data
            that captures key environmental factors, such as daily temperature,
            humidity, and wildfire occurrence. [CLICK] 
            '''
        )
        # SLIDE 04:  ===========================================================
        # SOME NON LINEAR MODELS APPEAR
        self.next_slide(
            notes=
            '''The policymaker might ask: "Is there a consistent relationship
            between environmental variables and wildfire risk?" This can be seen
            from a mathematical point of view: identify a model that predicts
            fire risk based on temperature and humidity.
            '''
        )
        # SLIDE 05:  ===========================================================
        # REGRESSED LINE APPEARS
        # LINEAR REGRESSION TITLE APPEARS
        self.next_slide(
            notes=
            '''Among different possible models, in this project, we will
            introduce linear regression, to determine linear relationships
            between variables. One key reason for the popularity of linear
            regression is its interpretability.
            The key question is: "How can I construct this line mathematically?"
            '''
        )
        # SLIDE 06:  ===========================================================
        # Y=F(X) APPEARS
        # EMPTY AXIS APPEARS
        # X AND Y LABELS ARE DUPLICATEED FROM Y=F(X) TO AXIS LABELS
        self.next_slide(
            notes=
            '''Linear regression is a fundamental statistical technique used to
            model the relationship between an independent variable x, and a
            dependent variable y.
            '''
        )
        # SLIDE 07:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''This technique assumes that the relationship between the
            dependent and independent variables is described by the equation y
            equals m times x plus q.
            The coefficients m and q have clear and intuitive meanings.
            '''
        )
        # SLIDE 08:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Specifically, q represents the y-intercept, that is the predicted
            value when x equals 0.
            '''
        )
        # SLIDE 09:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''m, instead, represents the “slope” of the line, that is the rate
            of change of y with respect to x. In other words, [CLIK] ...
            '''
        )
        # SLIDE 10:  ===========================================================
        # RISE OVER RUN TRIANGLE APPEARS
        # DELTA_X LABEL APPEARS
        self.next_slide(
            notes=
            '''... if the independent variable x increases by a certain amount
            delta_x, [CLIK] ...
            '''
        )
        # SLIDE 11:  ===========================================================
        # DELTA_Y LABEL APPEARS
        self.next_slide(
            notes=
            '''... the predicted dependent variable y will increase by m*delta_x.
            Let us now see how to find a line from a cloud of points. [CLICK]
            '''
        )
        # SLIDE 12:  ===========================================================
        # AXES WITH TWO LONE POINTS APPEARS
        # UNIQUE LINE BETWEEN THEM IS DRAWN
        self.next_slide(
            notes=
            '''We know that for two distinct points, there is exactly one line
            that passes through them. [CLICK]
            '''
        )
        # SLIDE 13:  ===========================================================
        # LINE DISAPPEARS, WHOLE POINT CLOUD APPEARS
        self.next_slide(
            notes=
            '''But what happens when there are many points? [CLICK]
            '''
        )
        # SLIDE 14:  ===========================================================
        # MANY CANDIDATE LINES APPEAR
        self.next_slide(
            notes=
            '''In this case, there are many lines that pass through pairs of
            points, but none of them will perfectly represent all the points in
            our dataset. So how do we choose the most representative line? How
            do we define which line is the "best" among them? [CLICK]
            '''
        )
        # SLIDE 15:  ===========================================================
        # ONLY ONE LINE FOR EXPLANATION REMAINS
        self.next_slide(
            notes=
            '''To find the "best"-fitting line, we must first define what we
            mean by "best". [CLICK]
            '''
        )
        # SLIDE 16:  ===========================================================
        # LABELS FOR P1, P2, Pi APPEAR IN SUCCESSION
        # ONLY LABEL OF Pi REMAINS
        self.next_slide(
            notes=
            '''Let (x_1, y_1) be the first data point, (x_2, y_2) the second,
            and so on for all n points. [CLICK]
            '''
        )
        # SLIDE 17:  ===========================================================
        # POINT REPRESENTING PREDICTION APPEARS
        self.next_slide(
            notes=
            '''For any given point (x_i, y_i), we define y^_i as the predicted
            value from the model, which is m * x_i + q.
            '''
        )
        # SLIDE 18:  ===========================================================
        # RESIDUAL BRACKET WITH LABEL APPEARS
        self.next_slide(
            notes=
            '''The difference y_i - y^_i is called the residual, and it measures
            the error between the actual and predicted values. [CLICK]
            '''
        )
        # SLIDE 19:  ===========================================================
        # LINES BETWEEN ALL DATA POINTS AND THE LINE APPEAR
        self.next_slide(
            notes=
            '''Our objective is to find the line that minimizes these residuals,
            making the differences between the observed data and the model's
            predictions as small as possible.
            '''
        )
        # SLIDE 20:  ===========================================================
        # GRAPH SHIFTS UP
        # FORMULA FOR SQUARED ERROR APPEARS
        # 'E' IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''More precisely, we look for the line that minimizes the sum of
            the squares of the residuals, that we denote by E.
            E is a number that quantifies the total error between observed and
            predicted data.
            '''
        )
        # SLIDE 21:  ===========================================================
        # COUNTERS FOR m AND q APPEAR
        # m AND q VARY, CHANGING THE LINE
        self.next_slide(
            notes=
            '''By varying m and q, we obtain different lines. For each
            combination of m and q, we can compute the corresponding error E,
            [CLICK]
            '''
        )
        # SLIDE 22:  ===========================================================
        # counter
        self.next_slide(
            notes=
            '''which is the sum of squared residuals. [CLICK]
            '''
        )
        # SLIDE 23:  ===========================================================
        # LINE CHANGES AGAIN WITH m AND q, DISPLAYING THE CORRESPONDING E VALUE
        self.next_slide(
            notes=
            '''Out of the infinitely many lines generated by varying m and q,
            we will select the one that minimizes E, achieving the best fit to
            the data.
            '''
        )
        # SLIDE 24:  ===========================================================
        # GRAPH FADES OUT
        # MINIMIZATION PROBLEM FORMULAZION APPEARS
        self.next_slide(
            notes=
            '''Mathematically, the regression line is defined as the one
            obtained with the coefficients m and q that minimize the quantity E.
            This is the mathematical definition that we give to the expression
            "best-fitting line" that we have used before.
            '''
        )
        # SLIDE 25:  ===========================================================
        # LEAST SQUARES TITLE APPEARS
        # 'SQUARED' HIGHLIGHTED IN FORMULA
        self.next_slide(
            notes=
            '''This approach is known as "least-squares" regression, as it
            focuses on minimizing the "squares" of the residuals.
            '''
        )
        # SLIDE 26:  ===========================================================
        # PARTIAL DERIVATIVES OF E SET TO ZERO APPEAR
        # IMLICATION: FORMULA FOR 
        self.next_slide(
            notes=
            '''By performing the necessary calculations, we find that the
            coefficients satisfying these conditions are given [CLICK] by the
            following expressions. These values ensure that the derivative of E
            with respect to both unknown coefficients m and q is equal to zero,
            that is to say no further improvement is possible.
            '''
        )
        # SLIDE 27:  ===========================================================
        # PARTIAL DERIVATIVES OF E SET TO ZERO APPEAR
        # IMLICATION: FORMULA FOR 
        self.next_slide(
            notes=
            '''We need a computer code to perform these operations automatically.
            In the upcoming videos, we will explore how to implement these
            formulas in Python and MATLAB.
            '''
        )