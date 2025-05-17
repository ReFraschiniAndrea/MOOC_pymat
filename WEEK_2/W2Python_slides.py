import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from manim_slides import Slide
from colab_utils import *


class W2Python_slides(Slide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Let's explore how linear regression can be implemented in Python.
            Our ultimate goal is to answer key questions for policymakers, such
            as: To what extent do variables like temperature and humidity affect
            the risk of wildfires?
            '''
        )
        # SLIDE 02:  ===========================================================
        # COLAB NOTEBOOK FADES IN
        # HAND CURSOR GROWS FROM CENTER AND MOVES TO FOLDER ICON
        self.next_slide(
            notes=
            '''Let's open the notebook. First, we need to load the data. By
            clicking on the folder icon on the left, [CLICK]
            '''
        )
        # SLIDE 03:  ===========================================================
        # FOLDER ICON SI CLICKED
        # SIDE MENU APPEARS
        self.next_slide(
            notes=
            '''a side bar appears showing the list of available files.
            Let us click on the upload button, [CLICK]
            '''
        )
        # SLIDE 04:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''and select from your local file system the file
            Algerian_forest_dataset.csv. [CLICK]
            This dataset includes detailed features related to a set of forest
            fires recorded in Algeria.
            '''
        )
        # SLIDE 05:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Next, we load the modules we are going to use in this project,
            and [CLICK] 
            '''
        )
        # SLIDE 06:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''now we load the dataset. [CLICK]
            '''
        )
        # SLIDE 07:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''We need to use module "pandas", imported with the name "pd"
            '''
        )
        # SLIDE 08:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''With the function "read_csv" we can read CSV files, Comma
            Separated Values files. Make sure the file name matches exactly!

            '''
        )
        # SLIDE 09:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''The output of the function is stored in a variable named
            my_dataset, which is an instance of the class DataFrame from pandas.
            A class is like a blueprint for organizing and working with data. It
            defines the attributes, that is characteristics, and methods, that
            is functions to perform actions, that an object can have.
            For example, the class DatFrame has attributes like its columns and rows, 
            '''
        )
        # SLIDE 10:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''and methods like head() to show the first rows
            '''
        )
        # SLIDE 11:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Let's explore our dataset to get familiar with this type of data
            structure and to see the data firsthand, which is always a good
            practice! In doing this we use attributes and methods.

            '''
        )
        # SLIDE 12:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''First of all, we find the number of rows and columns. We use
            the attribute of the dataset called shape to find the number of
            rows and columns.
            '''
        )
        # SLIDE 13:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''In this case we have 59 rows and 4 columns. 
            '''
        )
        # SLIDE 14:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''[CLICK] Next, we use the method head. The line dataset.head(5)z
            displays the first 5 rows of the dataframe.
            '''
        )
        # SLIDE 15:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''In the table, each row represents a different fire event, while
            each column corresponds to a specific variable associated with
            that event:
            '''
        )
        # SLIDE 16:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''the Temperature: expressed in Celsius degrees
            '''
        )
        # SLIDE 17:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''the Relative Humidity: expressed as a percentage
            '''
        )
        # SLIDE 18:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''the Build-Up Index, which represents the total quantity of
            combustible material in the environment.
            '''
        )
        # SLIDE 19:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''the Fire Weather Index, which evaluates the overall risk of
            forest fires.
            '''
        )
        # SLIDE 20:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''In our context, temperature, relative humidity, and Build-up
            index can (separately) play the role of variable X, while the fire
            weather index plays the role of variable Y.
            '''
        )
        # SLIDE 21:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Now we are ready to implement linear regression.
            '''
        )
        # SLIDE 22:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''We are going to write a function that [CLICK] takes as inputs the datapoints, organized in two lists.
            '''
        )
        # SLIDE 23:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''The first list, contained in the variable “x”, contains the x-coordinates of the points, 
            '''
        )
        # SLIDE 24:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''while the second list, named “y” contains the corresponding y-coordinates.
            '''
        )
        # SLIDE 25:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''The function will return [CLICK] the coefficients of the regression lines, namely m and q
            '''
        )
        # SLIDE 26:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''This is the structure of the Python function that we will write. The function will have two inputs (x and y) and two outputs (m and q). What we need to do now is to fill in the dots.
            '''
        )
        # SLIDE 27:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''This function will perform linear regression by computing the coefficients m and q according to these formulas.
At first glance, this might seem overwhelming, but let's simplify it by breaking the task into smaller steps. 

            '''
        )
        # SLIDE 28:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''First we will compute the results of each sum, and then we will combine the results.
            '''
        )
        # SLIDE 29:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Before starting the implementation, it’s worth noting an important observation: some sums are repeated.
[CLICK] The sum over yi appears twice 
            '''
        )
        # SLIDE 30:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''] and the sum over xi even three times!
We can take advantage of this, and compute these terms once and reuse the results wherever needed

            '''
        )
        # SLIDE 31:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''We are going to write now a python code that computes these four terms.
            '''
        )
        # SLIDE 32:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''These sums can be computed by writing suitable “for” loops. However, we will compute them in a more concise and readable way by leveraging the np.sum function 
            '''
        )
        # SLIDE 33:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''from the NumPy library (imported as np). This function allows us to compute directly the sum of all elements in an array
            '''
        )
        # SLIDE 34:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''For example, np.sum(x) calculates the sum of all the x-coordinates
            '''
        )
        # SLIDE 35:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Similarly, np.sum(y) calculates the sum of all the yi
            '''
        )
        # SLIDE 36:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''To compute the last two terms, we need to introduce a powerful concept in Python: vectorized operations.
            '''
        )
        # SLIDE 37:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''] let us consider the vectors x and y, containing the elements xi and yi, respectively
            '''
        )
        # SLIDE 38:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''the operation x * y creates a new array, [CLICK] whose first entry is the product of the first entries of x and y, [CLICK] the second entry is the product of the second entries, all the way up to the last element
            '''
        )
        # SLIDE 39:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''As a consequence, with np.sum(x * y) we compute the sum of all the products xi times yi, that is the term called sum_xy
            '''
        )
        # SLIDE 40:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Similarly, the element-wise power operation x ** 2 is also vectorized, meaning it applies the power operation to each element of the array individually. 
[CLICK] so that combining this operation with np.sum gives the last term.
            Very good. Now the hardest part is behind us. We just need to combine these quantities to finalize the computation.

            '''
        )
        # SLIDE 41:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''First, we compute the numerator of the expression giving m
            '''
        )
        # SLIDE 42:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Then, we compute the denominator.
            '''
        )
        # SLIDE 43:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''and we divide the numerator by the denominator to obtain the value of m. 
            '''
        )
        # SLIDE 44:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Finally, we calculate q, making use of the m value we just determined.
            This completes the computation of the regression coefficients m and q

            '''
        )
        # SLIDE 45:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Great! We have completed all the necessary steps for the implementation of our function. Let us quickly revise it.
            '''
        )
        # SLIDE 46:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''the function takes two arrays as inputs, containing the x and y coordinates of the data points
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''First we compute the sums needed to perform the linear regression
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''next, we combine these terms thus getting the optimal coefficients m and q
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Finally, we return m, q.
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Now that the function has been written we can run the block and it is ready to be used
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''In this way, we are ready to apply it to the Algerian forest dataset.
            We wonder how the temperature influences the Fire Weather Index,
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''To this goal, we chose the temperature as the x, using this code 
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''The label “Temperature” extracts the corresponding column from the dataset 
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''and the attribute “values” returns the array.
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''
            '''
        )