import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from manim_slides import Slide
from Generic_mooc_utils import *
from colab_utils import *
from W2Anim import LinearRegressionEquations

config.renderer='cairo'
config.background_color = WHITE
config.pixel_width=960
config.pixel_height=720
# config.pixel_width=1440
# config.pixel_height=1080


class W2Python_slides(Slide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # AXIS WITH DATA POINTS APPEAR
        # REGRESSION LINE IS DRAWN
        # FORMULAS FOR m, q  APPEAR
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
        cl_env = ColabEnv(r'Assets\W2\colabSLR.png')
        hand_cursor = Cursor()
        self.play(FadeIn(cl_env))
        self.wait(1)
        self.play(GrowFromCenter(hand_cursor))

        # SLIDE 03:  ===========================================================
        # HAND CURSOR MOVES TO FOLDER ICON
        # FOLDER ICON IS CLICKED AND SIDE MENU APPEARS
        self.next_slide(
            notes=
            '''a side bar appears showing the list of available files.
            Let us click on the upload button, [CLICK]
            '''
        )
        self.play(hand_cursor.animate.move_to(cl_env.MENU_))
        self.play(hand_cursor.Click())
        cl_env.set_image(r'Assets\W2\colabSLR_sidemenu.png')
        self.wait(0.1)

        # SLIDE 04:  ===========================================================
        # HAND CURSOR MOVES TO UPLOAD BUTTON
        # UPLOAD BUTTON IS CLICKED, UOLUADED FILE APPEARS
        self.next_slide(
            notes=
            '''and select from your local file system the file
            Algerian_forest_dataset.csv. [CLICK]
            This dataset includes detailed features related to a set of forest
            fires recorded in Algeria.
            '''
        )
        self.play(hand_cursor.animate.move_to(cl_env.UPLOAD_))
        self.play(hand_cursor.Click())
        cl_env.set_image(r'Assets\W2\colabSLR_uploaded.png'),
        self.wait(0.1)

        # SLIDE 05:  ===========================================================
        # NEW CODE CELL IS CREATED AND ZOOMED IN
        self.next_slide(
            notes=
            '''Next, we load the modules we are going to use in this project,
            and load the dataset.[CLICK] 
            '''
        )
        self.play(hand_cursor.animate.move_to(cl_env.PLUS_CODE_))
        empty_cell = ColabCodeBlock(code='')
        cl_env.add_cell(empty_cell)
        self.play(hand_cursor.Click())
        cl_env.set_image(r'Assets\W2\colabSLR.png')
        self.wait(0.8)
        self.play(cl_env.OutofColab(empty_cell), FadeOut(hand_cursor))

        # SLIDE 06:  ===========================================================
        # IMPORT NUMPY LINE IS WRITTEN
        self.next_slide(
            notes=
            '''We need the module "numpy", imported with the name "np", for
            vector oparations, [CLICK] 
            '''
        )
        import_code = ColabCode(
            code=r'''
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            # Load the dataset
            my_dataset = pd.read_csv('Algerian_forest_dataset.csv')
            '''
        )
        import_code.center()
        self.play(import_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 07:  ===========================================================
        # IMPORT PANDAS LINE IS WRITTEN
        self.next_slide(
            notes=
            '''Next, we use module "pandas", imported as "pd", to work with
            datasets, [CLICK]
            '''
        )
        self.play(import_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 08:  ===========================================================
        # IMPORT MATPLOTLIB LINE IS WRITTEN
        self.next_slide(
            notes=
            '''Finally, we will use "matplotlib" to visualize the data. [CLICK]
            '''
        )
        self.play(import_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 09:  ===========================================================
        # READ_CSV LINES WRITTEN
        self.next_slide(
            notes=
            '''To load the dataset, we can use the function "read_csv", which
            reads CSV files: Comma Separated Values files. Make sure the file
            name matches exactly! [CLICK]
            '''
        )
        self.play(import_code.TypeLetterbyLetter(lines=[4]))
        self.wait(0.5)
        self.play(import_code.TypeLetterbyLetter(lines=[5]))

        # SLIDE 09:  ===========================================================
        # DATAFRAME VARIABLE IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''The output of the function is stored in a variable named
            my_dataset, which is an instance of the class DataFrame from pandas.
            [CLICK]
            '''
        )
        my_dataset_highlight = HighlightRectangle(import_code[5][12:22])
        self.play(Create(my_dataset_highlight))

        # SLIDE 09:  ===========================================================
        # INTO COLAB, FIRST CELL IS RUN
        self.next_slide(
            notes=
            '''A class is like a blueprint for organizing and working with data. It
            defines the attributes, that is characteristics, and methods, that
            is functions to perform actions, that an object can have.
            For example, the class DataFrame has attributes like its columns and rows, 
            '''
        )

        # SLIDE 10:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''and methods like head() to show the first rows
            '''
        )
        self.play(FadeOut(my_dataset_highlight))
        # SLIDE 11:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Let's explore our dataset to get familiar with this type of data
            structure and to see the data firsthand, which is always a good
            practice! In doing this we use attributes and methods.
            '''
        )
        import_code.add_background_window(empty_cell.colabCode.window)
        cl_env.clear()
        self.play(import_code.IntoColab(cl_env))
        self.wait(1)
        self.play(cl_env.cells[0].Run())
        hand_cursor : Cursor = cl_env.cells[0].cursor
        self.wait(0.5)
        self.play(hand_cursor.animate.move_to(cl_env.PLUS_CODE_))
        self.play(hand_cursor.Click())
        cl_env.add_cell(empty_cell)
        self.play(cl_env.OutofColab(cl_env.cells[1]))

        # SLIDE 12:  ===========================================================
        # FIRST COMMENT LINE IS WRITTEN
        self.next_slide(
            notes=
            '''First of all, we find the number of rows and columns. We use
            the attribute of the dataset called [CLICK] 
            '''
        )
        dataset_size_code = ColabCode(
            r'''
            # Dataset dimensions
            print(my_dataset.shape)
            '''
        ).center()
        self.play(dataset_size_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 12:  ===========================================================
        # PRINT SHAPE LINE IS WRITTEN
        self.next_slide(
            notes=
            '''...shape to find the number of rows and columns.[CLICK] 
            '''
        )
        self.play(dataset_size_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 13:  ===========================================================
        # INTO COLAB, CELL IS RUN, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''In this case we have 59 rows and 4 columns. [CLICK]
            '''
        )
        dataset_size_code.add_background_window(empty_cell.colabCode.window)
        cl_env.clear()
        self.play(dataset_size_code.IntoColab(cl_env))
        cl_env.cells[0].add_output('(59,  4)')
        self.play(cl_env.cells[0].Run())

        # SLIDE 14:  ===========================================================
        # RETURN TO OUT OF COLAB
        # HEAD CODE IS WRITTEN
        self.next_slide(
            notes=
            '''Next, we use the method head. The line dataset.head(5)
            displays [CLICK]
            '''
        )
        DSS = DynamicSplitScreen(COLAB_LIGHTGRAY, WHITE)
        self.play(FadeIn(DSS.mainRect))
        self.remove(cl_env)
        dataset_head_code = ColabCode(
            r'''
            # Showing the data
            print("First 5 rows of the dataset:")
            dataset.head(5)
            '''
        ).center()
        self.play(dataset_head_code.TypeLetterbyLetter(lag_ratio=0))

        # SLIDE 14:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''...the first 5 rows of the dataframe. [CLICK]
            '''
        )
        dataset_head_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(dataset_head_code.IntoColab(cl_env))
        head_text = ColabBlockOutputText('First 5 rows of the dataset:')
        # creating the table
        dataset = np.genfromtxt(r'WEEK_2\supplementary_material\ALgerian_forest_dataset.csv', delimiter=',')
        row_labels = [Text(str(i), color=BLACK, font=CODE_FONT, weight=ULTRAHEAVY) for i in range(5)]
        col_labels = [Text(label,  color=BLACK, font=CODE_FONT, weight=ULTRAHEAVY) for label in ['Temperature', 'RH', 'BUI', 'FWI']]
        head_table = Table(dataset[1:6], row_labels=row_labels, col_labels=col_labels,
                          add_background_rectangles_to_entries=False,
                          element_to_mobject=CustomDecimalNumber,
                          element_to_mobject_config={'font':CODE_FONT,'color': BLACK, 'mob_class': Text, 'num_decimal_places':1},
                          line_config={'stroke_width':0},
                          arrange_in_grid_config={'cell_alignment': ORIGIN})
        for i in range(6):
            for j in range(5):
                color = WHITE if i % 2 ==0 else COLAB_LIGHTGRAY
                head_table.add_highlighted_cell((i+1,j+1),color=color)
                # color in table constructor does not work 
                head_table.get_entries((i+1, j+1)).set_color(BLACK)  
        head_table.scale(0.25).next_to(head_text, DOWN).align_to(head_text, LEFT)
        cl_env.cells[-1].add_output(VGroup(head_text, head_table))
        self.play(cl_env.cells[-1].Run())

        # SLIDE 15:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''In the table, each row represents a different fire event, while
            each column corresponds to a specific variable associated with
            that event:
            '''
        )
        self.play(cl_env.cells[-1].animate.focus_output(scale=0.8))

        # SLIDE 16:  ===========================================================
        # HIGHLIGHT TEMPERATURE
        self.next_slide(
            notes=
            '''the Temperature: expressed in Celsius degrees; [CLICK]
            '''
        )
        highlight_colors = [BLUE, TEAL, ORANGE, PINK]
        full_labels = VGroup(Text(label, font=CODE_FONT, color=BLACK) for label in
                       ['Temperature', 'Relative Humidity', 'Build-Up Index', 'Fire Weather Index'])
        full_labels.arrange_in_grid((4,1), cell_alignment=LEFT).next_to(head_table, RIGHT).shift(RIGHT)
        colored_dots = VGroup(Dot(color=highlight_colors[i], fill_opacity=0.4, stroke_width=0).next_to(full_labels[i], LEFT) for i in range(4))
        column_highlights = VGroup(HighlightRectangle(head_table.get_columns()[i+1][1:], color = highlight_colors[i]) for i in range(4))

        self.play(
            Create(column_highlights[0]),
            Create(colored_dots[0]),
            AddTextLetterByLetter(full_labels[i], rate_func=linear, time_per_char=0.01)
        )

        # SLIDE 17:  ===========================================================
        # HIGHLIGHT RELATIVE HUMIDITY
        self.next_slide(
            notes=
            '''the Relative Humidity: expressed as a percentage; [CLICK]
            '''
        )
        self.play(
            Create(column_highlights[1]),
            Create(colored_dots[1]),
            AddTextLetterByLetter(full_labels[1], rate_func=linear, time_per_char=0.01)
        )
        # SLIDE 18:  ===========================================================
        # HIGHLIGHT BUILD-UP INDEX
        self.next_slide(
            notes=
            '''the Build-Up Index, which represents the total quantity of
            combustible material in the environment. [CLICK]
            '''
        )
        self.play(
            Create(column_highlights[2]),
            Create(colored_dots[2]),
            AddTextLetterByLetter(full_labels[2], rate_func=linear, time_per_char=0.01)
        )
        # SLIDE 19:  ===========================================================
        # HIGHLIGHT FIRE WEATHER INDEX
        self.next_slide(
            notes=
            '''the Fire Weather Index, which evaluates the overall risk of
            forest fires. [CLICK]
            '''
        )
        self.play(
            Create(column_highlights[3]),
            Create(colored_dots[3]),
            AddTextLetterByLetter(full_labels[3], rate_func=linear, time_per_char=0.01)
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
        # NEW EMPTY EMPTY SCREEN FADES IN
        self.next_slide(
            notes=
            '''Now we are ready to implement linear regression. [CLICK]
            '''
        )
        linear_regression_code = ColabCode(
            r'''
            # Linear regression
            def linear_regression(x, y):
                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xy = np.sum(x * y)
                sum_x2 = np.sum(x ** 2)

                n = len(x)
                numerator = n*sum_xy - sum_x*sum_y
                denominator = n*sum_x2 - sum_x**2
                m = numerator / denominator
                q = (sum_y - m*sum_x)/n
                
                return m, q
            '''
        ).center()
        DSS.reset()
        self.play(FadeIn(DSS))
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[0]))

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
            '''The first list, contained in the variable "x", contains the x-coordinates of the points, 
            '''
        )
        # SLIDE 24:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''while the second list, named "y" contains the corresponding y-coordinates.
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
        # LINEAR REGRESSION FORMUALS APPEAR
        self.next_slide(
            notes=
            '''This function will perform linear regression by computing the coefficients m and q according to these formulas.
At first glance, this might seem overwhelming, but let's simplify it by breaking the task into smaller steps. 

            '''
        )
        self.play(DSS.bringOut())
        self.wait(0.2)
        LR_equations = LinearRegressionEquations().scale(0.75)
        DSS.add_side_obj(LR_equations)
        self.play(DSS.bringIn())
        # SLIDE 28:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''First we will compute the results of each sum, and then we will combine the results.
            Before starting the implementation, it's worth noting that some sums are repeated. [CLICK]
            '''
        )   
        self.play(
            AnimationGroup(
                *[Circumscribe(term, color=BLUE, run_time=3) 
                  for term in [
                    LR_equations.m_sum_x[0],
                    LR_equations.m_sum_x[1],
                    LR_equations.m_sum_y,
                    LR_equations.m_sum_x_y,
                    LR_equations.m_sum_x_sq,
                    LR_equations.q_sum_x,
                    LR_equations.q_sum_y,
                ]]
            )
        )

        # SLIDE 29:  ===========================================================
        # HIGHLIGHT SUMS OF y_i
        self.next_slide(
            notes=
            '''The sum over y_i appears twice, [CLICK] 
            '''
        )
        self.play(
            AnimationGroup(
                *[Circumscribe(term, color=BLUE, run_time=3) 
                  for term in [
                    LR_equations.m_sum_y,
                    LR_equations.q_sum_y,
                ]]
            )
        )

        # SLIDE 30:  ===========================================================
        # HIGHLIGHT SUMS OF x_i
        self.next_slide(
            notes=
            ''' and the sum over x_i even three times!
We can take advantage of this, and compute these terms once and reuse the results wherever needed
            '''
        )
        self.play(
            AnimationGroup(
                *[Circumscribe(term, color=BLUE, run_time=3) 
                  for term in [
                    LR_equations.m_sum_x[0],
                    LR_equations.m_sum_x[1],
                    LR_equations.q_sum_x,
                ]]
            )
        )

        # SLIDE 31:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''We are going to write now a python code that computes these four terms.
            '''
        )
        sum_terms = VGroup(LR_equations.m_sum_x.copy(), LR_equations.m_sum_y.copy(), LR_equations.m_sum_x_y.copy(), LR_equations.m_sum_x_sq.copy())
        sum_terms.arrange().scale(1.2).move_to(DSS.secondaryRect)
        # SLIDE 32:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''These sums can be computed by writing suitable "for" loops. However, we will compute them in a more concise and readable way by leveraging the np.sum function 
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
            '''The label "Temperature" extracts the corresponding column from the dataset 
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''and the attribute "values" returns the array.
            '''
        )
        # SLIDE :  ===========================================================
        # 
        self.next_slide(
            notes=
            '''
            '''
        )