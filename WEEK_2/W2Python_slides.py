import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from mooc_utils.colab import *
from W2Anim import *
import matplotlib.pyplot as plt

config.update(RELEASE_CONFIG)
config.max_files_cached = 200  # this presentation is particularly long

class W2Python_slides(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # AXIS WITH DATA POINTS APPEAR
        # REGRESSION LINE IS DRAWN
        # FORMULAS FOR m, q  APPEAR
        self.next_slide(
            notes=
            '''Let's explore how linear regression can be implemented in Python.
            Our ultimate goal is to answer key questions, such
            as: To what extent do variables like temperature and humidity affect
            the risk of wildfires? [CLICK]
            '''
        )
        LR_equations = LinearRegressionEquations().to_edge(UP).shift(UP*1.5)
        LR_equations.save_state()
        X_RANGE = (0, 1.5)
        ax = Axes(
            x_range=[X_RANGE[0], X_RANGE[1] + 0.1, 1],
            y_range=[0, 1.2, 1],
            x_length=9,
            y_length=9*1.2/(X_RANGE[1]+0.1),
            x_axis_config={'stroke_color':BLACK, 'include_ticks':False},
            y_axis_config={'stroke_color':BLACK, 'include_ticks':False}
        ).center().shift(DOWN*1.5)
        ax_labels = custom_get_axis_labels(ax, MathTex('x', color=BLUE).scale(0.75), MathTex('y', color=ORANGE).scale(0.75))
        dataset = generate_regression_dataset(func= lambda x: 1.5*(0.4*x-0.75)**3 + 0.8, x_range=(0.1, 1.5), n=20, sigma=0.15, seed=0)
        dataset_points = points_from_data(dataset, ax=ax, color=PURPLE_A).set_z_index(1)
        linear_fit = np.polynomial.polynomial.Polynomial.fit(dataset[:,0], dataset[:,1], 1).convert().coef
        reg_line = RegressionLine(linear_fit[1], linear_fit[0], ax, x_range=X_RANGE)

        ror_dx = 0.25
        ror_x = (X_RANGE[1] - X_RANGE[0])/2 - ror_dx/2
        rise_over_run = Polygon(
            reg_line.eval_to_point(ror_x),
            ax.c2p(ror_x+ror_dx, reg_line.eval(ror_x), 0),
            reg_line.eval_to_point(ror_x + ror_dx),
            color = PURPLE_C,
            fill_opacity=1,
            stroke_width=0
        ).set_z_index(2)
        slope_label = MathTex(r'\hat{m}', color=BLACK).scale(0.75).next_to(rise_over_run, UP).set_z_index(2)
        intercept_dot = Dot(ax.c2p(0, reg_line.intercept.get_value()), color=PURPLE_C)
        intercept_label = MathTex(r'\hat{q}', color=BLACK).scale(0.75).next_to(intercept_dot, LEFT)

        self.play(
            Succession(
                FadeIn(ax, ax_labels, dataset_points),
                Create(reg_line),
                FadeIn(rise_over_run, slope_label,intercept_dot, intercept_label),
                FadeIn(LR_equations)
            )
        )
        initial_graph = VGroup(ax, ax_labels, dataset_points, reg_line, rise_over_run, slope_label,intercept_dot, intercept_label, LR_equations)

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
        self.play(
            Succession(
                FadeOut(initial_graph),
                Wait(0.2),
                FadeIn(cl_env),
                Wait(1),
                GrowFromCenter(hand_cursor)
            )
        )

        # SLIDE 03:  ===========================================================
        # HAND CURSOR MOVES TO FOLDER ICON
        # FOLDER ICON IS CLICKED AND SIDE MENU APPEARS
        self.next_slide(
            notes=
            '''a side bar appears showing the list of available files. Let us
            click on the upload button, [CLICK]
            '''
        )
        self.play(Succession(hand_cursor.animate.move_to(cl_env.MENU_), hand_cursor.Click()))
        cl_env.set_image(r'Assets\W2\colabSLR_sidemenu.png')

        # SLIDE 04:  ===========================================================
        # HAND CURSOR MOVES TO UPLOAD BUTTON
        # UPLOAD BUTTON IS CLICKED, UOLUADED FILE APPEARS
        self.next_slide(
            notes=
            '''and select from your local file system the file
            Algerian_forest_dataset.csv. [CLICK] This dataset includes detailed
            features related to a set of forest fires recorded in Algeria.
            [CLICK]
            '''
        )
        self.play(Succession(hand_cursor.animate.move_to(cl_env.UPLOAD_), hand_cursor.Click()))
        cl_env.set_image(r'Assets\W2\colabSLR_uploaded.png'),

        # SLIDE 05:  ===========================================================
        # NEW CODE CELL IS CREATED AND ZOOMED IN
        self.next_slide(
            notes=
            '''next, we start coding. The first step is to import the modules we are going to use in the project. [CLICK]
            '''
        )
        empty_cell = ColabCodeBlock(code='')
        self.play(Succession(hand_cursor.animate.move_to(cl_env.PLUS_CODE_), hand_cursor.Click()))
        cl_env.set_image(r'Assets\W2\colabSLR.png')
        cl_env.add_cell(empty_cell)
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
            r'''
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            # Load the dataset
            my_dataset = pd.read_csv('Algerian_forest_dataset.csv')
            '''
        ).center()
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
        self.play(
            Succession(
                import_code.TypeLetterbyLetter(lines=[4]),
                Wait(0.5),
                import_code.TypeLetterbyLetter(lines=[5]),
            )
        )

        # SLIDE 10:  ===========================================================
        # DATAFRAME VARIABLE IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''The output of the function is stored in a variable named
            my_dataset, 
            [CLICK] ...
            '''
        )
        my_dataset_highlight = HighlightRectangle(import_code[5][:10])
        self.play(Create(my_dataset_highlight))

        # SLIDE 10:  ===========================================================
        # INTO COLAB, CODE IS RUN
        self.next_slide(
            notes=
            '''...which is an instance of the class DataFrame from pandas. 
            [CLICK]
            '''
        )
        DSS = DynamicSplitScreen(COLAB_LIGHTGRAY, WHITE, buff=0.5)
        self.add(DSS); self.remove(cl_env.cells[0].colabCode.window)
        cl_env.clear(self)
        import_code.add_background_window(DSS.mainRect.suspend_updating())
        
        self.play(FadeOut(my_dataset_highlight))
        self.play(import_code.IntoColab(cl_env))
        self.play(cl_env.Run(cell=0))

        # SLIDE 11:  ===========================================================
        # CLASS SCHEME APPEARS
        self.next_slide(
            notes=
            '''A class is more than just a data type: it defines the attributes, the characteristics that an object can have, but also the methods, the "actions" that we can perform on the and with the object.
            [CLICK]
            '''
        )
        class_title = Text("Class", font=CODE_FONT, color = COLAB_TEAL, weight=BOLD).scale(1.3)
        cs = VGroup(
            Dot(color=COLAB_BROWN), 
            Text("Attributes", font=CODE_FONT, color = COLAB_BROWN, weight=BOLD),
            Arrow(ORIGIN,RIGHT*1.5, color=BLACK),
            Text("Characteristics", font=CODE_FONT, color = BLACK),
            Dot(color=COLAB_BLUE), 
            Text("Methods", font=CODE_FONT, color = COLAB_BLUE, weight=BOLD),
            Arrow(ORIGIN,RIGHT*1.5, color=BLACK),
            Text('"Actions"', font=CODE_FONT, color = BLACK),
        )
        cs.arrange_in_grid(2, 4, buff = 0.5, cell_alignment=LEFT)

        class_title.next_to(cs,UP, buff = 0.7)
        
        css = VGroup(class_title, cs).scale(0.8).move_to( (cl_env.cells[0].get_bottom() + DOWN * FRAME_HEIGHT/2)/2)
        surrounding_css = SurroundingRectangle(VGroup(cs, class_title), fill_color=COLAB_LIGHTGRAY, fill_opacity=1, stroke_width=0.5,
                                              stroke_color=BLACK, corner_radius=0.2, buff=0.5)
        
        self.play(FadeIn(surrounding_css, css))

        # SLIDE 14:  ===========================================================
        # INTO COLAB, CELL IS RUN
        # NEW CELL APPEARS, OUT OF COLAB AGAIN
        self.next_slide(
            notes=
            '''Let's explore our dataset using both attributes and methods. [CLICK]
            '''
        )
        hand_cursor = cl_env.cursor
        self.play(
            Succession(
                FadeOut(surrounding_css, css),
                ApplyMethod(hand_cursor.move_to, cl_env.PLUS_CODE_),
                hand_cursor.Click()
            )
        )
        new_empty_cell=ColabCodeBlock(code='')
        cl_env.add_cell(new_empty_cell)
        self.add(cl_env)  # to update
        self.wait(0.5)
        self.play(cl_env.OutofColab(new_empty_cell))
        self.remove(hand_cursor)

        # SLIDE 15:  ===========================================================
        # FIRST COMMENT LINE IS WRITTEN
        self.next_slide(
            notes=
            '''First of all, we find the number of rows and columns. We use the
            attribute of the dataset called [CLICK] ...
            '''
        )
        dataset_size_code = ColabCode(
            r'''
            # Dataset dimensions
            print(my_dataset.shape)
            '''
        ).center()
        self.play(dataset_size_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 16:  ===========================================================
        # PRINT SHAPE LINE IS WRITTEN
        self.next_slide(
            notes=
            '''..."shape" to find the number of rows and columns. [CLICK]
            '''
        )
        self.play(dataset_size_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 17:  ===========================================================
        # INTO COLAB, CELL IS RUN, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''In this case we have 59 rows and 4 columns. [CLICK]
            '''
        )
        dataset_size_code.add_background_window(new_empty_cell.colabCode.window)
        cl_env.clear(self)
        self.play(dataset_size_code.IntoColab(cl_env))
        cl_env.cells[0].add_output('(59,  4)')
        self.play(cl_env.Run())

        # SLIDE 18:  ===========================================================
        # RETURN TO OUT OF COLAB
        # HEAD CODE IS WRITTEN
        self.next_slide(
            notes=
            '''Next, we use the method head. The line dataset.head(5) displays
            [CLICK]
            '''
        )
        DSS.reset()
        dataset_head_code = ColabCode(
            r'''
            # Showing the data
            print("First 5 rows of the dataset:")
            dataset.head(5)
            '''
        ).center()

        self.play(FadeIn(DSS.mainRect))
        self.play(dataset_head_code.TypeLetterbyLetter(lag_ratio=0))

        # SLIDE 19:  ===========================================================
        # DISPLAY HEAD TABLE AFTER RUNNNING CODE
        self.next_slide(
            notes=
            '''...the first 5 rows of the dataframe. [CLICK]
            '''
        )
        dataset_head_code.add_background_window(DSS.mainRect.suspend_updating())
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

        self.play(dataset_head_code.IntoColab(cl_env))
        cl_env.cells[1].add_output(VGroup(head_text, head_table))
        self.play(cl_env.Run(1, new_cursor=False))

        # SLIDE 20:  ===========================================================
        # FOCUS ON THE TABLE
        self.next_slide(
            notes=
            '''In the table, each row represents a different fire event, while
            each column corresponds to a specific variable associated with that
            event: [CLICK]
            '''
        )
        self.play(cl_env.focus_output(cell=1 ,scale=0.5, alignment=LEFT))

        # SLIDE 21:  ===========================================================
        # HIGHLIGHT TEMPERATURE COLUMN
        self.next_slide(
            notes=
            '''the Temperature: expressed in Celsius degrees; [CLICK]
            '''
        )
        highlight_colors = [BLUE, TEAL, ORANGE, PINK]
        full_labels = VGroup(Text(label, font=CODE_FONT, color=BLACK) for label in
                       ['Temperature', 'Relative Humidity', 'Build-Up Index', 'Fire Weather Index'])
        full_labels.scale(0.7).arrange_in_grid(4,1, cell_alignment=LEFT).next_to(head_table, RIGHT).shift(RIGHT)
        colored_dots = VGroup(Dot(color=highlight_colors[i], radius=0.15, fill_opacity=0.4, stroke_width=0).next_to(full_labels[i], LEFT) for i in range(4))
        column_highlights = VGroup(HighlightRectangle(head_table.get_columns()[i+1][1:], color = highlight_colors[i]) for i in range(4))

        self.play(
            Create(column_highlights[0]),
            GrowFromCenter(colored_dots[0]),
            AddTextLetterByLetter(full_labels[0], rate_func=linear, time_per_char=0.01)
        )

        # SLIDE 22:  ===========================================================
        # HIGHLIGHT RELATIVE HUMIDITY COLUMN
        self.next_slide(
            notes=
            '''the Relative Humidity: expressed as a percentage; [CLICK]
            '''
        )
        self.play(
            Create(column_highlights[1]),
            GrowFromCenter(colored_dots[1]),
            AddTextLetterByLetter(full_labels[1], rate_func=linear, time_per_char=0.01)
        )

        # SLIDE 23:  ===========================================================
        # HIGHLIGHT BUILD-UP INDEX COLUMN
        self.next_slide(
            notes=
            '''the Build-Up Index, which represents the total quantity of
            combustible material in the environment. [CLICK]
            '''
        )
        self.play(
            Create(column_highlights[2]),
            GrowFromCenter(colored_dots[2]),
            AddTextLetterByLetter(full_labels[2], rate_func=linear, time_per_char=0.01)
        )

        # SLIDE 24:  ===========================================================
        # HIGHLIGHT FIRE WEATHER INDEX COLUMN
        self.next_slide(
            notes=
            '''the Fire Weather Index, which evaluates the overall risk of
            forest fires. [CLICK]
            '''
        )
        self.play(
            Create(column_highlights[3]),
            GrowFromCenter(colored_dots[3]),
            AddTextLetterByLetter(full_labels[3], rate_func=linear, time_per_char=0.01)
        )

        # SLIDE 25:  ===========================================================
        # BRACES UNDER COLUMNS APPEAR WITH x_i, y_i labels
        self.next_slide(
            notes=
            '''In our context, temperature, relative humidity, and Build-up
            index can (separately) play the role of variable X, while the fire
            weather index plays the role of variable Y. [CLICK]
            '''
        )
        moving_brace = Brace(column_highlights[0], DOWN, color=BLACK)
        brace_x_label = MathTex('x_i', color=BLACK).scale(0.75)
        moving_brace.put_at_tip(brace_x_label)
        x_brace = VGroup(moving_brace, brace_x_label)
        second_brace = moving_brace.copy().next_to(column_highlights[3], DOWN).align_to(moving_brace, UP)
        brace_y_label = MathTex('y_i', color=BLACK).scale(0.75)
        second_brace.put_at_tip(brace_y_label)
        y_brace = VGroup(second_brace, brace_y_label)

        self.play(
            Succession(
                FadeIn(x_brace),
                Wait(0.5),
                ApplyMethod(x_brace.match_x, column_highlights[1]),
                Wait(0.5),
                ApplyMethod(x_brace.match_x, column_highlights[2]),
                Wait(0.5),
                FadeIn(y_brace)
            )
        )

        # SLIDE 26:  ===========================================================
        # NEW EMPTY EMPTY SCREEN FADES IN
        # # LINEAR REGRESSION WRITTEN
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
        DSS.mainRect.set_z_index(1)
        self.play(FadeIn(DSS))
        cl_env.clear(self)
        self.remove(cl_env, *full_labels, *colored_dots, *column_highlights, x_brace, y_brace)
        DSS.mainRect.set_z_index(-1)
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 27:  ===========================================================
        # SCHEMATIC DRAWING OF THE FUNCTION IS BROUGHT IN
        self.next_slide(
            notes=
            '''We are going to write a function that [CLICK] takes as inputs the
            datapoints, organized in two lists. [CLICK]
            '''
        )
        fscheme = FunctionAbstraction(scale=0.7)
        DSS.add_side_obj(fscheme)
        DSS.add_main_obj(linear_regression_code[0], follow_obj=linear_regression_code[1:])

        self.play(DSS.bringIn())
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 28:  ===========================================================
        # X INPUT WITH ITS ARRAY APPEARS
        self.next_slide(
            notes=
            '''The first list, contained in the variable "x", contains the
            x-coordinates of the points, [CLICK]
            '''
        )
        fscheme.add_inputs("x", "y")
        self.remove(fscheme.InputArrows, fscheme.InputLabels)
        x_vector = MathTex(r'[x_1, x_2, \dots, x_n]', color=BLACK, tex_to_color_map={'x_1':BLUE, 'x_2':BLUE,'x_n':BLUE}).next_to(fscheme.InputLabels[0], LEFT, buff=0.5)
        y_vector = MathTex(r'[y_1, y_2, \dots, y_n]', color=BLACK, tex_to_color_map={'y_1':ORANGE, 'y_2':ORANGE,'y_n':ORANGE}).next_to(fscheme.InputLabels[1], LEFT, buff=0.5)
        
        self.play(FadeIn(x_vector, fscheme.InputArrows[0], fscheme.InputLabels[0]))

        # SLIDE 29:  ===========================================================
        # Y INPUT WITH ITS ARRAY APPEARS
        self.next_slide(
            notes=
            '''while the second list, named "y" contains the corresponding
            y-coordinates. [CLICK]
            '''
        )
        self.play(FadeIn(y_vector, fscheme.InputArrows[1], fscheme.InputLabels[1]))
        
        # SLIDE 30:  ===========================================================
        # OUTPUTS M,Q APPEAR
        # RETURN LINE IS WRITTEN
        self.next_slide(
            notes=
            '''The function will return the coefficients of the regression
            lines, namely m and q. This is the structure of the Python function
            that we will write. The function will have two inputs (x and y) and
            two outputs (m and q). What we need to do now is to fill in the
            dots. [CLICK]
            '''
        )
        fscheme.add_outputs("m", "q")
        m_label = MathTex(r"\hat{m}", color=BLACK).next_to(fscheme.OutputLabels[0], RIGHT, buff=1)
        q_label = MathTex(r"\hat{q}", color=BLACK).next_to(fscheme.OutputLabels[1], RIGHT, buff=1)

        short_linear_regression_code = ColabCode(
            r'''
            ...
            return m, q
            '''
        ).align_to(linear_regression_code[2:4], DL)

        self.play(
            Succession(
                FadeIn(fscheme.OutputArrows, fscheme.OutputLabels, m_label, q_label),
                Wait(0.5),
                short_linear_regression_code.TypeLetterbyLetter()
            )
        )

        self.play
        # SLIDE 31:  ===========================================================
        # LINEAR REGRESSION FORMULAS APPEAR
        self.next_slide(
            notes=
            '''This function will perform linear regression by computing the
            coefficients m and q according to these formulas. At first glance,
            this might seem overwhelming, but let's simplify it by breaking the
            task into smaller steps. [CLICK]
            '''
        )
        DSS.remove_main_obj()
        self.play(
            DSS.bringOut(),
            VGroup(x_vector, y_vector, m_label, q_label).animate.shift(UP*DSS.secondaryRect.height),
        )
        self.wait(0.2)
        LR_equations.restore()
        DSS.add_side_obj(LR_equations)
        self.play(DSS.bringIn())
        DSS.add_main_obj(linear_regression_code[:2], linear_regression_code[2:])

        # SLIDE 32:  ===========================================================
        # ALL SUMS ARE HIGHLIGHTED
        self.next_slide(
            notes=
            '''First we will compute the results of each sum, and then we will
            combine the results. Before starting the implementation, it's worth
            noting that some sums are repeated. The sum over y_i appears twice, and the sum over x_i even three times! We can take advantage
            of this, and compute these terms once and reuse the results wherever
            needed. [CLICK]
            '''
        ) 
        sums_highlights = VGroup(
            HighlightRectangle(LR_equations.m_sum_x_y, BLUE),
            HighlightRectangle(LR_equations.m_sum_x[0], TEAL),
            HighlightRectangle(LR_equations.m_sum_x[1], TEAL),
            HighlightRectangle(LR_equations.q_sum_x, TEAL),
            HighlightRectangle(LR_equations.m_sum_y, PINK),
            HighlightRectangle(LR_equations.q_sum_y, PINK),
            HighlightRectangle(LR_equations.m_sum_x_sq, ORANGE),
        )  
        # sum_y_highlights = VGroup(sums_highlights[2], sums_highlights[6])
        # sum_x_highlights = VGroup(sums_highlights[i] for i in [0,1,5])
        # sum_xy_x2_highlights =VGroup(sums_highlights[i] for i in [3,4])
        
        self.play(FadeIn(sums_highlights))

        # SLIDE 35:  ===========================================================
        # THE NON RPEATED SUM TERMS ARE EXTRACTED FROM THE EQUATIONS
        self.next_slide(
            notes=
            '''We are going to write now a python code that computes these four
            terms. [CLICK]
            '''
        )
        self.play(FadeOut(sums_highlights))
        sum_terms = LR_equations.get_sums_without_repetition().arrange(RIGHT, buff=1).scale(1.2).move_to(DSS.secondaryRect)
        self.play(LR_equations.ExtractSumTerms(target=sum_terms))

        # SLIDE 36:  ===========================================================
        # SAMPLE SUM FOR LOOP CODE IS WRITTEN
        self.next_slide(
            notes=
            '''These sums can be computed by writing suitable "for" loops.
            However, we will compute them in a more concise and readable way by
            leveraging the np.sum function from the NumPy library (imported as
            np). [CLICK]
            '''
        )
        for_sum_code = ColabCode(
            r'''
            sum_x = 0
            for i in range(len(x)):
                sum_x = sum_x + x[i]
            '''
        ).align_to(linear_regression_code[2], UL)

        self.play(
            Succession(
                FadeOut(short_linear_regression_code),
                for_sum_code.TypeLetterbyLetter()
            )
        )

        # SLIDE 37:  ===========================================================
        # SAMPLE SUM FOR LOOP CODE TRANSFORMS INTO NP.SUM(X)
        self.next_slide(
            notes=
            '''This function allows us to compute directly the sum of all
            elements in an array. For example, np.sum(x) calculates the sum of
            all the x-coordinates. [CLICK]
            '''
        )
        code_sum_highlights = [
            HighlightRectangle(linear_regression_code[2], TEAL),
            HighlightRectangle(sum_terms[0], TEAL),
            HighlightRectangle(linear_regression_code[3], PINK),
            HighlightRectangle(sum_terms[1], PINK),
        ]

        self.play(ReplacementTransform(for_sum_code, linear_regression_code[2]))
        self.play(
            Create(code_sum_highlights[0]),
            Create(code_sum_highlights[1]),
        )

        # SLIDE 38:  ===========================================================
        # NP.SUM(Y) LINE WRITTEN
        self.next_slide(
            notes=
            '''Similarly, np.sum(y) calculates the sum of all the yi. [CLICK]
            '''
        )
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[3]))
        self.play(
            Create(code_sum_highlights[2]),
            Create(code_sum_highlights[3]),
        )

        # SLIDE 39:  ===========================================================
        # EMPTY RECTANGLE BROUGHT IN ON TOP
        # VECTORIZED OPERATIONS TITLE WRITTEN
        self.next_slide(
            notes=
            '''To compute the last two terms, we need to introduce a powerful
            concept in Python: vectorized operations. [CLICK]
            '''
        )
        DSS.add_side_obj(sum_terms)
        DSS.remove_main_obj()
        self.play(FadeOut(*code_sum_highlights))
        self.play(DSS.bringOut())

        title = Text('Vectorized operations', font=SANS_SERIF_FONT, weight=LIGHT, font_size=64, color=BLACK, stroke_color=BLACK)
        x_vector =  VectorArray(arrangement='vertical', include_dots=True, array=[f'x[{i}]' for i in [0,1,2,'n']]).scale(0.6)
        y_vector =  VectorArray(arrangement='vertical', include_dots=True, array=[f'y[{i}]' for i in [0,1,2,'n']]).scale(0.6)
        xy_vector = VectorArray(arrangement='vertical', include_dots=True, array=[f'x[{i}]*y[{i}]' for i in [0,1,2,'n']]).scale(0.6)
        x2_vector = VectorArray(arrangement='vertical', include_dots=True, array=[f'x[{i}]**2' for i in [0,1,2,'n']]).scale(0.6)
        vector_labels = [Text(s, color=BLACK, font=CODE_FONT) for s in ['x', 'y', 'x*y', 'x**2']]

        y_vector.next_to(x_vector, RIGHT)
        xy_vector.next_to(y_vector, RIGHT, buff=1)
        vector_labels[0].next_to(x_vector, UP)
        vector_labels[1].next_to(y_vector, UP).align_to(vector_labels[0], UP)
        vector_labels[2].next_to(xy_vector, UP).align_to(vector_labels[1], DOWN)
        first_group=VGroup(x_vector, y_vector, xy_vector, *vector_labels[:3])
        title.next_to(first_group, UP, buff=0.5)
        first_group.add(title)
        first_group.center()

        mixed_sum_terms = sum_terms[2:].scale(0.8).arrange_in_grid(2, 1, buff=1, cell_alignment=LEFT).next_to(x_vector, LEFT, buff = 0.8)

        # DSS.add_empty_side_obj(first_group.height)
        # DSS.add_main_obj(linear_regression_code[:4], linear_regression_code[4:])
        # self.play(DSS.bringIn(consider_follow=linear_regression_code[4:6]))
        # first_group.move_to(DSS.secondaryRect)
        DSS2 = DynamicSplitScreen(WHITE, COLAB_LIGHTGRAY, direction=DOWN, buff = 0.5)
        DSS2.add_empty_side_obj(FRAME_HEIGHT)
        DSS2.hard_bring_in()
        DSS2.add_side_obj(linear_regression_code[:4], linear_regression_code[4:])
        mixed_sum_terms.shift(UP*FRAME_HEIGHT)
        self.add(DSS2); self.remove(DSS)

        self.play(
            DSS2.bringOut(),
            mixed_sum_terms.animate.shift(DOWN*FRAME_HEIGHT)
        )

        self.play(Write(title))

        # SLIDE 40:  ===========================================================
        # X, Y COLUMN VECTORS APPEAR
        self.next_slide(
            notes=
            '''Let us consider the vectors x and y, containing the elements xi
            and yi, respectively. [CLICK]
            '''
        )
        self.play(
            Create(x_vector),
            Create(y_vector),
            FadeIn(*vector_labels[:2])
        )
        
        # SLIDE 41:  ===========================================================
        # EMPTY X*Y VECTOR APPEARS
        self.next_slide(
            notes=
            '''The operation x * y creates a new array, [CLICK]...
            '''
        )
        xy_term_highlights = VGroup(
            HighlightRectangle(vector_labels[2][1]), # * in x*y
            HighlightRectangle(mixed_sum_terms[0][5:]),    # x_i y_i
        )

        self.play(
            Create(xy_vector.get_lines()),
            FadeIn(vector_labels[2])
        )
        self.play(Create(xy_term_highlights[0]), Create(xy_term_highlights[1]))

        # SLIDE 42:  ===========================================================
        # X, Y TERMS ANIMATED INTO X*Y TERMS
        self.next_slide(
            notes=
            '''...whose first entry is the product of the first entries of x and
            y, the second entry is the product of the second entries, all the
            way up to the last element. [CLICK]
            '''
        )
        self.play(
            AnimationGroup(
                *[AnimationGroup(
                    ReplacementTransform(y_vector.get_entries((i, 1)).copy().set_opacity(0), xy_vector.get_entries((i, 1))[-4:]),
                    ReplacementTransform(x_vector.get_entries((i, 1)).copy().set_opacity(0), xy_vector.get_entries((i, 1))[:4]),
                    FadeIn(xy_vector.get_entries((i, 1))[4]),
                    lag_ratio=0.2
                ) if i !=4 else FadeIn(xy_vector.get_entries((i, 1)))
                for i in range(1,6)],
                lag_ratio=0.2
            )
        )

        # SLIDE 43:  ===========================================================
        # NP.SUM(X * Y) LINE WRITTEN
        self.next_slide(
            notes=
            '''As a consequence, with np.sum(x * y) we compute the sum of all
            the products xi times yi, that is the term called sum_xy. [CLICK]
            '''
        )
        DSS2.add_side_obj(linear_regression_code[:4], linear_regression_code[4:], consider_follow=linear_regression_code[4:6], center_horizontally=False)
        DSS2.add_main_obj(VGroup(first_group, mixed_sum_terms))

        self.play(FadeOut(xy_term_highlights))
        self.play(DSS2.bringIn())
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[4]))

        second_xy_term_highlights = VGroup(   
            HighlightRectangle(mixed_sum_terms[0]),    # whole sum
            HighlightRectangle(linear_regression_code[4]),    # line of code
        )

        self.play(Create(second_xy_term_highlights[0]), Create(second_xy_term_highlights[1]))

        # SLIDE 44:  ===========================================================
        # EMPTY X**2 APPEARS
        # ANIMATE X TERMS INTO X**2 TERMS
        self.next_slide(
            notes=
            '''Similarly, the element-wise power operation x ** 2 is also
            vectorized, meaning it applies the power operation to each element
            of the array individually. [CLICK]
            '''
        )
        DSS2.add_side_obj(linear_regression_code[:5], linear_regression_code[5:]) #, consider_follow=linear_regression_code[5])
        self.play(FadeOut(second_xy_term_highlights))
        self.play(DSS2.bringOut())

        x2_vector.move_to(xy_vector)
        vector_labels[-1].next_to(x2_vector, UP).align_to(vector_labels[0], DOWN)
        x2_term_highlights = VGroup(
            HighlightRectangle(vector_labels[3][1:]), # **2
            HighlightRectangle(mixed_sum_terms[1][6]),    # ^2 in the sum
        )

        self.play(
            Succession(
                FadeOut(y_vector, xy_vector, *vector_labels[1:3]) ,
                AnimationGroup(
                    Create(x2_vector.get_lines()),
                    FadeIn(vector_labels[-1])
                ),
                AnimationGroup(
                    Create(x2_term_highlights[0]),
                    Create(x2_term_highlights[1])
                )
            )
        )
        self.play(
            AnimationGroup(
                *[AnimationGroup(
                    ReplacementTransform(x_vector.get_entries((i, 1)).copy().set_opacity(0), x2_vector.get_entries((i, 1))[:4]),
                    FadeIn(x2_vector.get_entries((i, 1))[4:]),
                    lag_ratio=0.2
                ) if i !=4 else FadeIn(x2_vector.get_entries((i, 1)))
                for i in range(1,6)],
                lag_ratio=0.2
            )
        )

        # SLIDE 45:  ===========================================================
        # NP.SUM(X**2) LINE WRITTEN
        self.next_slide(
            notes=
            '''...so that combining this operation with np.sum gives the last
            term. [CLICK]
            '''
        )
        self.play(FadeOut(x2_term_highlights))

        DSS2.add_main_obj(VGroup(title, x_vector, x2_vector, vector_labels[0], vector_labels[-1], mixed_sum_terms))
        self.play(DSS2.bringIn())

        second_x2_term_highlights = VGroup(
            HighlightRectangle(mixed_sum_terms[1]),    # whole sum
            HighlightRectangle(linear_regression_code[5]),    # line of code
        )

        self.play(linear_regression_code.TypeLetterbyLetter(lines=[5]))
        self.play(Create(second_x2_term_highlights[0]), Create(second_x2_term_highlights[1]))

        # SLIDE 46:  ===========================================================
        # VECTORIZED OPERATIONS BROUGHT OUT OF FRAME
        # 'm', 'q' EQUATIONS BROUGHT BACK IN
        self.next_slide(
            notes=
            '''Very good. Now the hardest part is behind us. We just need to
            combine these quantities to finalize the computation. [CLICK]
            '''
        )
        self.play(FadeOut(second_x2_term_highlights))
        # need to switch back to primary DSS
        DSS.reset()
        DSS.add_empty_side_obj(DSS2.mainRect.height)
        DSS.hard_bring_in()
        DSS.add_side_obj( VGroup(x_vector, x2_vector, vector_labels[0], vector_labels[-1], title, mixed_sum_terms))
        DSS2.remove_side_obj(); self.add(DSS2); self.remove(DSS2)
        self.add(DSS)

        self.play(DSS.bringOut())

        DSS.add_side_obj(LR_equations.restore())
        DSS.add_main_obj(linear_regression_code[:6], follow_obj=linear_regression_code[6:])
        self.play(DSS.bringIn())

        # SLIDE 47:  ===========================================================
        # N = LEN(X) WRITTEN
        self.next_slide(
            notes=
            '''First, we compute the length of x, [CLICK] ...
            '''
        )
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[7]))

        # SLIDE 47:  ===========================================================
        # NUMERATOR LINE WRITTEN
        self.next_slide(
            notes=
            '''...and then the numerator of the expression giving m.
            [CLICK]
            '''
        )
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[8]))

        # SLIDE 48:  ===========================================================
        # DENOMINATOR LINE WRITTEN
        self.next_slide(
            notes=
            '''Then, we compute the denominator, [CLICK] ...
            '''
        )
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[9]))

        # SLIDE 49:  ===========================================================
        # NUMERATOR/DENOMINATOR LINE WRITTEN
        self.next_slide(
            notes=
            '''...and we divide the numerator by the denominator to obtain the
            value of m. [CLICK]
            '''
        )
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[10]))

        # SLIDE 50:  ===========================================================
        # Q LINE WRITTEN
        # RETURN M, Q LINE WRITTEN
        self.next_slide(
            notes=
            '''Finally, we calculate q, making use of the m value we just
            determined. This completes the computation of the regression
            coefficients m and q. [CLICK]
            '''
        )
        self.play(
            Succession(
                linear_regression_code.TypeLetterbyLetter(lines=[11]),
                Wait(1),
                linear_regression_code.TypeLetterbyLetter(lines=[13])
            )
        )

        # SLIDE 51:  ===========================================================
        # BRING OUT TOP RECTANGLE TO LEAVE ONLY THE FUNCTION AS THE FOCUS
        self.next_slide(
            notes=
            '''Great! We have completed all the necessary steps for the
            implementation of our function. [CLICK]
            '''
        )
        DSS.add_main_obj(linear_regression_code[:])
        self.play(DSS.bringOut())
        
        # SLIDE 56:  ===========================================================
        # INTO COLAB, FUNCTION DEFINITION CELL IS RUN
        self.next_slide(
            notes=
            '''Now that the function has been written we can run the block and
            it is ready to be used. [CLICK]
            '''
        )
        linear_regression_code.add_background_window(DSS.mainRect.suspend_updating())
        cl_env.clear(self)
        self.play(linear_regression_code.IntoColab(cl_env))
        self.play(cl_env.Run())

        # SLIDE 57:  ===========================================================
        # NEW EMPTY SCREEN FADES IN
        # FIRST COMMENT LINE IS WRITTEN
        self.next_slide(
            notes=
            '''In this way, we are ready to apply it to the Algerian forest
            dataset. We wonder how the temperature influences the Fire Weather
            Index, [CLICK] ...
            '''
        )
        LR_example_code = ColabCode(
            r'''
            # Perform simple linear regression
            x = my_dataset['Temperature'].values
            y = my_dataset['FWI'].values

            m, q = linear_regression(x, y)

            # Print results
            print('Linear model results:')
            print(f'Slope (m): {m}')
            print(f'Y-intercept (q): {q}')
            '''
        ).center()
        DSS.reset()
        self.play(FadeIn(DSS))

        self.play(LR_example_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 58:  ===========================================================
        # X = TEMPERATURE CODE LINE APPEARS
        self.next_slide(
            notes=
            '''To this goal, we chose the temperature as the x, using this code:
            [CLICK]
            '''
        )
        self.play(LR_example_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 59:  ===========================================================
        # HIGHLIGHT 'TEMPERATURE'
        self.next_slide(
            notes=
            '''The label "Temperature" extracts the corresponding column from
            the dataset, [CLICK] ...
            '''
        )
        temperature_highlight = HighlightRectangle(LR_example_code[1][13:26])
        values_highlight = HighlightRectangle(LR_example_code[1][28:])

        self.play(Create(temperature_highlight))
        
        # SLIDE 60:  ===========================================================
        # HIGHLIGHT VALUES
        self.next_slide(
            notes=
            '''and the attribute "values" returns the array. [CLICK]
            '''
        )
        self.play(ReplacementTransform(temperature_highlight, values_highlight))
        
        # SLIDE 61:  ===========================================================
        # Y = FWI CODE LINE APPEARS
        self.next_slide(
            notes=
            '''And similarly, for FWI which becomes y. [CLICK]
            '''
        )
        self.play(FadeOut(values_highlight))
        self.play(LR_example_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 62:  ===========================================================
        # LINEAR REGRESSION FUNCTION CALL LINE IS WRITTEN
        self.next_slide(
            notes=
            '''These two arrays are passed as inputs to our linear_regression
            function. The function returns two outputs, which we store in two
            separate variables. Please note that the order of these variables
            must match the order specified in the return statement of the
            function. [CLICK]
            '''
        )
        self.play(LR_example_code.TypeLetterbyLetter(lines=[4]))

        # SLIDE 63:  ===========================================================
        # PRINT LINES ARE WRITTEN
        self.next_slide(
            notes=
            '''Finally, we print the results: the regression coefficients m and
            q. [CLICK]
            '''
        )
        self.play(LR_example_code.TypeLetterbyLetter(lines=range(6, 10), lag_ratio=0))

        # SLIDE 64:  ===========================================================
        # 'f' F-STRINGS HIGHLIGHTED
        self.next_slide(
            notes=
            '''By putting the letter f in front of a string, [CLICK] ...
            '''
        )
        f_string_highlights = VGroup(
            HighlightRectangle(LR_example_code[8][6]),
            HighlightRectangle(LR_example_code[9][6])
        )

        self.play(Create(h) for h in f_string_highlights)

        # SLIDE 65:  ===========================================================
        # 'm', 'q' IN F-STRINGS HIGHLIGHTED
        self.next_slide(
            notes=
            '''...you allow Python to
            interpret variables inside curly braces directly within the string. [CLICK]
            '''
        )
        mq_f_string_highlights = VGroup(
            HighlightRectangle(LR_example_code[8][18]),
            HighlightRectangle(LR_example_code[9][24])
        )

        self.play(ReplacementTransform(fh, dot_h) for fh, dot_h in zip(f_string_highlights, mq_f_string_highlights))

        # SLIDE 66:  ===========================================================
        # INTO COLAB
        # CELL IS RUN, RESULT OUTPUT APPEARS
        self.next_slide(
            notes=
            '''Running this cell in the notebook, we see the results printed on
            the screen. [CLICK]
            '''
        )
        self.play(FadeOut(mq_f_string_highlights))
        # cl_env.clear()
        LR_example_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(LR_example_code.IntoColab(cl_env))
        cl_env.cells[1].add_output(
            'Linear model results:\n'
            'Slope (m): 1.421975321551814\n'
            'Y-intercept (q): -36.219188539161344'
        )
        self.play(cl_env.Run(1, new_cursor=False))

        # SLIDE 67:  ===========================================================
        # RETURN TO EMPTY SCREEN
        # WRITE # PLOTTING
        self.next_slide(
            notes=
            '''Let's now visualize the datapoints together with the regression
            line. By using the module matplotlib imported as plt, [CLICK]
            '''
        )
        plotting_code = ColabCode(
            r'''
            # Plotting
            plt.figure(figsize=(16, 10))
            plt.scatter(x, y, color='blue', alpha=0.5, label='Data points')
            plt.plot(x, m*x+q, color='red', label='Regression line')

            plt.xlabel('Temperature')
            plt.ylabel('FWI')
            plt.title('Linear Regression: FWI vs Temperature')
            plt.grid(True)
            plt.legend()
            '''
        ).center()
        # plotting_code.code.save_state()
        DSS.reset()
        self.play(FadeIn(DSS))
        self.add(cl_env); self.remove(cl_env)
        cl_env.clear(self)
        self.play(plotting_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 68:  ===========================================================
        # PLT.FIGURE LINE WRITTEN
        self.next_slide(
            notes=
            '''...we create a new figure, and we use the function [CLICK] ...
            '''
        )
        def draw_plot(fig) -> ImageMobject:
            fig.canvas.draw()
            buf1 = np.array(fig.canvas.buffer_rgba())
            return ImageMobject(buf1)
        
        PLOT_WIDTH=7
        
        # create the plot with matplotlib
        temp, RH, FWI = dataset[1:, 0], dataset[1:, 1], dataset[1:, -1]
        q, m = np.polynomial.polynomial.Polynomial.fit(temp, FWI, 1).convert().coef
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        DSS.add_side_obj(draw_plot(fig).scale_to_fit_width(PLOT_WIDTH))
        
        self.play(plotting_code.TypeLetterbyLetter(lines=[1]))
        DSS.add_main_obj(plotting_code[:2], follow_obj=plotting_code[2:])
        self.play(DSS.bringIn(consider_follow = plotting_code[2:4]))

        # SLIDE 69:  ===========================================================
        # PLT.SCATTER LINE WRITTEN
        self.next_slide(
            notes=
            '''...scatter to plot the datapoints as small blue circles in the
            x-y plane. [CLICK]
            '''
        )
        ax.scatter(temp, FWI, color='blue', alpha=0.5, label='Data points')
        new_plot = draw_plot(fig).scale_to_fit_width(PLOT_WIDTH).move_to(DSS.secondaryObj)

        self.play(plotting_code.TypeLetterbyLetter(lines=[2]))
        self.play(DSS.secondaryObj.animate.become(new_plot))

        # SLIDE 70:  ===========================================================
        # PLOT LINE APPEARS
        self.next_slide(
            notes=
            '''Then, we use the function "plot" to display the regression line
            in red. Finally, we can enhance the plot's readability by adding:
            [CLICK]
            '''
        )
        ax.plot(temp, m*temp+q, color='red', label='Regression line')
        new_plot = draw_plot(fig).scale_to_fit_width(PLOT_WIDTH).move_to(DSS.secondaryObj)

        self.play(plotting_code.TypeLetterbyLetter(lines=[3]))
        self.play(DSS.secondaryObj.animate.become(new_plot))

        # SLIDE 71:  ===========================================================
        # X, Y LABEL LINES AND RESULT SHOWN
        self.next_slide(
            notes=
            '''...axis labels, [CLICK]
            '''
        )
        ax.set_xlabel('Temperature')
        ax.set_ylabel('FWI')
        new_plot = draw_plot(fig).scale_to_fit_width(PLOT_WIDTH).move_to(DSS.secondaryObj)
        DSS.add_main_obj(plotting_code[:4], plotting_code[4:])

        self.play(DSS._MainObjIntoPosition()) # recenter considering the entire code
        self.play(plotting_code.TypeLetterbyLetter(lines=[5, 6], lag_ratio=0))
        self.play(DSS.secondaryObj.animate.become(new_plot))

        # SLIDE 72:  ===========================================================
        # GRID, TITLE LEGEND LINES AND RESULT SHOWN
        self.next_slide(
            notes=
            '''...a title, a grid and a legend. Please notice that the "legend"
            function leverages the [CLICK] ...
            '''
        )
        ax.set_title('Linear Regression: FWI vs Temperature')
        ax.grid(True)
        ax.legend()
        new_plot = draw_plot(fig).scale_to_fit_width(PLOT_WIDTH).move_to(DSS.secondaryObj)

        self.play(plotting_code.TypeLetterbyLetter(lines=[7, 8, 9]))
        self.play(DSS.secondaryObj.animate.become(new_plot))

        # SLIDE 73:  ===========================================================
        # LABEL KEYWORDS HIGHLIGHTED
        self.next_slide(
            notes=
            '''...argument "label" which we previously specified when plotting
            the data points and the regression line, to identify them in the
            plot. [CLICK]
            '''
        )
        plot_label_highlights = [
            HighlightRectangle(plotting_code[2][39:44]),
            HighlightRectangle(plotting_code[3][29:34])
        ]
        self.play(*[Create(highlight) for highlight in plot_label_highlights])

        # SLIDE 74:  ===========================================================
        # INTO COLAB
        # CELL IS RUN, FINAL PLOT APPEARS
        self.next_slide(
            notes=
            '''Running this cell, the plot is shown in the notebook. [CLICK]
            '''
        )
        cl_env.clear(self)
        plotting_code.add_background_window(DSS.mainRect.suspend_updating())
        DSS.remove_main_obj()
        self.play(FadeOut(*plot_label_highlights))
        self.play(
            plotting_code.IntoColab(cl_env),
            DSS.bringOut()
        )

        # save figure as array
        temp_fwi_plot = draw_plot(fig).scale_to_fit_width(6)
        
        cl_env.cells[0].add_output(temp_fwi_plot)
        self.play(cl_env.Run())
        
        # SLIDE 75:  ===========================================================
        # FOCUS ON PLOT
        self.next_slide(
            notes=
            '''The slope is positive, indicating that an increase in the
            temperature corresponds to a higher overall risk of forest fires.
            [CLICK]
            '''
        )
        self.play(cl_env.focus_output(0, scale=0.7))

        # SLIDE 76:  ===========================================================
        # PLOT CODE COMES BACK IN
        # 'TEMPERATURE' IS REPLACED WITH 'RH' IN THE CODE
        self.next_slide(
            notes=
            '''Now, let's repeat the above procedure using the relative humidity
            instead of the temperature. To do that, just replace the string
            'Temperature' with 'RH' in the previous lines of code. [CLICK]
            '''
        )
        # NOTE: the trailing space is added to the line x = my_dataset.., otherwise the font size changes??? I'm so done.
        LR_example_code_2 = ColabCode(LR_example_code.code_string).center()
        replacement_code = ColabCode(LR_example_code.code_string.replace(r"""'Temperature'].values""", r"""'RH'].values """))
        replacement_code.move_to(LR_example_code_2).align_to(LR_example_code_2, UL)

        DSS.reset()
        self.play(
            Succession(
                FadeIn(DSS.mainRect.set_z_index(0)),
                FadeIn(LR_example_code_2),
                Wait(1)
            )
        )
        cl_env.clear(self)
        DSS.mainRect.set_z_index(-1)
        self.play(
            Transform(LR_example_code_2[1][14:25], replacement_code[1][14:16]),
            Transform(LR_example_code_2[1][25:], replacement_code[1][16:]),
        )
        self.add(replacement_code); self.add(LR_example_code_2); self.remove(LR_example_code_2) # need to readd otherwise remove does not work...

        # create other plot
        q, m = np.polynomial.polynomial.Polynomial.fit(RH, FWI, 1).convert().coef
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.scatter(RH, FWI, color='blue', alpha=0.5, label='Data points')
        ax.plot(RH, m*RH+q, color='red', label='Regression line')
        ax.set_xlabel('RH')
        ax.set_ylabel('FWI')
        ax.set_title('Linear Regression: FWI vs RH')
        ax.grid(True)
        ax.legend()
        # save figure as array
        fig.canvas.draw()
        buf1 = np.array(fig.canvas.buffer_rgba())
        rh_fwi_plot = ImageMobject(buf1).scale_to_fit_width(6)

        cl_env.add_cell(ColabCodeBlock(replacement_code.code_string))
        cl_env.add_cell(ColabCodeBlock(plotting_code.code_string.replace('Temperature','RH')))
        replacement_code.add_background_window(DSS.mainRect.suspend_updating())
        
        self.play(replacement_code.IntoColab(cl_env, target_cell=0))
        self.play(cl_env.Run(0))
        cl_env.cells[1].add_output(rh_fwi_plot)
        self.remove(cl_env.cells[1].output) # it gets shown too early
        self.play(cl_env.Run(1, new_cursor=False))

        # SLIDE 77:  ===========================================================
        # FOCUS ON THE SECOND PLOT
        self.next_slide(
            notes=
            '''From the plot it can be seen an opposite trend with respect to
            the previous case: this means that an increase in relative humidity
            decreases the overall risk of forest fire. [CLICK]
            '''
        )
        self.play(cl_env.focus_output(cell=1, scale=0.7))

        # SLIDE 78:  ===========================================================
        # THE TWO PLOTS APPEAR SIDE BY SIDE
        self.next_slide(
            notes=
            '''Having in mind the initial question from the policymaker, we can
            conclude that temperature and humidity play opposing roles in
            determining fire risk, and understanding their relationship is
            crucial for effective fire risk management. [CLICK]
            '''
        )
        self.play(rh_fwi_plot.animate.scale(0.65).move_to(HALF_SCREEN_RIGHT))
        temp_fwi_plot.scale_to_fit_width(rh_fwi_plot.width).move_to(HALF_SCREEN_LEFT)
        self.play(FadeIn(temp_fwi_plot, shift=FRAME_WIDTH/4*LEFT))
        
        # SLIDE 78:  ===========================================================
        # THE TWO PLOTS APPEAR SIDE BY SIDE
        self.next_slide(
            notes=
            '''In general, linear
            regression helps us to extract valuable insights from the data,
            enabling us to quantify the relationships between variables. [END]
            '''
        )
        self.remove(LR_equations)
        initial_graph.remove(LR_equations)
        LR_equations.restore().shift(DOWN*0.5)
        initial_graph.scale(0.65)

        # initial_graph.scale(0.6).to_edge(UP).shift(UP*1.5)
        # self.play(Group(rh_fwi_plot, temp_fwi_plot).animate.next_to(initial_graph, DOWN))
        # self.play(FadeIn(initial_graph))

        self.play(
            AnimationGroup(
                AnimationGroup(
                    temp_fwi_plot.animate.scale(0.6).next_to(initial_graph[0], LEFT),
                    rh_fwi_plot.animate.scale(0.6).next_to(initial_graph[0], RIGHT),
                ),
                FadeIn(initial_graph, LR_equations),
                lag_ratio=0.5
            )
        )



class Test(Scene):
    def construct(self):
        LR_equations = LinearRegressionEquations().to_edge(UP).shift(UP*1.5)
        LR_equations.save_state()
        X_RANGE = (0, 1.5)
        axs = Axes(
            x_range=[X_RANGE[0], X_RANGE[1] + 0.1, 1],
            y_range=[0, 1.2, 1],
            x_length=9,
            y_length=9*1.2/(X_RANGE[1]+0.1),
            x_axis_config={'stroke_color':BLACK, 'include_ticks':False},
            y_axis_config={'stroke_color':BLACK, 'include_ticks':False}
        ).center().shift(DOWN*1.5)
        ax_labels = custom_get_axis_labels(axs, MathTex('x', color=BLUE).scale(0.75), MathTex('y', color=ORANGE).scale(0.75))
        dataset = generate_regression_dataset(func= lambda x: 1.5*(0.4*x-0.75)**3 + 0.8, x_range=(0.1, 1.5), n=20, sigma=0.15, seed=0)
        dataset_points = points_from_data(dataset, ax=axs, color=PURPLE_A).set_z_index(1)
        linear_fit = np.polynomial.polynomial.Polynomial.fit(dataset[:,0], dataset[:,1], 1).convert().coef
        reg_line = RegressionLine(linear_fit[1], linear_fit[0], axs, x_range=X_RANGE)

        ror_dx = 0.25
        ror_x = (X_RANGE[1] - X_RANGE[0])/2 - ror_dx/2
        rise_over_run = Polygon(
            reg_line.eval_to_point(ror_x),
            axs.c2p(ror_x+ror_dx, reg_line.eval(ror_x), 0),
            reg_line.eval_to_point(ror_x + ror_dx),
            color = PURPLE_C,
            fill_opacity=1,
            stroke_width=0
        ).set_z_index(2)
        slope_label = MathTex(r'\hat{m}', color=BLACK).scale(0.75).next_to(rise_over_run, UP).set_z_index(2)
        intercept_dot = Dot(axs.c2p(0, reg_line.intercept.get_value()), color=PURPLE_C)
        intercept_label = MathTex(r'q', color=BLACK).scale(0.75).next_to(intercept_dot, LEFT)

        initial_graph = VGroup(axs, ax_labels, dataset_points, reg_line, rise_over_run, slope_label,intercept_dot, intercept_label, LR_equations)
        initial_graph.remove(LR_equations)
        initial_graph.scale(0.65)
        LR_equations.shift(DOWN*0.5)

        dataset = np.genfromtxt(r'WEEK_2\supplementary_material\ALgerian_forest_dataset.csv', delimiter=',')

        temp, RH, FWI = dataset[1:, 0], dataset[1:, 1], dataset[1:, -1]

        q, m = np.polynomial.polynomial.Polynomial.fit(RH, FWI, 1).convert().coef
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.scatter(RH, FWI, color='blue', alpha=0.5, label='Data points')
        ax.plot(RH, m*RH+q, color='red', label='Regression line')
        ax.set_xlabel('RH')
        ax.set_ylabel('FWI')
        ax.set_title('Linear Regression: FWI vs RH')
        ax.grid(True)
        ax.legend()
        # save figure as array
        fig.canvas.draw()
        buf1 = np.array(fig.canvas.buffer_rgba())
        rh_fwi_plot = ImageMobject(buf1).scale_to_fit_width(6)
        temp_fwi_plot = rh_fwi_plot.copy()
        rh_fwi_plot.move_to(HALF_SCREEN_RIGHT)
        temp_fwi_plot.move_to(HALF_SCREEN_LEFT)

        self.add(rh_fwi_plot, temp_fwi_plot)
        self.wait()
        self.play(
            AnimationGroup(

            AnimationGroup(

            temp_fwi_plot.animate.scale(0.6).next_to(initial_graph[0], LEFT),
            rh_fwi_plot.animate.scale(0.6).next_to(initial_graph[0], RIGHT),
            ),
            FadeIn(initial_graph, LR_equations),
            lag_ratio=0.5
            )
            )
        # self.play(FadeIn(initial_graph, LR_equations))
        self.wait()


