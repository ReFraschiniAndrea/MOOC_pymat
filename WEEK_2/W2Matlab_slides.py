import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from config import *
from Generic_mooc_utils import *
from matlab_utils import *
from W2Anim import *


config.update(TEST_CONFIG)

class W2Python_slides(MOOCSlide):
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

        self.play(
            Succession(
                FadeIn(ax, ax_labels, dataset_points, LR_equations),
                Create(reg_line)
            )
        )

        # SLIDE 02:  ===========================================================
        # M FADES IN
        # HAND CURSOR GROWS FROM CENTER AND MOVES TO FOLDER ICON
        self.next_slide(
            notes=
            '''Let's open Matlab. First, we need to load the data: to ensure that MATLAB can access the file,
            click on the "Browse folder button" [CLICK]
            '''
        )
        mat_env = MatlabEnv(r'Assets\W2\colabSLR.png')
        hand_cursor = Cursor()
        self.play(
            Succession(
                FadeOut(ax, ax_labels, dataset_points, LR_equations, reg_line),
                Wait(0.2),
                FadeIn(mat_env),
                Wait(1),
                GrowFromCenter(hand_cursor)
            )
        )

        # SLIDE 03:  ===========================================================
        # HAND CURSOR MOVES TO FOLDER ICON
        # FOLDER ICON IS CLICKED AND SIDE MENU APPEARS
        self.next_slide(
            notes=
            '''and select the folder containing the  "Algerian_forest_dataset.csv" file. Then, check that the file appears in the "Current Folder" panel [CLICK].
            '''
        )
        self.play(Succession(hand_cursor.animate.move_to(mat_env.MENU_), hand_cursor.Click()))
        mat_env.set_image(r'Assets\W2\colabSLR_sidemenu.png')

        # SLIDE 04:  ===========================================================
        # HAND CURSOR MOVES TO UPLOAD BUTTON
        # UPLOAD BUTTON IS CLICKED, UOLUADED FILE APPEARS
        self.next_slide(
            notes=
            '''This confirms that MATLAB can locate the file before we proceed with reading it.
           [CLICK]
            '''
        )
        self.play(Succession(hand_cursor.animate.move_to(mat_env.UPLOAD_), hand_cursor.Click()))
        mat_env.set_image(r'Assets\W2\colabSLR_uploaded.png'),

        # SLIDE 09:  ===========================================================
        # READ_CSV LINES WRITTEN
        self.next_slide(
            notes=
            '''Let us create a new script, names "week2.m" [CLICK]
            '''
        )
        # SLIDE 09:  ===========================================================
        # READ_CSV LINES WRITTEN
        self.next_slide(
            notes=
            '''With the function “readtable” we can read CSV files, Comma-Separated Values files. Make sure the file name matches exactly! [CLICK]
            '''
        )

        import_code = MatlabCode(
            r'''
            % Load the dataset
            my_dataset = readtable('Algerian_forest_dataset.csv');
            '''
        ).center()
        self.play(import_code.TypeLetterbyLetter(lines=[0]))
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
            '''The output of the function is stored in a variable named my_dataset, which is an object of type "table".
            [CLICK]
            '''
        )
        my_dataset_highlight = HighlightRectangle(import_code[5][:10])
        self.play(Create(my_dataset_highlight))

        # SLIDE 11:  ===========================================================
        # CLASS DEFINITION SNIPPET APPEARS AT TOP
        self.next_slide(
            notes=
            '''In MATLAB, a “table” is a data type for storing tabular data, where columns represent variables and rows correspond to observations. It allows easy indexing, filtering, and manipulation using column names, making data analysis more efficient.
            '''
        )
        class_code = ColabCode(
            r'''
            class DataFrame():
                
                def shape(self):
                    ...
                
                def head(self):
                    ...
            
            '''
        )
        class_code.add_background_window()
        DSS = DynamicSplitScreen(COLAB_LIGHTGRAY, WHITE)
        self.add(DSS); self.remove(mat_env.cells[0].colabCode.window)
        DSS.add_side_obj(class_code.window)
        DSS.add_main_obj(VGroup(import_code, my_dataset_highlight))

        self.play(DSS.bringIn())
        class_code.code.move_to(class_code.window)
        self.play(class_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 14:  ===========================================================
        # INTO COLAB, CELL IS RUN
        # NEW CELL APPEARS, OUT OF COLAB AGAIN
        self.next_slide(
            notes=
            '''Let's explore our dataset to get familiar with this type of data
            structure and to see the data firsthand, which is always a good
            practice! In doing this we use attributes and methods. [CLICK]
            '''
        )
        hand_cursor = mat_env.cursor
        mat_env.clear(self)
        import_code.add_background_window(DSS.mainRect.suspend_updating())
        DSS.add_side_obj(class_code)
        DSS.remove_main_obj()
        
        self.play(FadeOut(my_dataset_highlight))
        self.play(
            import_code.IntoColab(mat_env),
            DSS.bringOut(),
        )
        self.wait(1)
        self.play(
            Succession(
                mat_env.Run(),
                Wait(0.5),
                ApplyMethod(hand_cursor.move_to, mat_env.PLUS_CODE_),
                hand_cursor.Click()
            )
        )
        new_empty_cell=ColabCodeBlock(code='')
        mat_env.add_cell(new_empty_cell)
        self.add(mat_env)  # to update
        self.wait(0.5)
        self.play(mat_env.OutofColab(new_empty_cell))
        self.remove(hand_cursor)

        # SLIDE 15:  ===========================================================
        # FIRST COMMENT LINE IS WRITTEN
        self.next_slide(
            notes=
            '''First of all, we find the number of rows and columns. We use the function [CLICK] ...
            '''
        )
        dataset_size_code = ColabCode(
            r'''
            # Dataset dimensions
            disp('Shape of the dataset:');
            size(my_dataset)
            '''
        ).center()
        self.play(dataset_size_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 16:  ===========================================================
        # PRINT SHAPE LINE IS WRITTEN
        self.next_slide(
            notes=
            '''...“size” to find the number of rows and columns. [CLICK] 
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
        mat_env.clear(self)
        self.play(dataset_size_code.IntoColab(mat_env))
        mat_env.cells[0].add_output('(59,  4)')
        self.play(mat_env.Run())

        # SLIDE 18:  ===========================================================
        # RETURN TO OUT OF COLAB
        # HEAD CODE IS WRITTEN
        self.next_slide(
            notes=
            '''Next, we display the first five rows of the dataset With the command my_dataset(1:5, :)
            we are using MATLAB indexing to extract a subset of the table: [CLICK]
            '''
        )
        DSS.reset()
        dataset_head_code = MatlabCode(
            r'''
            # Showing the data
            disp('First 5 rows of the dataset:');
            disp(my_dataset(1:5, :));
            '''
        ).center()

        self.play(FadeIn(DSS.mainRect))
        self.play(dataset_head_code.TypeLetterbyLetter(lag_ratio=0))

        # SLIDE 18:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''1:5 selects the first five rows of the dataset. The colon : in 1:5 represents a range from row 1 to row 5 [CLICK]
            '''
        )
        # SLIDE 18:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''":" in the column position means "all columns", so we keep all the variables in the table. [CLICK]
            '''
        )
        # SLIDE 19:  ===========================================================
        # DISPLAY HEAD TABLE AFTER RUNNNING CODE
        self.next_slide(
            notes=
            '''This syntax allows us to display only a portion of the dataset... [CLICK]
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

        self.play(dataset_head_code.IntoColab(mat_env))
        mat_env.cells[1].add_output(VGroup(head_text, head_table))
        self.play(mat_env.Run(1, new_cursor=False))

        # SLIDE 20:  ===========================================================
        # FOCUS ON THE TABLE
        self.next_slide(
            notes=
            '''where each row represents a different fire event, and each column corresponds to a specific variable: 
            '''
        )
        self.play(mat_env.focus_output(cell=1 ,scale=0.5, alignment=LEFT))

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
            weather index plays the role of variable Y.
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
        linear_regression_code = MatlabCode(
            r'''
            % Linear regression
            function [m, q] = my_linear_regression(x, y)
                sum_x = sum(x);
                sum_y = sum(y);
                sum_xy = sum(x.*y);
                sum_x2 = sum(x.^2);

                n = length(x);
                numerator = n*sum_xy - sum_x*sum_y;
                denominator = n*sum_x2 - sum_x^2;
                m = numerator / denominator;
                q = (sum_y - m*sum_x)/n;
            end
            '''
        ).center()
        DSS.reset()
        DSS.mainRect.set_z_index(1)
        self.play(FadeIn(DSS))
        mat_env.clear(self)
        self.remove(mat_env, *full_labels, *colored_dots, *column_highlights, x_brace, y_brace)
        DSS.mainRect.set_z_index(-1)
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 27:  ===========================================================
        # SCHEMATIC DRAWING OF THE FUNCTION IS BROUGHT IN
        self.next_slide(
            notes=
            '''We are going to write a function that takes as inputs the
            datapoints, organized in two lists. [CLICK]
            '''
        )
        fscheme = FunctionAbstraction(scale=0.7)
        DSS.add_side_obj(fscheme)
        DSS.add_main_obj(linear_regression_code[0], follow_obj=linear_regression_code[1:])

        self.play(DSS.bringIn())
        # self.play(linear_regression_code.TypeLetterbyLetter(lines=[1]))

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
        x_vector = MathTex(r'[x_1, x_2, \dots, x_n]', color=BLACK, tex_to_color_map={'x_1':BLUE, 'x_2':BLUE,'x_n':BLUE}).next_to(fscheme.InputLabels[0], LEFT, buff=1)
        y_vector = MathTex(r'[y_1, y_2, \dots, y_n]', color=BLACK, tex_to_color_map={'y_1':ORANGE, 'y_2':ORANGE,'y_n':ORANGE}).next_to(fscheme.InputLabels[1], LEFT, buff=1)
        
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
        self.next_slide(
            notes=
            '''The function will return the coefficients of the regression
            lines, namely m and q. [CLICK]
            '''
        )
        fscheme.add_outputs("m", "q")
        m_label = MathTex("m", color=BLACK).next_to(fscheme.OutputLabels[0], RIGHT, buff=1)
        q_label = MathTex("q", color=BLACK).next_to(fscheme.OutputLabels[1], RIGHT, buff=1)

        self.play(FadeIn(fscheme.OutputArrows, fscheme.OutputLabels, m_label, q_label))

        # SLIDE 30:  ===========================================================
        # SHOSRT FUNCCTION DEFINITION IS WRITTEN
        self.next_slide(
            notes=
            '''This is the structure of the Matlab function
            that we will write. We will create a new file named “my_linear_regression.m” and that will contain the function definition.  The function will have two inputs (x and y) and
            two outputs (m and q). What we need to do now is to fill in the
            dots. [CLICK]
            '''
        )
        short_linear_regression_code = MatlabCode(
            r'''
            function [m, q] = my_linear_regression(x, y)
                ...
            end
            '''
        ).align_to(linear_regression_code[1], UL)

        self.play(
            short_linear_regression_code.TypeLetterbyLetter()
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
            '''First we will compute the results of each sum, and then we will combine the results.
            Before starting the implementation, it's worth noting that some sums are repeated. [CLICK]
            '''
        ) 
        sums_highlights = VGroup(
            HighlightRectangle(term, opacity = 0.3) for term in
                [
                    LR_equations.m_sum_x[0],
                    LR_equations.m_sum_x[1],
                    LR_equations.m_sum_y,
                    LR_equations.m_sum_x_y,
                    LR_equations.m_sum_x_sq,
                    LR_equations.q_sum_x,
                    LR_equations.q_sum_y,
                ]                  
        )  
        sum_y_highlights = VGroup(sums_highlights[2], sums_highlights[6])
        sum_x_highlights = VGroup(sums_highlights[i] for i in [0,1,5])
        sum_xy_x2_highlights =VGroup(sums_highlights[i] for i in [3,4])
        
        self.play(FadeIn(sum_y_highlights, sum_x_highlights, sum_xy_x2_highlights))

        # SLIDE 33:  ===========================================================
        # HIGHLIGHT SUMS OF y_i
        self.next_slide(
            notes=
            '''The sum over y_i appears twice, [CLICK] 
            '''
        )
        self.play(FadeOut(sum_x_highlights, sum_xy_x2_highlights))

        # SLIDE 34:  ===========================================================
        # HIGHLIGHT SUMS OF x_i
        self.next_slide(
            notes=
            '''...and the sum over x_i even three times! We can take advantage
            of this, and compute these terms once and reuse the results wherever
            needed. [CLICK]
            '''
        )
        self.play(
            Succession(
                FadeOut(sum_y_highlights),
                FadeIn(sum_x_highlights)
                )
            )

        # SLIDE 35:  ===========================================================
        # THE NON RPEATED SUM TERMS ARE EXTRACTED FROM THE EQUATIONS
        self.next_slide(
            notes=
            '''We are going to write now a Matlab code that computes these four
            terms.  [CLICK]
            '''
        )
        self.play(FadeOut(sum_x_highlights))
        sum_terms = LR_equations.get_sums_without_repetition().arrange(RIGHT, buff=1).scale(1.2).move_to(DSS.secondaryRect)
        self.play(LR_equations.ExtractSumTerms(target=sum_terms))

        # SLIDE 36:  ===========================================================
        # SAMPLE SUM FOR LOOP CODE IS WRITTEN
        self.next_slide(
            notes=
            '''These sums can be computed by writing suitable "for" loops.
            However, we will compute them in a more concise and readable way by
            leveraging the "sum". [CLICK]
            '''
        )
        for_sum_code = MatlabCode(
            r'''
            sum_x = 0;
            for i = 1:length(x)
                sum_x += x(i)
            end
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
            elements in an array. For example, sum(x) calculates the sum of
            all the x-coordinates. [CLICK]
            '''
        )
        self.play(ReplacementTransform(for_sum_code, linear_regression_code[2]))

        # SLIDE 38:  ===========================================================
        # NP.SUM(Y) LINE WRITTEN
        self.next_slide(
            notes=
            '''Similarly, sum(y) calculates the sum of all the yi. [CLICK]
            '''
        )
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[3]))

        # SLIDE 39:  ===========================================================
        # EMPTY RECTANGLE BROUGHT IN ON TOP
        # VECTORIZED OPERATIONS TITLE WRITTEN
        self.next_slide(
            notes=
            '''To compute the last two terms, we need to introduce a powerful
            concept in Matlab: vectorized operations. [CLICK]
            '''
        )
        DSS.add_side_obj(sum_terms)
        DSS.add_main_obj(linear_regression_code[:4], linear_regression_code[4:])
        self.play(DSS.bringOut())

        title = Text('Vectorized operations', font=SANS_SERIF_FONT, weight=LIGHT, font_size=64, color=BLACK, stroke_color=BLACK)
        x_vector =  VectorArray(arrangement='vertical', include_dots=True, array=[f'x[{i}]' for i in [0,1,2,'n']]).scale(0.6)
        y_vector =  VectorArray(arrangement='vertical', include_dots=True, array=[f'y[{i}]' for i in [0,1,2,'n']]).scale(0.6)
        xy_vector = VectorArray(arrangement='vertical', include_dots=True, array=[f'x[{i}]*y[{i}]' for i in [0,1,2,'n']]).scale(0.6)
        x2_vector = VectorArray(arrangement='vertical', include_dots=True, array=[f'x[{i}]^2' for i in [0,1,2,'n']]).scale(0.6)
        vector_labels = [Text(s, color=BLACK, font=CODE_FONT) for s in ['x', 'y', 'x.*y', 'x.^2']]

        y_vector.next_to(x_vector, RIGHT)
        xy_vector.next_to(y_vector, RIGHT, buff=1)
        vector_labels[0].next_to(x_vector, UP)
        vector_labels[1].next_to(y_vector, UP).align_to(vector_labels[0], UP)
        vector_labels[2].next_to(xy_vector, UP).align_to(vector_labels[1], DOWN)
        first_group=VGroup(x_vector, y_vector, xy_vector, *vector_labels[:3])
        title.next_to(first_group, UP)
        first_group.add(title)

        DSS.add_empty_side_obj(first_group.height)
        self.play(DSS.bringIn())
        first_group.move_to(DSS.secondaryRect)
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
            '''The operation x .* y creates a new array, [CLICK]...
            '''
        )
        self.play(
            Create(xy_vector.get_lines()),
            FadeIn(vector_labels[2])
        )

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

        # SLIDE 42:  ===========================================================
        # '.*' HIGHLIGHTED
        self.next_slide(
            notes=
            '''Please notice the "." Before the "*", which indicates that the
            operation must act element by element. [CLICK]
            '''
        )
        dot_star_highlight = HighlightRectangle(vector_labels[2][1:3])
        self.play(Create(dot_star_highlight))

        # SLIDE 43:  ===========================================================
        # SUM(X .* Y) LINE WRITTEN
        self.next_slide(
            notes=
            '''As a consequence, with sum(x .* y) we compute the sum of all
            the products xi times yi, that is the term called sum_xy. [CLICK]
            '''
        )
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[4]))

        # SLIDE 44:  ===========================================================
        # EMPTY X**2 APPEARS
        # ANIMATE X TERMS INTO X**2 TERMS
        self.next_slide(
            notes=
            '''Similarly, the element-wise power operation x .^ 2 is also
            vectorized, meaning it applies the power operation to each element
            of the array individually. [CLICK]
            '''
        )
        self.play(FadeOut(y_vector, xy_vector, *vector_labels[1:3]) )
        x2_vector.move_to(xy_vector)
        vector_labels[-1].next_to(x2_vector, UP).align_to(vector_labels[0], DOWN)
        self.play(
            Create(x2_vector.get_lines()),
            FadeIn(vector_labels[-1])
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
            '''...so that combining this operation with sum gives the last
            term. [CLICK]
            '''
        )
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[5]))

        # SLIDE 46:  ===========================================================
        # VECTORIZED OPERATIONS BROUGHT OUT OF FRAME
        # 'm', 'q' EQUATIONS BROUGHT BACK IN
        self.next_slide(
            notes=
            '''Very good. Now the hardest part is behind us. We just need to
            combine these quantities to finalize the computation. [CLICK]
            '''
        )
        DSS.add_side_obj( VGroup(x_vector, x2_vector, vector_labels[0], vector_labels[-1], title))
        DSS.add_main_obj(linear_regression_code[:6], follow_obj=linear_regression_code[6:])
        self.play(DSS.bringOut())

        DSS.add_side_obj(LR_equations.restore())
        self.play(DSS.bringIn())

        # SLIDE 47:  ===========================================================
        # NUMERATOR LINE WRITTEN
        self.next_slide(
            notes=
            '''First, we compute the numerator of the expression giving m.
            [CLICK]
            '''
        )
        self.play(linear_regression_code.TypeLetterbyLetter(lines=[7, 8]))

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
                linear_regression_code.TypeLetterbyLetter(lines=[12])
            )
        )

        # SLIDE 51:  ===========================================================
        # BRING OUT TOP RECTANGLE TO LEAVE ONLY THE FUNCTION AS THE FOCUS
        self.next_slide(
            notes=
            '''Great! We have completed all the necessary steps for the
            implementation of our function. Let us quickly revise it. [CLICK]
            '''
        )
        DSS.add_main_obj(linear_regression_code)
        self.play(DSS.bringOut())

        # SLIDE 52:  ===========================================================
        # HIGHLIGHT FUNCTION DEFINITION
        self.next_slide(
            notes=
            '''The function takes two arrays as inputs, containing the x and y
            coordinates of the data points. [CLICK]
            '''
        )
        code_recap_highlights = [
            HighlightRectangle(code_snippet) for code_snippet in
            [linear_regression_code[1][14:],
             linear_regression_code[2:6],
             linear_regression_code[7:12],
             linear_regression_code[1][8:13]]
        ]
        self.play(Create(code_recap_highlights[0]))

        # SLIDE 53:  ===========================================================
        # HIGHLIGHT NP.SUM LINES
        self.next_slide(
            notes=
            '''First we compute the sums needed to perform the linear
            regression, [CLICK] ...
            '''
        )
        self.play(ReplacementTransform(code_recap_highlights[0], code_recap_highlights[1]))

        # SLIDE 54:  ===========================================================
        # HIGHLIGHT CODE CORRESPONDING TO M, Q FORMULAS
        self.next_slide(
            notes=
            '''...next, we combine these terms thus getting the optimal
            coefficients m and q. [CLICK]
            '''
        )
        self.play(ReplacementTransform(code_recap_highlights[1], code_recap_highlights[2]))

        # SLIDE 55:  ===========================================================
        # HIGHLIGHT RETURN LINE
        self.next_slide(
            notes=
            '''Finally, we return m, q. [CLICK]
            '''
        )
        self.play(ReplacementTransform(code_recap_highlights[2], code_recap_highlights[3]))
        
        # SLIDE 56:  ===========================================================
        # *************************************
        self.next_slide(
            notes=
            '''Now we are ready to use this function with the Algerian forest dataset. Let us go back to the “week2.m” script, and make sure that the file “my_linear_regression.m” is available in the current directory. 
            '''
        )
        self.play(FadeOut(code_recap_highlights[3]))
        linear_regression_code.add_background_window(DSS.mainRect.suspend_updating())
        mat_env.clear(self)
        self.play(linear_regression_code.IntoColab(mat_env))
        self.play(mat_env.Run())

        # SLIDE 57:  ===========================================================
        # NEW EMPTY SCREEN FADES IN
        # FIRST COMMENT LINE IS WRITTEN
        self.next_slide(
            notes=
            '''We wonder how the temperature influences the Fire Weather
            Index,
            '''
        )
        LR_example_code = MatlabCode(
            r'''
            % Perform simple linear regression
            x = my_dataset.Temperature;
            y = my_dataset.FWI;

            [m, q] = linear_regression(x, y);

            % Print results
            disp('Linear model results:');
            fprintf('Slope (m): %.4f\n', m);
            fprintf('Y-intercept (q): %.4f\n', q);
            '''
        ).center()
        DSS.reset()
        self.play(FadeIn(DSS))

        self.play(LR_example_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 58:  ===========================================================
        # X = TEMPERATURE CODE LINE APPEARS
        self.next_slide(
            notes=
            '''To this goal, we chose the temperature as the x, using this code.
            [CLICK]
            '''
        )
        self.play(LR_example_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 59:  ===========================================================
        # HIGHLIGHT 'TEMPERATURE'
        self.next_slide(
            notes=
            '''"my_dataset.Temperature" extracts the corresponding column from
            the dataset, which is stored in the variable "x", [CLICK]...
            '''
        )
        temperature_highlight = HighlightRectangle(LR_example_code[1][13:26])
        # values_highlight = HighlightRectangle(LR_example_code[1][28:])

        self.play(Create(temperature_highlight))
        
        # SLIDE 61:  ===========================================================
        # Y = FWI CODE LINE APPEARS
        self.next_slide(
            notes=
            '''...and similarly the content of the column FWI is stored in the
            variable "y". [CLICK] 
            '''
        )
        self.play(FadeOut(temperature_highlight))
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
        # '%.4f' F-STRINGS HIGHLIGHTED
        self.next_slide(
            notes=
            '''In MATLAB, the "fprintf" function is used to format and display
            output. The "%.4f" specifies that the variable should be displayed
            as a floating-point number with four decimal places. In this case,
            `m` and `q` are the variables being printed with four decimal places
            of precision. [CLICK]
            '''
        )
        perc4f_highlights = VGroup(
            HighlightRectangle(LR_example_code[8][6]),
            HighlightRectangle(LR_example_code[9][6])
        )

        self.play(Create(h) for h in perc4f_highlights)

        # SLIDE 66:  ===========================================================
        # INTO COLAB
        # CELL IS RUN, RESULT OUTPUT APPEARS
        self.next_slide(
            notes=
            '''Running this code, we see the results printed on the screen.
            [CLICK]
            '''
        )
        self.play(FadeOut(perc4f_highlights))
        LR_example_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(LR_example_code.IntoColab(mat_env))
        mat_env.cells[1].add_output(
            'Linear model results:\n'
            'Slope (m): 1.4220\n'
            'Y-intercept (q): -36.2192'
        )
        self.play(mat_env.Run(1, new_cursor=False))

        # SLIDE 67:  ===========================================================
        # RETURN TO EMPTY SCREEN
        # WRITE # PLOTTING
        self.next_slide(
            notes=
            '''Let's now visualize the datapoints together with the regression
            line. [CLICK]
            '''
        )
        plotting_code = MatlabCode(
            r'''
            % Plotting
            figure;
            scatter(x, y, 'blue', 'filled', 'MarkerFaceAlpha', 0.5);
            hold on;
            plot(x, m*x+q, 'red', 'LineWidth', 2);

            xlabel('Temperature');
            ylabel('FWI');
            title('Linear Regression: FWI vs Temperature');
            girf on;
            legend('Data points', 'Regression line');
            '''
        ).center()
        # plotting_code.code.save_state()
        DSS.reset()
        self.play(FadeIn(DSS))
        self.add(mat_env); self.remove(mat_env)
        mat_env.clear(self)
        self.play(plotting_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 68:  ===========================================================
        # PLT.FIGURE LINE WRITTEN
        self.next_slide(
            notes=
            '''We create a new figure, and we use the function [CLICK] ...
            '''
        )
        def draw_plot(fig) -> ImageMobject:
            fig.canvas.draw()
            buf1 = np.array(fig.canvas.buffer_rgba())
            return ImageMobject(buf1)
        
        PLOT_WIDTH=6
        
        # create the plot with matplotlib
        temp, RH, FWI = dataset[1:, 0], dataset[1:, 1], dataset[1:, -1]
        q, m = np.polynomial.polynomial.Polynomial.fit(temp, FWI, 1).convert().coef
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        DSS.add_side_obj(draw_plot(fig).scale_to_fit_width(PLOT_WIDTH))
        
        self.play(plotting_code.TypeLetterbyLetter(lines=[1]))
        DSS.add_main_obj(plotting_code[:2], follow_obj=plotting_code[2:])
        self.play(DSS.bringIn())

        # SLIDE 69:  ===========================================================
        # PLT.SCATTER LINE WRITTEN
        self.next_slide(
            notes=
            '''..."scatter" to display the datapoints as small blue circles in the
            x-y plane. [CLICK]
            '''
        )
        self.play(plotting_code.TypeLetterbyLetter(lines=[2]))
        ax.scatter(temp, FWI, color='blue', alpha=0.5, label='Data points')
        new_plot = draw_plot(fig).scale_to_fit_width(PLOT_WIDTH).move_to(DSS.secondaryObj)
        DSS.secondaryObj.become(new_plot)

        # SLIDE 70:  ===========================================================
        # PLOT LINE APPEARS
        self.next_slide(
            notes=
            '''Then, we use the function "plot" to display the regression line
            in red. Finally, we can enhance the plot's readability by adding: 
            [CLICK]
            '''
        )
        self.play(plotting_code.TypeLetterbyLetter(lines=[3]))
        ax.plot(temp, m*temp+q, color='red', label='Regression line')
        new_plot = draw_plot(fig).scale_to_fit_width(PLOT_WIDTH).move_to(DSS.secondaryObj)
        DSS.secondaryObj.become(new_plot)

        # SLIDE 71:  ===========================================================
        # X, Y LABEL LINES AND RESULT SHOWN
        self.next_slide(
            notes=
            '''...axis labels, [CLICK]
            '''
        )
        self.play(plotting_code.TypeLetterbyLetter(lines=[5, 6], lag_ratio=0))
        ax.set_xlabel('Temperature')
        ax.set_ylabel('FWI')
        new_plot = draw_plot(fig).scale_to_fit_width(PLOT_WIDTH).move_to(DSS.secondaryObj)
        DSS.secondaryObj.become(new_plot)

        # SLIDE 72:  ===========================================================
        # GRID, TITLE LEGEND LINES AND RESULT SHOWN
        self.next_slide(
            notes=
            '''...a title, a grid and a legend. Please notice that the labels passed to the function "legend" must match the order in which the different plots were created. 
            '''
        )
        self.play(plotting_code.TypeLetterbyLetter(lines=[7, 8, 9]))
        ax.set_title('Linear Regression: FWI vs Temperature')
        ax.grid(True)
        ax.legend()
        new_plot = draw_plot(fig).scale_to_fit_width(PLOT_WIDTH).move_to(DSS.secondaryObj)
        DSS.secondaryObj.become(new_plot)

        # SLIDE 74:  ===========================================================
        # INTO MATLAB
        # CELL IS RUN, FINAL PLOT APPEARS
        self.next_slide(
            notes=
            '''Running this cell, a new figure showing the figure is displayed.
            [CLICK]
            '''
        )
        mat_env.clear(self)
        plotting_code.add_background_window(DSS.mainRect.suspend_updating())
        DSS.remove_main_obj()
        # self.play(FadeOut(*plot_label_highlights))
        self.play(
            plotting_code.IntoColab(mat_env),
            DSS.bringOut()
        )

        # save figure as array
        temp_fwi_plot = draw_plot(fig).scale_to_fit_width(6)
        
        mat_env.cells[0].add_output(temp_fwi_plot)
        self.play(mat_env.Run())
        
        # SLIDE 75:  ===========================================================
        # FOCUS ON PLOT
        self.next_slide(
            notes=
            '''The slope is positive, indicating that an increase in the
            temperature corresponds to a higher overall risk of forest fires.
            [CLICK]
            '''
        )
        self.play(mat_env.focus_output(0, scale=0.7))

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
        LR_example_code_2 = MatlabCode(LR_example_code.code_string).center()
        replacement_code = MatlabCode(LR_example_code.code_string.replace(r"""'Temperature'].values""", r"""'RH'].values """))
        replacement_code.move_to(LR_example_code_2).align_to(LR_example_code_2, UL)

        DSS.reset()
        self.play(
            Succession(
                FadeIn(DSS.mainRect.set_z_index(0)),
                FadeIn(LR_example_code_2),
                Wait(1)
            )
        )
        mat_env.clear(self)
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

        mat_env.add_cell(ColabCodeBlock(replacement_code.code_string))
        mat_env.add_cell(ColabCodeBlock(plotting_code.code_string.replace('Temperature','RH')))
        replacement_code.add_background_window(DSS.mainRect.suspend_updating())
        
        self.play(replacement_code.IntoColab(mat_env, target_cell=0))
        self.play(mat_env.Run(0))
        mat_env.cells[1].add_output(rh_fwi_plot)
        self.remove(mat_env.cells[1].output) # it gets shown too early
        self.play(mat_env.Run(1, new_cursor=False))

        # SLIDE 77:  ===========================================================
        # FOCUS ON THE SECOND PLOT
        self.next_slide(
            notes=
            '''From the plot it can be seen an opposite trend with respect to
            the previous case: this means that an increase in relative humidity
            decreases the overall risk of forest fire.
            '''
        )
        self.play(mat_env.focus_output(cell=1, scale=0.7))

        # SLIDE 78:  ===========================================================
        # THE TWO PLOTS APPEAR SIDE BY SIDE
        self.next_slide(
            notes=
            '''Having in mind the initial question from the policymaker, we can
            conclude that temperature and humidity play opposing roles in
            determining fire risk, and understanding their relationship is
            crucial for effective fire risk management. In general, linear
            regression helps us to extract valuable insights from the data,
            enabling us to quantify the relationships between variables. [END]
            '''
        )
        self.play(rh_fwi_plot.animate.scale(0.65).move_to(HALF_SCREEN_LEFT))
        temp_fwi_plot.scale_to_fit_width(rh_fwi_plot.width).move_to(HALF_SCREEN_RIGHT)
        self.play(FadeIn(temp_fwi_plot, shift=FRAME_WIDTH/4*RIGHT))
