import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from mooc_utils.colab import ColabCodeWithLogo, ColabCode, COLAB_LIGHTGRAY
from mooc_utils.matlab import MatlabCodeWithLogo
from W2Anim import *

config.update(RELEASE_CONFIG)

LABELS_SIZE=0.75
ICONS_HEIGHT = 0.6

class W2Wrapup_slides(ThreeDMOOCSlide):
    def construct(self):
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
        X_RANGE = (0, 1.5)
        ax = Axes(
            x_range=[X_RANGE[0], X_RANGE[1]+0.1, 1],
            y_range=[0, 1.2, 1],
            x_length=9,
            y_length=9*1.2/(X_RANGE[1]+0.1),
            x_axis_config={'stroke_color':BLACK, 'include_ticks':False},
            y_axis_config={'stroke_color':BLACK, 'include_ticks':False}
        ).center()
        ax.set_z_index(-1)
        variable_ax_lab = custom_get_axis_labels(ax,
            Text('Temperature' , color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).scale(LABELS_SIZE/2),
            Text('Fire risk', color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).scale(LABELS_SIZE/2)
        )

        wildfire_icon = WildFireIcon().scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(-0.1, 1, 0))
        forest_icon = SVGMobject(r'Assets\W2\pine_trees_icon.svg').scale_to_fit_height(wildfire_icon.forest_icon.height).move_to(ax.c2p(-0.1, 0.3, 0))
        low_temp_icon = SVGMobject(r'Assets\W2\low_temperature_icon.svg').set_color(BLUE).scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(0.3, -0.1, 0))
        high_temp_icon = SVGMobject(r'Assets\W2\high_temperature_icon.svg').set_color(RED).scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(X_RANGE[1]-0.1, -0.1, 0))

        dataset = generate_regression_dataset(func= lambda x: 1.5*(0.4*x-0.75)**3 + 0.8, x_range=(0.1, 1.5), n=20, sigma=0.15, seed=0)
        dataset_points = points_from_data(dataset, ax=ax, color=PURPLE_A).set_z_index(1)
        linear_fit = np.polynomial.polynomial.Polynomial.fit(dataset[:, 0], dataset[:, 1], 1).convert().coef
        reg_line = RegressionLine(linear_fit[1], linear_fit[0], ax, x_range=X_RANGE).suspend_updating()

        self.play(
            Create(ax),
            FadeIn(forest_icon, wildfire_icon, low_temp_icon, high_temp_icon, variable_ax_lab),
        )
        self.play(
            AnimationGroup(
                *[GrowFromCenter(p) for p in dataset_points],
                run_time=2, lag_ratio=0.5)
        )
        self.play(Create(reg_line))

        # SLIDE 02:  ===========================================================
        # RESIDUAL LINES ARE DRAWN
        # SQUARED ERRORS SUM FORMULA APPEARS
        self.next_slide(
            notes=
            '''First, we saw that the line that best fits a set of data points
            can be defined as the one that minimizes the sum of the squared
            residuals, [CLICK]
            '''
        )
        minim_problem = MathTex(r'\min_{m,q} \sum_{i=1}^n (r_i)^2', color=BLACK).to_edge(UP).shift(UP)
        example_lr_plot = VGroup(ax, variable_ax_lab, reg_line, dataset_points,
                                 forest_icon, wildfire_icon, high_temp_icon, low_temp_icon)
        reg_line.suspend_updating()

        self.play(example_lr_plot.animate.shift(DOWN))
        self.play(FadeIn(minim_problem))

        reg_line.add_dataset(dataset_points)
        self.play(
            AnimationGroup(
                *[Succession(
                    Create(l), GrowFromCenter(p),
                    lag_ratio=0.5
                )
                for l, p in zip(reg_line.proj_lines, reg_line.proj_points)],
                lag_ratio=0.2,
                run_time=2
            )
        )

        # SLIDE 03:  ===========================================================
        # LINEAR REGRESSION COEFFICIENTS FORMULA REPLACES 'E'
        self.next_slide(
            notes=
            '''... and that the coefficients of this line can be found through
            appropriate calculations. We then saw how to implement these
            calculations in Python and MATLAB. [CLICK]
            '''
        )
        LR_equations = LinearRegressionEquations().scale(0.8).move_to(minim_problem)

        self.play(ReplacementTransform(minim_problem, LR_equations))

        # SLIDE 04:  ===========================================================
        # SUM TERMS ARE EXTRACTED FROM THE FORMULA AND GO TO TOP
        # CODE WINDOWS APPEAR WITH THE FIRST SUMS
        self.next_slide(
            notes=
            '''To do so efficiently, we first precomputed the terms that appear
            multiple times in the expressions, and we used vectorized
            computations, which allow us to operate directly on arrays without
            explicit loops. [CLICK]
            '''
        )
        sum_terms = LR_equations.get_sums_without_repetition().arrange(RIGHT, buff=1).scale(1.4).to_edge(UP)
        sum_python_code = ColabCodeWithLogo(
            r'''
            np.sum(x)

            np.sum(y)

            np.sum(x * y)

            np.sum(x ** 2)
            '''
        ).move_to(HALF_SCREEN_LEFT).shift(DR)
        sum_matlab_code = MatlabCodeWithLogo(
            r'''
            sum(x)

            sum(y)

            sum(x.*y)

            sum(x.^2)
            '''
        ).move_to(HALF_SCREEN_RIGHT).shift(DL)

        self.play(FadeOut(example_lr_plot, reg_line.proj_lines, reg_line.proj_points))
        self.play(
            LR_equations.ExtractSumTerms(target=sum_terms),
            FadeIn(sum_python_code, sum_matlab_code)
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
        dot_operator_highlights = VGroup(
            HighlightRectangle(sum_matlab_code.codeMobject[4][5:7]),
            HighlightRectangle(sum_matlab_code.codeMobject[6][5:7]),
        )

        self.play(Create(dot_operator_highlights))

        # SLIDE 06:  ===========================================================
        # '*' AND '**' HIGHLIGHTED IN PYTHON CODE
        self.next_slide(
            notes=
            '''In Python with NumPy, on the other hand, element-wise behavior is
            the default when working with arrays. [CLICK]
            '''
        )
        numpy_highlights = VGroup(
            HighlightRectangle(sum_python_code.codeMobject[4][8]),
            HighlightRectangle(sum_python_code.codeMobject[6][8:10]),
        )

        self.play(
            ReplacementTransform(dot_operator_highlights[0], numpy_highlights[0]),
            ReplacementTransform(dot_operator_highlights[1], numpy_highlights[1]),
        )
        
        # SLIDE 07:  ===========================================================
        # 'np.sum' HIGHLIGHTED IN THE CODE
        self.next_slide(
            notes=
            '''Finally, for computing these summations, in Python we used the
            sum function from the NumPy module, [CLICK] ...
            '''
        )
        sum_highlights_py = VGroup(HighlightRectangle(sum_python_code.codeMobject[i*2][:6]) for i in range(4))
        sum_highlights_mat = VGroup(HighlightRectangle(sum_matlab_code.codeMobject[i*2][:3]) for i in range(4))
        
        self.play(FadeOut(numpy_highlights))
        self.play(Create(sum_highlights_py))

        # SLIDE 08:  ===========================================================
        # 'sum' HIGHLIGHTED IN THE CODE
        self.next_slide(
            notes=
            '''...while in MATLAB we used the built-in sum function. [CLICK]
            '''
        )
        self.play(
            AnimationGroup(
                ReplacementTransform(sum_highlights_py[i], sum_highlights_mat[i]) for i in range(4)
            )
        )

        # SLIDE 09:  ===========================================================
        # SCHEME OF THE TWO FUNCTION DEFINITIONS APPEAR
        self.next_slide(
            notes=
            '''We then implemented a function that takes as input two arrays
            containing the x- and y-coordinates of the available data points,
            and computes the coefficients m and q of the regression line. To
            define functions, the syntax differs slightly between the two
            languages: [CLICK]
            '''
        )
        python_function_code = ColabCodeWithLogo(
            r'''
            def linear_regression(x, y):
                ...
                return m, q
            ''',
            logo_pos=LEFT, logo_shift_buff=0.1
        )
        matlab_function_code = MatlabCodeWithLogo(
            r'''
            function [m, q] = linear_regression(x, y):
                ...
            end
            ''',
            logo_pos=LEFT, logo_shift_buff=0.1
        ).next_to(python_function_code, DOWN, buff=1)
        python_function_code.align_to(matlab_function_code, LEFT)
        Group(python_function_code, matlab_function_code).center()

        self.play(FadeOut(sum_python_code, sum_matlab_code, sum_highlights_mat, sum_terms))
        self.play(FadeIn(python_function_code, matlab_function_code))
        
        # SLIDE 10:  ===========================================================
        # HIGHLIGHT DEF AND RETURN IN IN PYTHON
        self.next_slide(
            notes=
            '''In Python, we use the keyword def to start the function
            definition and return results using the return statement. [CLICK]
            '''
        )
        py_func_highlights = VGroup(
            HighlightRectangle(python_function_code.codeMobject[0][:3]),
            HighlightRectangle(python_function_code.codeMobject[2]),
        )

        self.play(Create(py_func_highlights))

        # SLIDE 11:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''In MATLAB, on the other hand, functions are introduced with the
            keyword function, and the output variables are listed directly in
            the function header. [CLICK]
            '''
        )
        mat_func_highlights = HighlightRectangle(matlab_function_code.codeMobject[0][:13])
        
        self.play(FadeOut(py_func_highlights))
        self.play(FadeIn(mat_func_highlights))

        # SLIDE 12:  ===========================================================
        # HEAD TABLE DATASET APPEARS
        self.next_slide(
            notes=
            '''Using the functions we implemented on a dataset that includes
            detailed features related to a set of forest fires recorded in
            Algeria, [CLICK]
            '''
        )
        algerian_dataset = np.genfromtxt(r'WEEK_2\supplementary_material\ALgerian_forest_dataset.csv', delimiter=',')
        row_labels = [Text(str(i), color=BLACK, font=CODE_FONT, weight=ULTRAHEAVY) for i in range(5)]
        col_labels = [Text(label,  color=BLACK, font=CODE_FONT, weight=ULTRAHEAVY) for label in ['Temperature', 'RH', 'BUI', 'FWI']]
        head_table = Table(
            algerian_dataset[1:6], row_labels=row_labels, col_labels=col_labels,
            add_background_rectangles_to_entries=False,
            element_to_mobject=CustomDecimalNumber,
            element_to_mobject_config={'font':CODE_FONT,'color': BLACK, 'mob_class': Text, 'num_decimal_places':1},
            line_config={'stroke_width':0},
            arrange_in_grid_config={'cell_alignment': ORIGIN}
        ).scale(0.75).center()
        for i in range(6):
            for j in range(5):
                color = WHITE if i % 2 ==0 else COLAB_LIGHTGRAY
                head_table.add_highlighted_cell((i+1,j+1),color=color)
                # color in table constructor does not work 
                head_table.get_entries((i+1, j+1)).set_color(BLACK)

        self.play(FadeOut(python_function_code, matlab_function_code, mat_func_highlights))
        self.play(FadeIn(head_table))

        # SLIDE 13:  ===========================================================
        # PLOT OF LINEAR REGRESSION (TEMP. VS FWI) APPEARS
        self.next_slide(
            notes=
            '''we obtained a regression line linking temperature to fire risk.
            This allowed us to gain a quantitative insight into how an increase
            in temperature raises the likelihood of wildfires. [CLICK]
            '''
        )
        self.play(FadeOut(head_table))
        self.clear()

        algerian_dataset = np.genfromtxt(r'WEEK_2\supplementary_material\ALgerian_forest_dataset.csv', delimiter=',')
        temperature = algerian_dataset[1:, 0]
        RH = algerian_dataset[1:, 1]
        FWI = algerian_dataset[1:, -1]
        t_range = (26, 38)   # temperature
        rh_range = (45, 85)  # relative humidity
        fwi_range = (0, 32)  # fire weather index

        # display first dataset
        ax_temp = NumberPlane(
            x_range=(*t_range, 2),
            y_range=(*fwi_range, 5),
            x_length=12,
            y_length=9,
            x_axis_config={'stroke_color': BLACK, 'include_ticks': True, 'include_tip':True, 'include_numbers': True, 'label_direction':DOWN, 'label_constructor':MathTex},
            y_axis_config={'stroke_color': BLACK, 'include_ticks': True, 'include_tip':True, 'include_numbers': True, 'label_direction':LEFT, 'label_constructor':MathTex},
            background_line_style={'stroke_color': BLACK, 'stroke_width': 0.5}
        ).center()
        ax_temp.x_axis.numbers.set_color(BLACK)
        ax_temp.y_axis.numbers.set_color(BLACK)
        temp_labels =custom_get_axis_labels(ax_temp,
            Text('Temperature' , color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).scale(LABELS_SIZE/2),
            Text('FWI', color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).scale(LABELS_SIZE/2),
        )

        point_config = {'color': PURPLE_A, 'radius': DEFAULT_DOT_RADIUS}
        temp_dataset = points_from_data(np.column_stack((temperature, FWI)), ax_temp, **point_config)
        _temp_fit = np.polynomial.polynomial.Polynomial.fit(temperature, FWI, 1).convert().coef
        temp_reg_line = RegressionLine(_temp_fit[1], _temp_fit[0], ax_temp, x_range=t_range)
        
        self.play(FadeIn(ax_temp, temp_labels, temp_dataset))
        self.play(Create(temp_reg_line))

        # SLIDE 14:  ===========================================================
        # PLOT OF LINEAR REGRESSION (HUMIDITY. VS FWI) REPLACES FIRST ONE
        self.next_slide(
            notes=
            '''Similarly, we saw how higher humidity can mitigate this risk,
            reducing the overall fire danger. [CLICK]
            '''
        )
        ax_3d = ThreeDAxes(
            x_range=(*rh_range, 5),
            y_range=(*t_range, 2),
            z_range=(*fwi_range, 5),
            x_length=12,
            y_length=12,
            z_length=9,
        ).set_color(BLACK).center()
        ax_3d.x_axis.rotate(PI/2, X_AXIS)
        ax_3d.y_axis.rotate(PI/2, Y_AXIS)
        threeD_labels = VGroup(
            Text('RH' , color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).scale(LABELS_SIZE/2).rotate(PI/2, X_AXIS).next_to(ax_3d.get_axis(0).get_corner(OUT+RIGHT), OUT),
            Text('Temperature' , color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).scale(LABELS_SIZE/2).rotate(PI/2, X_AXIS).next_to(ax_3d.get_axis(1).get_corner(OUT+UP), OUT),
            Text('FWI', color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).scale(LABELS_SIZE/2).rotate(PI/2, RIGHT).next_to(ax_3d.get_axis(2).get_corner(OUT+RIGHT), RIGHT),
        )

        ax_rh = NumberPlane(
            x_range=(*rh_range, 5),
            y_range=(*fwi_range, 5),
            x_length=12,
            y_length=9,
            x_axis_config={'stroke_color': BLACK, 'include_ticks': True, 'include_tip':True, 'include_numbers': True, 'label_direction':DOWN, 'label_constructor':MathTex},
            y_axis_config={'stroke_color': BLACK, 'include_ticks': True, 'include_tip':True, 'include_numbers': True, 'label_direction':LEFT, 'label_constructor':MathTex},
            background_line_style={'stroke_color': BLACK, 'stroke_width': 0.5}
        )
        ax_rh.x_axis.numbers.set_color(BLACK)
        ax_rh.y_axis.numbers.set_color(BLACK)
        ax_rh.rotate(PI/2, X_AXIS)
        ax_rh.scale(ax_3d.x_axis.width/ax_rh.x_axis.width*1.01)
        ax_rh.shift(ax_3d.c2p(rh_range[0], t_range[0], fwi_range[0])-ax_rh.c2p(rh_range[0], fwi_range[0]))
        
        dataset_3d = VGroup(
            Dot(ax_3d.c2p(rh, t_range[0], fwi), **point_config).rotate(PI/2, X_AXIS) for rh, fwi in zip(RH, FWI)
        )
        _rh_fit = np.polynomial.polynomial.Polynomial.fit(RH, FWI, 1).convert().coef
        rh_reg_line = RegressionLine(_rh_fit[1], _rh_fit[0], ax_rh, x_range=rh_range)

        self.play(FadeOut(ax_temp, temp_labels, temp_dataset, temp_reg_line))
        self.set_camera_orientation(phi=90 * DEGREES, theta=-90 * DEGREES, gamma=0*DEGREES, zoom=0.7)
        self.play(FadeIn(ax_3d.x_axis, ax_3d.z_axis, ax_rh.background_lines, ax_rh.x_axis.numbers, ax_rh.y_axis.numbers, dataset_3d,
                         threeD_labels[0], threeD_labels[2]))
        self.play(Create(rh_reg_line))

        # SLIDE 15:  ===========================================================
        # HEAD TABLE WITH BUI HIGHLIGHTED REAPPEARS
        self.next_slide(
            notes=
            '''As an exercise, you can try to compute the regression line for
            the Build-Up Index, which is the last quantity in the dataset that
            we did not consider. [CLICK]
            '''
        )
        self.play(FadeOut(ax_3d.x_axis, ax_3d.z_axis, ax_rh.background_lines, ax_rh.x_axis.numbers, ax_rh.y_axis.numbers, dataset_3d,
                        threeD_labels[0], threeD_labels[2], rh_reg_line))
        
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, gamma=0, zoom=1)
        BUI_highlight = HighlightRectangle(head_table.get_columns()[3][1:], color = ORANGE)

        self.play(
            Succession(
                Wait(0.5),
                FadeIn(head_table),
                Wait(0.5),
                Create(BUI_highlight)
            )
        )

        # SLIDE 16:  ===========================================================
        # RETURN TO SIMPLE LINEAR REGRESSION PLOT
        self.next_slide(
            notes=
            '''Now, a key question that we didn't address is the following: how
            well the regression line is fitting the available data? In other
            words, how effectively does the regression line explain the
            relationship between x and y? [CLICK]
            '''
        )
        self.play(FadeOut(head_table, BUI_highlight))
        self.clear()
        
        ax.save_state()
        ax.center()
        linear_dataset_1 = generate_regression_dataset(reg_line.eval, 20, x_range=X_RANGE, sigma=0.04, seed=1)
        tight_dataset = points_from_data(linear_dataset_1, ax, **point_config)
        _tight_fit = np.polynomial.polynomial.Polynomial.fit(linear_dataset_1[:, 0], linear_dataset_1[:, 1], 1).convert().coef
        tight_reg_line = RegressionLine(_tight_fit[1],  _tight_fit[0], ax, x_range=X_RANGE)
        tight_reg_line.line.clear_updaters()
        
        self.play(FadeIn(ax, tight_dataset, tight_reg_line))

        # SLIDE 17:  ===========================================================
        # TWO LINEAR REGRESSION PLOTS SIDE BY SIDE
        self.next_slide(
            notes=
            '''It is clear that the data on the left are better fitted by a line
            compared to the data on the right. But how can we quantify this
            intuition? [CLICK]
            '''
        )
        ax_2 = ax.copy().move_to(HALF_SCREEN_RIGHT)
        linear_dataset_2 = generate_regression_dataset(reg_line.eval, 20, x_range=X_RANGE, sigma=0.15, seed=2)
        loose_dataset = points_from_data(linear_dataset_2, ax_2, **point_config)
        _loose_fit = np.polynomial.polynomial.Polynomial.fit(linear_dataset_2[:, 0], linear_dataset_2[:, 1], 1).convert().coef
        loose_reg_line = RegressionLine(_loose_fit[1],  _loose_fit[0], ax_2, x_range=X_RANGE)
        loose_reg_line.suspend_updating()
        loose_plot = VGroup(ax_2, loose_dataset, loose_reg_line).move_to(HALF_SCREEN_RIGHT)
        tight_plot = VGroup(ax, tight_dataset, tight_reg_line)

        self.play(
            Succession(
                tight_plot.animate.scale(0.6).move_to(HALF_SCREEN_LEFT),
                FadeIn(loose_plot.scale(0.6))
            )
        )

        # SLIDE 18:  ===========================================================
        # COEFFICIENT OF DETERMINATION TITLE APPEARS
        self.next_slide(
            notes=
            '''One commnly used metric for this purpose is the Coefficient of
            determination, [CLICK] ...
            '''
        )
        title = Text('Coefficient of Determination', font_size=64, color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).to_edge(UP).shift(UP*0.5)
        self.play(Write(title))

        # SLIDE 19:  ===========================================================
        # FORMULA FOR R2 APPEARS
        self.next_slide(
            notes=
            '''... or R squared, which is defined as 1 - E / E bar, [CLICK] ...
            '''
        )
        r2_formula = MathTex(r'R^2 = 1 - \frac{E}{\bar{E}}', color=BLACK)

        self.play(FadeOut(tight_plot, loose_plot))
        self.play(FadeIn(r2_formula))

        # SLIDE 20:  ===========================================================
        # E DEFINITION APPEARS
        self.next_slide(
            notes=
            '''...where E is the sum of squared residuals that we defined
            before, [CLICK] ...
            '''
        )
        E_formula = MathTex(r'E = \sum_{i=1}^n r_i^2 = \sum_{i=1}^n (\widehat{y_i} - y_i)^2', color=BLACK,
                            tex_to_color_map={'y_i':ORANGE, r'\widehat{y_i}': ORANGE})
        
        self.play(r2_formula.animate.shift(UP))
        E_formula.next_to(r2_formula, DOWN, buff=0.5)
        self.play(FadeIn(E_formula))

        # SLIDE 21:  ===========================================================
        # DEFINITIONS OF E BAR AND OF Y BAR APPEAR
        self.next_slide(
            notes=
            '''...and E bar is the sum of the squares of the distances of the
            data points from the mean of y, denoted as y bar. [CLICK]
            '''
        )
        Ebar_formula = MathTex(r'\bar{E} = \sum_{i=1}^n (y_i - \bar{y})^2, \ \bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i', color=BLACK,
                               tex_to_color_map={'y_i':ORANGE, r'\widehat{y_i}': ORANGE})
        Ebar_formula.next_to(E_formula, DOWN, buff=0.5)
        
        self.play(FadeIn(Ebar_formula))

        # SLIDE 22:  ===========================================================
        # R^2 IN [0, 1] APPEARS
        self.next_slide(
            notes=
            '''R2 is a value between 0 and 1 that measures the fraction of the
            variability in the data that is explained by the model. [CLICK]
            '''
        )
        r2_formula_ext = MathTex(r'R^2 = 1 - \frac{E}{\bar{E}} \in [0, 1]', color=BLACK)

        self.play(
            AnimationGroup(
                FadeOut(E_formula, Ebar_formula),
                ReplacementTransform(r2_formula[0], r2_formula_ext[0][:9]),
                FadeIn(r2_formula_ext[0][9:]),
                lag_ratio=0.5
            )
        )

        # SLIDE 23:  ===========================================================
        # GRAPH WITH POOR FIT APPEARS
        self.next_slide(
            notes=
            '''If R2 is close to 0, it indicates a poor fit of the data, [CLICK]
            '''
        )
        ax.restore()
        dataset_poor = generate_regression_dataset(reg_line.eval, 20, x_range=X_RANGE, sigma=0.15, seed=3)
        r2_showcase_points = points_from_data(dataset_poor, ax, **point_config).set_z_index(1)
        reg_line = RegressionLine(*linear_reg_coeffs(dataset_poor), ax, x_range=X_RANGE)
        reg_line.add_dataset(r2_showcase_points)
        counter = R2Counter(reg_line, r2_showcase_points, num_decimal_places=3).set_color(BLACK).move_to(title)
        
        self.play(
            AnimationGroup(
                FadeOut(title, r2_formula_ext[0][3:]),
                AnimationGroup(
                    ReplacementTransform(r2_formula_ext[0][:2], counter.label[0]),  # ugh, to avoid wrong interpolation
                    ReplacementTransform(r2_formula_ext[0][2], counter.label[1]),
                ),
                FadeIn(ax, r2_showcase_points, reg_line, reg_line.proj_lines, reg_line.proj_points),
                FadeIn(counter.value),
                lag_ratio=0.5
            )
        )

        # SLIDE 24:  ===========================================================
        # LINE IS ANIMATED TO WORST CASE SCENARIO
        self.next_slide(
            notes=
            '''...and in the worst case scenario, a baseline model which always
            predicts y bar will have R2 = 0. [CLICK]
            '''
        )
        dataset_worst = generate_regression_dataset(lambda x: 0.7 +0*x, 20, x_range=X_RANGE, sigma=0.2, seed=3)

        def MoveDatasetAndLine(rline: RegressionLine, points: VGroup, new_dataset: np.ndarray, new_coeffs = None):
            if new_coeffs  is None:
                new_coeffs = linear_reg_coeffs(new_dataset)
            return AnimationGroup(
                AnimationGroup(
                    points[i].animate.move_to(rline.ax.c2p(*new_dataset[i]))
                    for i in range(len(r2_showcase_points))
                ),
                rline.slope.animate.set_value(new_coeffs[0]),
                rline.intercept.animate.set_value(new_coeffs[1])
            )
        
        self.play(MoveDatasetAndLine(reg_line, r2_showcase_points, dataset_worst, new_coeffs=(0, np.mean(dataset_worst[:, 1]))))

        intercept_dot = Dot(ax.c2p(0, reg_line.intercept.get_value()), color=PURPLE_C)
        intercept_label = MathTex(r'\bar{y}', color=BLACK).scale(LABELS_SIZE).next_to(intercept_dot, LEFT)

        self.play(
            GrowFromCenter(intercept_dot),
            FadeIn(intercept_label)
        )

        # SLIDE 25:  ===========================================================
        # LINE IS ANIMATED TO CASE WITH GOOD FIT
        self.next_slide(
            notes=
            '''Vice versa, if R2 is close to 1 it indicates a good fit. [CLICK]
            '''
        )
        dataset_good = generate_regression_dataset(lambda x: 0.35*x +0.35, 20, x_range=X_RANGE, sigma=0.05, seed=3)
        
        self.play(FadeOut(intercept_dot, intercept_label))
        self.play(MoveDatasetAndLine(reg_line, r2_showcase_points, dataset_good))

        # SLIDE 26:  ===========================================================
        # PERFECT FIT CASE IS SHOWN
        self.next_slide(
            notes=
            '''In the best case, the predicted values exactly match the observed
            values, which results E=0 and R2 = 1. [CLICK]
            '''
        )
        dataset_best = generate_regression_dataset(reg_line.eval, 20, x_range=X_RANGE, sigma=0.0, seed=3)

        self.play(MoveDatasetAndLine(reg_line, r2_showcase_points, dataset_best))

        # SLIDE 27:  ===========================================================
        # CODE SNIPPETS AS SOLUTION EXAMPLES APPEAR
        self.next_slide(
            notes=
            '''Try yourself to implement a function that computes the
            coefficient of determination and evaluate which one among
            temperature, relative humidity and buildup index regression lines
            has the best fit. [END]
            '''
        )
        sample_r2_code_py = ColabCodeWithLogo(
            r'''
            def R2(x, y):
                E = ...
                E_bar = ...
                r_2 = ...
                return r_2
            ''',
            logo_pos=LEFT
        )
        sample_r2_code_mat = MatlabCodeWithLogo(
            r'''
            function r2 = R2(x, y):
                E = ...
                E_bar = ...
                r_2 = ...
            end
            ''',
            logo_pos=LEFT
        ).next_to(sample_r2_code_py, DOWN, buff=1).align_to(sample_r2_code_py, LEFT)
        Group(sample_r2_code_py, sample_r2_code_mat).center()

        self.play(FadeOut(ax, reg_line, r2_showcase_points, reg_line.proj_lines, reg_line.proj_points, counter))
        self.play(
            Succession(
                Wait(0.5),
                FadeIn(sample_r2_code_py, sample_r2_code_mat)
            )
        )