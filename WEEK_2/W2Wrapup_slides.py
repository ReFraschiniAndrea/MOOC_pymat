import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from config import *
from W2Anim import *
from Generic_mooc_utils import *
from colab_utils import ColabCodeWithLogo, COLAB_LIGHTGRAY
from matlab_utils import MatlabCodeWithLogo

config.update(TEST_CONFIG)

LABELS_SIZE=0.75
ICONS_HEIGHT = 0.6

class W2Theory_slides(ThreeDMOOCSlide):
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
            Text('Temperature' , color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2),
            Text('Fire risk', color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2)
        )

        wildfire_icon = SVGMobject(r'Assets\W2\forest_fire_icon.svg').scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(-0.1, 1, 0))
        forest_icon = SVGMobject(r'Assets\W2\pine_trees_icon.svg').scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(-0.1, 0.3, 0))
        low_temp_icon = SVGMobject(r'Assets\W2\low_temperature_icon.svg').set_color(BLUE).scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(0.3, -0.1, 0))
        high_temp_icon = SVGMobject(r'Assets\W2\high_temperature_icon.svg').set_color(RED).scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(X_RANGE[1]-0.1, -0.1, 0))

        dataset = generate_regression_dataset(func= lambda x: 1.5*(0.4*x-0.75)**3 + 0.8, x_range=(0.1, 1.5), n=20, sigma=0.15, seed=0)
        dataset_points = points_from_data(dataset, ax=ax, color=PURPLE_A).set_z_index(1)
        linear_fit = np.polynomial.polynomial.Polynomial.fit(dataset[:, 0], dataset[:, 1], 1).convert().coef
        reg_line = RegressionLine(linear_fit[1], linear_fit[0], ax, x_range=X_RANGE)

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
            residuals,  [CLICK]
            '''
        )
        minim_problem = MathTex(r'\min_{m,q} \sum_{i=1}^n (r_i)^2', color=BLACK).to_edge(UP).shift(UP)
        example_lr_plot = VGroup(ax, variable_ax_lab, reg_line, dataset_points,
                                 forest_icon, wildfire_icon, high_temp_icon, low_temp_icon)

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
            '''... and that the coefficients of this line can be found
            through appropriate calculations.
            We then saw how to implement these calculations in Python and MATLAB.
            [CLICK]
            '''
        )
        LR_equations = LinearRegressionEquations().move_to(minim_problem)

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
        sum_terms = LR_equations.get_sums_without_repetition().arrange(RIGHT, buff=1).scale(1.2).to_edge(UP)
        sum_python_code = ColabCodeWithLogo(
            r'''
            np.sum(x)
            np.sum(y)
            np.sum(x * y)
            np.sum(x ** 2)
            '''
        ).move_to(HALF_SCREEN_LEFT).shift(DOWN)
        sum_matlab_code = MatlabCodeWithLogo(
            r'''
            sum(x)
            sum(y)
            sum(x.*y)
            sum(x.^2)
            '''
        ).move_to(HALF_SCREEN_RIGHT).shift(DOWN)

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
            HighlightRectangle(sum_matlab_code.codeMobject[2][17:19]),
            HighlightRectangle(sum_matlab_code.codeMobject[3][17:19]),
        )

        self.play(Create(dot_operator_highlights))

        # SLIDE 05:  ===========================================================
        # '*' AND '**' HIGHLIGHTED IN PYTHON CODE
        self.next_slide(
            notes=
            '''In Python with NumPy, on the other hand, element-wise behavior is
            the default when working with arrays. [CLICK]
            '''
        )
        numpy_highlights = VGroup(
            HighlightRectangle(sum_python_code.codeMobject[2][21]),
            HighlightRectangle(sum_python_code.codeMobject[3][21:23]),
        )

        self.play(
            ReplacementTransform(dot_operator_highlights[0], numpy_highlights[0]),
            ReplacementTransform(dot_operator_highlights[1], numpy_highlights[1]),
        )
        
        # SLIDE 06:  ===========================================================
        # 'np.sum' AND 'sum' HIGHLIGHTED IN THE TWO CODES
        self.next_slide(
            notes=
            '''Finally, for computing these summations, in Python we used the
            sum function from the NumPy module, while in MATLAB we used the
            built-in sum function. [CLICK]
            '''
        )
        sum_highlights = (
              VGroup(HighlightRectangle(sum_python_code.codeMobject[i][12:18]) for i in range(4))
            + VGroup(HighlightRectangle(sum_matlab_code.codeMobject[i][12:15]) for i in range(4))
        )

        self.play(FadeOut(numpy_highlights))
        self.play(Create(sum_highlights))

        # SLIDE 07:  ===========================================================
        # FUNCTION SCHEME?
        self.next_slide(
            notes=
            '''We then implemented a function that takes as input two arrays
            containing the x- and y-coordinates of the available data points,
            and computes the coefficients m and q of the regression line.
            [CLICK]
            '''
        )
        python_function_code = ColabCodeWithLogo(
            r'''
            def linear_regression(x, y):
                ...
                return m, q
            ''',
            logo_pos=LEFT
        )
        matlab_function_code = MatlabCodeWithLogo(
            r'''
            function [m, q] = linear_regression(x, y):
                ...
            end
            ''',
            logo_pos=LEFT
        ).next_to(python_function_code, DOWN)
        Group(python_function_code, matlab_function_code).center()

        self.play(FadeOut(sum_highlights, sum_python_code, sum_matlab_code, sum_terms))
        self.play(FadeIn(python_function_code, matlab_function_code))
        
        # SLIDE 08:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''To define functions, the syntax differs slightly between the two
            languages: in Python, we use the keyword def to start the function
            definition and return results using the return statement. [CLICK]
            '''
        )
        py_func_highlights = VGroup(
            HighlightRectangle(python_function_code.codeMobject[0][12:15]),
            HighlightRectangle(python_function_code.codeMobject[2][16:]),
        )

        self.play(Create(py_func_highlights))

        # SLIDE 09:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''In MATLAB, on the other hand, functions are introduced with the
            keyword function, and the output variables are listed directly in
            the function header. [CLICK]
            '''
        )
        mat_func_highlights = HighlightRectangle(matlab_function_code.codeMobject[0][12:27])
        
        self.play(FadeOut(py_func_highlights))
        self.play(FadeIn(mat_func_highlights))

        # SLIDE 10:  ===========================================================
        # HEAD TABLE DATASET APPEARS
        self.next_slide(
            notes=
            '''Using the functions we implemented on a dataset that includes
            detailed features related to a set of forest fires recorded in
            Algeria,  [CLICK]
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

        # SLIDE 11:  ===========================================================
        # PLOT OF LINEAR REGRESSION (TEMP. VS FWI) APPEARS
        self.next_slide(
            notes=
            '''we obtained a regression line linking temperature to fire risk.
            This allowed us to gain a quantitative insight into how an increase
            in temperature raises the likelihood of wildfires. [CLICK]
            '''
        )
        self.play(FadeOut(head_table))

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
            Text('Temperature' , color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2),
            Text('FWI', color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2),
        )

        point_config = {'color': PURPLE_A, 'radius': DEFAULT_DOT_RADIUS}
        temp_dataset = points_from_data(np.column_stack((temperature, FWI)), ax_temp, **point_config)
        _temp_fit = np.polynomial.polynomial.Polynomial.fit(temperature, FWI, 1).convert().coef
        temp_reg_line = RegressionLine(_temp_fit[1], _temp_fit[0], ax_temp, x_range=t_range)
        
        self.play(FadeIn(ax_temp, temp_labels, temp_dataset))
        self.play(Create(temp_reg_line))

        # SLIDE 12:  ===========================================================
        # PLOT OF LINEAR REGRESSION (HUMIDITY. VS FWI) REPLACES FIRST ONE
        self.next_slide(
            notes=
            '''Similarly, we saw how higher humidity can mitigate this risk,
            reducing the overall fire danger.
            But what happens if we consider both temperature and humidity? [CLICK]
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
            Text('RH' , color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2).rotate(PI/2, X_AXIS).next_to(ax_3d.get_axis(0).get_corner(OUT+RIGHT), OUT),
            Text('Temperature' , color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2).rotate(PI/2, X_AXIS).next_to(ax_3d.get_axis(1).get_corner(OUT+UP), OUT),
            Text('FWI', color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2).rotate(PI/2, RIGHT).next_to(ax_3d.get_axis(2).get_corner(OUT+RIGHT), RIGHT),
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

        self.set_camera_orientation(phi=90 * DEGREES, theta=-90 * DEGREES, gamma=0*DEGREES, zoom=0.7)
        self.play(FadeIn(ax_3d.x_axis, ax_3d.z_axis, ax_rh.background_lines, ax_rh.x_axis.numbers, ax_rh.y_axis.numbers, dataset_3d,
                         threeD_labels[0], threeD_labels[2]))
        self.play(Create(rh_reg_line))

        # SLIDE 12:  ===========================================================
        # PLOT OF DATASET BECOMES 3D
        self.next_slide(
            notes=
            '''But what happens if we consider both temperature and humidity?
            [CLICK]
            '''
        )
        dxs = np.linspace(rh_range[0], rh_range[1], 17)
        dys = np.linspace(t_range[0], t_range[1], 17)
        plane = VGroup( l for x, y in zip(dxs, dys) for l in(
            Line(ax_3d.c2p(x, t_range[0], fwi_range[0]), ax_3d.c2p(x, t_range[1], fwi_range[0]), stroke_color=BLACK, stroke_width=0.2),
            Line(ax_3d.c2p(rh_range[0], y, fwi_range[0]), ax_3d.c2p(rh_range[1], y, fwi_range[0]), stroke_color=BLACK, stroke_width=0.2)
            ) 
        )
        CAMERA_ROTATION_ANGLE = 40*DEGREES
        phi, theta, _, _, zoom = self.camera.get_value_trackers()
        self.play(FadeOut(ax_rh.background_lines, ax_rh.x_axis.numbers, ax_rh.y_axis.numbers, rh_reg_line))
        self.play(
            phi.animate.set_value(80*DEGREES),
            theta.animate.set_value(-90*DEGREES + CAMERA_ROTATION_ANGLE),
            zoom.animate.set_value(0.5),
            Create(ax_3d.y_axis), Create(plane),
            threeD_labels[0].animate.rotate(CAMERA_ROTATION_ANGLE, Z_AXIS),
            threeD_labels[2].animate.rotate(CAMERA_ROTATION_ANGLE, Z_AXIS),
            FadeIn(threeD_labels[1].rotate(CAMERA_ROTATION_ANGLE, Z_AXIS)),
            AnimationGroup(
                dataset_3d[i].animate.rotate(CAMERA_ROTATION_ANGLE, Z_AXIS).move_to(ax_3d.c2p(RH[i], temperature[i], FWI[i])) for i in range(len(RH))
            ),
        )

        # SLIDE 13:  ===========================================================
        # SIMPLE LINEAR REGRESSION FORMULA APPEARS AT TOP
        self.next_slide(
            notes=
            '''So far, we've focused on "simple" linear regression, where we
            have a single independent variable, x [CLICK]
            '''
        )
        linear_relation = MathTex(
            r'y = f(x) = {{mx+q}}',
            color = BLACK, tex_to_color_map={'y': ORANGE, 'x':BLUE}
        )
        self.add_fixed_in_frame_mobjects(linear_relation)
        linear_relation.to_edge(UP).shift(UP)
        self.play(FadeIn(linear_relation))

        # SLIDE 14:  ===========================================================
        # MULTIPLE LINEAR REGRESSION REPLACES SIMPLE ONE
        # REGRESSION PLANE APPEARS
        self.next_slide(
            notes=
            '''But in many cases, we have multiple independent variables—let's
            call them x1, x2, all the way up to xp. In this situation, the
            regression model includes p coefficients, [CLICK]
            '''
        )
        multiple_linear_relation = MathTex(
            r'y= f(x_1, x_2, \dots, \x_p) = {{ m_1 x_1 + m_2 x_2 + \dots + m_p x_p + q}}',
            color = BLACK, tex_to_color_map={'y': ORANGE, 'x_1':BLUE, 'x_2': BLUE, 'x_p':BLUE}
        ).move_to(linear_relation)

        # regression plane
        X = np.column_stack((RH, temperature))
        X = np.c_[X, np.ones(X.shape[0])]  # add bias term
        beta_hat = np.linalg.lstsq(X, FWI, rcond=None)[0]
        reg_plane = lambda x, y: beta_hat[0]*x + beta_hat[1]*y +beta_hat[2]
        reg_plane_plot = ax_3d.plot_surface(reg_plane,
                                            u_range=rh_range, v_range=t_range,
                                            checkerboard_colors=(BLUE, BLUE), fill_opacity=0.7, resolution=16)

        self.play(
            ReplacementTransform(linear_relation[:2], multiple_linear_relation[:2]), # y=f(
            ReplacementTransform(linear_relation[2], multiple_linear_relation[2:7]), # f argument
            ReplacementTransform(linear_relation[3], multiple_linear_relation[7]), # )=
            ReplacementTransform(linear_relation[4:6], multiple_linear_relation[8:14]), # m*x
            ReplacementTransform(linear_relation[-1], multiple_linear_relation[-1]), # +q
        )
        self.begin_ambient_camera_rotation(rate=0)  # so that manim checks z-depth every frame
        self.play(Create(reg_plane_plot))
        self.wait(0.5)
        
        # SLIDE 15:  ===========================================================
        # HIGHLIGHT 'm_i' COEFFICIENTS
        self.next_slide(
            notes=
            '''m1 through mp, one for each independent variable [CLICK]
            '''
        )
        m_i_highlights = VGroup(
            HighlightRectangle(multiple_linear_relation[8]),
            HighlightRectangle(multiple_linear_relation[10]),
            HighlightRectangle(multiple_linear_relation[12][-2:]),
        )

        self.play(Create(m_i_highlights))
        # SLIDE 15:  ===========================================================
        # HIGHLIGHT 'q'
        self.next_slide(
            notes=
            '''along with an intercept q. [CLICK]
            '''
        )
        q_highlight = HighlightRectangle(multiple_linear_relation[-1][-1])

        self.play(Create(q_highlight))

        # SLIDE 16:  ===========================================================
        # FORMULA FOR Y HAT PREDICTION APPEARS
        # LINES FROM DATA TO PREDICITIONS ARE DRAWN
        self.next_slide(
            notes=
            '''Just like in simple linear regression, in multiple linear
            regression, we calculate yi hat, which is the model's prediction for
            the i-th data point. [CLICK]
            '''
        )
        prediction_eq = MathTex(r'{{\hat{y_i}}} = m_1 (x_1)_i + \dots + m_p (x_p)_i',
                                color=BLACK, tex_to_color_map={'(x_1)_i':BLUE, '(x_p)_i':BLUE}
                                ).next_to(multiple_linear_relation, DOWN)
        prediction_eq[0].set_color(ORANGE)
        predictions = reg_plane(temperature, RH)
        prediction_dots = VGroup(Dot(ax_3d.c2p(t, rh, pred), color=TEAL).rotate(PI/2, X_AXIS).rotate(-40*DEGREES, Z_AXIS)
                                  for t, rh, pred in zip(temperature, RH, predictions))
        residual_lines = VGroup(Line3D(data.get_center(), pred.get_center(), color=TEAL) for data, pred in zip(dataset_3d, prediction_dots))

        self.play(FadeOut(m_i_highlights, q_highlight))
        self.play(FadeIn(prediction_eq))
        self.play(
            AnimationGroup(
                *[Succession(
                    Create(l), GrowFromCenter(p),
                    lag_ratio=0.5
                )
                for l, p in zip(residual_lines, prediction_dots)],
                lag_ratio=0.2,
                run_time=2
            )
        )
        self.stop_ambient_camera_rotation()

        # SLIDE 17:  ===========================================================
        # FORMULA FOR E APPEARS
        self.next_slide(
            notes=
            '''And again, the "best" fitting model is the one that minimizes the
            sum of the squares of the residuals, just as we defined before.
            Solving this minimization problem involves solving a linear system
            of equations. [CLICK]

            In the supplementary material, you can explore this technique
            further, in its extended form known as "multiple linear regression".
            '''
        )
        minim_problem = MathTex(r'\hat{m_1}, \dots, \hat{m_p} = \argmin_{m_1,\dots, m_p} \sum_{i=1}^n (r_i)^2', color=BLACK).move_to(prediction_eq)

        # SLIDE 18:  ===========================================================
        # RETURN TO SIMPLE LINEAR REGRESSION PLOT
        self.next_slide(
            notes=
            '''But a key question that we didn't address is the following: how
            well the regression line is fitting the available data? In other
            words, how effectively does the regression line explain the
            relationship between x and y? [CLICK]
            '''
        )
        linear_dataset_1 = generate_regression_dataset()
        linear_dataset_2 = generate_regression_dataset()
        _tight_fit = np.polynomial



        # SLIDE 19:  ===========================================================
        # TWO LINEAR REGRESSION PLOTS SIDE BY SIDE
        self.next_slide(
            notes=
            '''It is clear that the data on the left are better fitted by a line
            compared to the data on the right. But how can we quantify this
            intuition?  You can find the answer in the supplementary material.
            And now, it's your turn. [CLICK]
            '''
        )
        

class Test(ThreeDScene):
    def construct(self):
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
            Text('Temperature' , color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2),
            Text('FWI', color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2),
        )

        point_config = {'color': PURPLE_A, 'radius': DEFAULT_DOT_RADIUS}
        temp_dataset = points_from_data(np.column_stack((temperature, FWI)), ax_temp, **point_config)
        _temp_fit = np.polynomial.polynomial.Polynomial.fit(temperature, FWI, 1).convert().coef
        temp_reg_line = RegressionLine(_temp_fit[1], _temp_fit[0], ax_temp, x_range=t_range)
        self.add(ax_temp, temp_labels, temp_dataset)
        self.play(Create(temp_reg_line))

        self.wait() 
        self.play(FadeOut(ax_temp, temp_labels, temp_dataset, temp_reg_line))
        # display second one
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
        self.set_camera_orientation(phi=90 * DEGREES, theta=-90 * DEGREES, gamma=0*DEGREES, zoom=0.7)
        dataset_3d = VGroup(
            Dot(ax_3d.c2p(rh, t_range[0], fwi), **point_config).rotate(PI/2, X_AXIS) for rh, fwi in zip(RH, FWI)
        )
        ax_rh.scale(ax_3d.x_axis.width/ax_rh.x_axis.width*1.01)
        ax_rh.shift(ax_3d.c2p(rh_range[0], t_range[0], fwi_range[0])-ax_rh.c2p(rh_range[0], fwi_range[0]))
        threeD_labels = VGroup(
            Text('RH' , color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2).rotate(PI/2, X_AXIS).next_to(ax_3d.get_axis(0).get_corner(OUT+RIGHT), OUT),
            Text('Temperature' , color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2).rotate(PI/2, X_AXIS).next_to(ax_3d.get_axis(1).get_corner(OUT+UP), OUT),
            Text('FWI', color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2).rotate(PI/2, RIGHT).next_to(ax_3d.get_axis(2).get_corner(OUT+RIGHT), RIGHT),
        )
        _rh_fit = np.polynomial.polynomial.Polynomial.fit(RH, FWI, 1).convert().coef
        rh_reg_line = RegressionLine(_rh_fit[1], _rh_fit[0], ax_rh, x_range=rh_range)
        self.play(FadeIn(ax_3d.x_axis, ax_3d.z_axis, ax_rh.background_lines, ax_rh.x_axis.numbers, ax_rh.y_axis.numbers, dataset_3d,
                         threeD_labels[0], threeD_labels[2]))
        self.play(Create(rh_reg_line))
        self.wait()

        # shift to 3d
        dxs = np.linspace(rh_range[0], rh_range[1], 17)
        dys = np.linspace(t_range[0], t_range[1], 17)
        plane = VGroup( l for x, y in zip(dxs, dys) for l in(
            Line(ax_3d.c2p(x, t_range[0], fwi_range[0]), ax_3d.c2p(x, t_range[1], fwi_range[0]), stroke_color=BLACK, stroke_width=0.2),
            Line(ax_3d.c2p(rh_range[0], y, fwi_range[0]), ax_3d.c2p(rh_range[1], y, fwi_range[0]), stroke_color=BLACK, stroke_width=0.2)
            ) 
        )

        phi, theta, _, _, zoom = self.camera.get_value_trackers()
        CAMERA_ROTATION_ANGLE = 40*DEGREES
        self.play(FadeOut(ax_rh.background_lines, rh_reg_line))
        self.play(
            phi.animate.set_value(80*DEGREES),
            theta.animate.set_value(-90*DEGREES + CAMERA_ROTATION_ANGLE),
            zoom.animate.set_value(0.5),
            Create(ax_3d.y_axis), Create(plane),
            threeD_labels[0].animate.rotate(CAMERA_ROTATION_ANGLE, Z_AXIS),
            threeD_labels[2].animate.rotate(CAMERA_ROTATION_ANGLE, Z_AXIS),
            FadeIn(threeD_labels[1].rotate(CAMERA_ROTATION_ANGLE, Z_AXIS)),
            AnimationGroup(
                dataset_3d[i].animate.rotate(CAMERA_ROTATION_ANGLE, Z_AXIS).move_to(ax_3d.c2p(RH[i], temperature[i], FWI[i])) for i in range(len(RH))
            ),
        )

        # create regression plane
        X = np.column_stack((RH, temperature))
        X = np.c_[X, np.ones(X.shape[0])]  # add bias term
        beta_hat = np.linalg.lstsq(X, FWI, rcond=None)[0]
        reg_plane = lambda x, y: beta_hat[0]*x + beta_hat[1]*y +beta_hat[2]
        reg_plane_plot = ax_3d.plot_surface(reg_plane,
                                            u_range=rh_range, v_range=t_range,
                                            checkerboard_colors=(BLUE, BLUE), fill_opacity=0.7, resolution=16)

        self.begin_ambient_camera_rotation(rate=0)  # so that manim checks z-depth evry frame
        self.play(Create(reg_plane_plot))
        self.wait(0.5)

        # add predictions and residual lines
        predictions = reg_plane(temperature, RH)
        prediction_dots = VGroup(Dot(ax_3d.c2p(t, rh, pred), color=TEAL).rotate(PI/2, X_AXIS).rotate(-40*DEGREES, Z_AXIS)
                                  for t, rh, pred in zip(temperature, RH, predictions))
        residual_lines = VGroup(Line3D(data.get_center(), pred.get_center(), color=TEAL) for data, pred in zip(dataset_3d, prediction_dots))

        self.play(
            AnimationGroup(
                *[Succession(
                    Create(l), GrowFromCenter(p),
                    lag_ratio=0.5
                )
                for l, p in zip(residual_lines, prediction_dots)],
                lag_ratio=0.2,
                run_time=2
            )
        )
        self.stop_ambient_camera_rotation()

        # 


        
        


