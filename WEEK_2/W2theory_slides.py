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
        # ax_labels = ax.get_axis_labels(
        #     MathTex("x", color=BLACK).scale(LABELS_SIZE), MathTex("y", color=BLACK).scale(LABELS_SIZE)
        # )
        self.play(Create(ax))

        # SLIDE 02:  ===========================================================
        # HUMIDITY AND FIRE RISK ICONS
        # GRAPH WITH NEGATIVE CORRELATION APPEARS
        self.next_slide(
            notes=
            '''For example, we might expect that, the higher the humidity in the
            air, the lower the fire risk on that day. [CLICK]
            '''
        )
        t = ValueTracker(X_RANGE[0]+0.1)
        negative_corr_func = lambda t: 0.1*1/(t + 0.1) + 0.2
        tracing_dot = Dot(color=BLUE).add_updater(
            lambda m: m.move_to(ax.c2p(t.get_value(), negative_corr_func(t.get_value()), 0))
        )
        tracing_dot.update()
        trace = TracedPath(tracing_dot.get_center, stroke_color=BLUE, stroke_width=3)
        self.add(trace)

        # self.play(FadeIn(humidity_icon, fire_icon))
        self.play(GrowFromCenter(tracing_dot), run_time=0.5)
        self.wait(0.3)
        self.play(t.animate(run_time=1.5).set_value(X_RANGE[1]))

        # SLIDE 03:  ===========================================================
        # TEMPERATURE ICON REPLACES HUMIDITY ONE
        # GRAPH WITH POSITIVE CORRELATION APPEARS
        self.next_slide(
            notes=
            '''Similarly, we expect that high temperature increases the risk of
            fire. [CLICK]
            '''
        )
        self.play(
            FadeOut(trace, tracing_dot),
            # Succession(
            #     FadeOut(humidity_icon),
            #     FadeIn(temperature_icon),
            #     run_time=1
            # )
        )

        positive_corr_func = lambda t: 0.4*t**2 +0.2
        tracing_dot.clear_updaters().add_updater(
            lambda m: m.move_to(ax.c2p(t.get_value(), positive_corr_func(t.get_value()), 0))
        )
        t.set_value(X_RANGE[0]+0.1)
        tracing_dot.update()
        trace2 = TracedPath(tracing_dot.get_center, stroke_color=GREEN, stroke_width=3)

        self.play(GrowFromCenter(tracing_dot), run_time=0.5)
        self.add(trace2)
        self.wait(0.3)
        self.play(t.animate(run_time=1.5).set_value(X_RANGE[1]))

        # SLIDE 04:  ===========================================================
        # DATA POINTS APPEAR
        self.next_slide(
            notes=
            '''To answer this question quantitatively, we use real-world data
            that captures key environmental factors, such as daily temperature,
            humidity, and wildfire occurrence. [CLICK] 
            '''
        )
        self.play(FadeOut(trace2, tracing_dot))
        dataset = generate_regression_dataset(func= lambda x: 1.5*(0.4*x-0.75)**3 + 0.5, x_range=(0.1, 1.5), n=20, sigma=0.15, seed=0)
        dataset_points = points_from_data(dataset, ax=ax, color=PURPLE_A).set_z_index(1)
        self.play(
            AnimationGroup(
                *[GrowFromCenter(p) for p in dataset_points],
                run_time=2, lag_ratio=0.5)
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
        # fit a parabolic model, an exponential one and a cubic one
        x, y = dataset[:, 0], dataset[:, 1]
        linear_fit = np.polynomial.polynomial.Polynomial.fit(x, y, 1).convert().coef
        quadratic_fit = np.polynomial.polynomial.Polynomial.fit(x, y, 2).convert().coef
        cubic_fit = np.polynomial.polynomial.Polynomial.fit(x, y, 3).convert().coef
        log_fit = np.polynomial.polynomial.Polynomial.fit(np.log(x), y, 1).convert().coef

        quadratic_plot = ax.plot(lambda t: np.polynomial.polynomial.polyval(t, quadratic_fit), x_range=X_RANGE, color=BLUE)
        cubic_plot = ax.plot(lambda t: np.polynomial.polynomial.polyval(t, cubic_fit), x_range=X_RANGE, color=BLUE)
        log_plot = ax.plot(lambda t: log_fit[0]+ log_fit[1]*np.log(t), x_range=(X_RANGE[0]+0.01, X_RANGE[1]), color=BLUE, use_smoothing=False)

        self.play(Create(quadratic_plot))
        self.wait(1)
        self.play(ReplacementTransform(quadratic_plot, cubic_plot))
        self.wait(1)
        self.play(ReplacementTransform(cubic_plot, log_plot))

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
        reg_line = RegressionLine(linear_fit[1], linear_fit[0], ax, x_range=X_RANGE)
        reg_line.suspend_updating()
        LR_title = Text('Linear Regression', font_size=64, color=BLACK, font='Microsoft JhengHei', weight=LIGHT).to_edge(UP).shift(UP*0.5)
        self.play(ReplacementTransform(log_plot, reg_line))
        self.play(
            AnimationGroup(
                VGroup(ax, reg_line, dataset_points).animate.shift(DOWN),
                Write(LR_title),
                lag_ratio=0.5
            )
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
        x_lab = Text('x', color=BLUE).scale(LABELS_SIZE)
        y_lab = Text('y', color=ORANGE).scale(LABELS_SIZE)
        ax.get_axis_labels(x_lab, y_lab)
        self.play(Write(x_lab))
        self.wait(1)
        self.play(Write(y_lab))

        # SLIDE 07:  ===========================================================
        # LINEAR RELATION EQUATION APPEARS
        # LINE IS DRAWN IN THE AXIS
        self.next_slide(
            notes=
            '''This technique assumes that the relationship between the
            dependent and independent variables is described by the equation y
            equals m times x plus q.
            The coefficients m and q have clear and intuitive meanings.
            '''
        )
        # linear_relation = MathTex(r'y = f({{x}}) = m {{x}} + q')

        # SLIDE 08:  ===========================================================
        # 'q' HIGHLIGHTED IN THE EQUATION AND ON THE GRAPH
        self.next_slide(
            notes=
            '''Specifically, q represents the y-intercept, that is the predicted
            value when x equals 0.
            '''
        )
        intercept_dot = Dot(ax.c2p(0, reg_line.intercept.get_value()), color=PURPLE_C)
        intercept_label = Text('q', color=BLACK).scale(LABELS_SIZE).next_to(intercept_dot, LEFT)

        self.play(
            GrowFromCenter(intercept_dot),
            Write(intercept_label)
        )

        # SLIDE 09:  ===========================================================
        # 'm' HIGHLIGHTED IN THE EQUATION
        # RISE OVER RUN IS DRAWN TO EXPLAIN 'm'
        self.next_slide(
            notes=
            '''m, instead, represents the "slope" of the line, that is the rate
            of change of y with respect to x. In other words, [CLIK] ...
            '''
        )
        ror_dx = 0.25
        ror_x = (X_RANGE[1] - X_RANGE[0])/2 - ror_dx/2
        rise_over_run = Polygon(
            reg_line.eval_to_point(ror_x),
            ax.c2p(ror_x+ror_dx, reg_line.eval(ror_x), 0),
            reg_line.eval_to_point(ror_x + ror_dx),
            color = PURPLE_C,
            fill_opacity=1,
            stroke_width=0
        )
        slope_label = Text('m', color=BLACK).scale(LABELS_SIZE).next_to(rise_over_run, UP)
        
        self.play(
            Create(rise_over_run),
            Write(slope_label)
        )

        # SLIDE 10:  ===========================================================
        # RISE OVER RUN TRIANGLE APPEARS
        # DELTA_X LABEL APPEARS
        self.next_slide(
            notes=
            '''... if the independent variable x increases by a certain amount
            delta_x, [CLICK] ...
            '''
        )
        ror_brace_x = Brace(rise_over_run, DOWN, color=BLACK)
        ror_x_lab = Text('dx', color=BLACK).scale(LABELS_SIZE) # MathTex
        ror_brace_x.put_at_tip(ror_x_lab)
        ror_brace_y = Brace(rise_over_run, RIGHT, color=BLACK)
        ror_y_lab = Text('m*dy', color=BLACK).scale(LABELS_SIZE) # MathTex
        ror_brace_y.put_at_tip(ror_y_lab)

        self.play(FadeIn(ror_brace_x, ror_x_lab))

        # SLIDE 11:  ===========================================================
        # DELTA_Y LABEL APPEARS
        self.next_slide(
            notes=
            '''... the predicted dependent variable y will increase by m*delta_x. [CLICK]
            '''
        )
        self.play(FadeIn(ror_brace_y, ror_y_lab))

        # SLIDE 11:  ===========================================================
        # EVERYTHING FADES OUT TO LEAVE AXES EMPTY
        self.next_slide(
            notes=
            '''Let us now see how to find a line from a cloud of points. [CLICK]
            '''
        )
        self.play(FadeOut(reg_line, rise_over_run, ror_brace_x, ror_brace_y, ror_x_lab, ror_y_lab,
                          slope_label, intercept_label, intercept_dot, dataset_points))

        # SLIDE 12:  ===========================================================
        # AXES WITH TWO LONE POINTS APPEARS
        # UNIQUE LINE BETWEEN THEM IS DRAWN
        self.next_slide(
            notes=
            '''We know that for two distinct points, there is exactly one line
            that passes through them. [CLICK]
            '''
        )
        passing_line = RegressionLine(
            *mq_throgh_points(dataset[5], dataset[15]),
            axes=ax, x_range=X_RANGE, color=BLUE).set_length(5)
        self.play(GrowFromCenter(dataset_points[5]), GrowFromCenter(dataset_points[15]))
        self.wait(1)
        self.play(Create(passing_line))

        # SLIDE 13:  ===========================================================
        # LINE DISAPPEARS, WHOLE POINT CLOUD APPEARS
        self.next_slide(
            notes=
            '''But what happens when there are many points? [CLICK]
            '''
        )
        self.play(FadeOut(passing_line))
        self.play(
            AnimationGroup(
                *[GrowFromCenter(p) for p in dataset_points-VGroup(dataset_points[5], dataset_points[15])],
                run_time=2, lag_ratio=0.5)
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
        RNG = np.random.default_rng(seed=0)
        perturbed_mg = [(linear_fit[1], linear_fit[0])+RNG.normal(0 , scale=linear_fit[1]/2, size=2) for _ in range(5)]
        lines = VGroup(
            RegressionLine(m,q, ax, x_range=X_RANGE) for m,q in perturbed_mg
        )
        lines.add(reg_line)
        self.play(Create(lines, lag_ratio=0.2))

        # SLIDE 15:  ===========================================================
        # ONLY ONE LINE FOR EXPLANATION REMAINS
        self.next_slide(
            notes=
            '''To find the "best"-fitting line, we must first define what we
            mean by "best". [CLICK]
            '''
        )
        self.play(FadeOut(lines[:-1], lag_ratio=0.2))

        # SLIDE 16:  ===========================================================
        # LABELS FOR P1, P2, Pi APPEAR IN SUCCESSION
        # ONLY LABEL OF Pi REMAINS
        self.next_slide(
            notes=
            '''Let (x_1, y_1) be the first data point, (x_2, y_2) the second,
            and so on for all n points. [CLICK]
            '''
        )
        p1 = dataset_points[0]
        p2 = dataset_points[1]
        pi = dataset_points[10]
        lines_p1 = ax.get_lines_to_point(p1.get_center(), color=BLACK)
        lines_p2 = ax.get_lines_to_point(p2.get_center(), color=BLACK)
        lines_pi = ax.get_lines_to_point(pi.get_center(), color=BLACK)

        self.play(FadeIn(lines_p1))
        self.play(FadeIn(lines_p2))
        self.play(FadeIn(lines_pi))

        # SLIDE 17:  ===========================================================
        # POINT REPRESENTING PREDICTION APPEARS
        self.next_slide(
            notes=
            '''For any given point (x_i, y_i), we define y^_i as the predicted
            value from the model, which is m * x_i + q.
            '''
        )
        self.play(FadeOut(lines_p1, lines_p2))
        ax.get_x_axis()
        predicted_point = Dot(reg_line.eval_to_point(ax.p2c(pi.get_center()[0])))
        prediction_eq = Text(r'\hat{y_i} =m x_i + q').move_to(LR_title)
        prediction_label = Text(r'\hat{y_i}', color=BLACK).scale(LABELS_SIZE).next_to(predicted_point, UP)

        self.play(FadeOut(LR_title))
        self.play(FadeIn(prediction_eq))
        self.play(
            GrowFromCenter(predicted_point),
            ReplacementTransform(prediction_eq, prediction_label)
        )

        # SLIDE 18:  ===========================================================
        # RESIDUAL BRACE WITH LABEL APPEARS
        self.next_slide(
            notes=
            '''The difference y_i - y^_i is called the residual, and it measures
            the error between the actual and predicted values. [CLICK]
            '''
        )
        residual_brace = BraceBetweenPoints(predicted_point.get_center(), pi.get_center, LEFT, color=BLACK)
        residual_label = Text(r'r_i', color=BLACK).scale(LABELS_SIZE)
        residual_brace.put_at_tip(residual_label)
        
        self.play(ReplacementTransform())
        self.play(
            FadeIn(residual_brace),
            ReplacementTransform()
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
        E_formula = MathTex(r'E = \sum_{i=1}^n r_i^2 = \sum_{i=1}^n (y_i - \hat{y_i})^2', color=BLACK).scale(LABELS_SIZE)

        # SLIDE 21:  ===========================================================
        # COUNTERS FOR 'm' AND 'q' APPEAR
        # 'm' AND 'q' VARY, CHANGING THE LINE
        self.next_slide(
            notes=
            '''By varying m and q, we obtain different lines. For each
            combination of m and q, we can compute the corresponding error E,
            [CLICK]
            '''
        )
        # m_counter = Variable(expl_line.slope.get_value(), label='m', num_decimal_places=2).set_color(BLACK)
        # m_counter.tracker = expl_line.slope
        # q_counter = Variable(expl_line.intercept.get_value(), label='q', num_decimal_places=2).set_color(BLACK)
        # q_counter.tracker = expl_line.intercept
        # VGroup(m_counter, q_counter).arrange(RIGHT, buff=1).next_to(ax)
        reg_line.save_state()

        # self.play(FadeIn(m_counter, q_counter))
        self.play(
            reg_line.slope.animate.set_value(1),
            reg_line.intercept.animate.set_value(0.25)
        )
        self.play(
            reg_line.slope.animate.set_value(0.33),
            reg_line.intercept.animate.set_value(0.75)
        )
        self.play(reg_line.animate.restore())

        # SLIDE 22:  ===========================================================
        # E COUNTER APPEARS
        self.next_slide(
            notes=
            '''which is the sum of squared residuals. [CLICK]
            '''
        )
        # E_counter = ECounter(expl_line, dataset, num_decimal_places=2).to_edge(UP)
        # self.play(ReplacementTransform(E_formula, E_counter))

        # SLIDE 23:  ===========================================================
        # LINE CHANGES AGAIN WITH m AND q, DISPLAYING THE CORRESPONDING E VALUE
        self.next_slide(
            notes=
            '''Out of the infinitely many lines generated by varying m and q,
            we will select the one that minimizes E, achieving the best fit to
            the data.
            '''
        )
        self.play(
            reg_line.slope.animate.set_value(1),
            reg_line.intercept.animate.set_value(0.25)
        )
        self.play(
            reg_line.slope.animate.set_value(0.33),
            reg_line.intercept.animate.set_value(0.75)
        )
        self.play(
            reg_line.slope.animate.set_value(linear_fit[1]),
            reg_line.intercept.animate.set_value(linear_fit[0])
        )

        # SLIDE 24:  ===========================================================
        # GRAPH FADES OUT
        # MINIMIZATION PROBLEM FORMULATION APPEARS
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
