import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from manim_slides import Slide
from W2Anim import *
from Generic_mooc_utils import HighlightRectangle

config.renderer='cairo'
config.background_color = WHITE
config.pixel_width=1440
config.pixel_height=1080
# config.pixel_width=960
# config.pixel_height=720

LABELS_SIZE=0.75
ICONS_HEIGHT = 0.6

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
        tri = Triangle().scale(3).center()
        circles = [Circle(stroke_color=BLACK, fill_color=WHITE, radius= ICONS_HEIGHT*2 + 0.5, stroke_width=6, fill_opacity=0).move_to(tri.get_vertices()[i]) for i in range(3)]
        
        wildfire_icon = SVGMobject(r'Assets\W2\forest_fire_icon.svg').scale_to_fit_height(ICONS_HEIGHT*4).move_to(tri.get_vertices()[0])
        high_temp_icon = SVGMobject(r'Assets\W2\high_temperature_icon.svg').set_color(RED).scale_to_fit_height(ICONS_HEIGHT*4).move_to(tri.get_vertices()[1])
        humidty_icon = SVGMobject(r'Assets\W2\humidity_icon.svg').set_color(BLUE).scale_to_fit_height(ICONS_HEIGHT*4).move_to(tri.get_vertices()[2])
       
        causal_arrows = VGroup(
            Arrow(circles[1].get_top(), wildfire_icon.get_critical_point(DL), color=BLACK),
            Arrow(circles[2].get_top(), wildfire_icon.get_critical_point(DR), color=BLACK),
        )

        self.play(
            Succession(
                AnimationGroup(Create(circles[1]), FadeIn(high_temp_icon)),
                AnimationGroup(Create(circles[2]), FadeIn(humidty_icon))
            )
        )
        self.wait(1)
        self.play(Create(circles[0]), FadeIn(wildfire_icon))
        self.play(FadeIn(causal_arrows))

        # SLIDE 02:  ===========================================================
        # HUMIDITY AND FIRE RISK ICONS
        # GRAPH WITH NEGATIVE CORRELATION APPEARS
        self.next_slide(
            notes=
            '''For example, we might expect that, the higher the humidity in the
            air, the lower the fire risk on that day. [CLICK]
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
        variable_ax_lab = ax.get_axis_labels(
            Text('Humidity' , color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2),
            Text('Fire risk', color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2)
        )

        forest_icon = SVGMobject(r'Assets\W2\pine_trees_icon.svg').scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(-0.1, 0.3, 0))
        dry_icon = humidty_icon.copy().set_color(RED).scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(0.3, -0.1, 0))
        self.play(
            FadeOut(*circles, causal_arrows, high_temp_icon),
            wildfire_icon.animate.scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(-0.1, 1, 0)),
            humidty_icon.animate.scale_to_fit_height(ICONS_HEIGHT).move_to(ax.c2p(X_RANGE[1]-0.1, -0.1, 0)),
            )
        self.play(
            Create(ax),
            FadeIn(forest_icon, dry_icon),
            FadeIn(variable_ax_lab)
        )

        t = ValueTracker(X_RANGE[0]+0.1)
        negative_corr_func = lambda t: 0.2*1/(t + 0.1) + 0.1
        tracing_dot = Dot(color=BLUE).add_updater(
            lambda m: m.move_to(ax.c2p(t.get_value(), negative_corr_func(t.get_value()), 0))
        )
        tracing_dot.update()
        trace = TracedPath(tracing_dot.get_center, stroke_color=BLUE, stroke_width=4)

        self.play(GrowFromCenter(tracing_dot), run_time=0.5)
        self.add(trace)
        self.wait(0.3)
        self.play(t.animate(run_time=1.5).set_value(X_RANGE[1]))
        trace.clear_updaters()
        self.wait(0.5)

        # SLIDE 03:  ===========================================================
        # TEMPERATURE ICON REPLACES HUMIDITY ONE
        # GRAPH WITH POSITIVE CORRELATION APPEARS
        self.next_slide(
            notes=
            '''Similarly, we expect that high temperature increases the risk of
            fire. [CLICK]
            '''
        )
        low_temp_icon = SVGMobject(r'Assets\W2\low_temperature_icon.svg').set_color(BLUE).scale_to_fit_height(ICONS_HEIGHT).move_to(dry_icon)
        high_temp_icon.scale_to_fit_height(ICONS_HEIGHT).move_to(humidty_icon)
        self.play(
            FadeOut(tracing_dot, trace),
            ReplacementTransform(dry_icon, low_temp_icon),
            ReplacementTransform(humidty_icon, high_temp_icon),
            Transform(variable_ax_lab[0], Text('Temperature', color=BLACK, font='Microsoft JhengHei', weight=LIGHT).scale(LABELS_SIZE/2).move_to(variable_ax_lab[0]))
        )

        positive_corr_func = lambda t: 0.4*t**2 +0.2
        tracing_dot.clear_updaters().add_updater(
            lambda m: m.move_to(ax.c2p(t.get_value(), positive_corr_func(t.get_value()), 0))
        )
        t.set_value(X_RANGE[0]+0.1)
        tracing_dot.update()
        trace2 = TracedPath(tracing_dot.get_center, stroke_color=BLUE, stroke_width=4)

        self.play(GrowFromCenter(tracing_dot), run_time=0.5)
        self.add(trace2)
        self.wait(0.3)
        self.play(t.animate(run_time=1.5).set_value(X_RANGE[1]))
        trace2.clear_updaters()
        self.wait(0.5)

        # SLIDE 04:  ===========================================================
        # DATA POINTS APPEAR
        self.next_slide(
            notes=
            '''To answer this question quantitatively, we use real-world data
            that captures key environmental factors, such as daily temperature,
            humidity, and wildfire occurrence. [CLICK] 
            '''
        )
        dataset = generate_regression_dataset(func= lambda x: 1.5*(0.4*x-0.75)**3 + 0.8, x_range=(0.1, 1.5), n=20, sigma=0.15, seed=0)
        dataset_points = points_from_data(dataset, ax=ax, color=PURPLE_A).set_z_index(1)
        
        self.play(FadeOut(tracing_dot, trace2))
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
            fire risk based on temperature and humidity. [CLICK]
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
        log_plot = ax.plot(lambda t: log_fit[0]+ log_fit[1]*np.log(t), x_range=(np.exp(-log_fit[0]/log_fit[1]), X_RANGE[1], 0.01), color=BLUE, use_smoothing=False)

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
            [CLICK]
            '''
        )
        reg_line = RegressionLine(linear_fit[1], linear_fit[0], ax, x_range=X_RANGE)
        reg_line.suspend_updating()
        LR_title = Text('Linear Regression', font_size=64, color=BLACK, font='Microsoft JhengHei', weight=LIGHT).to_edge(UP).shift(UP*0.5)

        self.play(FadeOut(wildfire_icon, forest_icon, low_temp_icon, high_temp_icon, variable_ax_lab))
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
            dependent variable y. [CLICK]
            '''
        )
        generic_relation = MathTex(r'y = {{f(x)}}', color=BLACK,
                                   tex_to_color_map={'x':BLUE, 'y':ORANGE}).move_to(LR_title)
        x_lab = MathTex('x', color=BLUE).scale(LABELS_SIZE)
        y_lab = MathTex('y', color=ORANGE).scale(LABELS_SIZE)
        ax.get_axis_labels(x_lab, y_lab)

        self.play(FadeOut(LR_title, dataset_points, reg_line))
        self.play(FadeIn(generic_relation))
        self.wait(0.5)
        self.play(
            AnimationGroup(
                ReplacementTransform(generic_relation[3].copy(), x_lab),
                ReplacementTransform(generic_relation[0].copy(), y_lab),
                lag_ratio=0.5
            )
        )

        # SLIDE 07:  ===========================================================
        # LINEAR RELATION EQUATION APPEARS
        # LINE IS DRAWN IN THE AXIS
        self.next_slide(
            notes=
            '''This technique assumes that the relationship between the
            dependent and independent variables is described by the equation y
            equals m times x plus q.
            The coefficients m and q have clear and intuitive meanings. [CLICK]
            '''
        )
        linear_relation = MathTex(r'y = {{m x}} + {{q}}', color=BLACK,
                                  tex_to_color_map={'x':BLUE, 'y':ORANGE}
                                  )
        linear_relation.shift(generic_relation[0].get_center()-linear_relation[0].get_center())

        self.play(
            ReplacementTransform(generic_relation[:2], linear_relation[:2]),
            ReplacementTransform(generic_relation[2:], linear_relation[2:]),
            )
        self.play(Create(reg_line))

        # SLIDE 08:  ===========================================================
        # 'q' HIGHLIGHTED IN THE EQUATION AND ON THE GRAPH
        self.next_slide(
            notes=
            '''Specifically, q represents the y-intercept, that is the predicted
            value when x equals 0. [CLICK]
            '''
        )
        intercept_dot = Dot(ax.c2p(0, reg_line.intercept.get_value()), color=PURPLE_C)
        intercept_label = MathTex('q', color=BLACK).scale(LABELS_SIZE).next_to(intercept_dot, LEFT)
        q_higlights = [
            HighlightRectangle(linear_relation[-1]),
            HighlightRectangle(intercept_label)
        ]

        self.play(
            GrowFromCenter(intercept_dot),
            Write(intercept_label)
        )
        self.play(Create(h) for h in q_higlights)

        # SLIDE 09:  ===========================================================
        # 'm' HIGHLIGHTED IN THE EQUATION
        # RISE OVER RUN IS DRAWN TO EXPLAIN 'm'
        self.next_slide(
            notes=
            '''m, instead, represents the "slope" of the line, that is the rate
            of change of y with respect to x. In other words, [CLICK] ...
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
        ).set_z_index(2)
        slope_label = MathTex('m', color=BLACK).scale(LABELS_SIZE).next_to(rise_over_run, UP).set_z_index(2)
        m_highlights= [
            HighlightRectangle(linear_relation[2]),
            HighlightRectangle(slope_label)
        ]
        
        self.play(FadeOut(*q_higlights))
        self.play(
            Create(rise_over_run),
            Write(slope_label)
        )
        self.play(Create(h) for h in m_highlights)

        # SLIDE 10:  ===========================================================
        # RISE OVER RUN TRIANGLE APPEARS
        # DELTA_X LABEL APPEARS
        self.next_slide(
            notes=
            '''... if the independent variable x increases by a certain amount
            delta_x, [CLICK] ...
            '''
        )
        ror_brace_x = Brace(rise_over_run, DOWN, color=BLACK, sharpness=1)
        ror_x_lab = MathTex(r'\Delta x', color=BLACK).scale(LABELS_SIZE)
        ror_brace_x.put_at_tip(ror_x_lab)
        ror_brace_y = Brace(rise_over_run, RIGHT, color=BLACK, sharpness=1)
        ror_y_lab = MathTex(r'm \Delta x', color=BLACK).scale(LABELS_SIZE)
        ror_brace_y.put_at_tip(ror_y_lab)
        VGroup(ror_brace_x, ror_brace_y, ror_x_lab, ror_y_lab).set_z_index(2)

        self.play(FadeOut(*m_highlights))
        self.play(FadeIn(ror_brace_x, ror_x_lab))

        # SLIDE 11:  ===========================================================
        # DELTA_Y LABEL APPEARS
        self.next_slide(
            notes=
            '''... the predicted dependent variable y will increase by
            m*delta_x. [CLICK]
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
                          slope_label, intercept_label, intercept_dot))

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
                *[GrowFromCenter(dataset_points[i]) for i in range(len(dataset_points)) if i not in (5, 15)],
                run_time=2, lag_ratio=0.5
            )
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
        RNG = np.random.default_rng(seed=1)
        pertubation = RNG.normal(0 , scale=linear_fit[1]/3, size = 5)
        perturbed_mg = [(linear_fit[1] + pertubation[i],
                         linear_fit[0] + RNG.uniform(0, linear_fit[0]/2)*(-np.sign(pertubation[i])))
                        for i in range(5)]
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
        self.play(FadeOut(lines[:-1]))

        # SLIDE 16:  ===========================================================
        # LABELS FOR P1, P2, Pi APPEAR IN SUCCESSION
        # ONLY LABEL OF Pi REMAINS
        self.next_slide(
            notes=
            '''Let (x_1, y_1) be the first data point, (x_2, y_2) the second,
            and so on for all n points. [CLICK]
            '''
        )
        p = {
            '1' : dataset_points[0],
            '2' : dataset_points[3],
            'i' : dataset_points[9]
        }
        lines_to_points = [ax.get_lines_to_point(point.get_center(), color=point.get_color()) for point in p.values()]
        point_labels = [
            VGroup(
                MathTex(f'x_{index}', color=BLUE).scale(LABELS_SIZE*0.9).next_to(lines_to_points[i][1], DOWN),
                MathTex(f'y_{index}', color=ORANGE).scale(LABELS_SIZE*0.9).next_to(lines_to_points[i][0], LEFT),
            )
            for index, i in zip(p.keys(), range(3))
        ]
        
        self.play(
            Succession(
                AnimationGroup(
                    Create(lines_to_points[i]),
                    FadeIn(point_labels[i])
                )
                for i in range(3)
            )
        )

        # SLIDE 17:  ===========================================================
        # PREDICTION FORMULA APPEARS
        # POINT REPRESENTING PREDICTION APPEARS
        self.next_slide(
            notes=
            '''For any given point (x_i, y_i), we define y^_i as the predicted
            value from the model, which is m * x_i + q. [CLICK]
            '''
        )
        self.play(FadeOut(lines_to_points[0], lines_to_points[1], point_labels[0], point_labels[1]))
        reg_line.add_data_point(p['i'])
        predicted_point = reg_line.proj_points[0]
        predicted_point_line = reg_line.proj_lines[0]

        prediction_eq = MathTex(r'{{\hat{y_i}}} =m {{x_i}} + q', color=BLACK,
                                tex_to_color_map={'x_i':BLUE}).move_to(LR_title)
        prediction_eq[0].set_color(ORANGE)

        prediction_horiz_line = ax.get_horizontal_line(reg_line.proj_points[0].get_center(), color=p['i'].get_color())
        prediction_label = MathTex(r'\hat{y_i}', color=ORANGE).scale(LABELS_SIZE).next_to(prediction_horiz_line, LEFT)

        self.play(ReplacementTransform(linear_relation, prediction_eq))
        self.play(
            GrowFromCenter(predicted_point),
            Create(predicted_point_line),
            Create(prediction_horiz_line),
            ReplacementTransform(prediction_eq[0].copy(), prediction_label)
        )

        # SLIDE 18:  ===========================================================
        # RESIDUAL BRACE WITH LABEL APPEARS
        self.next_slide(
            notes=
            '''The difference y_i - y^_i is called the residual, and it measures
            the error between the actual and predicted values. [CLICK]
            '''
        )
        residual_eq = MathTex(r'{{r_i}} = {{y_i}} - {{\hat{y_i}}}', color=BLACK).next_to(prediction_eq, DOWN)
        residual_eq[2].set_color(ORANGE)
        residual_eq[4].set_color(ORANGE)
        residual_brace = BraceBetweenPoints(predicted_point.get_center(), p['i'].get_center(), LEFT, color=BLACK, sharpness=1).set_z_index(2)
        residual_label = MathTex(r'r_i', color=BLACK).scale(LABELS_SIZE).set_z_index(2)
        residual_brace.put_at_tip(residual_label)
        
        self.play(FadeIn(residual_eq))
        self.play(
            FadeIn(residual_brace),
            ReplacementTransform(residual_eq[0].copy(), residual_label)
        )

        # SLIDE 19:  ===========================================================
        # LINES BETWEEN ALL DATA POINTS AND THE LINE APPEAR
        self.next_slide(
            notes=
            '''Our objective is to find the line that minimizes these residuals,
            making the differences between the observed data and the model's
            predictions as small as possible. [CLICK]
            '''
        )
        reg_line.add_dataset(dataset_points)
        self.play(
            AnimationGroup(
                *[Succession(
                    Create(l), GrowFromCenter(p),
                    lag_ratio=0.5
                )
                for l, p in zip(reg_line.proj_lines[1:], reg_line.proj_points[1:])],
                lag_ratio=0.2,
                run_time=2
            )
        )

        # SLIDE 20:  ===========================================================
        # FORMULA FOR SQUARED ERROR APPEARS
        # 'E' IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''More precisely, we look for the line that minimizes the sum of
            the squares of the residuals, that we denote by E.
            E is a number that quantifies the total error between observed and
            predicted data. [CLICK]
            '''
        )
        E_formula = MathTex(r'E = \sum_{i=1}^n r_i^2 = r_1^2 + r_2^2 + \dots + r_n^2', color=BLACK).move_to(prediction_eq)
        self.play(FadeOut(prediction_eq, residual_eq))
        self.play(FadeIn(E_formula[0][:11]))
        self.play(ReplacementTransform(reg_line.proj_lines.copy(), E_formula[0][11:], lag_ratio=0.1, run_time=3))

        # SLIDE 21:  ===========================================================
        # COUNTERS FOR 'm' AND 'q' APPEAR
        # 'm' AND 'q' VARY, CHANGING THE LINE
        self.next_slide(
            notes=
            '''By varying m and q, we obtain different lines. For each
            combination of m and q, we can compute the corresponding error E,
            [CLICK] ...
            '''
        )
        m_counter = Variable(reg_line.slope.get_value(), label='m', num_decimal_places=2).set_color(BLACK)
        m_counter.tracker = reg_line.slope
        q_counter = Variable(reg_line.intercept.get_value(), label='q', num_decimal_places=2).set_color(BLACK)
        q_counter.tracker = reg_line.intercept
        VGroup(m_counter, q_counter).arrange(RIGHT, buff=1).next_to(ax, DOWN)
        reg_line.save_state()

        self.play(FadeOut(residual_brace, residual_label, lines_to_points[-1], point_labels[-1], prediction_label, prediction_horiz_line))
        self.play(FadeIn(m_counter, q_counter))
        self.play(reg_line.slope.animate(rate_func=there_and_back, run_time=2).set_value(0.7))
        self.play(reg_line.intercept.animate(rate_func=there_and_back, run_time=2).set_value(0.05))

        # SLIDE 22:  ===========================================================
        # E COUNTER APPEARS
        self.next_slide(
            notes=
            '''which is the sum of squared residuals. [CLICK]
            '''
        )
        E_counter = ECounter(reg_line, dataset, num_decimal_places=2).move_to(E_formula).set_color(BLACK)
        self.play(
            AnimationGroup(
                ReplacementTransform(E_formula[0][:2], E_counter.label),
                Succession(
                    FadeOut(E_formula[0][2:]),
                    FadeIn(E_counter.value),
                    run_time = 1
                )
            )
        )

        # SLIDE 23:  ===========================================================
        # LINE CHANGES AGAIN WITH m AND q, DISPLAYING THE CORRESPONDING E VALUE
        self.next_slide(
            notes=
            '''Out of the infinitely many lines generated by varying m and q,
            we will select the one that minimizes E, achieving the best fit to
            the data. [CLICK]
            '''
        )
        self.play(
            reg_line.slope.animate.set_value(linear_fit[1]*1.4),
            reg_line.intercept.animate.set_value(linear_fit[0]*0.6)
        )
        self.wait(0.5)
        self.play(
            reg_line.slope.animate.set_value(linear_fit[1]*1.2),
            reg_line.intercept.animate.set_value((linear_fit[0]*0.8))
        )
        self.wait(0.5)
        self.play(
            reg_line.slope.animate.set_value(linear_fit[1]*0.6),
            reg_line.intercept.animate.set_value((linear_fit[0]*1.4))
        )
        self.wait(0.5)
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
            "best-fitting line" that we have used before. [CLICK]
            '''
        )
        minim_problem = MathTex(r'\min_{m,q} \sum_{i=1}^n (r_i)^2', color=BLACK).move_to(E_counter)
        self.play(FadeOut(m_counter, q_counter, E_counter))
        self.play(FadeIn(minim_problem))

        # SLIDE 25:  ===========================================================
        # LEAST SQUARES TITLE APPEARS
        # 'SQUARED' HIGHLIGHTED IN FORMULA
        self.next_slide(
            notes=
            '''This approach is known as "least-squares" regression, as it
            focuses on minimizing the "squares" of the residuals. [CLICK]
            '''
        )
        ls_title = Text('Least Squares Regression', font_size=64, color=BLACK, font='Microsoft JhengHei', weight=LIGHT).to_edge(UP).shift(UP*0.5)
        self.play(FadeOut(ax, x_lab, y_lab, reg_line, reg_line.proj_lines, reg_line.proj_points, dataset_points))
        self.play(
            minim_problem.animate.center(),
            Write(ls_title)
        )

        # SLIDE 26:  ===========================================================
        # PARTIAL DERIVATIVES OF E SET TO ZERO APPEAR
        self.next_slide(
            notes=
            '''To find the values of m and q, we must ensure that the derivative
            of E with respect to both unknown coefficients is equal to zero,
            that is to say no further improvement is possible.
            By performing the necessary calculations, we find that the
            coefficients satisfying these conditions are given [CLICK] ...
            '''
        )

        necessary_condition = MathTex(r'\frac{\partial E}{\partial m}=0, \ \frac{\partial E}{\partial q}=0', color=BLACK)
        imply = MathTex(r'\Rightarrow', color=BLACK).rotate(-PI/2)
        self.play(minim_problem.animate.shift(UP))
        imply.next_to(minim_problem, DOWN, buff=0.2)
        necessary_condition.next_to(imply, DOWN, buff=0.2)
        self.play(FadeIn(necessary_condition, imply))

        # SLIDE 27:  ===========================================================
        # IMPLICATION: FORMULA FOR 'm', 'q'
        self.next_slide(
            notes=
            ''' by the following expressions. [CLICK] 
            '''
        )
        mq_eqs = LinearRegressionEquations()
        self.play(VGroup(minim_problem, necessary_condition, imply).animate.shift(UP*0.5))
        imply2 = imply.copy().next_to(necessary_condition, DOWN, buff=0.2)
        mq_eqs.next_to(imply2, DOWN, buff=0.2)
        self.play(FadeIn(mq_eqs, imply2))

        # SLIDE 28:  ===========================================================
        # FINAL FORMULAS GO TO CENTER
        self.next_slide(
            notes=
            '''We need a computer code to perform these operations automatically.
            In the upcoming videos, we will explore how to implement these
            formulas in Python and MATLAB. [END]
            '''
        )
        self.play(
            AnimationGroup(
                FadeOut(minim_problem, necessary_condition, imply, imply2),
                mq_eqs.animate.center(),
                lag_ratio=0.5
            )
        )
