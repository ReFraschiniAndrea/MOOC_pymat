from manim import *
from manim_slides import ThreeDSlide
import os
from W3Anim import DoubleArm, double_arm_kinematics, GD

env = os.environ
env["PATH"] = r"C:\Users\rfand\AppData\Local\Programs\MiKTeX\miktex\bin\x64;" + env["PATH"]

config.background_color=WHITE
config.renderer="cairo"
# test resolution
config.pixel_width = 960 
config.pixel_height = 720 
# release resolution
config.pixel_width = 1440 
config.pixel_height = 1080 

class WEEK3Anim_GradientDescent(ThreeDSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        #  VIDEO OF THE ROBOTIC ARM IS SHOWN
        self.next_slide(
            notes=
            '''Nowadays, seeing robotic arms performing work is the norm.
            But how are they controlled from a mathematical standpoint? 
            Let's figure it out together. [CLICK]
            '''
        )

        # SLIDE 02:  ===========================================================
        # ROBOT ARM AND AXIS ARE DRAWN
        self.next_slide(
            notes=
            '''As a simplified situation, let's consider a double-jointed
            mechanical arm moving in the x-y plane.
            In particular, we want to guide [CLICK] ...
            '''
        )
        ax_2d = Axes(
            x_range=[-4,4,1],
            y_range=[-4,4,1],
            x_length=9,
            y_length=9,
            x_axis_config={'stroke_color':BLACK, 'include_ticks':False},
            y_axis_config={'stroke_color':BLACK, 'include_ticks':False}
        )
        ax_2d.set_z_index(-1)
        ax_2d_labels = ax_2d.get_axis_labels(
            MathTex("x", color=BLACK).scale(0.75), MathTex("y", color=BLACK).scale(0.75)
        )
        robot_arm = DoubleArm(1.3, 1.8, PI*2/3, PI*4/5)
        robot_arm_VG = VGroup(robot_arm.origin, robot_arm.arm1, robot_arm.joint, robot_arm.arm2, robot_arm.hand)

        self.play(AnimationGroup(
            *[Create(s) for s in robot_arm_VG],
            lag_ratio=0.2))
        self.play(Create(ax_2d, run_time=1))
        self.play(Write(ax_2d_labels))

        # SLIDE 03:  ===========================================================
        # HAND AND TARGET ARE INDICATED
        # HAND MOVES TO TARGET
        self.next_slide(
            notes=
            '''... the end effector, the tip of the arm, to a reachable target
            point.[CLICK]
            '''
        )

        t1, t2 = -PI/4, -PI*11/12  # target point angles
        target = Star(color=ORANGE, fill_opacity=1).scale(0.1).move_to(
            double_arm_kinematics(robot_arm.l1, robot_arm.l2, t1, t2, 1, center=robot_arm.center)
            )

        self.play(Indicate(robot_arm.hand))
        self.wait(0.3)
        self.play(GrowFromCenter(target))
        self.wait(0.2)
        robot_arm.animate_to_angle(self, t1, t2, run_time = 3)

        # SLIDE 04:  ===========================================================
        # ARM MOVES BACK AND DISTANCE ARROW APPEARS
        self.next_slide(
            notes=
            '''This requires to minimize the distance between the tip and the 
            target point. From the mathematical point of view, this is a 
            minimization problem. It consists in finding the smallest value of a
            given function, typically called cost function, subjected to certain
            constraints.
            '''
        )
        distance_arrow = DoubleArrow(
            robot_arm.hand, target, 
            buff=0.1,
            stroke_color=BLACK, 
            stroke_width=1,
            max_tip_length_to_length_ratio=0.05)
        dist_label = MathTex('d', color=BLACK).scale(0.5).next_to(distance_arrow, RIGHT, buff=-0.5).shift(UP*0.5)
        
        robot_arm.animate_to_angle(self, PI/3, PI/8, run_time = 1.5)
        self.wait(0.5)
        self.play(FadeIn(distance_arrow), Write(dist_label))


        T1, T2 = PI*4/5, PI*11/12   # Arm angular positions for explanation

        # SLIDE 05:  ===========================================================
        # ARM IS MOVED AND AXIS ARE MODFIED TO THE EXPLANATION CONFIGURATION
        self.next_slide(
            notes=
            '''As we said, we consider the mechanical arm moving in the plane,
            with the foot fixed at the origin of the axis.
            '''
        )
        self.play(FadeOut(distance_arrow, dist_label, target))
        robot_arm.animate_to_angle(self, -PI*2/3, -PI/7) # intermediate position
        self.wait(0.3)
        robot_arm.animate_to_angle(self, T1, T2) # Position for explanation

        # SLIDE 06:  ===========================================================
        # ARM LABLES APPEAR
        self.next_slide(
            notes=
            '''Let's define its geometry: We have two joint segments, having
            fixed length L1 and L2.
            '''
        )
        L1_label = MathTex(r'L_1', color=BLACK).scale(0.5).next_to(robot_arm.arm1, LEFT, buff=-0.5).shift(DOWN*0.15)
        L2_label = MathTex(r'L_2', color=BLACK).scale(0.5).next_to(robot_arm.arm2, DOWN, buff=-0.1)

        self.play(Write(L1_label))
        self.play(Write(L2_label))

        # SLIDE 07:  ===========================================================
        # ANGLES AND ANGLE LABELS APPEAR
        self.next_slide(
            notes=
            '''theta_1 and theta_2 are the angles of the first and the second
            segment relative to the x-axis, respectively.
            '''
        )
        sub_false = Line(robot_arm.origin.get_center(), robot_arm.origin.get_center()+RIGHT)
        subline = DashedLine(robot_arm.joint.get_center(), robot_arm.joint.get_center()+RIGHT*0.3, stroke_color=BLACK, stroke_width=0.3)
        angle1 = Angle(sub_false, robot_arm.arm1, color=BLUE_B, radius=0.25)
        angle2 = Angle(subline, robot_arm.arm2, color=BLUE_B, radius=0.25)
        t1_label = MathTex(r'\theta_1', color=BLACK).scale(0.5).next_to(angle1,UP*0.25).shift(RIGHT*0.1)
        t2_label = MathTex(r'\theta_2', color=BLACK).scale(0.5).next_to(angle2,UP*0.25)

        self.play(Create(angle1), Write(t1_label))
        self.play(Create(angle2), Create(subline), Write(t2_label))

        # SLIDE 08:  ===========================================================
        # TARGET STAR AND ITS LABEL APPEAR
        self.next_slide(
            notes=
            '''Our target is the orange star, which has coordinates (x_p, y_p).
            '''
        )
        target_coord = MathTex(r'(x_p, y_p)', color=BLACK).scale(0.5).next_to(target, DOWN)

        self.play(GrowFromCenter(target))
        self.play(Write(target_coord))

        # SLIDE 09:  ===========================================================
        # ROBOT HAND IS INDICATED, AND ITS COORSINATES APPEAR
        self.next_slide(
            notes=
            '''The green point (x,y) is the position of the tip, which depends
            only on the two angles theta_1 and theta_2. 
            '''
        )

        hand_coord = MathTex(r'(x, y)', color=BLACK).scale(0.5).next_to(robot_arm.hand, UP)

        self.play(Indicate(robot_arm.hand), Write(hand_coord))

        # SLIDE 10:  ===========================================================
        # ARMS ARE PROJECTED ON THE AXES
        self.next_slide(
            notes=
            '''Using some trigonometry, we can project the arms on the x and y
            axes, obtaining these expressions.
            '''
        )

        l1_projlines = ax_2d.get_lines_to_point(robot_arm.joint.get_center(), color=BLACK)
        l2_projlines = ax_2d.get_lines_to_point(robot_arm.hand.get_center(), color=BLACK)

        b11 = Brace(robot_arm.arm1, direction=DOWN, color=BLACK, buff=0.2)
        b12 = Brace(robot_arm.arm1, direction=RIGHT, color=BLACK, buff=0.2)
        b21 = BraceBetweenPoints(
            (robot_arm.l1*np.cos(T1))*RIGHT,
            (robot_arm.l1*np.cos(T1) + robot_arm.l2*np.cos(T2))*RIGHT,
            direction=DOWN, color=BLACK, buff=0.2)
        b22 = BraceBetweenPoints(
            (robot_arm.l1*np.sin(T1))*UP,
            (robot_arm.l1*np.sin(T1) + robot_arm.l2*np.sin(T2))*UP,
            direction=RIGHT, color=BLACK, buff=0.2)

        b11text = MathTex(r"L_1 \cos(\theta_1)", color=BLACK).scale(0.5)
        b12text = MathTex(r"L_1 \sin(\theta_1)", color=BLACK).scale(0.5)
        b21text = MathTex(r"L_2 \cos(\theta_2)", color=BLACK).scale(0.5)
        b22text = MathTex(r"L_2 \sin(\theta_2)", color=BLACK).scale(0.5)

        b11.put_at_tip(b11text)
        b12.put_at_tip(b12text)
        b21.put_at_tip(b21text)
        b22.put_at_tip(b22text)

        self.play(Create(l1_projlines, lag_ratio=0))
        self.play(FadeIn(b11, b11text, b12, b12text))
        self.wait(1)
        self.play(Create(l2_projlines, lag_ratio=0))
        self.play(FadeIn(b21, b21text, b22, b22text))

        # SLIDE 11:  ===========================================================
        # COMBINE INTO FORMULA FOR X, Y AS A FUNCTION OF ANGLES
        self.next_slide(
            notes=
            ''' 
            '''
        )

        projection_objectsVG = VGroup(l1_projlines, l2_projlines, b11,b12, b21, b22, b11text, b12text, b21text, b22text)
        every_minus_proj = VGroup(ax_2d, ax_2d_labels, robot_arm.ARM1, robot_arm.ARM2,
                                  L1_label, L2_label, angle1, angle2, subline, t1_label, t2_label,
                                  target, target_coord, hand_coord)
        robot_kinematics_drawingVG = every_minus_proj + projection_objectsVG
        x_eq = MathTex(r'{{x =}} {{L_1 \cos(\theta_1)}} + {{L_2 \cos(\theta_2)}}', color=BLACK).next_to(robot_kinematics_drawingVG, DOWN*1.1)
        y_eq = MathTex(r'{{y =}} {{L_1 \sin(\theta_1)}} + {{L_2 \sin(\theta_2)}}', color=BLACK).next_to(x_eq, DOWN)

        self.play(robot_kinematics_drawingVG.animate.to_edge(UP))
        self.play(Write(x_eq[0]), Write(y_eq[0]))
        self.play(
            FadeOut(b11, b12, b21, b22, l1_projlines, l2_projlines),
            TransformMatchingTex(VGroup(b11text, b21text), x_eq),
            TransformMatchingTex(VGroup(b12text, b22text), y_eq)
        )

        # SLIDE 12:  ===========================================================
        # FINAL DISTANCE FORMULA APPEARS
        self.next_slide(
            notes=
            '''We express the distance between the tip (x,y) and the target
            point (x_p, y_p) in terms of the angles theta_1 and theta_2. Now we
            are ready to formulate the minimization problem. 
            '''
        )

        distance_arrow = DoubleArrow(
            robot_arm.hand, target, 
            buff=0.1,
            stroke_color=BLACK, 
            stroke_width=1,
            max_tip_length_to_length_ratio=0.05)
        dist_label = MathTex('d_p', color=BLACK).scale(0.5).next_to(distance_arrow, LEFT, buff=-0.5).shift(DOWN*0.5)
        xy_eqs=VGroup(x_eq, y_eq)
        d_b_eq = MathTex(r'{{d_p}}(\theta_1, \theta_2) = \sqrt{(x(\theta_1, \theta_2)-x_p)^2 + (y(\theta_1, \theta_2)-y_p)^2}',
                         color=BLACK).move_to(xy_eqs.get_center())

        self.play(Create(distance_arrow), Write(dist_label))
        self.play(FadeOut(x_eq, y_eq))
        self.play(FadeIn(d_b_eq))
        self.wait(0.5)
        self.play(Indicate(dist_label), Indicate(d_b_eq[0]))

        # SLIDE 13:  ===========================================================
        # MINMIMIZATION PROBLEM FORMULATION APPEARS
        self.next_slide(
            notes=
            '''Then, our problem is finding the joint angles theta1 and theta2 
            that minimize the distance d_p, positioning the tip closest to the
            desired target. An equivalent and simpler problem is the following:
            [CLICK]
            '''
        )

        first_minim_problem = MathTex(r'\min \ J(\theta_1, \theta_2), \text{with} \ J=d_p',
                                       color=BLACK).move_to(d_b_eq.get_center())

        self.play(ReplacementTransform(d_b_eq, first_minim_problem))

        # SLIDE 14:  ===========================================================
        # CHANGE TO SQUARED DISTANCE
        self.next_slide(
            notes=
            '''find the angles theta1 and theta2 that minimize the cost function
            J which is defined as the square of the distance. [CLICK]
            '''
        )
        minim_problem = MathTex(r'\min \ J(\theta_1, \theta_2), \text{with} \ J={{d_p^2}}',
                                 color=BLACK).move_to(d_b_eq.get_center())
       
        self.play(TransformMatchingTex(first_minim_problem, minim_problem))
        self.wait(1.5)
        self.play(Indicate(minim_problem[1]))

        # SLIDE 15:  ===========================================================
        # 3D PLOT OF THE OBJECTIVE FUNCTION FADES IN
        self.next_slide(
            notes=
            '''Before solving the problem, let's have a look at the function J.
            It is a surface within the space R3. [CLICK]
            '''
        )
        self.play(FadeOut(every_minus_proj, minim_problem, 
                          distance_arrow, dist_label,
                          hand_coord, target, target_coord))
        self.wait(0.2)
        # 3D reference system
        ax_3d = ThreeDAxes(
            x_range=[0, 2*PI, PI],
            y_range=[0, 2*PI, PI],
            z_range=[0, 35, 7],
            x_length=2*PI,
            y_length=2*PI,
            z_length=2*PI,
            x_axis_config={'stroke_color':BLACK},
            y_axis_config={'stroke_color':BLACK},
            z_axis_config={'stroke_color':BLACK}
        )
        plane = VGroup()
        for i in range(16):
            dx = i/16*(2*PI)
            plane.add(
                Line(ax_3d.c2p(dx, 0, 0), ax_3d.c2p(dx, 15/16*2*PI, 0), stroke_color=BLACK, stroke_width=0.2),
                Line(ax_3d.c2p(0, dx, 0), ax_3d.c2p(15/16*2*PI, dx, 0), stroke_color=BLACK, stroke_width=0.2),
            )
        ax_3d_labels = ax_3d.get_axis_labels(
            x_label = MathTex(r'\theta_1', color=BLACK).rotate(0).rotate(+PI/2, axis=RIGHT), 
            y_label = MathTex(r'\theta_2', color=BLACK).rotate(+PI/2, axis=RIGHT).rotate(-PI/2), 
            z_label = MathTex(r'J', color=BLACK).rotate(0).rotate(0, RIGHT)
            )
        ax_3d_labels[0].shift(DOWN)
        ax_3d_labels[1].shift(LEFT)
        ref_sys = VGroup(ax_3d, plane, ax_3d_labels).move_to(ORIGIN)

        # NOTE: the target point is now different to make the plot more clear
        T1_3d, T2_3d = PI/1.8, +PI/6
        target_point = double_arm_kinematics(robot_arm.l1, robot_arm.l2, T1_3d, T2_3d)
        def objective_func(t1, t2):
            return (
                np.square(robot_arm.l1*np.cos(t1) + robot_arm.l2*np.cos(t2)- target_point[0]) +
                np.square(robot_arm.l1*np.sin(t1) + robot_arm.l2*np.sin(t2)- target_point[1])
                )
        obj_surf = ax_3d.plot_surface(
            objective_func,
            u_range=[0, 2*PI], v_range=[0, 2*PI], 
            colorscale=[BLUE, TEAL,YELLOW],
            colorscale_axis=2,
            fill_opacity=0.9
            )
        
        self.set_camera_orientation(phi=75 * DEGREES, theta=-120 * DEGREES, zoom=0.75)
        # self.begin_ambient_camera_rotation(rate=0.005)
        self.play(FadeIn(obj_surf, ref_sys))  # order is important

        # SLIDE 16:  ===========================================================
        # AXIS LABELS ARE HIGHLIGHTED
        self.next_slide(
            notes=
            '''On the two horizontal axes, we have the angles theta1 and theta2
            in radians, while the value of J (the squared distance) is reported
            on the vertical one.
            '''
        )
        self.play(Indicate(ax_3d_labels[0]), Indicate(ax_3d_labels[1]))
        self.wait(1)
        self.play(Indicate(ax_3d_labels[2]))
        
        # SLIDE 17:  ===========================================================
        # ANIMATION 13: show the target point
        self.next_slide(
            notes=
            '''Our goal is to identify the angles theta1* and theta2* that are
            associated to the minimum point, the star! Those angles are the ones
            that bring the tip closest to the target point. The question is: how
            can we find them?
            '''
        )

        angular_target = Star(color=ORANGE).scale(0.2).move_to(ax_3d.c2p(*[T1_3d, T2_3d,0]))
        target_projline = ax_3d.get_lines_to_point(angular_target.get_center(), color=ORANGE, stroke_width=1.5)
        t1star_label = MathTex(r'\theta_1^*', color=BLACK).scale(0.75).move_to(ax_3d.c2p(T1_3d,-1,0)).rotate(0).rotate(+PI/2, axis=RIGHT)
        t2star_label = MathTex(r'\theta_2^*', color=BLACK).scale(0.75).move_to(ax_3d.c2p(-1,T2_3d,0)).rotate(+PI/2, axis=RIGHT)
        
        self.play(GrowFromPoint(angular_target))
        self.play(Create(target_projline))
        self.play(Write(t1star_label), Write(t2star_label))
        self.wait(0.5)
        self.play(Indicate(t1star_label), Indicate(t2star_label))

        # SLIDE 18:  ===========================================================
        # GRADIENT DESCENT TITLE IS WRITTEN
        self.next_slide(
            notes=
            '''We will use a popular numerical algorithm, the so-called the
            Gradient Descent method. Let us briefly explain how it works.
            '''
        )
        
        self.play(FadeOut(obj_surf, ref_sys, angular_target, target_projline, t1star_label, t2star_label))
        # self.stop_ambient_camera_rotation()
        self.wait(0.2)

        self.set_camera_orientation(phi=0 , theta=-PI/2, zoom=1)

        GD_title = Tex('Gradient Descent', color=BLACK).scale(1.5).to_edge(UP).shift(UP*0.5)

        self.play(Write(GD_title))
        self.add_fixed_in_frame_mobjects(GD_title)

        # SLIDE 19:  ===========================================================
        # GRADIENT DEFINITION IS WRITTEN
        self.next_slide(
            notes=
            '''The method is based on the gradient of a function. We recall that
            a gradient of a function J(theta1, theta2) is defined by this
            expression.
            '''
        )
        gradient_def = VGroup(
            MathTex(r"{{\nabla J(\theta_1, \theta_2)}}", "=", color=BLACK),
            Matrix([[r"\frac{\partial J}{\partial \theta_1}"], [ r"\frac{\partial J}{\partial \theta_2}"]],
                   v_buff= 1.5)).arrange_in_grid(1,2).center()
        gradient_def[1].set_color(BLACK)

        self.play(Write(gradient_def))

        # SLIDE 20:  ===========================================================
        # CHAIN RULE SUBSTITUTION IS APPLIED
        self.next_slide(
            notes=
            '''In our case, using the chain rule (2) we get this formula, 
            '''
        )
        gradient_with_chain_rule = VGroup(
            MathTex(r"{{\nabla J(\theta_1, \theta_2)}}", "=", color=BLACK),
            Matrix([
            [r"{{\frac{\partial J}{\partial x}}} {{\frac{\partial x}{\partial \theta_1}}} {{+}}" 
             r"{{\frac{\partial J}{\partial y}}} {{\frac{\partial y}{\partial \theta_1}}}"],
            [r"{{\frac{\partial J}{\partial x}}} {{\frac{\partial x}{\partial \theta_2}}} {{+}}" 
             r"{{\frac{\partial J}{\partial y}}} {{\frac{\partial y}{\partial \theta_2}}}"]
            ], v_buff= 1.5)).arrange_in_grid(1,2).center()
        gradient_with_chain_rule[1].set_color(BLACK)

        self.play(ReplacementTransform(gradient_def, gradient_with_chain_rule))
        
        # SLIDE 19:  ===========================================================
        # GRADIENT FORMULA SLIDES UP
        # CHAIN RULE COMPONENTS EVALUATIONS APPEAR
        self.next_slide(
            notes=
            '''where the factors are given by these formulas.'''
        )
        self.play(gradient_with_chain_rule.animate.next_to(GD_title, DOWN).shift(DOWN*0.3))
        ch_eq_1 = MathTex(r"\frac{\partial J}{\partial x}", "=", "2(x-x_p)", color=BLACK)
        ch_eq_2 = MathTex(r"\frac{\partial J}{\partial y}", "=", "2(y-y_p)", color=BLACK)

        ch_eq_3 = MathTex(r"\frac{\partial x}{\partial \theta_1}", "=", r"-L_1 \sin(\theta_1)", color=BLACK)
        ch_eq_4 = MathTex(r"\frac{\partial x}{\partial \theta_2}", "=", r"-L_2 \sin(\theta_2)", color=BLACK)
        ch_eq_5 = MathTex(r"\frac{\partial y}{\partial \theta_1}", "=", r"L_1 \cos(\theta_1)" , color=BLACK)
        ch_eq_6 = MathTex(r"\frac{\partial y}{\partial \theta_2}", "=", r"L_2 \cos(\theta_2)" , color=BLACK)
        
        ch_eqs = VGroup(ch_eq_1, ch_eq_2, ch_eq_3, ch_eq_4, ch_eq_5, ch_eq_6
                        ).arrange_in_grid(3,2).scale(0.8).next_to(gradient_with_chain_rule, DOWN).shift(DOWN*0.3)
        # fisrtVG = VGroup(ch_eq_1, ch_eq_2).arrange_in_grid(1,2).scale(0.8).next_to(gradient_with_chain_rule, DOWN*1.2)
        # secondVG = VGroup(ch_eq_3, ch_eq_4, ch_eq_5, ch_eq_6).scale(0.8).arrange_in_grid(2,2).next_to(fisrtVG, DOWN*1.2)        

        self.play(FadeIn(*ch_eqs[:2]))
        self.play(FadeIn(*ch_eqs[2:]))

        # final subsitution ?
        # ieq1 = MathTex(r"{{\frac{\partial J}{\partial x}}} {{\frac{\partial x}{\partial \theta_1}}}",
        #                 "=", r"-L sin(\theta_1)")

        # SLIDE 20:  ===========================================================
        # OBJECTIVE FUNCTION SURFACE FADES BACK IN
        # POINT TRAVELS DOWN THE SLOPE
        self.next_slide(
            notes=
            '''Intuitively, The Gradient Descent is an iterative algorithm that,
            starting from an initial guess, follows the opposite of the gradient 
            of J and builds a sequence of angles that gradually approaches 
            the minimum.
            '''
        )
        self.play(FadeOut(ch_eq_1,ch_eq_2, ch_eq_3,ch_eq_4,ch_eq_5,ch_eq_6,
                          gradient_with_chain_rule))
        self.wait(0.2)

        starting_gd_guess = [7/6*PI, 6/6*PI]
        gd_point = Dot3D(ax_3d.c2p(starting_gd_guess[0], starting_gd_guess[1], objective_func(starting_gd_guess[0], starting_gd_guess[1])),
                    color=GREEN, radius = 0.15)
        # gd_point.set_z_index(1)
        GD_trajectory = GD(
            starting_point=[7/6*PI, 6/6*PI],
            target = double_arm_kinematics(robot_arm.l1,robot_arm.l2,T1_3d,T2_3d)[:2],
            l1=robot_arm.l1, l2=robot_arm.l2,
            tol=1e-3
            )
        curve = VGroup().set_points_as_corners(ax_3d.c2p(GD_trajectory))
        trace = TracedPath(gd_point.get_center, stroke_color=GREEN, stroke_width=3)
        self.add(trace)

        # start animation
        self.set_camera_orientation(phi=75 * DEGREES, theta=-120 * DEGREES, zoom=0.75)
        
        self.begin_ambient_camera_rotation(rate=0.005)
        self.play(FadeIn(obj_surf, ref_sys, angular_target, target_projline, t1star_label, t2star_label))
        self.wait(1)
        self.play(FadeIn(gd_point))
        self.wait(0.8)

        self.play(MoveAlongPath(gd_point, curve), run_time=5)#, rate_func=linear
        self.wait(1)
        self.play(FadeOut(obj_surf, ref_sys, 
                          angular_target, target_projline, t1star_label, t2star_label,
                          GD_title,
                          gd_point, trace))
        self.stop_ambient_camera_rotation()

        # SLIDE 01:  ===========================================================
        # ANIMATION 20: show title
        self.next_slide(
            notes=
            '''Here is how the algorithm looks like.
            '''
        )
        self.set_camera_orientation(phi=0 , theta=-PI/2, zoom=1)
        pc = Tex(r"{{\textbf{Algorithm:} Gradient Descent Method \newline}}"
                 r"{{\textbf{Require:} $ (x_p, y_p), L_1, L_2,(\theta_1^0, \theta_2^0),tol, \alpha, N_{iter} \geq 1$ \newline}}"
                 r"{{1: $i = 1$ \newline}}"
                 r"{{2: \textbf{while} $i \leq N_{iter} $ \textbf{do}: \newline}}"
                 r"{{3: \quad $(\theta_1^i, \theta_2^i) = (\theta_1^{i-1}, \theta_2^{i-1}) - \alpha \nabla J(\theta_1^{i-1}, \theta_2^{i-1})$ \newline}}"
                 r"{{4: \quad \textbf{if} $ J (\theta_1^i, \theta_2^i) < tol $} \textbf{then} \textbf{stop} \newline}}"
                 r"{{5: \quad \textbf{end if} \newline}}"
                 r"{{6: \quad $i = i +1 $ \newline}}"
                 r"{{7: \textbf{end while}}}",
                 color=BLACK)
        for i in range(len(pc)):
            pc[i].align_on_border(LEFT)
        pc.scale(1).center()

        self.play(Write(pc[0]))

        # SLIDE 01:  ===========================================================
        # ANIMATION 21: show algorithm requirements
        self.next_slide(
            notes=
            '''Starting from an initial guess (theta_1^0, theta_1_2^0),
            '''
        )
        self.play(Write(pc[1]))

        # SLIDE 01:  ===========================================================
        # ANIMATION 22: show while loop
        self.next_slide(
            notes=
            ''' for an iteration index "i" that goes from 1 to  N_iter,
            '''
        )
        self.play(Write(pc[2]), Write(pc[3]))

        # SLIDE 01:  ===========================================================
        # ANIMATION 23: GD update equation
        self.next_slide(
            notes=
            ''' We build a sequence of angles with this formula,
            '''
        )
        GD_update_eq = MathTex(
            r"\boldsymbol{\theta}^n = \boldsymbol{\theta}^{n-1} -{{\alpha}} \nabla J (\boldsymbol{\theta}^{n-1})",
            color=BLACK).shift(DOWN)
        self.play(Write(GD_update_eq))
        self.play(Circumscribe(GD_update_eq))
        
        # SLIDE 01:  ===========================================================
        # ANIMATION 24: highlight learning rate for explanation
        self.next_slide(
            notes=
            '''where alpha is a positive number called learning rate or step
            length, and it determines the size of the step in the direction
            opposite to the gradient.
            '''
        )
        self.play(Indicate(GD_update_eq[1]))

        # SLIDE 01:  ===========================================================
        # ANIMATION 25: transform into pseudo code
        self.next_slide(
            notes=
            '''
            '''
        )
        self.play(ReplacementTransform(GD_update_eq, pc[4]))

        # SLIDE 01:  ===========================================================
        # ANIMATION 26: stop if tolerance low enough
        self.next_slide(
            notes=
            '''We stop the computation when J is small enough, say it is less
            than a certain tolerance tol that we fixed in advance. This means
            that the distance between the effector and the target point is small.
            '''
        )
        self.play(Write(pc[5]), Write(pc[6]))
        self.play(Write(pc[7]), Write(pc[8]))

        # SLIDE 01:  ===========================================================
        # ANIMATION 27: stop if number of iterations exceeds limit
        self.next_slide(
            notes=
            '''Note that the iterations stop anyway if the maximum number of
            iterations N_iter is exceeded. Now, we need a computer code to
            perform these operations automatically. In the next videos we will
            learn how to write the algorithm. 
            '''
        )
        self.play(Circumscribe(pc[3]))
        self.wait(1)

        '''END'''