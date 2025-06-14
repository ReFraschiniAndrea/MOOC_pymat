import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from config import *
from W3Anim import *
from Generic_mooc_utils import *
import skimage

config.update(RELEASE_CONFIG)

LABELS_SIZE = 0.75
LBELS_SIZE_3D = 1.5

class W3Theory_slides(ThreeDMOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # VIDEO OF THE ROBOTIC ARM IS SHOWN
        self.next_slide(
            notes=
            '''Nowadays, seeing robotic arms performing work is the norm.
            How can we control their position? We can write the problem using
            mathematics. Let's figure it out together. [CLICK]
            '''
        )
        placeholder = Text('Placeholder robot_arm.mp4', color=BLACK)
        self.add(placeholder)
        self.wait(.1)
        self.remove(placeholder)

        # SLIDE 02:  ===========================================================
        # ROBOT ARM AND AXIS ARE DRAWN
        self.next_slide(
            notes=
            '''As a simplified situation, let's consider a double-jointed
            mechanical arm moving in the x-y plane.
            In particular, we want to position [CLICK] ...
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
            MathTex("x", color=BLACK).scale(LABELS_SIZE),
            MathTex("y", color=BLACK).scale(LABELS_SIZE)
        )
        robot_arm = NewDB(ax_2d, 1, 1.5, PI*2/3, PI*4/5)

        self.play(AnimationGroup(
            *[Create(s) for s in robot_arm.submobjects[1:]],
            lag_ratio=0.2))
        self.play(Create(ax_2d, run_time=1))
        self.play(Write(ax_2d_labels, run_time=1))
        self.play(robot_arm.MoveByAngles(2*PI, -2*PI), run_time=3)
        # we have increased by 2PI and decrease it manually to avoid problems later
        robot_arm.theta1.set_value(PI*2/3)  
        robot_arm.theta2.set_value(PI*4/5)

        # SLIDE 03:  ===========================================================
        # HAND AND TARGET ARE INDICATED
        # HAND MOVES TO TARGET
        self.next_slide(
            notes=
            '''... the end effector, the tip of the arm, on a target point.
            In other words, we want to minimize the distance [CLICK] ...
            '''
        )

        t1, t2 = -PI/4, -PI*11/12  # target point angles
        target = Star(color=ORANGE, fill_opacity=1).scale(0.1).move_to(robot_arm._hand_coord(t1, t2))

        self.play(Indicate(robot_arm.hand, scale_factor=1.5, run_time=1.5))
        self.wait(0.3)
        self.play(GrowFromCenter(target))
        self.wait(0.2)
        self.play(robot_arm.MoveToAngles(t1, t2, run_time = 2))

        # SLIDE 04:  ===========================================================
        # ARM MOVES BACK AND DISTANCE ARROW APPEARS
        self.next_slide(
            notes=
            '''... between the tip and the target point. From the mathematical
            point of view, this is a minimization problem. It consists in
            finding the smallest value of a given function, typically called cost
            function. [CLICK]
            '''
        )
        self.play(robot_arm.MoveToAngles(PI/3, PI/8, run_time = 1.5))

        distance_arrow = DoubleArrow(
            robot_arm.hand, target, 
            buff=0.1,
            stroke_color=RED, 
            stroke_width=2,
            max_tip_length_to_length_ratio=0.05)
        self.wait(0.5)
        self.play(Create(distance_arrow))#, Write(dist_label))

        # SLIDE 05:  ===========================================================
        # ARM IS MOVED AND AXES ARE MODIFIED TO THE EXPLANATION CONFIGURATION
        self.next_slide(
            notes=
            '''As we said, we consider the mechanical arm moving in the plane,
            with the abse fixed at the origin of the axis. [CLICK]
            '''
        )
        self.play(FadeOut(distance_arrow))#, dist_label))
        self.play(robot_arm.MoveToAngles(-PI*2/3, -PI/7)) # intermediate position
        self.wait(0.3)
        T1, T2 = PI*4/5, PI*11/12   # Arm angular positions for explanation
        self.play(robot_arm.MoveToAngles(T1, T2)) # Position for explanation

        ax_ex =  Axes(
                x_range=[-4,1,1],
                y_range=[-1.7,2.7,1],
                x_length=9,
                y_length=9*(5-0.6)/5,
                x_axis_config={'stroke_color':BLACK, 'include_ticks':False},
                y_axis_config={'stroke_color':BLACK, 'include_ticks':False}
            ).center()
        robot_arm_ex = NewDB(ax_ex, 1, 1.5, T1, T2)
        ax_ex_labels = ax_ex.get_axis_labels(ax_2d_labels[0].copy(), ax_2d_labels[1].copy())
        robot_arm.suspend_updating()
        self.play(
            Transform(ax_2d, ax_ex), # replacement transform breaks c2p in the arm updaters
            ReplacementTransform(robot_arm, robot_arm_ex),
            ax_2d_labels[0].animate.move_to(ax_ex_labels[0].shift(RIGHT*0.3)),
            ax_2d_labels[1].animate.move_to(ax_ex_labels[1]),
            target.animate.move_to(robot_arm_ex._hand_coord(t1, t2))
        )
        robot_arm = robot_arm_ex # change back name for convenience
        self.remove(ax_2d)
        self.add(ax_ex)

        # SLIDE 06:  ===========================================================
        # ARM LABELS APPEAR
        self.next_slide(
            notes=
            '''Let's define its geometry: We have two joint segments, having
            fixed length L1 and L2. [CLICK]
            '''
        )
        L1_label = MathTex(r'L_1', color=BLACK).scale(LABELS_SIZE).next_to(robot_arm.arm1, LEFT, buff=-0.5).shift(DOWN*0.15)
        L2_label = MathTex(r'L_2', color=BLACK).scale(LABELS_SIZE).next_to(robot_arm.arm2, DOWN, buff=-0.1)

        self.play(Write(L1_label))
        self.play(Write(L2_label))

        # SLIDE 07:  ===========================================================
        # ANGLES AND ANGLE LABELS APPEAR
        self.next_slide(
            notes=
            '''theta_1 and theta_2 are the angles of the first and the second
            segment relative to the x-axis, respectively. [CLICK]
            '''
        )
        sub_false = Line(robot_arm.foot.get_center(), robot_arm.foot.get_center()+RIGHT)
        subline = DashedLine(robot_arm.joint.get_center(), robot_arm.joint.get_center()+RIGHT*0.3, stroke_color=BLACK, stroke_width=0.3)
        angle1 = Angle(sub_false, robot_arm.arm1, color=BLUE_B, radius=0.25)
        angle2 = Angle(subline, robot_arm.arm2, color=BLUE_B, radius=0.25)
        t1_label = MathTex(r'\theta_1', color=BLACK).scale(LABELS_SIZE).next_to(angle1, UP*0.25).shift(RIGHT*0.2)
        t2_label = MathTex(r'\theta_2', color=BLACK).scale(LABELS_SIZE).next_to(angle2, UP*0.25)

        self.play(Create(angle1), Write(t1_label))
        self.play(Create(angle2), Create(subline), Write(t2_label))

        # SLIDE 08:  ===========================================================
        # TARGET STAR AND ITS LABEL APPEAR
        self.next_slide(
            notes=
            '''Our target is the orange star, which has coordinates (x_p, y_p).
            [CLICK]
            '''
        )
        target_coord = MathTex(r'(x_p, y_p)', color=BLACK).scale(LABELS_SIZE).next_to(target, DOWN)

        self.play(Indicate(target, scale_factor=1.5, run_time=1.5), Write(target_coord))

        # SLIDE 09:  ===========================================================
        # ROBOT HAND IS INDICATED, AND ITS COORDINATES APPEAR
        self.next_slide(
            notes=
            '''The point (x,y) is the position of the tip, which depends
            only on the two angles theta_1 and theta_2. [CLICK]
            '''
        )
        hand_coord = MathTex(r'(x, y)', color=BLACK).scale(LABELS_SIZE).next_to(robot_arm.hand, UP)

        self.play(Indicate(robot_arm.hand, scale_factor=1.5, run_time=1.5), Write(hand_coord))

        # SLIDE 10:  ===========================================================
        # ARMS ARE PROJECTED ON THE AXES
        self.next_slide(
            notes=
            '''Using some trigonometry, we can project the arms on the x and y
            axes, [CLICK]...
            '''
        )
        l1_projlines = ax_ex.get_lines_to_point(robot_arm.joint.get_center(), color=BLACK)
        l2_projlines = ax_ex.get_lines_to_point(robot_arm.hand.get_center(), color=BLACK)

        b11 = Brace(robot_arm.arm1, direction=DOWN,  color=BLACK, buff=0.2)
        b12 = Brace(robot_arm.arm1, direction=RIGHT, color=BLACK, buff=0.3)
        b21 = Brace(robot_arm.arm2, direction=DOWN,  color=BLACK, buff=0.2).align_to(b11, DOWN)
        b22 = Brace(robot_arm.arm2, direction=RIGHT, color=BLACK, buff=0.3).align_to(b12, RIGHT)

        b11text = MathTex(r"L_1 \cos(\theta_1)", color=BLACK).scale(LABELS_SIZE*0.75)
        b12text = MathTex(r"L_1 \sin(\theta_1)", color=BLACK).scale(LABELS_SIZE*0.75)
        b21text = MathTex(r"L_2 \cos(\theta_2)", color=BLACK).scale(LABELS_SIZE*0.75)
        b22text = MathTex(r"L_2 \sin(\theta_2)", color=BLACK).scale(LABELS_SIZE*0.75)

        b11.put_at_tip(b11text)
        b12.put_at_tip(b12text)
        b21.put_at_tip(b21text)
        b22.put_at_tip(b22text)

        self.play(Create(l1_projlines, lag_ratio=0))
        self.play(FadeIn(b11, b11text, b12, b12text))
        self.play(Create(l2_projlines, lag_ratio=0))
        self.play(FadeIn(b21, b21text, b22, b22text))

        # SLIDE 11:  ===========================================================
        # DRAWING IS SHIFTED UP
        # COMBINE INTO FORMULA FOR X, Y AS A FUNCTION OF ANGLES
        self.next_slide(
            notes=
            '''obtaining these expressions for the tip position. [CLICK]
            '''
        )

        projection_objectsVG = VGroup(l1_projlines, l2_projlines, b11,b12, b21, b22, b11text, b12text, b21text, b22text)
        every_minus_proj = VGroup(ax_ex, ax_2d_labels,
                                  L1_label, L2_label, angle1, angle2, subline, t1_label, t2_label,
                                  target, target_coord, hand_coord)
        robot_kinematics_drawingVG = every_minus_proj + projection_objectsVG

        self.play(robot_kinematics_drawingVG.animate.shift(UP*1))

        x_eq = MathTex(r'{{x =}} {{L_1 \cos(\theta_1)}} + {{L_2 \cos(\theta_2)}}', color=BLACK).next_to(robot_kinematics_drawingVG, DOWN*1.2)
        y_eq = MathTex(r'{{y =}} {{L_1 \sin(\theta_1)}} + {{L_2 \sin(\theta_2)}}', color=BLACK).next_to(x_eq, DOWN*0.8)
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
            '''Let us write the distance between the tip (x,y) and the target
            point (x_p, y_p) in terms of the angles theta_1 and theta_2. We are
            now ready to formulate the minimization problem. [CLICK]
            '''
        )

        distance_arrow = DoubleArrow(
            robot_arm.hand, target, 
            buff=0.1,
            stroke_color=RED, 
            stroke_width=2,
            max_tip_length_to_length_ratio=0.05)
        dist_label = MathTex('d_p', color=BLACK).scale(LABELS_SIZE).next_to(distance_arrow, LEFT, buff=-0.5).shift(DOWN*0.5)
        d_b_eq = MathTex(r'{{d_p}}(\theta_1, \theta_2) = \sqrt{(x(\theta_1, \theta_2)-x_p)^2 + (y(\theta_1, \theta_2)-y_p)^2}',
                         color=BLACK).move_to(VGroup(x_eq, y_eq).get_center())

        self.play(Create(distance_arrow), Write(dist_label))
        self.play(FadeOut(x_eq, y_eq))
        self.play(FadeIn(d_b_eq))
        self.wait(0.5)
        dp_highlight = [
            HighlightRectangle(dist_label),
            HighlightRectangle(d_b_eq[0]),
        ]
        self.play(Create(dp_highlight[0]), Create(dp_highlight[1]))

        # SLIDE 13:  ===========================================================
        # MINMIMIZATION PROBLEM FORMULATION APPEARS
        self.next_slide(
            notes=
            '''We want to find the joint angles theta1 and theta2 that minimize
            the distance d_p positioning the tip closest to the desired target.
            [CLICK]
            '''
        )
        self.play(FadeOut(*dp_highlight))
        minim_problem = MathTex(r'\min \ J(\theta_1, \theta_2), \text{with} \ J={{d_p}}',
                                       color=BLACK).move_to(d_b_eq)

        self.play(ReplacementTransform(d_b_eq, minim_problem))

        # SLIDE 14:  ===========================================================
        # CHANGE TO SQUARED DISTANCE
        self.next_slide(
            notes=
            '''Equivalently, we can minimize the square of the distance, [CLICK]
            '''
        )
        d_p_squared = MathTex(r'd_p^2',color=BLACK).move_to(minim_problem[1])
    
        self.play(Transform(minim_problem[1], d_p_squared))

        # SLIDE 15:  ===========================================================
        # HIGHLIGHT SQURED DISTANCE TERM
        self.next_slide(
            notes=
            '''... setting J equal to d_p^2. [CLICK]
            '''
        )
        d_p_highlight = HighlightRectangle(minim_problem[1])
        self.play(Create(d_p_highlight))

        # SLIDE 16:  ===========================================================
        # 3D PLOT OF THE OBJECTIVE FUNCTION FADES IN
        self.next_slide(
            notes=
            '''Before solving the problem, let's have a look at the function J.
            It can be represented as a surface within the space R3. [CLICK]
            '''
        )
        self.play(FadeOut(robot_arm, every_minus_proj,
                          minim_problem, d_p_highlight,
                          distance_arrow, dist_label,
                          hand_coord, target, target_coord))
        self.wait(0.2)
        # 3D reference system
        ax_3d = ThreeDAxes(
            x_range=[0, 2*PI*17/16, PI],
            y_range=[0, 2*PI*17/16, PI],
            z_range=[0, 30, 7],
            x_length=2*PI*17/16,
            y_length=2*PI*17/16,
            z_length=2*PI*17/16,
            x_axis_config={'stroke_color':BLACK},
            y_axis_config={'stroke_color':BLACK},
            z_axis_config={'stroke_color':BLACK}
        )
        # hand create a 2D number plane because it does not work
        plane = VGroup()
        for i in range(17):
            dx = i/16*(2*PI)
            plane.add(
                Line(ax_3d.c2p(dx, 0, 0), ax_3d.c2p(dx, 2*PI, 0), stroke_color=BLACK, stroke_width=0.2),
                Line(ax_3d.c2p(0, dx, 0), ax_3d.c2p(2*PI, dx, 0), stroke_color=BLACK, stroke_width=0.2),
            )
        ax_3d_labels = ax_3d.get_axis_labels(
            x_label = MathTex(r'\theta_1', color=BLACK).rotate(+PI/2, axis=X_AXIS).rotate(-PI/6).scale(LBELS_SIZE_3D), 
            y_label = MathTex(r'\theta_2', color=BLACK).rotate(+PI/2, axis=X_AXIS).rotate(-120*DEGREES).scale(LBELS_SIZE_3D), 
            z_label = MathTex(r'J', color=BLACK).rotate(-PI/6, axis=Y_AXIS).scale(LBELS_SIZE_3D)
            )
        ax_3d_labels[0].shift(DOWN)
        ax_3d_labels[1].shift(LEFT)
        ax_3d_labels[2].shift([0,0,0.5])
        ref_sys = VGroup(ax_3d, plane, ax_3d_labels).move_to(ORIGIN).save_state()

        T1_3d, T2_3d = PI/1.8, +PI/6
        target_point = double_arm_kinematics(1, 1.5, T1_3d, T2_3d)
        gd = RobotGradientDescent(1, 1.5, target = target_point[:2])
        obj_surf = ax_3d.plot_surface(
            gd.J,
            u_range=[0, 2*PI], v_range=[0, 2*PI], 
            colorscale=[BLUE, TEAL, YELLOW],
            colorscale_axis=2,
            fill_opacity=0.9
            )
        obj_surf.save_state()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-120 * DEGREES, zoom=0.75)
        self.play(FadeIn(obj_surf, ax_3d, plane))

        # SLIDE 17:  ===========================================================
        # AXIS LABELS ARE HIGHLIGHTED
        self.next_slide(
            notes=
            '''On the two horizontal axes, we have the angles theta1 and theta2
            in radians, [CLICK] ...
            '''
        )
        for lb in ax_3d_labels:
            lb.save_state()
        self.play(Write(ax_3d_labels[0])) 
        self.play(Write(ax_3d_labels[1])) 
        
        # SLIDE 18:  ===========================================================
        # AXIS LABELS ARE HIGHLIGHTED
        self.next_slide(
            notes=
            '''while the value of J (the squared distance) is reported
            on the vertical one. [CLICK]
            '''
        )
        self.play(Write(ax_3d_labels[2]))

        # SLIDE 19:  ===========================================================
        # CREATE TARGET STAR AND ADD ITS LABELS
        self.next_slide(
            notes=
            '''Our goal is to identify the angles theta1* and theta2* [CLICK]
            that are associated to the minimum point, the star! Those angles are
            the ones that bring the tip closest to the target point. The question
            is: how can we find them? [CLICK]
            '''
        )
        angular_target = Star(color=ORANGE, fill_opacity=1).scale(0.2).move_to(ax_3d.c2p(T1_3d, T2_3d,0))
        target_projline = ax_3d.get_lines_to_point(angular_target.get_center(), color=ORANGE, stroke_width=1.5)
        tstar_labels = VGroup(
             MathTex(r'\theta_1^*', color=BLACK).move_to(ax_3d.c2p(T1_3d,-1,0)).rotate(+PI/2, axis=X_AXIS).rotate(-PI/6).scale(LBELS_SIZE_3D),
             MathTex(r'\theta_2^*', color=BLACK).move_to(ax_3d.c2p(-1,T2_3d,0)).rotate(+PI/2, axis=X_AXIS).rotate(-PI/6).scale(LBELS_SIZE_3D)
        ).save_state()
        target_3d_plot = VGroup(angular_target, target_projline, tstar_labels).save_state()
    
        self.play(GrowFromCenter(angular_target))
        self.play(Create(target_projline))
        self.play(Write(tstar_labels))
        self.wait(0.5)

        # SLIDE 20:  ===========================================================
        # GRADIENT DESCENT TITLE IS WRITTEN
        self.next_slide(
            notes=
            '''We will use a popular numerical algorithm, the so-called Gradient
            Descent method. Let us briefly explain how it works. Our goal is to
            reach the lowest point [CLICK]
            '''
        )
        GD_title = Text('Gradient Descent', font_size=64, color=BLACK, font='Arial', weight=LIGHT).to_edge(UP).shift(UP*0.75)
        self.add_fixed_in_frame_mobjects(GD_title)
        self.play(Write(GD_title))

        # SLIDE 21:  ===========================================================
        # POINT TRAVELS DOWN THE SLOPE
        self.next_slide(
            notes=
            '''Intuitively, The Gradient Descent starts from an initial guess,
            and builds a sequence of solutions that gradually approaches the
            minimum. [CLICK] We want to descend as fast as possible, how can we
            choose the best directions?
            '''
        )
        starting_gd_guess = [7/6*PI, 6/6*PI]
        gd_point = Dot3D(ax_3d.c2p(starting_gd_guess[0], starting_gd_guess[1], gd.J(starting_gd_guess[0], starting_gd_guess[1])),
                         color=GREEN, radius = 0.15)
        GD_trajectory = gd.run(
            starting_point=starting_gd_guess,
            tol=1e-6
        )
        curve = VGroup().set_points_as_corners(ax_3d.c2p(GD_trajectory))
        trace = TracedPath(gd_point.get_center, stroke_color=GREEN, stroke_width=3)
        self.add(trace)

        # start animation
        self.begin_ambient_camera_rotation(rate=0.005)
        self.wait(1)
        self.play(FadeIn(gd_point))
        self.wait(0.8)

        self.play(MoveAlongPath(gd_point, curve), run_time=5)#, rate_func=linear
        self.wait(1)
        self.begin_ambient_camera_rotation(rate=0)

        # SLIDE 22:  ===========================================================
        # CONTOUR LINES APPEAR
        # SURFACE PLOT IS FLATTENED
        # CAMERA ANGLES GOES BACK TO 2D
        self.next_slide(
            notes=
            '''To answer, let us plot the level curves, connecting points at the
            same "elevation". [CLICK]
            '''
        )
        self.play(FadeOut(gd_point, trace))
        self.stop_ambient_camera_rotation()  # keep it here so z-axis is still checked by the renderer

        # create contours
        def compute_contours(surface: Surface, levels=None):
            u_values, v_values = surface._get_u_values_and_v_values()
            u, v = np.meshgrid(u_values, v_values, indexing='ij')
            image = gd.J(u, v)
            if levels is None:
                levels = np.linspace(image.min(), image.max(), 12)
            Vcontours = VGroup()
            for level in levels:
                contours = skimage.measure.find_contours(image, level=level)
                # rescale contours appropriately (they are given in pixels)
                for i in range(len(contours)):
                    contours[i][:, 0] *= (u_values[-1]-u_values[0])/(len(u_values)-1) 
                    contours[i][:, 1] *= (v_values[-1]-v_values[0])/(len(v_values)-1)
                    contours[i][:, 0] += u_values[0]
                    contours[i][:, 1] += v_values[0]
                    c = np.column_stack((contours[i], np.ones(len(contours[i]))*level ))
                    vcont = VGroup().set_points_as_corners(ax_3d.c2p(c))
                    Vcontours.add(vcont)
            return Vcontours

        contours = compute_contours(obj_surf)
        contours.set_color_by_gradient(BLUE, TEAL,YELLOW).set_stroke(width=2)
        self.play(AnimationGroup(*[Create(c) for c in contours], run_time=1, lag_ratio=0.1))

        # flatten plot
        target_surface = ax_3d.plot_surface(
            lambda u,v: 0,
            u_range=[0, 2*PI], v_range=[0, 2*PI], 
            color=WHITE,
            fill_opacity=0
        )
        self.play(
            *[contour.animate.set_z(plane.get_z()) for contour in contours],
            Transform(obj_surf, target_surface)
        )
        self.remove(obj_surf)

        # change camera angles
        theta_axis_labels = VGroup(
            ax_3d.get_x_axis_label(MathTex(r'\theta_1', color=BLACK).scale(1)),
            ax_3d.get_y_axis_label(MathTex(r'\theta_2', color=BLACK).rotate(-PI/2).scale(1))
        )
        theta_star_labels = VGroup(
            MathTex(r'\theta_1^*', color=BLACK).move_to(ax_3d.c2p(T1_3d,-0.5,0)).scale(1),
            MathTex(r'\theta_2^*', color=BLACK).move_to(ax_3d.c2p(-0.5,T2_3d,0)).scale(1)
        )
        phi, theta, _, _, zoom = self.camera.get_value_trackers()
        ref_sys_2d = VGroup(plane, ax_3d.x_axis, ax_3d.y_axis, *ax_3d_labels[:2])
        self.play(
            phi.animate.set_value(0),
            theta.animate.set_value(-PI/2),
            zoom.animate.set_value(1),
            FadeOut(ax_3d.z_axis, ax_3d_labels[2]),
            Transform(ax_3d_labels[0], theta_axis_labels[0]),
            Transform(ax_3d_labels[1], theta_axis_labels[1]),
            Transform(tstar_labels, theta_star_labels),
        )

        # SLIDE 23:  ===========================================================
        # GRADIENT DEFINITION IS WRITTEN
        # GRADIENT VECTOR FIELD APPEARS
        self.next_slide(
            notes=
            '''And let us plot the gradient of the function, given by this
            expression. As you can see, at each point it is a vector. The
            opposite of these vectors guarantee the steepest descent.[CLICK]
            '''
        )
        gradient_def = VGroup(
            MathTex(r"{{\nabla J(\theta_1, \theta_2)}}", "=", color=BLACK),
            Matrix([[r"\frac{\partial J}{\partial \theta_1}"], [ r"\frac{\partial J}{\partial \theta_2}"]],
                   v_buff= 1.5)).arrange_in_grid(1,2).scale(0.9).move_to(HALF_SCREEN_LEFT)
        gradient_def[1].set_color(BLACK)
        self.play(
            AnimationGroup(
                VGroup(contours, ref_sys_2d, target_3d_plot).animate.scale_to_fit_width(FRAME_WIDTH/2*0.9).move_to(HALF_SCREEN_RIGHT),
                Write(gradient_def),
                lag_ratio=0.5
            )
        )

        def grad_J(theta):
            dx, dy = gd.gradient_obj_function(theta[0], theta[1])
            return np.array((dx, dy, 0))
        gradient_vector_field =  ArrowVectorField(
            grad_J, 
            x_range=[0, 2*PI], y_range=[0, 2*PI], 
            color=RED, opacity=1.0,
            length_func=lambda norm: 0.45 * sigmoid(norm/7)
        )
        gradient_vector_field.scale_to_fit_width(plane.width).move_to(plane)
        self.play(AnimationGroup(
            *[Create(arrow) for arrow in gradient_vector_field.submobjects],
            run_time=1, lag_ratio=0.05)
        )

        # SLIDE 24:  ===========================================================
        # CHAIN RULE SUBSTITUTION IS APPLIED
        self.next_slide(
            notes=
            '''Now we need to do some computations: since J depends on x and y,
            which depend on the angles, we use the chain rule to get this [CLICK]
            formula,
            '''
        )
        gradient_with_chain_rule = VGroup(
            MathTex(r"{{\nabla J(\theta_1, \theta_2)}}", "=", color=BLACK),
            Matrix([
            [r"{{\frac{\partial J}{\partial x}}} {{\frac{\partial x}{\partial \theta_1}}} {{+}}" 
             r"{{\frac{\partial J}{\partial y}}} {{\frac{\partial y}{\partial \theta_1}}}"],
            [r"{{\frac{\partial J}{\partial x}}} {{\frac{\partial x}{\partial \theta_2}}} {{+}}" 
             r"{{\frac{\partial J}{\partial y}}} {{\frac{\partial y}{\partial \theta_2}}}"]
            ], v_buff= 1.5)).arrange_in_grid(1,2).scale(0.9).move_to(HALF_SCREEN_LEFT)
        gradient_with_chain_rule[1].set_color(BLACK)

        self.play(ReplacementTransform(gradient_def, gradient_with_chain_rule))
        
        # SLIDE 25:  ===========================================================
        # GRADIENT FORMULA SLIDES UP
        # CHAIN RULE COMPONENTS EVALUATIONS APPEAR
        self.next_slide(
            notes=
            '''where the factors are given by these expressions. Putting
            everything together, we have the gradient, the main ingredient of
            the method.
            '''
        )
        self.play(gradient_with_chain_rule.animate.shift(UP*1.5))
        ch_eq_1 = MathTex(r"\frac{\partial J}{\partial x}", "=", "2(x-x_p)", color=BLACK)
        ch_eq_2 = MathTex(r"\frac{\partial J}{\partial y}", "=", "2(y-y_p)", color=BLACK)

        ch_eq_3 = MathTex(r"\frac{\partial x}{\partial \theta_1}", "=", r"-L_1 \sin(\theta_1)", color=BLACK)
        ch_eq_4 = MathTex(r"\frac{\partial x}{\partial \theta_2}", "=", r"-L_2 \sin(\theta_2)", color=BLACK)
        ch_eq_5 = MathTex(r"\frac{\partial y}{\partial \theta_1}", "=", r"L_1 \cos(\theta_1)" , color=BLACK)
        ch_eq_6 = MathTex(r"\frac{\partial y}{\partial \theta_2}", "=", r"L_2 \cos(\theta_2)" , color=BLACK)
        
        ch_eqs = VGroup(ch_eq_1, ch_eq_2, ch_eq_3, ch_eq_4, ch_eq_5, ch_eq_6
                        ).arrange_in_grid(3,2, cell_alignment=LEFT).scale(0.7).next_to(gradient_with_chain_rule, DOWN).shift(DOWN*0.3)
       
        self.play(FadeIn(*ch_eqs[:2]))
        self.play(FadeIn(*ch_eqs[2:]))

        # SLIDE 26:  ===========================================================
        # TITLE OF PSEUDO CODE IS WRITTEN
        self.next_slide(
            notes=
            '''Here is how the algorithm looks like. [CLICK]
            '''
        )
        self.play(FadeOut(ch_eq_1,ch_eq_2, ch_eq_3,ch_eq_4,ch_eq_5,ch_eq_6,
                          gradient_with_chain_rule,
                          contours, ref_sys_2d, target_3d_plot,
                          gradient_vector_field, GD_title))
        self.wait(0.5)

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

        # SLIDE 27:  ===========================================================
        # ALGORITHM DATA REQUIREMENTS
        self.next_slide(
            notes=
            '''Starting from an initial guess (theta_1^0, theta_1_2^0), [CLICK]
            '''
        )
        self.play(Write(pc[1]))

        # SLIDE 28:  ===========================================================
        # WHILE LOOP INSTRUCTIONS WRITTEN
        self.next_slide(
            notes=
            '''... we perform iterations, counted with an index i, and at each
            iteration [CLICK] ...
            '''
        )
        self.play(Write(pc[2]), Write(pc[3]))

        # SLIDE 29:  ===========================================================
        # GRADIENT DESCENT ANGLES UPDATE EQUATION WRITTEN
        self.next_slide(
            notes=
            '''...we build a new approximation of the angles with [CLICK] this
            formula,
            '''
        )
        GD_update_eq = MathTex(
            r"\boldsymbol{\theta}^i = \boldsymbol{\theta}^{i-1} -{{\alpha}} \nabla J (\boldsymbol{\theta}^{i-1})",
            color=BLACK).shift(DOWN)
        self.play(Write(GD_update_eq))
        
        # SLIDE 30:  ===========================================================
        # HIGHLIGHT LEARNING RATE
        self.next_slide(
            notes=
            '''...where alpha  is a positive number called learning rate or step
            length, and it determines the size of the step [CLICK] ...
            '''
        )
        alpha_highlight = HighlightRectangle(GD_update_eq[1])
        self.play(Create(alpha_highlight))

        # SLIDE 31:  ===========================================================
        # GD UPDATE EQUATION TRANSFORMS INTO PSEUDO CODE
        self.next_slide(
            notes=
            '''...in the direction opposite to the gradient. [CLICK]
            '''
        )
        self.play(FadeOut(alpha_highlight))
        self.play(ReplacementTransform(GD_update_eq, pc[4]))

        # SLIDE 32:  ===========================================================
        # TOLERANCE CHECK AND BREAK LOOP INSTRUCTION WRITTEN
        self.next_slide(
            notes=
            '''We stop the computation when J is small enough, say it is less
            than a certain tolerance tol that we fixed in advance. This means
            that the distance between the effector and the target point is
            small. [CLICK]
            '''
        )
        self.play(Write(pc[5]), Write(pc[6]))
        self.play(Write(pc[7]), Write(pc[8]))

        # SLIDE 33:  ===========================================================
        # MAX ITERATION NUMBER CHECK HIGHLIGHTED
        self.next_slide(
            notes=
            '''Note that the iterations stop anyway if the maximum number of
            iterations N_iter is exceeded. [CLICK]
            '''
        )
        max_iter_highlight = HighlightRectangle(pc[3])
        self.play(Create(max_iter_highlight))

        # SLIDE 34:  ===========================================================
        # LAST HIGHLIGHT FADES
        self.next_slide(
            notes=
            '''Now, we need a computer code to perform these operations
            automatically. In the next videos we will learn how to write the
            algorithm. [END]
            '''
        )
        self.play(FadeOut(max_iter_highlight))
