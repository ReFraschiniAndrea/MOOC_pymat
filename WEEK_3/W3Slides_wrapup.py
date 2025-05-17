import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from manim_slides import ThreeDSlide
from Generic_mooc_utils import HALF_SCREEN_LEFT, HALF_SCREEN_RIGHT, HighlightRectangle, CustomDecimalNumber
from matlab_utils import MatlabCodeWithLogo
from colab_utils import ColabCodeWithLogo, ColabCode
from W3Anim import double_arm_kinematics, NewDB, RobotGradientDescent

config.background_color=WHITE
config.renderer="cairo"
# config.pixel_width = 960
# config.pixel_height = 720
config.pixel_width = 1440 
config.pixel_height = 1080

LABELS_SIZE = 0.75
LBELS_SIZE_3D = 1.5

class W3WrapUp_GradientDescent(ThreeDSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # REFERENCE AXES AND ROBOT ARM APPEAR
        # LABELS AND DISTANCE ARROWS APPEAR
        self.next_slide(
            notes=
            '''In this lesson we were interested in finding the joint angles
            theta1 and theta2 that minimize the distance between the tip of a 
            robotic arm and a desired target. [CLICK]
            '''
        )
        # reference axes
        ax_2d = Axes(
                x_range=[-4,1,1],
                y_range=[-1.7,2.7,1],
                x_length=9,
                y_length=9*(5-0.6)/5,
                x_axis_config={'stroke_color':BLACK, 'include_ticks':False},
                y_axis_config={'stroke_color':BLACK, 'include_ticks':False}
        ).center()
        ax_2d.set_z_index(-1)
        ax_2d_labels = ax_2d.get_axis_labels(
            MathTex("x", color=BLACK).scale(LABELS_SIZE), MathTex("y", color=BLACK).scale(LABELS_SIZE)
        )
        # robot arm
        T1, T2 = PI*4/5, PI*11/12
        robot_arm = NewDB(ax_2d, 1, 1.5, T1, T2)
        # target point
        t1, t2 = -PI/4, -PI*11/12  # target point angles
        target = Star(color=ORANGE, fill_opacity=1).scale(0.1).move_to(
            robot_arm._hand_coord(t1, t2)
            )
        # distance arrow
        distance_arrow = DoubleArrow(
            robot_arm.hand, target, 
            buff=0.1,
            stroke_color=RED, 
            stroke_width=2,
            max_tip_length_to_length_ratio=0.05)
        dist_label = MathTex('d_p', color=BLACK).scale(LABELS_SIZE).next_to(distance_arrow, LEFT, buff=-0.5).shift(UP*0.5)
        # arms
        L1_label = MathTex(r'L_1', color=BLACK).scale(LABELS_SIZE).next_to(robot_arm.arm1, LEFT, buff=-0.5).shift(DOWN*0.15)
        L2_label = MathTex(r'L_2', color=BLACK).scale(LABELS_SIZE).next_to(robot_arm.arm2, DOWN, buff=-0.1)
        # angles
        sub_false = Line(robot_arm.foot.get_center(), robot_arm.foot.get_center()+RIGHT)
        subline = DashedLine(robot_arm.joint.get_center(), robot_arm.joint.get_center()+RIGHT*0.3, stroke_color=BLACK, stroke_width=0.3)
        angle1 = Angle(sub_false, robot_arm.arm1, color=BLUE_B, radius=0.25)
        angle2 = Angle(subline, robot_arm.arm2, color=BLUE_B, radius=0.25)
        t1_label = MathTex(r'\theta_1', color=BLACK).scale(LABELS_SIZE).next_to(angle1,UP*0.25).shift(RIGHT*0.2)
        t2_label = MathTex(r'\theta_2', color=BLACK).scale(LABELS_SIZE).next_to(angle2,UP*0.25)
        # coordinates
        hand_coord = MathTex(r'(x, y)', color=BLACK).scale(LABELS_SIZE).next_to(robot_arm.hand, UP)
        target_coord = MathTex(r'(x_p, y_p)', color=BLACK).scale(LABELS_SIZE).next_to(target, DOWN)
        hand_coord.add_updater(lambda mob: mob.next_to(robot_arm.hand, UP))
        target_coord.add_updater(lambda mob: mob.next_to(target, DOWN))
        Everything = VGroup(
            ax_2d, ax_2d_labels, 
            robot_arm, L1_label, L2_label, 
            angle1, angle2,  t1_label, t2_label, subline,
            target_coord, hand_coord, 
            target, distance_arrow, dist_label)
        Everything.center()

        # start inserting elements
        self.play(
            AnimationGroup( *[Create(s) for s in robot_arm.submobjects[1:]], lag_ratio=0.2),
            Create(ax_2d, run_time=1),
            Write(ax_2d_labels)
        )
        self.play(GrowFromCenter(target), Write(target_coord),  Write(hand_coord), Write(L1_label), Write(L2_label))
        self.play(Create(angle1), Write(t1_label))
        self.play(Create(angle2), Create(subline), Write(t2_label))
        self.play(FadeIn(distance_arrow), Write(dist_label))

        # SLIDE 02:  ===========================================================
        # ROBOT ARM SHIFTS UP
        # MINIMIZATION EQUATION APPEARS BELOW
        self.next_slide(
            notes=
            '''We mathematically formalized it by means of a minimization
            problem for the cost function J, defined as the square of the
            distance. [CLICK]
            '''
        )
        Everything.suspend_updating()
        self.play(Everything.animate.shift(UP*1))
        Everything.resume_updating()
        minim_problem = MathTex(r'\min \ J(\theta_1, \theta_2), \text{with} \ J={{d_p^2}}', color=BLACK).next_to(Everything, DOWN*1.2)
        self.play(FadeIn(minim_problem))
        self.wait(1.5)
        d_p_highlight = HighlightRectangle(minim_problem[1])
        self.play(Create(d_p_highlight))

        # SLIDE 03:  =========================================================== 
        # OBJECTIVE FUNCTION SURFACE APPEARS
        # GRADIENT DESCENT ANIMATION IS PLAYED
        self.next_slide(
            notes=
            '''To solve the problem numerically we introduced the gradient
            descent method, a popular iterative algorithm that starting from an
            initial guess, follows the opposite of the gradient of J and builds
            a sequence of angles that gradually approaches the minimum. [CLICK]
            '''
        )
        self.play(FadeOut(Everything, minim_problem, d_p_highlight))

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
        plane = VGroup()
        for i in range(17):
            dx = i/16*(2*PI)
            plane.add(
                Line(ax_3d.c2p(dx, 0, 0), ax_3d.c2p(dx, 2*PI, 0), stroke_color=BLACK, stroke_width=0.2),
                Line(ax_3d.c2p(0, dx, 0), ax_3d.c2p(2*PI, dx, 0), stroke_color=BLACK, stroke_width=0.2),
            )
        ax_3d_labels = ax_3d.get_axis_labels(
            x_label = MathTex(r'\theta_1', color=BLACK).rotate(+PI/2, axis=X_AXIS).rotate(-10*DEGREES).scale(LBELS_SIZE_3D), 
            y_label = MathTex(r'\theta_2', color=BLACK).rotate(+PI/2, axis=X_AXIS).rotate(-100*DEGREES).scale(LBELS_SIZE_3D), 
            z_label = MathTex(r'J', color=BLACK).rotate(-10*DEGREES, axis=Y_AXIS).scale(LBELS_SIZE_3D)
            )
        ax_3d_labels[0].shift(LEFT*1.2+DOWN*1.2)
        ax_3d_labels[1].shift(LEFT)
        ax_3d_labels[2].shift([0,0,0.5])
        T1_3d, T2_3d = PI/1.8, +PI/6
        target_point = double_arm_kinematics(1, 1.5, T1_3d, T2_3d)
        angular_target = Star(color=ORANGE, fill_opacity=1).scale(0.2).move_to(ax_3d.c2p(*[T1_3d, T2_3d,0]))
        target_projline = ax_3d.get_lines_to_point(angular_target.get_center(), color=ORANGE, stroke_width=1.5)
        tstar_labels = VGroup(
            MathTex(r'\theta_1^*', color=BLACK).move_to(ax_3d.c2p(T1_3d,-1,0)).rotate(+PI/2, axis=X_AXIS).rotate(-10*DEGREES).scale(LBELS_SIZE_3D),
            MathTex(r'\theta_2^*', color=BLACK).move_to(ax_3d.c2p(-1,T2_3d,0)).rotate(+PI/2, axis=X_AXIS).rotate(-10*DEGREES).scale(LBELS_SIZE_3D)
        )
        
        ref_sys = VGroup(ax_3d, plane, ax_3d_labels, angular_target, target_projline, tstar_labels
                         ).move_to(ORIGIN).rotate(100*DEGREES, axis=[0,0,1])

        self.set_camera_orientation(phi=75 * DEGREES, theta=0, zoom=0.7)
        ref_sys.shift(UP*5)
        
        gd = RobotGradientDescent(1, 1.5, target = target_point[:2])
        obj_surf = ax_3d.plot_surface(
            gd.J,
            u_range=[0, 2*PI], v_range=[0, 2*PI], 
            colorscale=[BLUE, TEAL,YELLOW],
            colorscale_axis=2,
            fill_opacity=0.9
            )
        starting_gd_guess = [7/6*PI, 6/6*PI]
        gd_point = Dot3D(ax_3d.c2p(starting_gd_guess[0], starting_gd_guess[1], gd.J(starting_gd_guess[0], starting_gd_guess[1])),
                         color=GREEN, radius = 0.15)
        GD_trajectory = gd.run(
            starting_point=starting_gd_guess,
            tol=1e-6
        )

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
        self.add_fixed_in_frame_mobjects(pc)
        pc.scale(0.7).to_edge(LEFT)
        
        # actual animation
        curve = VGroup().set_points_as_corners(ax_3d.c2p(GD_trajectory))
        trace = TracedPath(gd_point.get_center, stroke_color=GREEN, stroke_width=3)
        self.add(trace)
        self.play(FadeIn(pc, obj_surf, ref_sys))  # order is important
        self.wait(0.5)
        self.play(FadeIn(gd_point))
        self.wait(0.8)
        self.play(MoveAlongPath(gd_point, curve), run_time=5) #, rate_func=linear
        self.wait(1)

        # SLIDE 04:  =========================================================== 
        # PYTHON AND MATLAB SCRIPTS FOR GD APPEAR SIDE BY SIDE
        self.next_slide(
            notes=
            '''We implemented this alogrithm either in Python and in Matlab, by
            using the following scripts. [CLICK]
            '''
        )
        self.play(FadeOut(pc, obj_surf, ref_sys, 
                        #   angular_target, target_projline, tstar_labels,
                          gd_point, trace))
        self.set_camera_orientation(phi=0 , theta=-PI/2, zoom=1)
        
        GD_codesnipd_py = ColabCodeWithLogo(
            r'''
            # Gradient Descent Method 
            i = 1 
            while i <= Niter: 
                # Gradient of the objective function 
                grad = grad_J(theta, xp, L1, L2) 

                # Update theta with gradient 
                theta[0] -= alpha * grad[0]  
                theta[1] -= alpha * grad[1]  

                # Compute the current distance 
                Jval = J(theta, xp, L1, L2) 
                
                # Check for convergence 
                if Jval < tol: 
                    break 

                i = i + 1;
                ''',
            paragraph_config={'font_size':20},
            logo_pos=UP
        ).to_edge(LEFT, buff=0.5)

        GD_codesnipd_mat=MatlabCodeWithLogo(
            r'''
            % Gradient Descent Method
            i = 1; 
            while i <= Niter 
                % Gradient of the objective function 
                grad = grad_J(theta, xp, L1, L2); 
                
                % Update theta with gradient 
                theta = theta - alpha * grad;   

                % Compute the current distance 
                Jval = J(theta, xp, L1, L2); 

                % Check for convergence 
                if Jval < tol 
                    break 
                end 
                i = i + 1;
            end
            ''',
            paragraph_config={'font_size':20},
            logo_pos=UP
        ).next_to(GD_codesnipd_py, RIGHT).align_to(GD_codesnipd_py, DOWN)
        
        self.play(FadeIn(Group(GD_codesnipd_py, GD_codesnipd_mat).center()))
        
        # SLIDE 05:  ===========================================================
        # PROBLEM DATA APPEARS WITH FINAL CONFIGURATION PLOT
        self.next_slide(
            notes=
            '''We use the gradient descent method to find a final configuration
            for these problem's data. [CLICK]
            '''
        )
        self.play(FadeOut(GD_codesnipd_py, GD_codesnipd_mat))
        Everything.remove(dist_label, distance_arrow, L1_label, L2_label, t1_label, t2_label, angle1, angle2, subline)
        ax_2d.become(
            Axes(
            x_range=[-3,2,1],
            y_range=[-2,3,1],
            x_length=7,
            y_length=7,
            x_axis_config={'stroke_color':BLACK, 'include_ticks':False},
            y_axis_config={'stroke_color':BLACK, 'include_ticks':False}
            )
        )
        ax_2d_labels.become(ax_2d.get_axis_labels(
            MathTex("x", color=BLACK).scale(0.75), MathTex("y", color=BLACK).scale(0.75)
        ))
        Everything.center().to_edge(RIGHT)
        target.move_to(ax_2d.c2p(0.75, -1, 0))
        
        problem_data = ColabCode(
            r'''
            # Data
            xp = [0.75 , -1];   
            L1 = 1;   
            L2 = 1.5;  
            theta = [2.5 2.7]; 
            tol = 0.01;  
            alpha = 0.1;  
            Niter = 1000;
            ''',
            paragraph_config={'font_size':28},
        )
        problem_data.add_background_window()
        problem_data.to_edge(LEFT)

        self.play(FadeIn(Everything, problem_data))
        self.wait(0.5)
        self.play(robot_arm.MoveToAngles(0.6293184, 4.6875734, run_time=2))

        # SLIDE 06:  ===========================================================
        # SHOW CODE SNIPPETS FOR IMPORTING FUNCTIONS
        self.next_slide(
            notes=
            '''We learnt how to import/use external functions.
            '''
        )
        self.play(FadeOut(Everything, problem_data))
        
        import_codesnip_py = ColabCodeWithLogo(
            r'''from my_functions import plot_robot_arm''',
            paragraph_config={'font_size':28},
            logo_pos=UP
        )
        import_codesnip_mat=MatlabCodeWithLogo(
            r'''addpath('/path/to/folder/')''',
            paragraph_config={'font_size':28},
            logo_pos=UP
        ).next_to(import_codesnip_py, DOWN).align_to(import_codesnip_py, LEFT)
        
        self.play(FadeIn(Group(import_codesnip_py, import_codesnip_mat).center()))
        self.wait(0.1)

        # SLIDE 07:  ===========================================================
        # CODE TRANSFORMS INTO CODE FOR WHILE/BREAK
        self.next_slide(
            notes=
            '''Moreover we use the while loop and break instruction to code the algorithm.
            '''
        )
        break_codesnip_py = ColabCodeWithLogo(
            r'''
            i = 1 
            while i <= Niter: 
                # Instructions 
                # Break  
                if Jval < tol: 
                    # Instructions 
                    break 
                i = i+1''',
            paragraph_config={'font_size':28},
        )

        break_codesnip_mat=MatlabCodeWithLogo(
            r'''
            i = 1  
            while i <= Niter 
                % Instructions 
                % Break 
                if Jval < tol 
                    % Instructions  
                    break 
                end 
                i = i+1; 
            end ''',
            paragraph_config={'font_size':28},
        ).next_to(break_codesnip_py, RIGHT)

        Group(break_codesnip_py, break_codesnip_mat).center()

        self.play(Transform(import_codesnip_py, break_codesnip_py), 
                  Transform(import_codesnip_mat, break_codesnip_mat))
        self.wait(0.1)

        # SLIDE 08:  ===========================================================
        # NEW AXES APPEAR WITH ROBOT ARM
        # ROBOT ARM TRIPLICATES, MOVING TO RESPECTIVE STARTING POSTIONS
        # ROBOTS FOLOW RESPECTIVE GD TRAJECTORIES, WHILE COUNTERS KEEP TRACK OF
        # ITERATION NUMBER AND OBJECTIVE FUNCTION VALUE
        self.next_slide(
            notes=
            '''With the code we can perform different simulations by changing the initial conditions and fixing the target.
            '''
        )
        self.play(FadeOut(import_codesnip_py, import_codesnip_mat))

        # new trajectories
        L1, L2 = 1, 1.5
        startig_points = [
            [2.5, 2.7],
            [0, 0.5],
            [2.7, 3.14],
        ]
        common_target = [0.75, -1]
        gd_2 = RobotGradientDescent(L1, L2, target=common_target)
        gd_traj1 = gd_2.run(startig_points[0], learning_rate=0.1, tol=0.01)
        gd_traj2 = gd_2.run(startig_points[1], learning_rate=0.1, tol=0.01)
        gd_traj3 = gd_2.run(startig_points[2], learning_rate=0.1, tol=0.01)

        ax_2d_new = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            x_length=9,
            y_length=9,
            x_axis_config={'stroke_color':BLACK, 'include_ticks':True},
            y_axis_config={'stroke_color':BLACK, 'include_ticks':True}
        )

        db1 = NewDB(ax_2d_new, L1, L2, *startig_points[0], BLUE, z_index=3)
        db2 = NewDB(ax_2d_new, L1, L2, *startig_points[0], ORANGE, z_index=2)
        db3 = NewDB(ax_2d_new, L1, L2, *startig_points[0], GREEN, z_index=1)

        def my_counter(text, color, **kwargs):
            label = Text(text, color=color, font='Aptos Mono')
            label.add(Text('=', color=color, font='Aptos Mono').next_to(label, RIGHT))
            counter =  CustomDecimalNumber(font='Aptos Mono', color=color, mob_class = Text, **kwargs).next_to(label, RIGHT)
            return VGroup(label, counter)

        iter_1 = my_counter('iter', num_decimal_places=0, color=BLUE)
        iter_2 = my_counter('iter', num_decimal_places=0, color=ORANGE)
        iter_3 = my_counter('iter', num_decimal_places=0, color=GREEN)

        J_1 = my_counter('J', num_decimal_places=4, color=BLUE)
        J_2 = my_counter('J', num_decimal_places=4, color=ORANGE)
        J_3 = my_counter('J', num_decimal_places=4, color=GREEN)

        counter1 = VGroup(iter_1, J_1).arrange_in_grid(2, 1, cell_alignment=LEFT).to_corner(UR, buff = 1)
        counter2 = VGroup(iter_2, J_2).arrange_in_grid(2, 1, cell_alignment=LEFT).center().to_edge(RIGHT, buff = 1)
        counter3 = VGroup(iter_3, J_3).arrange_in_grid(2, 1, cell_alignment=LEFT).to_corner(DR, buff = 1)
        
        common_target = Star(color=RED,fill_opacity=1).scale(0.1).move_to( ax_2d_new.c2p(*common_target, 0))

        self.play(FadeIn(ax_2d_new, db1, db2, db3, common_target))
        # self.play(GrowFromCenter(common_target))
        self.play(
            db2.MoveToAngles(*startig_points[1]),
            db3.MoveToAngles(*startig_points[2])
        )
        self.play(
            AnimationGroup(
                VGroup(ax_2d_new, common_target).animate(run_time=0.5).to_edge(LEFT), 
                FadeIn(counter1, counter2, counter3, run_time=0.5),
                lag_ratio=0.75
            ))

        for i in range(100):
            anim = []
            run_time = 0.2* np.exp(-i/20)
            rf = linear if i>1 else rate_functions.ease_in_sine
            if i<len(gd_traj1): 
                anim.append(db1.MoveToAngles(gd_traj1[i, 0], gd_traj1[i, 1], rate_func=rf, run_time=run_time))
                iter_1[1].set_value(i)
                J_1[1].set_value(gd_traj1[i, 2])
            if i<len(gd_traj2): 
                anim.append(db2.MoveToAngles(gd_traj2[i, 0], gd_traj2[i, 1], rate_func=rf, run_time=run_time)); 
                iter_2[1].set_value(i); 
                J_2[1].set_value(gd_traj2[i, 2])
            if i<len(gd_traj3): 
                anim.append(db3.MoveToAngles(gd_traj3[i, 0], gd_traj3[i, 1], rate_func=rf, run_time=run_time)); 
                iter_3[1].set_value(i); 
                J_3[1].set_value(gd_traj3[i, 2])

            if anim:
                self.play(*anim)

        # SLIDE 09:  ===========================================================
        # EMPTY AXES WITH LABELS TO HINT AT THE SOLUTION
        self.next_slide(
            notes=
            '''Now it's your turn.
            Try collecting computed distances in a vector and plot them as a
            function of  iteration. [CLICK]
            '''
        )
        self.play(FadeOut(db1, db2, db3, ax_2d_new, counter1, counter2, counter3, common_target))

        plane = NumberPlane(
            x_range=[0, 45, 5],
            y_range=[-2, 2, 1] ,
            x_length=9,
            y_length=9,
            x_axis_config={'stroke_color': BLACK, 'include_ticks': False, 'include_tip':True, 'include_numbers': True, 'label_direction':DOWN, 'label_constructor':MathTex},
            y_axis_config={'stroke_color': BLACK, 'include_ticks': False, 'include_tip':True, 'include_numbers': True, 'label_direction':LEFT, 'label_constructor':MathTex,
                           'scaling': LogBase(custom_labels=True), 'numbers_to_include':[-2, -1, 0, 1]},
            background_line_style={'stroke_color': BLACK, 'stroke_width': 0.5}
        ).scale(0.65).move_to(HALF_SCREEN_LEFT)
        plane.x_axis.numbers.set_color(BLACK)
        plane.y_axis.labels.set_color(BLACK)
        plane_labels =plane.get_axis_labels(
            Text('iter', color=BLACK, font='Aptos Mono').scale(LABELS_SIZE),  #, font='Aptos'
            Text('J', color=BLACK, font='Aptos Mono').scale(LABELS_SIZE)
        )
        dots = [Dot(plane.c2p(i, gd_traj1[i, 2], 0), color=RED, radius=0.8*DEFAULT_DOT_RADIUS) for i in range(len(gd_traj1))]
        lines = [Line(dots[i].get_center(), dots[i+1].get_center(), color=RED, stroke_width=2) for i in range(len(gd_traj1)-1)]

        iter_c = my_counter('iter', color=BLACK, num_decimal_places=0)
        J_c = my_counter('J', color=BLACK, num_decimal_places=4).next_to(iter_c, DOWN)
        iter_c[1].set_value(0)
        J_c[1].set_value(gd_traj1[0, 2])
        counterc = VGroup(iter_c, J_c).arrange_in_grid(2,1, cell_alignment=LEFT).to_edge(UP).shift(UP)

        ax_2d_new.scale(0.65).move_to(HALF_SCREEN_RIGHT)
        common_target.move_to(ax_2d_new.c2p(0.75, -1, 0)).set_color(ORANGE)
        final_robot_arm = NewDB(ax_2d_new, 1 , 1.5, gd_traj1[0,0], gd_traj1[0,1], color=BLUE)
        distance_arrow = DoubleArrow(
            final_robot_arm.hand, common_target, 
            buff=0.1,
            stroke_color=RED, 
            stroke_width=2,
            max_tip_length_to_length_ratio=0.05)

        self.play(FadeIn(ax_2d_new, final_robot_arm , common_target, distance_arrow, plane, plane_labels, counterc, dots[0]))
        distance_arrow.add_updater(
            lambda m: m.become(DoubleArrow(
            final_robot_arm.hand, common_target, 
            buff=0.1,
            stroke_color=RED, 
            stroke_width=2,
            max_tip_length_to_length_ratio=0.05))
        )
        self.wait(0.5)
        # self.play(GrowFromCenter(dots[0], run_time=0.1))
        for i in range(1,len(dots)):
            dt = 0.1/(1+ i/10)
            self.play(AnimationGroup(
                final_robot_arm.MoveToAngles(gd_traj1[i, 0], gd_traj1[i, 1], rate_func=smooth, run_time=dt),
                GrowFromCenter(dots[i], run_time=dt),
                Create(lines[i-1], run_time=dt)
            ))
            iter_c[1].set_value(i)
            J_c[1].set_value(gd_traj1[i, 2])
        self.wait(0.05)

        # SLIDE 10:  ===========================================================
        # ROBOT ARM WITH CIRCLE TO SHOW REACH APPEARS
        self.next_slide(
            notes=
            '''Also, think about what happens when the target is not reachable
            by the robot. [END]
            '''
        )
        self.play(
            AnimationGroup(
                FadeOut(plane, *dots, *lines, plane_labels, distance_arrow, counterc),
                VGroup(ax_2d_new, common_target).animate.center().scale(1.4),
                lag_ratio=0.75   
            )
        )
        self.play(common_target.animate.move_to(ax_2d_new.c2p(2, 2.5,0)))
        robot_radius = Circle(radius=final_robot_arm.arm1.get_length()+final_robot_arm.arm2.get_length(),
                              color=RED, fill_opacity=0, stroke_width=2).move_to(ax_2d_new.c2p(0,0,0))
        self.play(FadeIn(robot_radius))
        self.play(final_robot_arm.MoveToAngles(PI/4, PI/4))
        self.wait(0.2)
