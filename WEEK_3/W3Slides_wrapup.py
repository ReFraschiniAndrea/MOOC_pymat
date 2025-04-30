from manim import *
from manim_slides import ThreeDSlide
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mooc_utils
from W3Anim import DoubleArm, double_arm_kinematics, GD, GD_D, NewDB

config.background_color=WHITE
config.renderer="cairo"
# test resolution
# config.pixel_width = 960 
# config.pixel_height = 720
# config.frame_rate=30
# release resolution
config.pixel_width = 1440 
config.pixel_height = 1080
config.frame_rate=60

LABELS_SIZE = 0.75

class W3WrapUp_GradientDescent(ThreeDSlide):
    def construct(self):
        
        # SLIDE 1: robot arm explanation reappears
        self.next_slide(
            notes=
            '''In this lesson we were interested in finding the joint angles theta1 and theta2 
            that minimize the distance between the tip of a robotic arm and a desired target. 
            '''
        )
        # reference axes
        ax_2d = Axes(
            x_range=[-4,1,1],
            y_range=[-2,3,1],
            x_length=9,
            y_length=9,
            x_axis_config={'stroke_color':BLACK, 'include_ticks':False},
            y_axis_config={'stroke_color':BLACK, 'include_ticks':False}
        ).center()
        ax_2d.set_z_index(-1)
        ax_2d_labels = ax_2d.get_axis_labels(
            MathTex("x", color=BLACK).scale(0.75), MathTex("y", color=BLACK).scale(0.75)
        )
        # robot arm
        T1, T2 = PI*4/5, PI*11/12
        robot_arm = NewDB(ax_2d, 1, 1.5, T1, T2)
        robot_arm_VG = VGroup(robot_arm.foot, robot_arm.arm1, robot_arm.joint, robot_arm.arm2, robot_arm.hand)
        # target point
        t1, t2 = -PI/4, -PI*11/12  # target point angles
        target = Star(color=ORANGE, fill_opacity=1).scale(0.1).move_to(
            robot_arm._hand_coord(t1, t2)
            )
        # distance arrow
        distance_arrow = DoubleArrow(
            robot_arm.hand, target, 
            buff=0.1,
            stroke_color=BLACK, 
            stroke_width=1,
            max_tip_length_to_length_ratio=0.05)
        dist_label = MathTex('d_p', color=BLACK).scale(LABELS_SIZE).next_to(distance_arrow, LEFT, buff=-0.5).shift(UP*0.5)
        # arms
        L1_label = MathTex(r'L_1', color=BLACK).scale(LABELS_SIZE).next_to(robot_arm.arm1, LEFT, buff=-0.5).shift(DOWN*0.15)
        L2_label = MathTex(r'L_2', color=BLACK).scale(LABELS_SIZE).next_to(robot_arm.arm2, DOWN, buff=-0.1)
        # L1_label.add_updater(lambda mob: mob.next_to(robot_arm.arm1, LEFT, buff=-0.5).shift(DOWN*0.15))
        # L2_label.add_updater(lambda mob: mob.next_to(robot_arm.arm2, DOWN, buff=-0.1))
        # angles
        sub_false = Line(robot_arm.foot.get_center(), robot_arm.foot.get_center()+RIGHT)
        subline = DashedLine(robot_arm.joint.get_center(), robot_arm.joint.get_center()+RIGHT*0.3, stroke_color=BLACK, stroke_width=0.3)
        angle1 = Angle(sub_false, robot_arm.arm1, color=BLUE_B, radius=0.25)
        angle2 = Angle(subline, robot_arm.arm2, color=BLUE_B, radius=0.25)
        t1_label = MathTex(r'\theta_1', color=BLACK).scale(LABELS_SIZE).next_to(angle1,UP*0.25).shift(RIGHT*0.2)
        t2_label = MathTex(r'\theta_2', color=BLACK).scale(LABELS_SIZE).next_to(angle2,UP*0.25)
        # t1_label.add_updater(lambda mob: mob.next_to(angle1,UP*0.25).shift(RIGHT*0.2))
        # t2_label.add_updater(lambda mob: mob.next_to(angle2,UP*0.25))
        # coordinates
        hand_coord = MathTex(r'(x, y)', color=BLACK).scale(LABELS_SIZE).next_to(robot_arm.hand, UP)
        target_coord = MathTex(r'(x_p, y_p)', color=BLACK).scale(LABELS_SIZE).next_to(target, DOWN)
        hand_coord.add_updater(lambda mob: mob.next_to(robot_arm.hand, UP))
        target_coord.add_updater(lambda mob: mob.next_to(target, DOWN))
        Everything = VGroup(
            ax_2d, ax_2d_labels, 
            robot_arm_VG, L1_label, L2_label, 
            angle1, angle2,  t1_label, t2_label, subline,
            target_coord, hand_coord, 
            target, distance_arrow, dist_label)
        Everything.center()
        # start inserting elements
        self.play(
            AnimationGroup( *[Create(s) for s in robot_arm_VG], lag_ratio=0.2),
            Create(ax_2d, run_time=1),
            Write(ax_2d_labels)
        )
        self.play(GrowFromCenter(target), Write(target_coord),  Write(hand_coord), Write(L1_label), Write(L2_label))
        
        self.play(Create(angle1), Write(t1_label))
        self.play(Create(angle2), Create(subline), Write(t2_label))
        
        self.play(FadeIn(distance_arrow), Write(dist_label))

        # SLIDE 2: recall minimization problem
        self.next_slide(
            notes=
            '''We mathematically formalized it by means of a minimization problem for the cost 
            function J, defined as the square of the distance. 
            '''
        )
        Everything.suspend_updating()
        self.play(Everything.animate.shift(UP*0.75))
        Everything.resume_updating()
        minim_problem = MathTex(r'\min \ J(\theta_1, \theta_2), \text{with} \ J={{d_p^2}}', color=BLACK).next_to(Everything, DOWN)
        self.play(FadeIn(minim_problem))
        self.wait(1.5)
        self.play(Indicate(minim_problem[1]))

        # SLIDE 3: recall gradient descent algorithm
        self.next_slide(
            notes=
            '''To solve the problem numerically we introduced the gradient descent method, a 
            popular iterative algorithm that starting from an initial guess, follows the opposite 
            of the gradient of J and builds a sequence of angles that gradually approaches the minimum.
            '''
        )
        self.play(FadeOut(Everything, minim_problem))

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
        ax_3d_labels[0].shift(DOWN+LEFT)
        ax_3d_labels[1].shift(LEFT)
        ref_sys = VGroup(ax_3d, plane, ax_3d_labels).move_to(ORIGIN).rotate(100*DEGREES, axis=[0,0,1])#.shift(UP*5)

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

        self.set_camera_orientation(phi=75 * DEGREES, theta=0, zoom=0.7)
        ref_sys.shift(UP*5)
        pc.scale(0.7).to_edge(LEFT)
        self.add_fixed_in_frame_mobjects(pc)

        # NOTE: the target point is now different to make the plot more clear
        T1_3d, T2_3d = PI/1.8, +PI/6
        target_point = double_arm_kinematics(1.3, 1.8, T1_3d, T2_3d)
        def objective_func(t1, t2):
            return (
                np.square(1.3*np.cos(t1) + 1.8*np.cos(t2)- target_point[0]) +
                np.square(1.3*np.sin(t1) + 1.8*np.sin(t2)- target_point[1])
                )
        obj_surf = ax_3d.plot_surface(
            objective_func,
            u_range=[0, 2*PI], v_range=[0, 2*PI], 
            colorscale=[BLUE, TEAL,YELLOW],
            colorscale_axis=2,
            fill_opacity=0.9
            )
        angular_target = Star(color=ORANGE).scale(0.2).move_to(ax_3d.c2p(*[T1_3d, T2_3d,0]))
        target_projline = ax_3d.get_lines_to_point(angular_target.get_center(), color=ORANGE, stroke_width=1.5)
        t1star_label = MathTex(r'\theta_1^*', color=BLACK).scale(0.75).move_to(ax_3d.c2p(T1_3d,-1,0)).rotate(0).rotate(+PI/2, axis=RIGHT)
        t2star_label = MathTex(r'\theta_2^*', color=BLACK).scale(0.75).move_to(ax_3d.c2p(-1,T2_3d,0)).rotate(+PI/2, axis=RIGHT)
        
        starting_gd_guess = [7/6*PI, 6/6*PI]
        gd_point = Dot3D(ax_3d.c2p(starting_gd_guess[0], starting_gd_guess[1], objective_func(starting_gd_guess[0], starting_gd_guess[1])),
                    color=GREEN, radius = 0.15)
        GD_trajectory = GD(
            starting_point=[7/6*PI, 6/6*PI],
            target = double_arm_kinematics(1.3,1.8,T1_3d,T2_3d)[:2],
            l1=1.3, l2=1.8,
            tol=1e-3
            )
        
        # actual animation
        curve = VGroup().set_points_as_corners(ax_3d.c2p(GD_trajectory))
        trace = TracedPath(gd_point.get_center, stroke_color=GREEN, stroke_width=3)
        self.add(trace)
        self.play(FadeIn(pc, obj_surf, ref_sys, angular_target, target_projline))  # order is important
        self.wait(0.5)
        self.play(FadeIn(gd_point))
        self.wait(0.8)
        self.play(MoveAlongPath(gd_point, curve), run_time=5) #, rate_func=linear
        self.wait(1)

        # SLIDE 4: show GD scripts
        self.next_slide(
            notes=
            '''We implemented this alogrithm either in Python and in Matlab, by using the following scripts.
            '''
        )
        self.play(FadeOut(pc, obj_surf, ref_sys, 
                          angular_target, target_projline,# t1star_label, t2star_label,
                          gd_point, trace))
        self.set_camera_orientation(phi=0 , theta=-PI/2, zoom=1)
        
        GD_codesnipd_py = mooc_utils.ColabCode(
            r'''
            # Gradient Descent Method 
            i = 1 
            while i <= Niter: 
                # Gradient of the objective function 
                grad = grad_J(angles, target, L1, L2) 

                # Update theta with gradient 
                angles[0] -= alpha * grad[0]  
                angles[1] -= alpha * grad[1]  

                # Compute the current distance 
                Jval = J(angles, target, L1, L2) 
                
                # Check for convergence 
                if Jval < tol: 
                    break 

                i = i + 1;
                ''',
            font_size=20,
            include_logo=True,
            logo_pos=UP
        ).to_edge(LEFT, buff=0.5)

        GD_codesnipd_mat=mooc_utils.MatlabCode(
            r'''
            % Gradient Descent Method
            i = 1; 
            while i <= Niter 
                % Gradient of the objective function 
                grad = grad_J(angles, target, L1, L2); 
                
                % Update theta with gradient 
                angles = angles - alpha * grad;   

                % Compute the current distance 
                Jval = J(angles, target, L1, L2); 

                % Check for convergence 
                if Jval < tol 
                    break 
                end 
                i = i + 1;
            end
            ''',
            font_size=20,
            include_logo=True,
            logo_pos=UP
        ).next_to(GD_codesnipd_py, RIGHT).align_to(GD_codesnipd_py, DOWN)
        
        self.play(FadeIn(Group(GD_codesnipd_py, GD_codesnipd_mat).center()))
        

        # SLIDE 5: show final arm configuration for problme data used
        self.next_slide(
            notes=
            '''We use the gradient descent method to find a final configuration for these problem's data.
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
        
        problem_data = mooc_utils.ColabCode(
            code='''
            # Data
            target = [0.75 , -1];   
            L1 = 1;   
            L2 = 1.5;  
            angles = [2.5 2.7]; 
            tol = 0.01;  
            alpha = 0.1;  
            Niter = 1000;
            ''',
            font_size=28
        ).to_edge(LEFT)

        self.play(FadeIn(Everything, problem_data))
        self.wait(0.5)
        self.play(robot_arm.MoveToAngles(0.6293184, 4.6875734, run_time=2))

        # SLIDE 6: Show code snippets for importing functions
        self.next_slide(
            notes=
            '''We learnt how to import/use external functions.
            '''
        )
        self.play(FadeOut(Everything, problem_data))
        
        import_codesnip_py = mooc_utils.ColabCode(
            '''from my_functions import plot_robot_arm''',
            font_size=28,
            include_logo=True,
            logo_pos=UP
        )
        import_codesnip_mat=mooc_utils.MatlabCode(
            '''addpath('/path/to/folder/')''',
            font_size=28,
            include_logo=True,
            logo_pos=UP
        ).next_to(import_codesnip_py, DOWN).align_to(import_codesnip_py, LEFT)
        
        self.play(FadeIn(Group(import_codesnip_py, import_codesnip_mat).center()))
        self.wait(0.1)

        # SLIDE 7: Code snippets for while loop / break instruction
        self.next_slide(
            notes=
            '''Moreover we use the while loop and break instruction to code the algorithm.
            '''
        )

        break_codesnip_py = mooc_utils.ColabCode(
            '''
            i = 1 
            while i <= Niter: 
                # Instructions 
                # Break  
                if Jval < tol: 
                    # Instructions 
                    break 
                i = i+1''',
            font_size=28,
            include_logo=True
        )

        break_codesnip_mat=mooc_utils.MatlabCode(
            '''
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
            font_size=28,
            include_logo=True
        ).next_to(break_codesnip_py, RIGHT)

        Group(break_codesnip_py, break_codesnip_mat).center()

        self.play(mooc_utils.CodeTransform(import_codesnip_py, break_codesnip_py), 
                  mooc_utils.CodeTransform(import_codesnip_mat, break_codesnip_mat))
        self.wait(0.1)


        # SLIDE 8: show 3 overlapped simulations for target starting point
        self.next_slide(
            notes=
            '''With the code we can perform different simulations by changing the target point
            by fixing the same initial conditions.
            '''
        )

        self.play(FadeOut(import_codesnip_py, import_codesnip_mat))

        L1, L2 = 1, 1.5
        start_t1, start_t2 = 2.5, 2.7
        ax_2d_new = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4],
            x_length=9,
            y_length=9,
            x_axis_config={'stroke_color':BLACK, 'include_ticks':True},
            y_axis_config={'stroke_color':BLACK, 'include_ticks':True}
        )

        db1 = NewDB(ax_2d_new, L1, L2, start_t1, start_t2, BLUE, z_index=3)
        db2 = NewDB(ax_2d_new, L1, L2, start_t1, start_t2, ORANGE, z_index=2)
        db3 = NewDB(ax_2d_new, L1, L2, start_t1, start_t2, GREEN, z_index=1)

        def my_counter(text, color, c_type=Integer, **kwargs):
            label = Text(text, color=color, font='Aptos Mono')
            label.add(Text('=', color=color, font='Aptos Mono').next_to(label, RIGHT))
            counter =  c_type(0, color=color, mob_class = Text, **kwargs).next_to(label, RIGHT)
            return VGroup(label, counter)

        iter_1 = my_counter('iter', color=BLUE)
        iter_2 = my_counter('iter', color=ORANGE)
        iter_3 = my_counter('iter', color=GREEN)

        J_1 = my_counter('J', color=BLUE, num_decimal_places=4).next_to(iter_1, DOWN)
        J_2 = my_counter('J', color=ORANGE, num_decimal_places=4).next_to(iter_2, DOWN)
        J_3 = my_counter('J', color=GREEN, num_decimal_places=4).next_to(iter_3, DOWN)

        counter1 = VGroup(iter_1, J_1).to_corner(UR, buff = 1)
        counter2 = VGroup(iter_2, J_2).center().to_edge(RIGHT, buff = 1)
        counter3 = VGroup(iter_3, J_3).to_corner(DR, buff = 1)
        
        target1 = Star(color=BLUE,fill_opacity=1).scale(0.1).move_to(  ax_2d_new.c2p(0.75, -1, 0))
        target2 = Star(color=ORANGE,fill_opacity=1).scale(0.1).move_to(ax_2d_new.c2p(1, 1,0))
        target3 = Star(color=GREEN,fill_opacity=1).scale(0.1).move_to( ax_2d_new.c2p(2, 2.5,0))

        gd_traj1 = GD_D([start_t1, start_t2],[0.75,  -1], L1, L2, learning_rate=0.1, tol=0.01)
        gd_traj2 = GD_D([start_t1, start_t2],[   1,   1], L1, L2, learning_rate=0.1, tol=0.01)
        gd_traj3 = GD_D([start_t1, start_t2],[   2, 2.5], L1, L2, learning_rate=0.1, tol=0.01)

        self.play(FadeIn(ax_2d_new, db1.ARM1, db1.ARM2, db2.ARM1, db2.ARM2, db3.ARM1, db3.ARM2))
        self.play(GrowFromCenter(target1), GrowFromCenter(target2), GrowFromCenter(target3))
        self.wait(0.5)
        self.play(
            AnimationGroup(
                VGroup(ax_2d_new, target1, target2, target3).animate.to_edge(LEFT), 
                FadeIn(counter1, counter2, counter3),
                lag_ratio=0.75
            ))
        self.next_slide(
            notes=
            '''With the code we can perform different simulations by changing the target point
            by fixing the same initial conditions. (robot arm animation)'''
        )

        for i in range(100):
            anim = []
            run_time = 0.3* np.exp(-i/20)
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

            if i<100:
                self.play(*anim)
        

        # SLIDE 9: show 3 overlapped simulations for different target point
        self.next_slide(
            notes=
            '''On the other hand, we can fix the target = [0.75 -1] and choose  different initial conditions.
            '''
        )
        # new trajectories
        common_target = [0.75, -1]
        gd_traj1 = GD_D([2.5, 2.7], common_target, L1, L2, learning_rate=0.1, tol=0.01)
        gd_traj2 = GD_D([0, 0.5],   common_target,L1, L2, learning_rate=0.1, tol=0.01)
        gd_traj3 = GD_D([2.7, 3.14],common_target, L1, L2, learning_rate=0.1, tol=0.01)
        # 3 targets become one
        self.play(target1.animate.set_color(RED))
        self.wait(0.5)
        self.play(target2.animate.become(target1), target3.animate.become(target1))
        self.wait(0.5)
        # reset counters and set the position
        self.play(
            db1.MoveToAngles(2.5, 2.7),
            db2.MoveToAngles(0, 0.5),
            db3.MoveToAngles(2.7, 3.14)
            )
        iter_1[1].set_value(0) 
        iter_2[1].set_value(0)
        iter_2[1].set_value(0)
        J_1[1].set_value(gd_traj1[0, 2])
        J_2[1].set_value(gd_traj1[0, 2])
        J_3[1].set_value(gd_traj1[0, 2])
        self.wait(0.5)

         # SLIDE 10: start the simulations
        self.next_slide(
            notes=
            '''On the other hand, we can fix the target = [0.75 -1] and choose  different initial conditions.
            (robot arm Gradient descent animation)
            '''
        )

        for i in range(100):
            anim = []
            run_time = 0.3* np.exp(-i/20)
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

        # NEXT SLIDE to ACtually start the animation

        self.next_slide(
            notes=
            '''Now it's your turn.
            Try collecting of computed distances in a vector and plot them as a function of  iteration.
            Also, think about what happens when the target is not reachable by the robot.
            '''
        )
        self.wait(0.5)
        '''END'''
        