from manim import *
from manim_slides import Slide
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mooc_utils import *

config.renderer='cairo'
config.background_color = WHITE
config.pixel_width=960
config.pixel_height=720
# config.pixel_width=1440
# config.pixel_height=1080



    

class CodeTest(Scene):
    def construct(self):
        DB = DynamicSplitScreen()
        pc1 = Text("Hello world", color=BLACK)
        pc2 = Text("Hello you too", color=BLACK)
        DB.add_main_obj(pc1)
        DB.add_side_obj(pc2)
        self.add(DB, pc1, pc2)
        self.wait()
        self.play(DB.bringIn())
        self.wait()
        self.play(DB.bringOut())
        self.wait()

class W3Slides_matlab(Slide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # COVER: MATLAB ENVIRONMENT APPEARS, PSEUDO CODE FADES IN ON TOP
        self.next_slide(
            notes=
            '''Let's open Matlab and start to learn how to code the Gradient 
            Descent algorithm. [CLICK]
            '''
        )
        mat_env = MatlabEnv(r'Assets\matlab.png')
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
        pc.scale_to_fit_width(FRAME_WIDTH*0.65).center()
        pc_rect = SurroundingRectangle(pc, color=WHITE, corner_radius=0.2, fill_opacity=1,
                                       stroke_color=BLACK, stroke_width=0.1)
        
        self.play(FadeIn(mat_env))
        self.wait(1)
        self.play(FadeIn(pc_rect, pc))

        # SLIDE 02:  ===========================================================
        # PSEUDO CODE FADES
        # CURSOR APPEARS ON 'NEW SCRIPT'
        self.next_slide(
            notes=
            '''By clicking on the New Script icon on the left [CLICK]...
            '''
        )
        self.play(FadeOut(pc_rect, pc))

        hand_cursor = Cursor().move_to(pixel2p(25, 437))  # new script position
        self.play(GrowFromCenter(hand_cursor))

        # SLIDE 03:  ===========================================================
        # 'NEW SCRIPT' IS CLICKED, NEW SCRIPT APPEARS
        # CURSOR MOVES TO 'SAVE' ICON
        self.next_slide(
            notes=
            ''' we open a new script file and save it [CLICK] as 
            "gradient_method.m".
            '''
        )

        self.play(Succession(
            hand_cursor.click(),
            mat_env.env_image.animate(run_time=0).become(ImageMobject(r'Assets\matlab.png').scale_to_fit_height(FRAME_HEIGHT).set_z_index(-1)),    #//
            lag_ratio=0.5
        ))
        self.wait(0.5)
        self.play(hand_cursor.animate.move_to(pixel2p(84, 290)))  # SAVE ICON POSITION
        
        # SLIDE 04:  ===========================================================
        # SAVE ICON IS CLICKED, SAVE PROMPT WINDOW APPEARS
        # ICON QUICKLY MOVES TO 'SAVE' AND CLICKS IT
        self.next_slide(
            notes=
            '''... as "gradient_method.m". [CLICK]
            '''
        )
        self.play(Succession(
            hand_cursor.click(),
            mat_env.env_image.animate(run_time=0).become(ImageMobject(r'Assets\matlab.png').scale_to_fit_height(FRAME_HEIGHT).set_z_index(-1)),
            lag_ratio=0.5
        ))
        self.wait(0.25)
        self.play(hand_cursor.animate.move_to(pixel2p(210, 105)))  # 'save' position
        self.play(Succession(
            hand_cursor.click(),
            mat_env.env_image.animate(run_time=0).become(ImageMobject(r'Assets\matlab.png').scale_to_fit_height(FRAME_HEIGHT).set_z_index(-1)),
            lag_ratio=0.5
        ))

        # SLIDE 05:  ===========================================================
        # HAND CURSOR FADES OUT
        self.next_slide(
            notes=
            '''We load in our working directory the file "plot_robot_arm.m" that
            we will use to plot the robot arm configuration.  This can be done by
            using the command [CLICK] ...
            '''
        )
        self.play(ShrinkToCenter(hand_cursor))
        # SLIDE 06:  ===========================================================
        # ADDPATH LINE IS WRITTEN
        self.next_slide(
            notes=
            '''... addpath /path/to/folder/, Where /path/to/folder/ is the path
            to the folder where plot_robot_arm.m is stored. [CLICK]
            '''
        )
        empty_cell = MatlabCodeBlock(code='')
        self.play(hand_cursor.click())
        mat_env.env_image.become(ImageMobject(r'Assets\colabGD.png').scale_to_fit_height(FRAME_HEIGHT).set_z_index(-1))
        self.add(mat_env)  # update change of background
        mat_env.add_cell(empty_cell)
        self.wait(0.2)
        self.play(mat_env.outof_colab(empty_cell), FadeOut(hand_cursor))
        self.wait(0.2)

        import_code = MatlabCode(
            r'''
            addpath /path/to/folder
            '''
        )
        input_data_code = MatlabCode(
            r'''
            % Definition of robotic arm variables and function
            target = [0.75 , -1];     % target point 
            L1 = 1;                   % length of arm1 
            L2 = 1.5;                 % lenght of arm2 
            angles = [2.5 2.7];       % inital angles 
            tol = 0.01;               % tolerance 
            alpha = 0.1;              % learning rate 
            Niter = 1000;             % max iteration number
            '''
        )
        import_code.window.become(empty_cell.colabCode.window)
        self.add(import_code.window)
        self.remove(empty_cell.colabCode.window)
        self.play(import_code.typeLetterbyLetter(lines=[0]))

        # SLIDE 07:  ===========================================================
        # PSEUDO CODE APPEARS, 
        # ONLY THE 'REQUIRE' LINE IS KEPT AND MOVES TO TOP, REST FADES OUT
        # NEW CODE WITH %DATA IS WRITTEN BELOW
        self.next_slide(
            notes=
            '''To translate our pseudo-code in Matlab we first define our input
            parameters: 
            '''
        )
        half_screen_rect = Rectangle(height=FRAME_HEIGHT, width=FRAME_HEIGHT*4/3 /2, color=WHITE, fill_opacity=1, stroke_width=0)
        half_screen_rect.to_edge(RIGHT, buff=0)
        left_rect = half_screen_rect.copy().set_color(WHITE).to_edge(LEFT, buff=0)
        
        pc.scale_to_fit_width(half_screen_rect.width*0.9).move_to(left_rect)
        self.play(
            import_code.window.animate.become(half_screen_rect),
            import_code.code.animate.scale_to_fit_width(half_screen_rect.width*0.9).move_to(half_screen_rect)
        )
        self.play(FadeIn(pc), import_code.code.animate.shift(UP*1.5))
        input_data_code.scale_to_fit_width(half_screen_rect.width*0.9
                ).next_to(import_code.code, DOWN).align_to(import_code.code, LEFT)
        self.play(input_data_code.typeLetterbyLetter(lines=[0]))

        # SLIDE 08:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''1 - the target point target = [0.75, -1] 
            [CLICK]
            '''
        )
        self.play(
            Indicate(pc[1][8:15]),
            input_data_code.typeLetterbyLetter(lines=[1])
            )
        
        # SLIDE 09:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''2 - the arm's length L1 = 1 and L2 = -1.5
            [CLICK]
            '''
        )
        self.play(
            Indicate(pc[1][16:21]),
            input_data_code.typeLetterbyLetter(lines=[2,3])
            )
        
        # SLIDE 11:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''3 - the initial guess for the angles in radians = [2.5, 2.7] 
            [CLICK]
            '''
        )
        self.play(
            Indicate(pc[1][22:31]),
            input_data_code.typeLetterbyLetter(lines=[4])
            )
        
        # SLIDE 12:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            ''' 4 - the tolerance for stopping the algorithm tol = 0.01 
            [CLICK]
            '''
        )
        self.play(
            Indicate(pc[1][32:35]),
            input_data_code.typeLetterbyLetter(lines=[5])
            )
    
        # SLIDE 13:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''5 - the learning rate alpha = 0.1
            [CLICK]
            '''
        )
        self.play(
            Indicate(pc[1][36]),
            input_data_code.typeLetterbyLetter(lines=[6])
            )
        
        # SLIDE 14:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''6 - The maximum number of iterations. Since the convergence in 
            not a-priori guaranteed, we fix the total iteration number Niter 
            equal to 1000. This prevents the algorithm from running endlessly.
            [CLICK]
            '''
        )
        self.play(
            Indicate(pc[1][38:]),
            input_data_code.typeLetterbyLetter(lines=[7])
            )
        
        # SLIDE 15:  ===========================================================
        # KINEMATICS FORMULAS ARE WRITTEN
        self.next_slide(
            notes=
            '''Now we want to compute the position of the robot according to 
            this formulas. To that end we define [CLICK]...
            '''
        )
        self.play(
            FadeOut(pc),
            FadeOut(import_code.code, input_data_code.code, 
                    shift=(import_code.code.height + input_data_code.code.height)*UP))
        kinematics_eq = VGroup(
            MathTex(r"x(\theta_1, \theta_2) = L_1 \cos(\theta_1) + L_2 \cos(\theta_2)", color=BLACK),
            MathTex(r"y(\theta_1, \theta_2) = L_1 \sin(\theta_1) + L_2 \sin(\theta_2)", color=BLACK)
            ).arrange(DOWN).scale_to_fit_width(pc.width).move_to(pc)
        kinematics_code=MatlabCode(
            r'''
            % Compute tip robot position
            function [x,y] = robot_position(theta,L1,L2)
                x = L1 * cos(theta(1)) + L2 * cos(theta(2));
                y = L1 * sin(theta(1)) + L2 * sin(theta(2));
            end

            % Plot robot arms
            [x,y] = robot_position(angles, L1, L2);
            fprintf('%f4.2 %f4.2 \n',[x,y]);
            plot_robot_arm(angles, target, [L1, L2]);
            '''
        )
        kinematics_code.window.become(half_screen_rect)
        self.add(kinematics_code.window)
        self.remove(import_code.window)
        kinematics_code.code.scale_to_fit_width(half_screen_rect.width*0.9).move_to(half_screen_rect)
        self.play(
            FadeIn(kinematics_eq),
            kinematics_code.typeLetterbyLetter(lines=[0])
            )
        
        # SLIDE 16:  ===========================================================
        # NEW BLOCK APPEARS, FUNCTION DEFINITION IS WRITTEN
        self.next_slide(
            notes=
            '''... the new function robot_position which takes
            the angles as input and returns the position in x-y plane. 
            Let us print the coordinates [CLICK] ...
            '''
        )
        self.play(
            kinematics_code.typeLetterbyLetter(lines=range(1,6))
            )
        
        # SLIDE 17:  ===========================================================
        # PRINT LINES ARE WRITTEN.
        # CAMERA ZOOMS OUT TO REVEAL COLAB ENVIRONMENT, RUN BUTTON IS CLICKED, 
        # OUTPUT APPEARS.
        self.next_slide(
            notes=
            '''... of the robot tip giving the initial guess for the angles and 
            visualize the corresponding robot position using the function we imported.
            [CLICK]
            '''
        )
        self.play(
            kinematics_code.typeLetterbyLetter(lines=[7,8,9])
            )
        self.wait(0.2)
        mat_env.remove_cell()
        mat_env.add_cell(MatlabCodeBlock(code=import_code.code_string + input_data_code.code_string))
        self.play(
            kinematics_code.into_colab(colab_env=mat_env),
            FadeOut(kinematics_eq)
            )
        self.play(mat_env.cells[-1].run())
        mat_env.cells[-1].add_output('[-2.1572518285725253, 1.2395419644547012]')
        self.add(mat_env.cells[-1].output, mat_env.cells[-1].outputWindow)

        
        # SLIDE 18:  ===========================================================
        # COLAB ENV FADES OUT
        # PSEUDO CODE FADES IN, WHILE LOOP HIGHLIGHTED.
        self.next_slide(
            notes=
            '''  Moving on, we now look at the iteration loop. [CLICK]
            '''
        )
        self.play(FadeOut(mat_env))
        self.play(FadeIn(pc.scale_to_fit_width(FRAME_HEIGHT*4/3*0.9).center()))
        self.wait(0.3)
        self.play(Circumscribe(pc[2:]))

        # SLIDE 19:  ===========================================================
        # HIGHLIGHT J AND NABLA J IN PSEUDO CODE.
        # FOURMULAS FOR J ABD NABLA J APPEAR
        self.next_slide(
            notes=
            '''  All we need is an update for the vector angles and an evaluation
            of two functions J and Nabla J. To that end we create two functions.
            [CLICK]  
            '''
        )
        J_pc = pc[5][4:14].copy()
        nabla_J_pc = pc[4][-15:].copy()
        J_eq = MathTex(r'{{J(\theta_1, \theta_2)}} = (x(\theta_1, \theta_2)-x_p)^2 + (y(\theta_1, \theta_2)-y_p)^2',
                         color=BLACK)
        nabla_J_eq = VGroup(
            MathTex(r"{{\nabla J(\theta_1, \theta_2)}}", "=", color=BLACK),
            Matrix([
            [r"{{\frac{\partial J}{\partial x}}} {{\frac{\partial x}{\partial \theta_1}}} {{+}}" 
             r"{{\frac{\partial J}{\partial y}}} {{\frac{\partial y}{\partial \theta_1}}}"],
            [r"{{\frac{\partial J}{\partial x}}} {{\frac{\partial x}{\partial \theta_2}}} {{+}}" 
             r"{{\frac{\partial J}{\partial y}}} {{\frac{\partial y}{\partial \theta_2}}}"]
            ], v_buff= 1.5)).set_color(BLACK).arrange_in_grid(1,2).next_to(J_eq, DOWN)
        VGroup(J_eq, nabla_J_eq).center()

        self.play(AnimationGroup(
            Indicate(pc[5][4:14]),  # J
            Indicate(pc[4][-15:])   # nabla J
        ))
        self.add(J_pc, nabla_J_pc)
        self.play(AnimationGroup(
            FadeOut(pc),
            AnimationGroup(Transform(nabla_J_pc, nabla_J_eq[0][0]), Transform(J_pc, J_eq[0])),
            FadeIn(J_eq[1:], nabla_J_eq[0][1], nabla_J_eq[1]),
            lag_ratio=0.6
        ))
        self.remove(J_pc, nabla_J_pc)    

        # SLIDE 20:  ===========================================================
        # SCREEN SPLIT BETWEEN FORMULAS AND CODE
        # DEFINITION OF J IS WRITTEN
        self.next_slide(
            notes=
            '''For the function J we use the following syntax.
            Using the robot_position function we can compute the value of J
            in this way, and print the value of J providing the initial guess
            for the target. [CLICK]
            '''
        )
        J_code = MatlabCode(
            r'''
            % Evaluation of J
            function Jval = J(theta, target, L1, L2) 
                [x, y] = robot_position(theta, L1, L2);
                Jval = (x-target(1))^2 + (y-target(2))^2;
            end

            Jval = J(angles, target, L1, L2);
            fprintf('%f4.2 \n',Jval);

            '''
        )
        J_code.window.become(half_screen_rect)
        J_code.code.scale_to_fit_width(half_screen_rect.width*0.9).move_to(half_screen_rect)
        self.play(AnimationGroup(
            AnimationGroup(
                J_eq.animate.move_to(left_rect).scale_to_fit_width(left_rect.width*0.9),
                FadeOut(nabla_J_eq)),
            FadeIn(half_screen_rect, J_code.code[0]),
            lag_ratio=0.5
            ))
        
        self.play(J_code.typeLetterbyLetter(lines=range(1,7)))
        mat_env.clear()
        self.add(J_code.window)
        self.remove(half_screen_rect)
        self.play(AnimationGroup(
            J_code.into_colab(mat_env),
            FadeOut(J_eq)))
        self.play(mat_env.cells[-1].run())
        mat_env.cells[-1].add_output(output = '13.467661405291913' )
        self.add(mat_env.cells[-1].output, mat_env.cells[-1].outputWindow)

        
        # SLIDE 21:  ===========================================================
        # WRITE FIRST LINES OF FUNCION GRAD_J
        self.next_slide(
            notes=
            '''Next, we define a Python function called "grad_J" to compute the 
            gradient of J at a given point, which takes in input the angles theta_1
            and theta_2  and outputs the values dJ_dt1 and dJ_dt2.
            Here, we use the analytical derivation of the gradient of J.
            For instance, 
            '''
        )
        nabla_J_code = MatlabCode(
            r'''
            % Evaluation of grad(J)
            function grad = grad_J(theta, target, L1, L2)
                [x, y] = robot_position(theta,L1,L2);

                dx_dt1 = - L1 * sin(theta(1));
                dx_dt2 = - L2 * sin(theta(2));
                dy_dt1 = L1 * cos(theta(1));
                dy_dt2 = L2 * cos(theta(2));

                dJ_dx = 2*(x - target(1));
                dJ_dy = 2*(y - target(2));
                
                DJ_dt1 = dJ_dx*dx_dt1 + dJ_dy*dy_dt1;
                DJ_dt2 = dJ_dx*dx_dt2 + dJ_dy*dy_dt2; 
                grad = [DJ_dt1, DJ_dt2];
                end
            '''
        )
        self.play(FadeOut(mat_env))
        nabla_J_eq.scale_to_fit_width(half_screen_rect.width*0.9).move_to(left_rect)
        nabla_J_code.scale_to_fit_width(half_screen_rect.width*0.9).move_to(half_screen_rect).shift(DOWN*0.5)
        nabla_J_code.window.become(half_screen_rect)
        self.play(AnimationGroup(
            FadeIn(nabla_J_code.window),
            FadeIn(nabla_J_eq, shift=UP*2),
            FadeIn(nabla_J_code.code[0], shift=UP*2)
        ))
        self.play(nabla_J_code.typeLetterbyLetter(lines=range(1, 3)))
        self.play(nabla_J_code.typeLetterbyLetter(lines=range(4, 8), lag_ratio=0))
        self.play(nabla_J_code.typeLetterbyLetter(lines=range(9, 11), lag_ratio=0))
    
        # SLIDE 22:  ===========================================================
        # PART OF NABLA J FORMULA AND CORRESPONDING CODE IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''... , the partial derivative of x with respect to theta_1
            is coded this a way[CLICK]
            '''
        )
        # 1->matrix; 0->content of matrix; 0 or 1-> first or second row; 2 or 7 -> nonsense
        # 16-> avoid some kinda of buffer?
        highlight_colors = [YELLOW, GREEN, PURPLE, ORANGE]
        pairs_to_highlight =  [
            [nabla_J_eq[1][0][0][2], nabla_J_code.code[4][16:]],
            [nabla_J_eq[1][0][0][7], nabla_J_code.code[5][16:]],
            [nabla_J_eq[1][0][1][2], nabla_J_code.code[6][16:]],
            [nabla_J_eq[1][0][1][7], nabla_J_code.code[7][16:]],
        ]
        highlight_rect = [
            BackgroundRectangle(pairs_to_highlight[j][i], color=highlight_colors[j], fill_opacity=0.4, buff=0.05,corner_radius=0.1)#.set_z_index(-0.5)
            for j in range(4) for i in range(2)]
        
        self.play(AnimationGroup(
            Create(highlight_rect[0]),
            Create(highlight_rect[1])
        ))

        # SLIDE 22:  ===========================================================
        # PART OF NABLA J FORMULA AND CORRESPONDING CODE IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''... , and similarly the other terms. [CLICK]
            '''
        )
        self.play(AnimationGroup(*[Create(highlight_rect[i]) for i in range(2, 8)]))

        # SLIDE 22:  ===========================================================
        # OTHER PARTS OF NABLA J FORMULA AND CORRESPONDING CODE ARE HIGHLIGHTED
        self.next_slide(
            notes=
            '''Finally, we use them to write the two components of the gradient
            of J. [CLICK]
            '''
        )
        self.play(nabla_J_code.typeLetterbyLetter(lines=range(12, 14), lag_ratio=0))
        self.play(nabla_J_code.typeLetterbyLetter(lines=[15]))

        # SLIDE 23:  ===========================================================
        # PSEUDO CODE FADES BACK IN CENTERED
        # THEN MOVES TO LEFT AND CODE BLOCK APPEARS
        self.next_slide(
            notes=
            '''Now we have all the ingredients to code the iteration loop. [CLICK]
            '''
        )
        self.play(FadeOut(nabla_J_eq, half_screen_rect, nabla_J_code, *highlight_rect))
        self.play(FadeIn(pc))
        GD_code = MatlabCode(
            r'''
            % Gradient Descent Method 
            i = 1;
            while i <= Niter
                % Compute the gradient of the objective function
                grad = grad_J(angles, target, L1, L2);
                
                % Update theta with gradient 
                angles = angles - alpha * grad;  

                % Compute the current distance
                Jval = J(angles, target, L1, L2);

                % Check for convergence
                if Jval < tol
                    fprintf('%s %d %s \n','Converged after', i, 'iterations.')
                    break
                end
                i = i + 1;
            end

            fprintf('%s %f4.2 %f4.2 \n','Final angle combination: ', angles) 
            fprintf('%s %f4.2 \n','Final distance: ', Jval) 
            plot_robot_arm(angles, target, [L1, L2]);
            ''')
        GD_code.window.become(half_screen_rect)
        GD_code.code.scale_to_fit_width(half_screen_rect.width*0.9).move_to(half_screen_rect).shift(DOWN)
        

        # SLIDE 24:  ===========================================================
        # WHILE LOOP INSTRUCTIONS HIGHLIGHTED
        self.next_slide(
            notes=
            '''For each iteration of the while loop we perform the following
            instructions: [CLICK]
            '''
        )
        self.play(AnimationGroup(
            pc.animate.scale_to_fit_width(half_screen_rect.width*0.9).move_to(left_rect),
            FadeIn(GD_code.window),
            FadeIn(GD_code.code[0], shift=UP)
        ))
        highlight_pairs_2 = [
            [pc[3][2:], pc[7][2:], GD_code.code[2][12:]],
            [pc[4][-15:], GD_code.code[4][16:]],
            [pc[4][2:], VGroup(GD_code.code[7][16:], GD_code.code[8][16:])],
            [pc[5][4:18], GD_code.code[11][16:], GD_code.code[14][16:]]
        ]
        highlight_rect_2 = [
            BackgroundRectangle(highlight_pairs_2[j][i], color=highlight_colors[j], fill_opacity=0.4, buff=0.05,corner_radius=0.1)#.set_z_index(-0.5)
            for j in range(4) for i in range(len(highlight_pairs_2[j]))
        ]

        self.play(GD_code.typeLetterbyLetter(lines=[1,2]))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in range(3)]))

        # SLIDE 25:  ===========================================================
        # HIGLIGHT NABLA J COMPUTATION
        self.next_slide(
            notes=
            '''1 - Compute the gradient using the current angles and the target
            point [CLICK]
            '''
        )
        self.play(GD_code.typeLetterbyLetter(lines=[3,4]))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in range(3, 5)]))

        # SLIDE 26:  ===========================================================
        # HIGLIGHT ANGLES UPDATE
        self.next_slide(
            notes=
            '''2. Update the angles. pay attention! Here we use the short hand
            notation for the updates, which is equivalent to this one. [CLICK]
            '''
        )
        self.play(GD_code.typeLetterbyLetter(lines=[6,7,8]))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in range(5,7)]))

        # SLIDE 28:  ===========================================================
        # TRANSFORM CORRESPONDING LINES IN EXTENDED FORM AND BACK
        self.next_slide(
            notes=
            '''...to this one. [CLICK]
            '''
        )
        explict_update_code = MatlabCode(
            '''
            angles[0] = angles[0] - alpha * grad[0]
            angles[1] = angles[1] - alpha * grad[1]
            '''
        )
        two_lines = VGroup(GD_code.code[7][16:], GD_code.code[8][16:])
        two_lines.save_state()
        highlight_rect_2[6].save_state()
        explict_update_code.code.scale_to_fit_height(two_lines.height).move_to(two_lines).align_to(two_lines, LEFT)
        target_high_rect = BackgroundRectangle(explict_update_code.code, color=highlight_colors[2], fill_opacity=0.4, buff=0.05,corner_radius=0.1)

        self.play(AnimationGroup(
            Transform(two_lines, explict_update_code.code, run_time=0.3),
            highlight_rect_2[6].animate(run_time=0.3).become(target_high_rect)
            ))
        self.wait(0.2)
        # SLIDE 29:  ===========================================================
        # HIGLIGHT STOPPING CRITERION
        self.next_slide(
            notes=
            '''3 - Perform the stopping criterion and print the number of 
            iterations reached at this point.
            To check the output of our algorithm we can print [CLICK] ...
            '''
        )
        self.play(AnimationGroup(two_lines.animate(run_time=0.3).restore(), highlight_rect_2[6].animate(run_time=0.3).restore()))
        self.play(GD_code.typeLetterbyLetter(lines=range(10,19)))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in [7,8,9]]))

        # SLIDE 30:  ===========================================================
        # CODE SCROLLS UP, PRINT LINES ARE WRITTEN
        self.next_slide(
            notes=
            '''... the final angles combination, the computed distance and the 
            final arm's configuration. [CLICK]
            '''
        )
        self.play(VGroup(GD_code.code[:19], *[highlight_rect_2[i] for i in [2,4,6,8,9]]).animate.shift(UP))
        GD_code.code[19:].shift(UP)
        self.play(GD_code.typeLetterbyLetter(lines=[20,21,22], lag_ratio=0))
        

        # SLIDE 31:  ===========================================================
        # ZOOM OUT TO COLAB, PRESS RUN, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''Let us run the script and comment the outputs: [CLICK]
            '''
        )
        mat_env.clear()
        self.play(FadeOut(*highlight_rect_2, pc))
        self.play( GD_code.into_colab(mat_env))
        self.play(mat_env.cells[-1].run())
        final_output_text = ColabBlockOutputText(
            'Converged after 38 iterations.\n'
            'Final angle combination:  0.6293184.2 4.6875734.2 \n'
            'Final distance:  0.0083814.2'
        )
        final_output_img = ImageMobject(r'Assets\W3\RobotArmEnd_mat.png').scale(0.5).next_to(final_output_text, DOWN).align_to(final_output_text, LEFT)
        mat_env.cells[-1].add_output(
            Group(final_output_text, final_output_img)
        )
        self.add(mat_env.cells[-1].output)
        self.play(mat_env.focus_output(mat_env.cells[-1]))

        # SLIDE 32:  ===========================================================
        # HIGLIGHT FIRST LINE
        self.next_slide(
            notes=
            '''1 - the method converges in 38 iterations [CLICK]
            '''
        )
        self.play(Circumscribe(final_output_text[0], run_time=2, time_width=0.6))

        # SLIDE 33:  ===========================================================
        # HIGLIGHT SECOND LINE
        self.next_slide(
            notes=
            '''2 - The numerical solution is the pair of angles [0.62, 4.68] [CLICK]
            '''
        )
        self.play(Circumscribe(final_output_text[1], run_time=2, time_width=0.6))

        # SLIDE 34:  ===========================================================
        # HIGLIGHT THIRD LINE
        self.next_slide(
            notes=
            '''The final computed squared distance J between the arm's tip and
            the target point is 0.008. note that it is smaller than the tolerance.
            The method works! We found a pair of angles to reach the target point.
            [CLICK]
            '''
        )
        self.play(Circumscribe(final_output_text[2], run_time=2, time_width=0.6))

        # SLIDE 35:  ===========================================================
        self.next_slide(
            notes=
            '''END
            '''
        )
