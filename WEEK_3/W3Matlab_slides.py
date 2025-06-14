import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from config import *
from Generic_mooc_utils import *
from matlab_utils import *
from colab_utils import COLAB_LIGHTGRAY
from W3Anim import NewDB

config.update(RELEASE_CONFIG)

class W3Slides_matlab(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # COVER: MATLAB ENVIRONMENT APPEARS, PSEUDO CODE FADES IN ON TOP
        self.next_slide(
            notes=
            '''Let's open Matlab and start to learn how to code the Gradient 
            Descent algorithm. [CLICK]
            '''
        )
        mat_env = MatlabEnv(r'Assets\matlab_noscript.png')
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
        pc.save_state()
        surrounding_pc = SurroundingRectangle(pc, fill_color=WHITE, fill_opacity=1, stroke_width=0.5,
                                              stroke_color=BLACK, corner_radius=0.2, buff=0.5).set_z_index(-0.5)
        self.play(FadeIn(mat_env))
        self.wait(1)
        self.play(FadeIn(surrounding_pc, pc))

        # SLIDE 02:  ===========================================================
        # PSEUDO CODE FADES
        # CURSOR APPEARS ON 'NEW SCRIPT'
        self.next_slide(
            notes=
            '''By clicking on the New Script icon on the left [CLICK]...
            '''
        )
        self.play(FadeOut(surrounding_pc, pc))
        hand_cursor = Cursor() # new script position
        self.wait(1)
        self.play(GrowFromCenter(hand_cursor))

        # SLIDE 03:  ===========================================================
        # 'NEW SCRIPT' IS CLICKED, NEW SCRIPT APPEARS
        # CURSOR MOVES TO 'SAVE' ICON
        self.next_slide(
            notes=
            ''' we open a new script file and [CLICK] we can ... 
            '''
        )
        self.play(hand_cursor.animate.move_to(mat_env.NEW_SCRIPT_))
        self.play(hand_cursor.Click())
        mat_env.set_image(r'Assets\matlab.png')
        self.wait(0.1)
        
        # SLIDE 04:  ===========================================================
        # SAVE ICON IS CLICKED, SAVE PROMPT WINDOW APPEARS
        # ICON QUICKLY MOVES TO 'SAVE' AND CLICKS IT
        self.next_slide(
            notes=
            '''... save it as "gradient_method.m". [CLICK]
            '''
        )
        self.play(hand_cursor.animate.move_to(mat_env.SAVE_))  # SAVE ICON POSITION
        self.play(hand_cursor.Click())
        mat_env.set_image(r'Assets\W3\matlab_save_prompt_gd.png'),
        self.wait(0.25)
        self.play(hand_cursor.animate.move_to(mat_env.SAVE_PROMPT_BUTTON_))  # 'save' button in prompt window
        self.play(hand_cursor.Click())
        mat_env.set_image(r'Assets\W3\matlab_saved_gd.png'),

        # SLIDE 05:  ===========================================================
        # HAND CURSOR FADES OUT
        self.next_slide(
            notes=
            '''We need the function "plot_robot_arm.m" to plot the robot arm
            configuration.  We need to tell matlab where it is, using the
            command [CLICK] ...
            '''
        )
        self.play(ShrinkToCenter(hand_cursor))

        # SLIDE 06:  ===========================================================
        # ADDPATH LINE IS WRITTEN
        self.next_slide(
            notes=
            '''..."addpath" specifying the path to the folder where the function
            is stored. [CLICK]
            '''
        )
        empty_cell = MatlabCodeBlock(code='')
        mat_env.add_cell(empty_cell)
        self.play(mat_env.OutofMatlab(empty_cell))
        self.wait(0.2)

        init_code = MatlabCode(
            r'''
            addpath('path/to/folder')

            % Input parameters 
            xp = [0.75 , -1];     % target point 
            L1 = 1;               % length of arm1 
            L2 = 1.5;             % length of arm2 
            theta = [2.5 2.7];    % initial angles 
            tol = 0.01;           % tolerance 
            alpha = 0.1;          % learning rate 
            Niter = 1000;         % max iteration number
            '''
        )
        DSS = DynamicSplitScreen(main_color=WHITE, side_color=COLAB_LIGHTGRAY)
        self.add(DSS)
        self.remove(empty_cell.window)
        RequireLine = pc[1]
        DSS.add_side_obj(RequireLine)
        init_code.move_to(DSS.get_final_mainObj_pos())
        import_code = init_code[0]
        import_code.save_state()
        import_code.shift(DOWN)
        self.play(init_code.TypeLetterbyLetter(lines=[0]))

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
        self.play(DSS.bringIn(), import_code.animate.restore())
        self.wait(0.5)
        self.play(init_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 08:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''1 - the target point xp with its two components; [CLICK]
            '''
        )
        # dictionary to make it more legible
        input_data_words = {
            'target' : RequireLine[8:15],
            'arm_length' : RequireLine[16:21],
            'initial_guess': RequireLine[22:31],
            'tolerance': RequireLine[32:35],
            'alpha': RequireLine[36],
            'max_iter' : RequireLine[38:]
        }
        input_data_high_rect = {word : HighlightRectangle(input_data_words[word]) for word in input_data_words.keys()}
        
        self.play(
            Create(input_data_high_rect['target']),
            init_code.TypeLetterbyLetter(lines=[3])
        )
        # SLIDE 09:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''2 - the arms' lengths L1 and L2; [CLICK]
            '''
        )
        self.play(
            ReplacementTransform(input_data_high_rect['target'], input_data_high_rect['arm_length'], run_time=1),
            init_code.TypeLetterbyLetter(lines=[4, 5])
        )
        
        # SLIDE 10:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''3 - the initial guess for the angles in radians; [CLICK]
            '''
        )
        self.play(
            ReplacementTransform(input_data_high_rect['arm_length'], input_data_high_rect['initial_guess'], run_time=1),
            init_code.TypeLetterbyLetter(lines=[6])
        )
        
        # SLIDE 11:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            ''' 4 - the tolerance for stopping the algorithm "tol"; [CLICK]
            '''
        )
        self.play(
            ReplacementTransform(input_data_high_rect['initial_guess'], input_data_high_rect['tolerance'], run_time=1),
            init_code.TypeLetterbyLetter(lines=[7])
        )
    
        # SLIDE 12:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''5 - the learning rate alpha; [CLICK]
            '''
        )
        self.play(
            ReplacementTransform(input_data_high_rect['tolerance'], input_data_high_rect['alpha'], run_time=1),
            init_code.TypeLetterbyLetter(lines=[8])
        )
        
        # SLIDE 13:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''6 - The maximum number of iterations. Since the convergence in 
            not a-priori guaranteed, we fix the total number of iterations to
            1000. [CLICK]
            '''
        )
        self.play(
            ReplacementTransform(input_data_high_rect['alpha'], input_data_high_rect['max_iter'], run_time=1),
            init_code.TypeLetterbyLetter(lines=[9])
        )
        
        # SLIDE 14:  ===========================================================
        # ZOOM OUT TO MATLAB
        self.next_slide(
            notes=
            ''' This prevents the algorithm from running endlessly. [CLICK]
            '''
        )
        init_code.add_background_window(DSS.mainRect.suspend_updating())
        mat_env.clear()
        self.play(
            init_code.IntoMatlab(mat_env),
            DSS.bringOut(),
            input_data_high_rect['max_iter'].animate.shift(UP*DSS.secondaryRect.height)
        )
        self.remove(input_data_high_rect['max_iter'])  # take care of the last highlight

        # SLIDE 15:  ===========================================================
        # KINEMATICS FORMULAS ARE WRITTEN
        self.next_slide(
            notes=
            '''We also need to be able to compute the tip position, given the
            angles, as in these formulas. To do it we define [CLICK]...
            '''
        )
        kinematics_eq = VGroup(
            MathTex(r"x(\theta_1, \theta_2) = L_1 \cos(\theta_1) + L_2 \cos(\theta_2)", color=BLACK),
            MathTex(r"y(\theta_1, \theta_2) = L_1 \sin(\theta_1) + L_2 \sin(\theta_2)", color=BLACK)
            ).arrange(DOWN)
        kinematics_code=MatlabCode(
            r'''
            % Compute tip robot position
            function [x,y] = robot_position(theta, L1, L2)
                x = L1 * cos(theta(1)) + L2 * cos(theta(2));
                y = L1 * sin(theta(1)) + L2 * sin(theta(2));
            end

            % Plot robot arms
            [x,y] = robot_position(theta, L1, L2);
            fprintf('%f4.2 %f4.2 \n',[x,y]);
            plot_robot_arm(theta, xp, [L1, L2]);
            '''
        )
        DSS.reset()
        DSS.add_side_obj(kinematics_eq)
        kinematics_code.move_to(DSS.mainRect)
        self.play(FadeIn(DSS.mainRect))
        self.play(
            AnimationGroup(
                DSS.bringIn(),
                kinematics_code.TypeLetterbyLetter(lines=[0])
            )
        )
        self.add(mat_env); self.remove(mat_env)  # Add so that remove is in self.mobjects, so that remove actually works.

        # SLIDE 16:  ===========================================================
        # NEW BLOCK APPEARS, FUNCTION DEFINITION IS WRITTEN
        self.next_slide(
            notes=
            '''... the new function robot_position which takes the angles as
            input and returns the position in x-y plane. We can print the
            coordinates [CLICK] ...
            '''
        )
        self.play(kinematics_code.TypeLetterbyLetter(lines=range(1,5)))
        
        # SLIDE 17:  ===========================================================
        # PRINT LINES ARE WRITTEN.
        # CAMERA ZOOMS OUT TO REVEAL COLAB ENVIRONMENT
        self.next_slide(
            notes=
            '''... of the robot tip given the initial angles and, using the
            function we imported, [CLICK] ...
            '''
        )
        self.play(kinematics_code.TypeLetterbyLetter(lines=range(6, 10)))
        self.wait(0.2)
        kinematics_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(
            kinematics_code.IntoMatlab(mat_env),
            DSS.bringOut()
        )
        
        # SLIDE 18:  ===========================================================
        # RUN BUTTON IS CLICKED, OUTPUT APPEARS.
        self.next_slide(
            notes=
            '''...we can visualize the corresponding robot position. [CLICK]
            '''
        )
        mat_env.add_output(
            output_text='[-2.1572518285725253, 1.2395419644547012]', 
            output_image=r'Assets\W3\RobotArmStart_mat.png'
        )
        self.play(mat_env.Run())
        
        # SLIDE 19:  ===========================================================
        # MATLAB ENV FADES OUT
        # PSEUDO CODE FADES IN, WHILE LOOP HIGHLIGHTED.
        self.next_slide(
            notes=
            '''Moving on, we now look at the iteration loop. [CLICK]
            '''
        )
        self.play(FadeOut(mat_env))
        self.clear()
        self.play(FadeIn(pc.restore().scale_to_fit_width(FRAME_WIDTH*0.9).center()))
        self.wait(0.3)
        while_loop_highlight = HighlightRectangle(pc[2:])
        self.play(Create(while_loop_highlight))
       
        # SLIDE 20:  ===========================================================
        # HIGHLIGHT J AND NABLA J IN PSEUDO CODE.
        # FOURMULAS FOR J ABD NABLA J APPEAR
        self.next_slide(
            notes=
            '''Notice that at each iteration we will need an evaluation of the
            function J and its gradient. To this end, let's implement two
            functions. [CLICK]
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

        self.play(FadeOut(while_loop_highlight))
        self.play(AnimationGroup(
            Indicate(pc[5][4:14], run_time=1, color=BLUE),  # J
            Indicate(pc[4][-15:], run_time=1, color=BLUE)   # nabla J
        ))
        self.add(J_pc, nabla_J_pc)
        self.wait(0.5)
        self.play(AnimationGroup(
            FadeOut(pc),
            AnimationGroup(Transform(nabla_J_pc, nabla_J_eq[0][0]), Transform(J_pc, J_eq[0])),
            FadeIn(J_eq, nabla_J_eq),
            lag_ratio=0.6
        ))
        self.remove(J_pc, nabla_J_pc)   

        # SLIDE 21:  ===========================================================
        # SCREEN SPLIT BETWEEN FORMULAS AND CODE
        # DEFINITION OF J IS WRITTEN + ROBOT POSITION COMPUTATION
        self.next_slide(
            notes=
            '''For the function J we use the following syntax. Using the
            robot_position function we can compute the position (x, y), [CLICK]
            '''
        )
        J_code = MatlabCode(
            r'''
            % Evaluation of J
            function Jval = J(theta, xp, L1, L2) 
                [x, y] = robot_position(theta, L1, L2);
                Jval = (x-xp(1))^2 + (y-xp(2))^2;
            end

            Jval = J(theta, xp, L1, L2);
            fprintf('%f4.2 \n',Jval);
            '''
        )
        DSS.reset()
        DSS.add_side_obj(J_eq.copy())
        DSS.bring_in()
        J_code.move_to(DSS.mainRect)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(nabla_J_eq),
                    FadeIn(DSS.mainRect, DSS.secondaryRect),
                    ReplacementTransform(J_eq, DSS.secondaryObj)
                ),
                J_code.TypeLetterbyLetter(lines=[0]),
                lag_ratio=1
            )
        )
        self.play(J_code.TypeLetterbyLetter(lines=[1, 2]))

        # SLIDE 22:  ===========================================================
        # J COMPUTATION IS WRITTEN
        self.next_slide(
            notes=
            '''...so that we can compute the value of J using pythagoras
            theorem. [CLICK]
            '''
        )
        self.play(J_code.TypeLetterbyLetter(lines=[3, 4]))

        # SLIDE 23:  ===========================================================
        # PRINT CODE IS WRITTEN
        # INTO MATLAB AND RUN CODE
        self.next_slide(
            notes=
            '''...and, if we want to test the function, we can print the value
            of J for the initial angles. It's of course very high. [CLICK]
            '''
        )
        self.play(J_code.TypeLetterbyLetter(lines=[6, 7]))
        J_code.add_background_window(DSS.mainRect.suspend_updating())
        mat_env.clear()
        self.play(
            J_code.IntoMatlab(mat_env),
            DSS.bringOut()
        )
        mat_env.add_output(output_text = '13.467661405291913' )
        self.play(mat_env.Run())
        
        # SLIDE 24:  ===========================================================
        # WRITE FIRST LINES OF FUNCION GRAD_J
        self.next_slide(
            notes=
            '''Next, we define a Python function called "grad_J" which takes as
            input the angles theta_1 and theta_2  and outputs the the two
            components of the gradient. Given x and y as before, [CLICK]
            '''
        )
        nabla_J_code = MatlabCode(
            r'''
            % Evaluation of grad(J)
            function grad = grad_J(theta, xp, L1, L2)
                [x, y] = robot_position(theta,L1,L2);

                dx_dt1 = - L1 * sin(theta(1));
                dx_dt2 = - L2 * sin(theta(2));
                dy_dt1 =   L1 * cos(theta(1));
                dy_dt2 =   L2 * cos(theta(2));

                dJ_dx = 2*(x - xp(1));
                dJ_dy = 2*(y - xp(2));
                
                DJ_dt1 = dJ_dx*dx_dt1 + dJ_dy*dy_dt1;
                DJ_dt2 = dJ_dx*dx_dt2 + dJ_dy*dy_dt2; 
                grad = [DJ_dt1, DJ_dt2];
            end
            '''
        )
        DSS.reset()
        DSS.add_side_obj(nabla_J_eq.scale(0.6))
        nabla_J_code.move_to(DSS.get_final_mainObj_pos())
        self.play(FadeIn(DSS.mainRect))
        self.add(mat_env) ; self.remove(mat_env)  # remove mat_env from behind
        self.play(
            AnimationGroup(
                DSS.bringIn(),
                nabla_J_code.TypeLetterbyLetter(lines=[0])
            )
        )
        self.play(nabla_J_code.TypeLetterbyLetter(lines=range(1, 3)))
    
        # SLIDE 25:  ===========================================================
        # CODE FOR PARTIAL DERIVATIVES WRITTEN
        self.next_slide(
            notes=
            '''...We compute the partial derivatives needed, [CLICK] ...
            '''
        )
        self.play(nabla_J_code.TypeLetterbyLetter(lines=range(4, 8), lag_ratio=0))
        self.play(nabla_J_code.TypeLetterbyLetter(lines=range(9, 11), lag_ratio=0))

        # SLIDE 26:  ===========================================================
        # PART OF NABLA J FORMULA AND CORRESPONDING CODE IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''For instance, the partial derivative of x with respect to theta_1
            is coded this a way, [CLICK] ...
            '''
        )
        # 1->matrix; 0->content of matrix; 0 or 1-> first or second row; 2 or 7 -> nonsense
        # 16-> avoid some kinda of buffer?
        highlight_colors = [BLUE, TEAL, ORANGE, PINK]
        nabla_J_highlights = [
            {'eq': nabla_J_eq[1][0][0][2], 'code': nabla_J_code.code[4][16:]}, # dxdt1
            {'eq': nabla_J_eq[1][0][0][7], 'code': nabla_J_code.code[5][16:]}, # dydt1
            {'eq': nabla_J_eq[1][0][1][2], 'code': nabla_J_code.code[6][16:]}, # dxdt2
            {'eq': nabla_J_eq[1][0][1][7], 'code': nabla_J_code.code[7][16:]}  # dydt2
        ]
        nabla_J_rectangle_highlights = [
            HighlightRectangle(nabla_J_highlights[i][t], color = highlight_colors[i])
            for i in range(4) for t in ['eq', 'code'] 
        ]
        
        self.play(
            Create(nabla_J_rectangle_highlights[0]),
            Create(nabla_J_rectangle_highlights[1])
        )

        # SLIDE 27:  ===========================================================
        # OTHER PARTS OF NABLA J FORMULA AND CORRESPONDING CODE ARE HIGHLIGHTED
        self.next_slide(
            notes=
            '''... and similarly the other terms. [CLICK]
            '''
        )
        self.play(AnimationGroup(*[Create(r) for r in nabla_J_rectangle_highlights[2:]]))

        # SLIDE 28:  ===========================================================
        # LINES FOR COMPONENTS OF THE GRADIENT ARE WRITTEN
        self.next_slide(
            notes=
            '''Finally, we use all these terms to write the two components of
            the gradient of J. [CLICK]
            '''
        )
        self.play(nabla_J_code.TypeLetterbyLetter(lines=range(12, 14), lag_ratio=0))
        self.play(nabla_J_code.TypeLetterbyLetter(lines=[14, 15]))
        self.wait(0.1)

        # SLIDE 29:  ===========================================================
        # PSEUDO CODE FADES BACK IN CENTERED
        self.next_slide(
            notes=
            '''Now we have all the ingredients to code the iteration loop. [CLICK]
            '''
        )
        self.play(FadeOut(nabla_J_eq, DSS, nabla_J_code, *nabla_J_rectangle_highlights))
        self.play(FadeIn(pc))
        GD_code = MatlabCode(
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
                    fprintf('Converged after %d iterations.\n', i)
                    break
                end
                i = i + 1;
            end

            fprintf('Final angle combination: %f4.2 %f4.2 \n', theta) 
            fprintf('Final distance: %f4.2 \n', Jval) 
            plot_robot_arm(theta, xp, [L1, L2]);
            ''')

        # SLIDE 30:  ===========================================================
        # WHILE LOOP MOVES TO TOP
        # FIRST LINE 0F PSEUDO CODE WRITTEN
        # HIGLIGHT WHILE INSTRUCTION IN PSEUDO CODE AND CODE
        self.next_slide(
            notes=
            '''We implement a while loop: we set the iteration counter i to one,
            and while i is smaller than the maximum number of iterations we
            perform these instructions: [CLICK]
            '''
        )
        DSS.reset()
        DSS.add_side_obj(pc[2:].copy().scale(0.4))
        DSS.bring_in()
        GD_code.scale(0.85).move_to(DSS.get_final_mainObj_pos()).shift(DOWN*0.5)
        self.play(
            Succession(
                AnimationGroup(
                    FadeOut(pc[:2]),
                    AnimationGroup(
                        FadeIn(DSS.mainRect, DSS.secondaryRect),
                        ReplacementTransform(pc[2:], DSS.secondaryObj)
                    ),
                    lag_ratio=0.5
                ),
                GD_code.TypeLetterbyLetter(lines=[0])
            )
        )
        highlight_pairs_2 = [
            [pc[3][2:], pc[7][2:], GD_code.code[2][12:]],
            [pc[4][-15:], GD_code.code[4][16:]],
            [pc[4][2:], GD_code.code[7][16:],],
            [pc[5][4:18], GD_code.code[10][16:], GD_code.code[13][16:]]
        ]
        highlight_rect_2 = [
            HighlightRectangle(highlight_pairs_2[j][i], color=highlight_colors[j])
            for j in range(4) for i in range(len(highlight_pairs_2[j]))
        ]

        self.play(GD_code.TypeLetterbyLetter(lines=[1,2]))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in range(3)]))

        # SLIDE 31:  ===========================================================
        # HIGLIGHT NABLA J COMPUTATION
        self.next_slide(
            notes=
            '''1 - Compute the gradient using the current angles and the target
            point with the corresponding function [CLICK]
            '''
        )
        self.play(GD_code.TypeLetterbyLetter(lines=[3,4]))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in range(3, 5)]))

        # SLIDE 32:  ===========================================================
        # HIGLIGHT ANGLES UPDATE
        self.next_slide(
            notes=
            '''2. Update the angles. [CLICK]
            '''
        )
        self.play(GD_code.TypeLetterbyLetter(lines=[6,7]))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in range(5,7)]))

        # SLIDE 33:  ===========================================================
        # HIGLIGHT STOPPING CRITERION
        self.next_slide(
            notes=
            '''3 - Compute the cost function to check the stopping criterion.
            [CLICK]
            '''
        )
        self.play(GD_code.TypeLetterbyLetter(lines=range(9, 14)))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in [7,8,9]]))

        # SLIDE 34:  ===========================================================
        # STOP TRUE -> PRINT NUMBER OF ITERATIONS AND BREAK WRITTEN
        self.next_slide(
            notes=
            '''If it is satisfied, we print the number of  iterations reached at
            this point. The instruction "break" interrupts the loop. [CLICK] ...
            '''
        )
        self.play(GD_code.TypeLetterbyLetter(lines=range(14, 17)))

        # SLIDE 35:  ===========================================================
        # 'i=i+1' LINE WRITTEN
        self.next_slide(
            notes=
            '''Otherwise, we increment i and move to the next iteration. To
            check the output of our algorithm we can print [CLICK] ...
            '''
        )
        self.play(GD_code.TypeLetterbyLetter(lines=range(17, 19)))

        # SLIDE 36:  ===========================================================
        # CODE SCROLLS UP, PRINT LINES ARE WRITTEN
        self.next_slide(
            notes=
            '''...the final angles combination, the computed distance and the 
            final arm's configuration. [CLICK]
            '''
        )
        self.play(VGroup(GD_code.code[:19], *[highlight_rect_2[i] for i in [2,4,6,8,9]]).animate.shift(UP*0.5))
        GD_code.code[19:].shift(UP*0.5)
        self.play(GD_code.TypeLetterbyLetter(lines=[20,21,22], lag_ratio=0))

        # SLIDE 37:  ===========================================================
        # ZOOM OUT TO MATLAB
        self.next_slide(
            notes=
            '''Let us run the script [CLICK] ...
            '''
        )
        mat_env.clear()
        self.play(FadeOut(*highlight_rect_2))
        GD_code.add_background_window(DSS.mainRect)
        self.play(
            GD_code.IntoMatlab(mat_env),
            DSS.bringOut()
        )

        # SLIDE 38:  ===========================================================
        # PRESS RUN, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''[CLICK]
            '''
        )
        final_output_text = MatlabOutputText(
            'Converged after 38 iterations.\n'
            'Final angle combination:  0.6293184 4.6875734 \n'
            'Final distance:  0.0083814'
        )
        mat_env.add_output(
            output_text=final_output_text,
            output_image=r'Assets\W3\RobotArmEnd_mat.png'
        )
        self.play(mat_env.Run())

        # SLIDE 39:  ===========================================================
        # OUTPUT TO FULLSCREEN
        self.next_slide(
            notes=
            '''...and have a look at the results: [CLICK]
            '''
        )
        self.play(mat_env.focus_output(scale = 0.8))

        # SLIDE 40:  ===========================================================
        # HIGLIGHT FIRST LINE
        self.next_slide(
            notes=
            '''1 - the method converges in 38 iterations [CLICK]
            '''
        )
        output_highlights = [HighlightRectangle(final_output_text[i]) for i in range(3)]
        self.play(Create(output_highlights[0]))

        # SLIDE 41:  ===========================================================
        # HIGLIGHT SECOND LINE
        self.next_slide(
            notes=
            '''2 - The numerical solution is the pair of angles [0.62, 4.68] [CLICK]
            '''
        )
        self.play(ReplacementTransform(output_highlights[0], output_highlights[1]))

        # SLIDE 42:  ===========================================================
        # HIGLIGHT THIRD LINE
        self.next_slide(
            notes=
            '''The final squared distance J between the arm's tip and the target
            point is 0.008. note that it is smaller than the tolerance. [CLICK]
            '''
        )
        self.play(ReplacementTransform(output_highlights[1], output_highlights[2]))

        # SLIDE 43:  ===========================================================
        # PYTHON PLOT TRANSFORMS INTO ROBOT ARM GRAPHIC WITH ANGLES TO BE CLEARER
        self.next_slide(
            notes=
            '''The method works! We found a pair of angles to reach the target
            point. Remember, they are in radians, in degrees they correspond to
            36 and 270 degrees. Do you think this solution is unique? Why? [END]
            '''
        )
        self.play(FadeOut(output_highlights[2]))

        ax = NumberPlane(
            x_range=[-2, 3],
            y_range=[-2, 2],
            x_length=5,
            y_length=4,
            x_axis_config={'stroke_color':BLACK, 'include_ticks':False, 'include_tip':False, 'stroke_width':0.5},
            y_axis_config={'stroke_color':BLACK, 'include_ticks':False, 'include_tip':False, 'stroke_width':0.5},
            background_line_style={'stroke_color':BLACK, 'stroke_width':0.5}
        ).scale(1).move_to(mat_env.output.image).shift(RIGHT*3).set_z(0)
        T1, T2 = 0.6293184, 4.6875734
        robot_arm=NewDB(ax, 1, 1.5,  T1, T2)
        robot_arm.suspend_updating()
        target = Star(color=ORANGE, fill_opacity=1).scale(0.1).move_to(
            ax.c2p(0.75, -1, 0))
        end = VGroup(ax, robot_arm, target)
        start = end.copy().scale_to_fit_height(mat_env.output.image.height).scale(0.8*0.66).move_to(mat_env.output.image).set_opacity(0).set_z(0)
        sub_false = DashedLine(robot_arm.foot.get_center(), robot_arm.foot.get_center()+RIGHT, stroke_color=BLACK, stroke_width=0.3)
        subline = DashedLine(robot_arm.joint.get_center(), robot_arm.joint.get_center()+RIGHT*0.3, stroke_color=BLACK, stroke_width=0.3)
        angle1 = Angle(sub_false, robot_arm.arm1, color=BLUE_B, radius=0.25)
        angle2 = Angle(subline, robot_arm.arm2, color=BLUE_B, radius=0.25)
        t1_label = Variable(var=T1, label=MathTex(r'\theta_1')).set_color(BLACK).scale(0.75).next_to(angle1, LEFT).shift(UP*0.2).suspend_updating()
        t2_label = Variable(var=T2, label=MathTex(r'\theta_2')).set_color(BLACK).scale(0.75).next_to(angle2, UP*0.25).suspend_updating()

        self.play(
            Group(mat_env.output.plot_window, mat_env.output.image).animate.shift(LEFT*3),
            Transform(start, end)
        )
        self.play(
            FadeIn(angle1, angle2,sub_false, subline, t1_label.label, t2_label.label),
            ReplacementTransform(final_output_text[1][22:31].copy(), t1_label.value),
            ReplacementTransform(final_output_text[1][31:].copy(), t2_label.value)
        )
