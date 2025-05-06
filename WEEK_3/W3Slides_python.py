import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from manim_slides import Slide
from Generic_mooc_utils import *
from colab_utils import *

config.renderer='cairo'
config.background_color = WHITE
# config.pixel_width=960
# config.pixel_height=720
config.pixel_width=1440
config.pixel_height=1080


class W3Slides_python(Slide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # COVER: TRANSLATE PSEUDO CODE INTO PYTHON CODE
        self.next_slide(
            notes=
            '''Let's open a notebook, and let's start to learn how to code the 
            Gradient Descent algorithm. [CLICK]
            '''
        )
        cl_env = ColabEnv(r'Assets\W3\colabGD.png')
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
        self.play(FadeIn(cl_env))
        self.wait(1)
        self.play(FadeIn(surrounding_pc, pc))

        # SLIDE 02:  ===========================================================
        # COLAB ENVIRONMENT REPLACES COVER
        self.next_slide(
            notes=
            '''Firstly, we want to load a script contains the function 
            plot_robot_arm.py that we will use to plot the robot arm configuration.
            By clicking [CLICK] on...
            '''
        )
        self.play(FadeOut(surrounding_pc, pc))
        hand_cursor = Cursor().move_to(pixel2p(25, 437))
        self.wait(1)
        self.play(GrowFromCenter(hand_cursor))

        # SLIDE 03:  ===========================================================
        # COLAB SIDE BAR APPEARS AFTER MOUSE CLICK
        # CURSOR MOVES TO NEXT BUTTON (UPLOAD)
        self.next_slide(
            notes=
            '''...the folder icon on the left a side bar appears showing the list
            of available files. Let us click [CLICK] on the upload button...
            '''
        )
        self.play(Succession(
            hand_cursor.Click(),
            cl_env.animate(run_time=0).set_image(r'Assets\W3\colabGD_sidemenu.png'),
            lag_ratio=0.5
        ))
        self.wait(1)
        self.play(hand_cursor.animate.move_to(pixel2p(84, 290)))
        
        # SLIDE 04:  ===========================================================
        # CLICK ON UPLOAD BUTTON AND NEW SIDE MENU APPEARS
        self.next_slide(
            notes=
            '''... and select from your local file system the file my_functions.py.
            To import this function we click on + Code [CLICK] ... 
            '''
        )
        self.play(hand_cursor.Click())
        self.wait(0.5)
        self.play(hand_cursor.animate.move_to(pixel2p(210, 105)))

        # SLIDE 05:  ===========================================================
        # CLICK ON + CODE
        # CODE BLOCK APPEARS AND FIRST CODE LINE IS WRITTEN. THEN BLOCK IS 
        # EXPANDED TO FULL SCREEN.
        self.next_slide(
            notes=
            '''... and insert the commands:
            from my_functions import plot_robot_arm
            Next we also import sine and cosine mathematical functions with
            [CLICK]
            '''
        )
        empty_cell = ColabCodeBlock(code='')
        self.play(hand_cursor.Click())
        cl_env.set_image(r'Assets\W3\colabGD.png')
        # self.add(cl_env)  # update change of background
        cl_env.add_cell(empty_cell)
        self.wait(0.8)
        self.play(cl_env.OutofColab(empty_cell), FadeOut(hand_cursor))
        self.wait(0.2)

        import_code = ColabCode(
            r'''
            from my_functions import plot_robot_arm  
            from math import cos, sin  
            from matplotlib import pyplot as plt
            '''
        )
        input_data_code = ColabCode(
            r'''
            # Input parameters 
            xp = [0.75, -1]       # target point 
            L1 = 1                # length of arm1 
            L2 = 1.5              # length of arm2 
            theta = [2.5, 2.7]    # initial angles 
            tol = 0.01            # tolerance 
            alpha = 0.1           # learning rate 
            Niter = 1000          # max iterations
            '''
        )
        DSS = DynamicSplitScreen(main_color=COLAB_LIGHTGRAY, side_color=WHITE)
        self.add(DSS)
        self.remove(empty_cell.colabCode.window)
        self.play(import_code.TypeLetterbyLetter(lines=[0]))
        self.wait(0.1)

        # SLIDE 06:  ===========================================================
        #  NEXT CODE LINES ARE WRITTEN
        self.next_slide(
            notes=
            '''  from math import cos, sin
            and the pyplot library used for the graphics: [CLICK]
            '''
        )
        self.play(import_code.TypeLetterbyLetter(lines=[1]))
        self.wait(0.05)

        # SLIDE 06:  ===========================================================
        #  NEXT CODE LINES ARE WRITTEN
        self.next_slide(
            notes=
            '''from matplotlib import pyplot as plt [CLICK]
            '''
        )
        self.play(import_code.TypeLetterbyLetter(lines=[2]))
        self.wait(0.05)

        # SLIDE 07:  ===========================================================
        # PSEUDO CODE APPEARS, NEW CODE BLOCK WITH #DATA REPLACES OLD ONE
        self.next_slide(
            notes=
            '''To translate our pseudo-code in Python we first define our input
            parameters: [CLICK]
            '''
        )
        RequireLine = pc[1]
        DSS.add_side_obj(RequireLine)
        
        self.play(DSS.bringIn(), import_code.code.animate.shift(UP*1.5))
        self.wait(0.5)
        input_data_code.code.next_to(import_code.code, DOWN)
        input_data_code.code.align_to(import_code.code, LEFT)
        self.play(input_data_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 08:  ===========================================================
        # DATA LINE IS WRITTEN
        self.next_slide(
            notes=
            '''1 - the target point target = [0.75, -1] 
            [CLICK]
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
            input_data_code.TypeLetterbyLetter(lines=[1])
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
            ReplacementTransform(input_data_high_rect['target'], input_data_high_rect['arm_length'], run_time=1),
            input_data_code.TypeLetterbyLetter(lines=[2,3])
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
            ReplacementTransform(input_data_high_rect['arm_length'], input_data_high_rect['initial_guess'], run_time=1),
            input_data_code.TypeLetterbyLetter(lines=[4])
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
            ReplacementTransform(input_data_high_rect['initial_guess'], input_data_high_rect['tolerance'], run_time=1),
            input_data_code.TypeLetterbyLetter(lines=[5])
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
            ReplacementTransform(input_data_high_rect['tolerance'], input_data_high_rect['alpha'], run_time=1),
            input_data_code.TypeLetterbyLetter(lines=[6])
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
            ReplacementTransform(input_data_high_rect['alpha'], input_data_high_rect['max_iter'], run_time=1),
            input_data_code.TypeLetterbyLetter(lines=[7])
        )
        
        # SLIDE 15:  ===========================================================
        # KINEMATICS FORMULAS ARE WRITTEN
        self.next_slide(
            notes=
            '''Now we want to compute the position of the robot according to 
            this formulas. To that end we define [CLICK]...
            '''
        )
        kinematics_eq = VGroup(
            MathTex(r"x(\theta_1, \theta_2) = L_1 \cos(\theta_1) + L_2 \cos(\theta_2)", color=BLACK),
            MathTex(r"y(\theta_1, \theta_2) = L_1 \sin(\theta_1) + L_2 \sin(\theta_2)", color=BLACK)
            ).arrange(DOWN)
        kinematics_code=ColabCode(
            r'''
            # Compute tip robot position 
            def robot_position(theta, L1, L2): 
                theta1, theta2 = theta 
                x = L1 * cos(theta1) + L2 * cos(theta2) 
                y = L1 * sin(theta1) + L2 * sin(theta2) 
                return [x, y] 

            # Plot robot arms
            print(robot_position(theta_0, L1, L2)) 
            plot_robot_arm(theta_0, xp, [L1, L2])
            '''
        )
        self.play(
            FadeOut(input_data_code, import_code),
            input_data_high_rect['max_iter'].animate.shift(UP*DSS.secondaryRect.height),
            DSS.bringOut()
        )
        self.remove(input_data_high_rect['max_iter'])  # take care of the last highlight
        DSS.add_side_obj(kinematics_eq)
        kinematics_code.move_to(DSS.mainRect)
        self.play(
            DSS.bringIn(),
            kinematics_code.TypeLetterbyLetter(lines=[0])
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
            kinematics_code.TypeLetterbyLetter(lines=range(1,6))
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
            kinematics_code.TypeLetterbyLetter(lines=[7,8,9])
        )
        self.wait(0.2)
        cl_env.clear()
        cl_env.add_cell(ColabCodeBlock(code=import_code.code_string + input_data_code.code_string))
        kinematics_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(AnimationGroup(
            FadeOut(kinematics_eq, DSS.secondaryRect),
            kinematics_code.IntoColab(colab_env=cl_env),
            lag_ratio=0.5
            ))
        starting_output_text = ColabBlockOutputText( '[-2.1572518285725253, 1.2395419644547012].'  )
        starting_output_img = ImageMobject(r'Assets\W3\RobotArmStart_py.png').scale(0.7).next_to(starting_output_text, DOWN).align_to(starting_output_text, LEFT)
        cl_env.cells[-1].add_output(Group(starting_output_text, starting_output_img))
        self.play(cl_env.cells[-1].Run())
        
        # SLIDE 18:  ===========================================================
        # COLAB ENV FADES OUT
        # PSEUDO CODE FADES IN, WHILE LOOP HIGHLIGHTED.
        self.next_slide(
            notes=
            '''  Moving on, we now look at the iteration loop. [CLICK]
            '''
        )
        self.play(FadeOut(cl_env))
        self.play(FadeIn(pc.restore().scale_to_fit_width(FRAME_WIDTH*0.9).center()))
        self.wait(0.3)
        self.play(Circumscribe(pc[2:], run_time=2, color=BLUE))

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

        # SLIDE 20:  ===========================================================
        # SCREEN SPLIT BETWEEN FORMULAS AND CODE
        # DEFINITION OF J IS WRITTEN
        self.next_slide(
            notes=
            '''For the function J we use the following syntax.
            Using the robot_position function we can compute the value of J
            in this way, and print [CLICK] ...
            '''
        )
        J_code = ColabCode(
            r'''
            # Evaluation of J 
            def J(theta, xp, L1, L2): 
                [x, y] = robot_position(theta, L1, L2) 
                Jval = (x-xp[0])**2 + (y-xp[1])**2 
                return Jval
            
            print(J(angles, target, L1, L2))
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
                    ReplacementTransform(J_eq, DSS.secondaryObj),
                    FadeIn(DSS.mainRect, DSS.secondaryRect)
                ),
                J_code.TypeLetterbyLetter(lines=[0]),
                lag_ratio=1
            )
        )
        self.play(J_code.TypeLetterbyLetter(lines=range(1,5)))
        
        # SLIDE 20:  ===========================================================
        # PRINT CODE IS WRITTEN
        # INTO COLAB AND RUN CELL
        self.next_slide(
            notes=
            '''... the value of J providing the initial guess for the target.
            [CLICK]
            '''
        )
        self.play(J_code.TypeLetterbyLetter(lines=[6]))
        J_code.add_background_window(DSS.mainRect.suspend_updating())
        cl_env.clear()
        self.play(AnimationGroup(
            J_code.IntoColab(cl_env),
            FadeOut(DSS.secondaryObj, DSS.secondaryRect)
            ))
        
        cl_env.cells[-1].add_output(output = '13.467661405291913' )
        self.play(cl_env.cells[-1].Run())
        
        # SLIDE 21:  ===========================================================
        # WRITE FIRST LINES OF FUNCION GRAD_J
        self.next_slide(
            notes=
            '''Next, we define a Python function called "grad_J" to compute the 
            gradient of J at a given point, which takes in input the angles theta_1
            and theta_2  and outputs the values dJ_dt1 and dJ_dt2.
            Here, we use the analytical derivation of the gradient of J.
            For instance, [CLICK]
            '''
        )
        nabla_J_code = ColabCode(
            r'''
            # Evaluation of Grad(J) 
            def grad_J(theta, xp, L1, L2): 
                [x, y] = robot_position(theta, L1, L2)

                dx_dt1 = - L1 * sin(theta[0]) 
                dx_dt2 = - L2 * sin(theta[1]) 
                dy_dt1 =   L1 * cos(theta[0]) 
                dy_dt2 =   L2 * cos(theta[1]) 

                dJ_dx = 2*(x - xp[0]) 
                dJ_dy = 2*(y - xp[1]) 
            
                DJ_dt1 = dJ_dx*dx_dt1 + dJ_dy*dy_dt1 
                DJ_dt2 = dJ_dx*dx_dt2 + dJ_dy*dy_dt2 

                return [DJ_dt1, DJ_dt2]
            '''
        )
        self.play(FadeOut(cl_env))
        DSS.reset()
        DSS.add_side_obj(nabla_J_eq.scale(0.6))
        nabla_J_code.move_to(DSS.get_final_mainObj_pos())
        self.play(AnimationGroup(
            FadeIn(DSS.mainRect),
            AnimationGroup(
                DSS.bringIn(),
                nabla_J_code.TypeLetterbyLetter(lines=[0])
            )
        ))
        self.play(nabla_J_code.TypeLetterbyLetter(lines=range(1, 3)))
        self.play(nabla_J_code.TypeLetterbyLetter(lines=range(4, 8), lag_ratio=0))
        self.play(nabla_J_code.TypeLetterbyLetter(lines=range(9, 11), lag_ratio=0))
    
        # SLIDE 22:  ===========================================================
        # PART OF NABLA J FORMULA AND CORRESPONDING CODE IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''... , the partial derivative of x with respect to theta_1
            is coded this a way [CLICK] ...
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

        # SLIDE 22:  ===========================================================
        # PART OF NABLA J FORMULA AND CORRESPONDING CODE IS HIGHLIGHTED
        self.next_slide(
            notes=
            '''... , and similarly the other terms. [CLICK]
            '''
        )
        self.play(AnimationGroup(*[Create(r) for r in nabla_J_rectangle_highlights[2:]]))

        # SLIDE 22:  ===========================================================
        # OTHER PARTS OF NABLA J FORMULA AND CORRESPONDING CODE ARE HIGHLIGHTED
        self.next_slide(
            notes=
            '''Finally, we use them to write the two components of the gradient
            of J. [CLICK]
            '''
        )
        self.play(nabla_J_code.TypeLetterbyLetter(lines=range(12, 14), lag_ratio=0))
        self.play(nabla_J_code.TypeLetterbyLetter(lines=[15]))
        self.wait(0.1)

        # SLIDE 23:  ===========================================================
        # PSEUDO CODE FADES BACK IN CENTERED
        self.next_slide(
            notes=
            '''Now we have all the ingredients to code the iteration loop. [CLICK]
            '''
        )
        self.play(FadeOut(nabla_J_eq, DSS, nabla_J_code, *nabla_J_rectangle_highlights))
        self.play(FadeIn(pc))
        GD_code = ColabCode(
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
                    print(f"Converged after {i} iterations.") 
                    break 

                i = i + 1 

            print("Final angle combination:", theta) 
            print("Final distance:", Jval) 
            plot_robot_arm(angles, xp, [L1, L2])
            ''')

        # SLIDE 24:  ===========================================================
        # WHILE LOOP MOVES TO TOP
        # FIRST LINE 0F PSEUDO CODE WRITTEN
        # HIGLIGHT WHILE INSTRUCTION IN PSEUDO CODE AND CODE
        self.next_slide(
            notes=
            '''For each iteration of the while loop we perform the following
            instructions: [CLICK]
            '''
        )
        DSS.add_side_obj(pc[2:].copy().scale(0.4))
        GD_code.scale(0.85).move_to(DSS.get_final_mainObj_pos()).shift(DOWN)
        self.play(
            Succession(
                AnimationGroup(
                    FadeOut(pc[:2]),
                    ReplacementTransform(pc[2:], DSS.secondaryObj),
                    FadeIn(DSS)
                ),
                GD_code.TypeLetterbyLetter(lines=[0])
            )
        )
        highlight_pairs_2 = [
            [pc[3][2:], pc[7][2:], GD_code.code[2][12:]],
            [pc[4][-15:], GD_code.code[4][16:]],
            [pc[4][2:], VGroup(GD_code.code[7][16:], GD_code.code[8][16:])],
            [pc[5][4:18], GD_code.code[11][16:], GD_code.code[14][16:]]
        ]
        highlight_rect_2 = [
            HighlightRectangle(highlight_pairs_2[j][i], color=highlight_colors[j])
            for j in range(4) for i in range(len(highlight_pairs_2[j]))
        ]

        self.play(GD_code.TypeLetterbyLetter(lines=[1,2]))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in range(3)]))

        # SLIDE 25:  ===========================================================
        # HIGLIGHT NABLA J COMPUTATION (CODE AND PSEUDO-CODE)
        self.next_slide(
            notes=
            '''1 - Compute the gradient using the current angles and the target
            point [CLICK]
            '''
        )
        self.play(GD_code.TypeLetterbyLetter(lines=[3,4]))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in range(3, 5)]))

        # SLIDE 26:  ===========================================================
        # HIGLIGHT ANGLES UPDATE (CODE AND PSEUDO-CODE)
        self.next_slide(
            notes=
            '''2. Update the angles. pay attention! Here we use the short hand
            notation for the updates, which is equivalent [CLICK] to this one. 
            '''
        )
        self.play(GD_code.TypeLetterbyLetter(lines=[6,7,8]))
        self.play(AnimationGroup(*[Create(highlight_rect_2[i]) for i in range(5,7)]))

        # SLIDE 28:  ===========================================================
        # TRANSFORM CORRESPONDING LINES IN EXTENDED FORM AND BACK
        self.next_slide(
            notes=
            '''...to this one. [CLICK]
            '''
        )
        explict_update_code = ColabCode(
            r'''
            theta[0] = theta[0] - alpha * grad[0]
            theta[1] = theta[1] - alpha * grad[1]
            '''
        )
        two_lines = VGroup(GD_code.code[7][16:], GD_code.code[8][16:])
        two_lines.save_state()
        highlight_rect_2[6].save_state()
        explict_update_code.code.scale_to_fit_height(two_lines.height).move_to(two_lines).align_to(two_lines, LEFT)
        target_high_rect = HighlightRectangle(explict_update_code.code, color=highlight_colors[2])

        self.play(AnimationGroup(
            Transform(two_lines, explict_update_code.code, run_time=0.3),
            highlight_rect_2[6].animate(run_time=0.3).become(target_high_rect)
            ))
        self.wait(0.2)
        # SLIDE 29:  ===========================================================
        # HIGLIGHT STOPPING CRITERION (CODE AND PSEUDO-CODE)
        self.next_slide(
            notes=
            '''3 - Perform the stopping criterion and print the number of 
            iterations reached at this point.
            To check the output of our algorithm we can print [CLICK] ...
            '''
        )
        self.play(AnimationGroup(two_lines.animate(run_time=0.3).restore(), highlight_rect_2[6].animate(run_time=0.3).restore()))
        self.play(GD_code.TypeLetterbyLetter(lines=range(10,19)))
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
        self.play(GD_code.TypeLetterbyLetter(lines=[20,21,22], lag_ratio=0))

        # SLIDE 31:  ===========================================================
        # INTO COLAB, PRESS RUN, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''Let us run the script and comment the outputs: [CLICK]
            '''
        )
        cl_env.clear()
        self.play(FadeOut(*highlight_rect_2))
        GD_code.add_background_window(DSS.mainRect)
        self.play(GD_code.IntoColab(cl_env), FadeOut(pc[2:], DSS.secondaryObj,  DSS.secondaryRect))
        final_output_text = ColabBlockOutputText(
            'Converged after 38 iterations.\n'  
            'Final angle combination: [0.6293180379169862, 4.687572633369985]\n'
            'Final distance: 0.008380667670762636 '
        )
        final_output_img = ImageMobject(r'Assets\W3\RobotArmEnd_py.png').scale(0.7).next_to(final_output_text, DOWN).align_to(final_output_text, LEFT)
        cl_env.cells[-1].add_output(
            Group(final_output_text, final_output_img)
        )
        self.play(cl_env.cells[-1].Run())
        self.wait(1)
        self.play(cl_env.cells[-1].animate.focus_output(scale = 0.8))

        # SLIDE 32:  ===========================================================
        # HIGLIGHT FIRST LINE
        self.next_slide(
            notes=
            '''1 - the method converges in 38 iterations [CLICK]
            '''
        )
        self.play(Circumscribe(final_output_text[0], run_time=2, color=BLUE))

        # SLIDE 33:  ===========================================================
        # HIGLIGHT SECOND LINE
        self.next_slide(
            notes=
            '''2 - The numerical solution is the pair of angles [0.62, 4.68] [CLICK]
            '''
        )
        self.play(Circumscribe(final_output_text[1], run_time=2, color=BLUE))

        # SLIDE 34:  ===========================================================
        # HIGLIGHT THIRD LINE
        self.next_slide(
            notes=
            '''The final computed squared distance J between the arm's tip and
            the target point is 0.008. Note that it is smaller than the tolerance.
            The method works! We found a pair of angles to reach the target point.
            [END]
            '''
        )
        self.play(Circumscribe(final_output_text[2], run_time=2, color=BLUE))


