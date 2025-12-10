import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from mooc_utils.colab import *
from W4Anim import DiscreteConvolutionPseudoCode, separable_box_blur, LiveConvolution, SATURATED_BLUE, Kernel3X3IndexAnimation
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO


config.update(RELEASE_CONFIG)


class W4Python_slides(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # EMPTY NOTEBOOK (FIRST CELL ALREADY PRESENT), PSEUDO-CODE APPEARS ON TOP
        self.next_slide(
            notes=
            '''Let's open a notebook, and let's start to learn how to code the
            discrete convolution algorithm to compute filtered images.
            '''
        )
        cl_env = ColabEnv(self, r'Assets\W4\colabDC.png')
        cl_env.add_cell()
        pc = DiscreteConvolutionPseudoCode()
        pc.scale_to_fit_width(FRAME_WIDTH*0.65).center()
        pc.save_state()
        surrounding_rect = SurroundingRectangle(pc, fill_color=WHITE, fill_opacity=1, stroke_width=0.5,
                                              stroke_color=BLACK, corner_radius=0.2, buff=0.5).set_z_index(-0.5)
        self.play(
            Succession(
                FadeIn(cl_env.background, *cl_env.cells),
                Wait(1),
                FadeIn(surrounding_rect, pc)
            )
        )

        # SLIDE 02:  ===========================================================
        # PSEUDO-CODE FADES, CURSOR APPEARS
        self.next_slide(
            notes=
            '''First, we need to load an image and some support functionalities.
            By clicking [CLICK] on
            '''
        )
        hand_cursor = Cursor()
        self.play(
            Succession(
                FadeOut(pc, surrounding_rect),
                Wait(0.5),
                GrowFromCenter(hand_cursor)
            )
        )

        # SLIDE 03:  ===========================================================
        # CURSOR MOVES TO FOLDER ICON AND CLICKS IT
        # SIDE BAR APPEARS, CELL SHIFTS RIGHT ACCORDINGLY
        self.next_slide(
            notes=
            '''on the folder icon on the left, [CLICK] a side bar appears
            showing the list of available files.
            '''
        )
        self.play(hand_cursor.animate.move_to(cl_env.MENU_))
        self.play(hand_cursor.Click())
        cl_env.set_image(r'Assets\W4\colabDC_sidemenu.png')
        cl_env.get_cell(0).shift(RIGHT*cl_env.SIDE_MENU_WIDTH_)

        # SLIDE 04:  ===========================================================
        # CURSOR MOVES TO UPLOAD BUTTON AND CLICKS IT
        # HELPERS AND IMAGE APPEAR IN SIDEBAR
        self.next_slide(
            notes=
            '''Let us click [CLICK] on the upload button and select from your
            local file system the file [CLICK] helper_functions.py and the image
            "part.png".
            '''
        )
        self.play(Succession(hand_cursor.animate.move_to(cl_env.UPLOAD_), hand_cursor.Click()))
        cl_env.add_file_to_sidemenu('helper_functions.py')
        cl_env.add_file_to_sidemenu('part.png')

        # SLIDE 05:  ===========================================================
        # FOLDER ICON CLICKED AGAIN, SIDE BAR DISAPPEARS, CELL SHIFTS LEFT
        # OUT OF COLAB
        # IMPORT NUMPY AND MATPLOTLIB WRITTEN
        self.next_slide(
            notes=
            '''We need to import the numpy and matplotlib modules.
            '''
        )
        self.play(Succession(hand_cursor.animate.move_to(cl_env.MENU_), hand_cursor.Click()))
        cl_env.clear_sidemenu()
        cl_env.set_image(r'Assets\W4\colabDC.png')
        cl_env.get_cell(0).shift(LEFT*cl_env.SIDE_MENU_WIDTH_)
        self.wait(0.3)
        # DSS = DynamicSplitScreen(main_color=COLAB_LIGHTGRAY, side_color=WHITE)
        import_code = ColabCode(
            r'''
            import numpy as np
            import matplotlib.pyplot as plt
            from PIL import Image
            from helper_functions import *
            '''
        )
        self.play(cl_env.OutofColab(cell=0), FadeOut(hand_cursor))
        self.play(import_code.TypeLetterbyLetter(lines=[0]))
        self.play(import_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 06:  ===========================================================
        # IMPORT PIL IMAGE WRITTEN
        self.next_slide(
            notes=
            '''Then, from the Python imaging library we import the module
            "Image",
            '''
        )
        self.play(import_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 07:  ===========================================================
        # IMPORT HELPER_FUNCTIONS WRITTEN
        self.next_slide(
            notes=
            '''and with this syntax we import all the functions contained in
            helper_functions.py, ...
            '''
        )
        self.play(import_code.TypeLetterbyLetter(lines=[3]))

        # SLIDE 08:  ===========================================================
        # INTO COLAB, RUN CELL
        self.next_slide(
            notes=
            '''... to help us visualize the results.
            '''
        )
        import_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()
        self.play(import_code.IntoColab(cl_env))
        self.play(cl_env.Run(cell=0))

        # SLIDE 09:  ===========================================================
        # CURSOR CLICKS +CODE, NEW CELL APPEARS, OUT OF COLAB
        # IMAGE.OPEN WRITTEN
        self.next_slide(
            notes=
            '''Let us load the image as a matrix: First, we use the Image module
            to load the image contained in "part.png".
            '''
        )
        self.play(Succession(cl_env.cursor.animate.move_to(cl_env.PLUS_CODE_), cl_env.cursor.Click()))
        cl_env.add_cell()
        self.wait(0.3)
        self.play(cl_env.OutofColab(cell=1))

        load_image_code = ColabCode(
            r'''
            file_name = "part.png"
            A_color = Image.open(file_name)
            A_g = A_color.convert('L')
            A = np.array(A_g)
            print("type", type(A), "shape", A.shape)
            '''
        )

        self.play(load_image_code.TypeLetterbyLetter(lines=[0]))
        self.play(load_image_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 10:  ===========================================================
        # CONVERT LINE WRITTEN, "L" HIGHLIGHTED
        self.next_slide(
            notes=
            '''Then, we convert A_color in greyscale using the method "convert"
            and the option "L",
            '''
        )
        L_highlight = HighlightRectangle(load_image_code[2][-4:-1])
        self.play(load_image_code.TypeLetterbyLetter(lines=[2]))
        self.play(Create(L_highlight))

        # SLIDE 11:  ===========================================================
        # NP.ARRAY() LINE WRITTEN
        self.next_slide(
            notes=
            '''And finally we convert this greyscale image into a numpy array
            with this syntax.
            '''
        )
        self.play(
            AnimationGroup(
                FadeOut(L_highlight),
                load_image_code.TypeLetterbyLetter(lines=[3]),
                lag_ratio=0.5
            )
        )

        # SLIDE 12:  ===========================================================
        # PRINT TYPE AND SHAPE LINE WRITTEN
        self.next_slide(
            notes=
            '''With the instructions "type" and "shape" we can extract some
            information on A:
            '''
        )
        self.play(load_image_code.TypeLetterbyLetter(lines=[4]))

        # SLIDE 13:  ===========================================================
        # INTO COLAB, RUN CELL, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''as we can see it is an n-dimensional numpy array of shape 12X12
            '''
        )
        load_image_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()
        self.play(load_image_code.IntoColab(cl_env))

        cl_env.get_cell(1).add_output(
            r"data type <class 'numpy.ndarray'> shape (12, 12)"
        )
        self.play(cl_env.Run(cell=1, new_cursor=False))
        
        # SLIDE 14:  ===========================================================
        # CURSOR CLICKS +CODE, NEW CELL APEARS, OUT OF COLAB
        # PRINT(A[1,2]) WRITTEN
        self.next_slide(
            notes=
            '''How does a 2D-numpy array work? To access to an element we use
            square brackets. For example the element in position 1, 2 can be
            accessed as A[1, 2].
            '''
        )
        sample_A = np.array(Image.open(r'Assets\W4\part.png'), dtype=np.uint8)

        bp = FullScreenBackground(COLAB_LIGHTGRAY)
        self.play(FadeIn(bp))
        cl_env.clear()

        square_brackets_code = ColabCode(
            r'''
            print(A[1, 2])
            '''
        )
        self.play(square_brackets_code.TypeLetterbyLetter())

        # SLIDE 15:  ===========================================================
        # INTO COLAB, RUN CELL, OUTOUT APPEARS
        self.next_slide(
            notes=
            '''For example the element in position 1, 2 can be accessed as A[1,
            2].
            '''
        )
        square_brackets_code.add_background_window(bp)
        cl_env.remove_cell()
        self.play(square_brackets_code.IntoColab(cl_env))
        cl_env.get_cell(0).add_output(
            str(sample_A[1,2])
        )
        self.play(cl_env.Run(cell=0))

        # SLIDE 16:  ===========================================================
        # CURSOR CLICKS +CODE, NEW CELL APEARS
        # WRITE LINE IN NEW CELL
        # CELL IS RUN, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''We can also print the matrix A. Notice the values 255 for the
            white parts of the image.
            '''
        )
        self.play(Succession(cl_env.cursor.animate.move_to(cl_env.PLUS_CODE_), cl_env.cursor.Click()))
        print_matrix_cell = ColabCodeBlock(
            r'''
            print(A)
            '''
        )
        cl_env.add_cell(print_matrix_cell)
        self.remove(print_matrix_cell.code)
        self.wait(0.3)
        self.play(print_matrix_cell.TypeLetterbyLetter())
       
        # self.play(square_brackets_code.TypeLetterbyLetter())
        # square_brackets_code.add_background_window(cl_env.get_cell(3).window.copy())
        # cl_env.remove_cell()
        # self.play(square_brackets_code.IntoColab())
        print_A_result_stream = StringIO()
        print(sample_A, file=print_A_result_stream)
        print_matrix_cell.add_output(
            print_A_result_stream.getvalue()
        )
        print_A_result_stream.close()
        self.play(cl_env.Run(cell=1, new_cursor=False))
    
        # SLIDE 17:  ===========================================================
        # CURSOR CLICKS +CODE, NEW CELL APPEARS, OUT OF COLAB
        # IMSHOW LINE WRITTEN
        self.next_slide(
            notes=
            '''Instead, by using imshow from matplot lib we can display the
            matrix as an image;
            '''
        )
        self.play(Succession(cl_env.cursor.animate.move_to(cl_env.PLUS_CODE_), cl_env.cursor.Click()))
        cl_env.add_cell()
        self.wait(0.3)
        self.play(cl_env.OutofColab(cell=2))
        imshow_code = ColabCode(
            r'''
            plt.imshow(A, "grey")
            '''
        )
        self.play(imshow_code.TypeLetterbyLetter())

        # SLIDE 18:  ===========================================================
        # INTO COLAB, RUN CELL, IMAGE APPEARS
        self.next_slide(
            notes=
            '''we just need to specify that we are working with a greyscale
            image.
            '''
        )
        imshow_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(sample_A, cmap='gray')
        image_plot = draw_plot(fig).scale_to_fit_height(0.3*FRAME_HEIGHT)

        self.play(imshow_code.IntoColab(cl_env))
        cl_env.get_cell(2).add_output(
            image_plot
        )
        self.play(cl_env.Run(cell=2, new_cursor=False))

        # SLIDE 19:  ===========================================================
        # FADEOUT ALL
        # PSEUDO-CODE APPEARS
        self.next_slide(
            notes=
            '''Now we can implement the local convolution applied to a specific
            pixel in position i.j as in this pseudocode.
            '''
        )
        self.play(cl_env.FadeOut())
        self.clear()
        cl_env.clear()
        pc.restore()

        self.play(FadeIn(pc))

        # SLIDE 20:  ===========================================================
        # PSEUDO-CODE MOVED TO TOP RECTANGLE
        # FUNCTION DEFINITION LINE WRITTEN
        self.next_slide(
            notes=
            '''Let us create a function called local_convolution.
            '''
        )
        DSS = DynamicSplitScreen(main_color=COLAB_LIGHTGRAY, side_color=WHITE)

        local_convolution_code = ColabCode(
            r'''
            def local_convolution(A, K, i, j):
                v = 0
                for m in range(3):      # kernel rows
                    for n in range(3):  # kernel columns
                        v += A[i - 1 + m, j - 1 + n] * K[m, n]
                return v
            '''
        ).move_to(DSS.mainRect)
        pc.set_z_index(-2)
        self.play(FadeIn(DSS))
        self.remove(pc)
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 21:  ===========================================================
        # PSEUDO-CODE MOVED TO TOP RECTANGLE
        self.next_slide(
            notes=
            '''This function takes as input the matrix, the kernel, and the
            indices of the pixel ...
            '''
        )
        fscheme1 = FunctionAbstraction(scale=0.7)
        DSS.add_side_obj(fscheme1)
        DSS.add_main_obj(local_convolution_code[0], local_convolution_code[1:]) 
        self.play(DSS.bringIn())     
        fscheme1.add_inputs("A", "K", "i", "j", font_size=32, relative_offset=0.5)
        self.remove(fscheme1.InputArrows, fscheme1.InputLabels)
        AKij_labels = VGroup(MathTex(lab, color=BLACK) for lab in ("A", "K", "i", "j")).arrange(DOWN).next_to(fscheme1.InputLabels, LEFT, buff=1)
        for i, lab in enumerate(AKij_labels):
            lab.match_y(fscheme1.InputLabels[i])

        self.play(
            Succession(
                FadeIn(fscheme1.InputLabels[0], fscheme1.InputArrows[0], AKij_labels[0]),
                FadeIn(fscheme1.InputLabels[1], fscheme1.InputArrows[1], AKij_labels[1]),
                FadeIn(fscheme1.InputLabels[2:], fscheme1.InputArrows[2:], AKij_labels[2:]),
            )
        )

        # SLIDE 22:  ===========================================================
        # PSEUDO-CODE MOVED TO TOP RECTANGLE
        self.next_slide(
            notes=
            '''... and returns the value of the filtered pixel.
            '''
        )
        fscheme1.add_outputs("v", font_size=32)
        v_label = MathTex("v", color=BLACK).next_to(fscheme1.OutputLabels, RIGHT, buff=1)

        self.play(
            FadeIn(fscheme1.OutputLabels, fscheme1.OutputArrows, v_label),
            local_convolution_code.TypeLetterbyLetter(lines=[-1])
        )

        # SLIDE 23:  ===========================================================
        # BRING OUT FUNCTION SCHEME, BRING IN PSEUDO-CODE
        # V = 0 LINE WRITTEN
        self.next_slide(
            notes=
            '''First of all, we declare the variable v and set it to zero.
            '''
        )
        DSS.remove_main_obj()
        self.play(DSS.bringOut(), VGroup(AKij_labels, v_label).animate.shift(UP*DSS.secondaryRect.height))
        # DSS.add_main_obj(VGroup(local_convolution_code[0], local_convolution_code[-1]), local_convolution_code[1:-1])
        DSS.add_side_obj(pc.restore().scale(0.6).set_z_index(1)); self.add(pc)
        self.play(DSS.bringIn())

        v_highlight = HighlightRectangle(pc[2][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[1]), Create(v_highlight))

        # SLIDE 24:  ===========================================================
        # FIRST FOR LOOP WRITTEN
        self.next_slide(
            notes=
            '''Then, we introduce two for loops: The first one to access the
            rows, ...
            '''
        )
        first_loop_highlight = HighlightRectangle(pc[3][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[2]), ReplacementTransform(v_highlight, first_loop_highlight))

        # SLIDE 25:  ===========================================================
        # SECOND FOR LOOP WRITTEN
        self.next_slide(
            notes=
            '''And the second to access the columns of the kernel. Notice that
            these two loops are NESTED, meaning, they are one inside the other.
            '''
        )
        second_loop_highlight = HighlightRectangle(pc[4][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[3]), ReplacementTransform(first_loop_highlight, second_loop_highlight))

        # SLIDE 26:  ===========================================================
        # CONVOLUTION LINE WRITTEN
        # (i,j) (m,n) INDEX ANIMATION SHOWN ON THE SIDE
        self.next_slide(
            notes=
            '''Now for each pixel i,j, and for each position in the kernel, we
            compute the product of the value in the kernel, and the
            corresponding pixel in the image... and we add this value to v.
            '''
        )
        self.play(FadeOut(second_loop_highlight))
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[4]))

        # convolution_highlight = HighlightRectangle(pc[5][2:])
        pc_A_K_highlights = VGroup(
            HighlightRectangle(pc[5][6:20]),
            HighlightRectangle(pc[5][-6:], color=ORANGE)
        )
        code_A_K_highlights = VGroup(
            HighlightRectangle(local_convolution_code[4][3:17]),
            HighlightRectangle(local_convolution_code[4][-6:], color=ORANGE)
        )

        self.play(
            Succession(
                Create(pc_A_K_highlights[1]),
                Create(code_A_K_highlights[1]),
                Wait(1),
                Create(pc_A_K_highlights[0]),
                Create(code_A_K_highlights[0]),
            )
        )

        # SLIDE 27:  ===========================================================
        # RETURN LINE WRITTEN
        self.next_slide(
            notes=
            '''Finally, we return v, the value of a single filtered pixel.
            '''
        )
        self.play(FadeOut(pc_A_K_highlights, code_A_K_highlights))

        return_highlight = HighlightRectangle(local_convolution_code[5])
        self.play(FadeIn(return_highlight))
        self.wait(1)
        self.play(FadeOut(return_highlight))

        DSS.remove_main_obj()
        local_convolution_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(
            DSS.bringOut(),
            local_convolution_code.IntoColab(cl_env)
        )
        self.play(cl_env.Run(cell=-1))

        # SLIDE 28:  ===========================================================
        # SLIDING KERNEL WHILE BLURRED IMAGE FILLS IN (3b1b ANIMATION)
        self.next_slide(
            notes=
            '''Now we need to loop over all the internal rows and columns of the
            original matrix to process the whole image!
            '''
        )
        self.play(cl_env.FadeOut())
        self.clear()

        sample_A_PA: PixelArray = PixelArray(sample_A, stroke_width=0.75).set_height(0.6*FRAME_HEIGHT)
        blurred_A = np.round(separable_box_blur(sample_A, 3, mode = "constant", cval=0)).astype(np.uint8)  # with zero padding for later
        blurred_A_PA: PixelArray = PixelArray(blurred_A[1:-1, 1:-1], stroke_color=WHITE, stroke_width=1).set_height(10/12*sample_A_PA.height)
        kernel_array, kernel_values = sample_A_PA.get_kernel_array(np.ones((3,3))/9, kernel_color=SATURATED_BLUE, kernel_stroke_width=4, add_values=True, kernel_tex = " 1 / 9", values_size_fator=0.5)
        kernel = VGroup(kernel_array, kernel_values)
        pixel_highlight = blurred_A_PA.get_pixel_highlight(color=SATURATED_BLUE, stroke_width=4)

        LC = LiveConvolution(sample_A_PA, blurred_A_PA, kernel, pixel_highlight, kernel_size=3,
                             background_color=LIGHTER_GRAY)
        self.play(FadeIn(sample_A_PA, blurred_A_PA, kernel, kernel, pixel_highlight))
        
        LC.setup()
        self.play(LC.SlideKernel(end=99, run_time=4))
        blurred_A_PA.pixel_array.set_fill(opacity=1)
        LC.clear_updaters()

        # SLIDE 29:  ===========================================================
        # SCHEMATIC DRAWING OF THE FUNCTION IS BROUGHT IN
        self.next_slide(
            notes=
            '''For this we create a second function, called im_filtering,, that
            takes as input the image, as a matrix and a kernel and returns....
            '''
        )
        DSS.reset()
        DSS.add_empty_side_obj(FRAME_HEIGHT)
        DSS.hard_bring_in()
        self.add(DSS)
        self.play(DSS.bringOut(), VGroup(sample_A_PA, blurred_A_PA, kernel, kernel, pixel_highlight).animate.shift(FRAME_HEIGHT*UP))
        self.remove(sample_A_PA, blurred_A_PA, kernel, kernel, pixel_highlight)

        im_filtering_code = ColabCode(
            r'''
            # Image Convolution
            def im_filtering(A, K):
                rows, cols = A.shape
                # Create an output matrix for the result
                R = np.zeros(shape=(rows - 2, cols - 2))

                for i in range(1, rows - 1):      # internal rows
                    for j in range(1, cols - 1):  # internal columns
                        R[i - 1, j - 1] = local_convolution(A, K, i, j)

                return R
            '''
        )

        self.play(im_filtering_code.TypeLetterbyLetter(lines=[0]))

        fscheme = FunctionAbstraction(scale=0.7)
        DSS.add_side_obj(fscheme)
        DSS.add_main_obj(im_filtering_code[0], follow_obj=im_filtering_code[1:])
        self.play(DSS.bringIn())

        fscheme.add_inputs("A", "K")
        self.remove(fscheme.InputArrows, fscheme.InputLabels)
        A_label = MathTex("A", color=BLACK).next_to(fscheme.InputLabels[0], LEFT, buff=1)
        K_label = MathTex("K", color=BLACK).next_to(fscheme.InputLabels[1], LEFT, buff=1)
       
        self.play(
            Succession(
                im_filtering_code.TypeLetterbyLetter(lines=[1]),
                FadeIn(fscheme.InputArrows[0], fscheme.InputLabels[0], A_label),
                FadeIn(fscheme.InputArrows[1], fscheme.InputLabels[1], K_label)
            )
        )

        # SLIDE 30:  ===========================================================
        # OUTPUTS OF FUNCTION SCHEME APPEAR
        # RETURN LINE WRITTEN
        self.next_slide(
            notes=
            '''... returns the filtered image.
            '''
        )
        fscheme.add_outputs("R")
        R_label = MathTex("R", color=BLACK).next_to(fscheme.OutputLabels, RIGHT, buff=1)
       
        self.play(
            Succession(
                im_filtering_code.TypeLetterbyLetter(lines=[-1]),
                FadeIn(fscheme.OutputArrows, fscheme.OutputLabels, R_label)
            )
        )

        # SLIDE 31:  ===========================================================
        # FOR LOOPS WRITTEN
        self.next_slide(
            notes=
            '''As we have seen, we only process the internal pixels of the image
            and so, our filtered image will be smaller then the original one! If
            rows and cols indicate the size of A, the resulting matrix R will be
            rows-2 times cols-2. The result can be initialized in this way as a
            matrix of zeros.
            '''
        )
        DSS.add_main_obj(VGroup(im_filtering_code[:2], im_filtering_code[-1]), follow_obj=im_filtering_code[2:-1])
        self.play(DSS.bringOut(), VGroup(A_label, K_label, R_label).animate.shift(UP*DSS.secondaryRect.height))

        self.play(im_filtering_code.TypeLetterbyLetter(lines=[2,3,4]))
        rows_cols_2_highlight = HighlightRectangle(im_filtering_code[4][18:31])
        self.play(Create(rows_cols_2_highlight))

        # SLIDE 32:  ===========================================================
        # FOR LOOPS WRITTEN
        self.next_slide(
            notes=
            '''Now, we perform two loops: one for the rows, one for the columns.
            '''
        )
        self.play(FadeOut(rows_cols_2_highlight))
        self.play(im_filtering_code.TypeLetterbyLetter(lines=[6]))
        self.wait(0.5)
        self.play(im_filtering_code.TypeLetterbyLetter(lines=[7]))

        # SLIDE 33:  ===========================================================
        # HIGHLIGHT THE INDICES IN THE CODE
        self.next_slide(
            notes=
            '''Notice that the index i starts from one and ends before rows-1,
            to skip the first and last rows of the original matrix, and
            similarly for the columns
            '''
        )
        indices_highlight = VGroup(
            HighlightRectangle(im_filtering_code[6][12:20]),
            HighlightRectangle(im_filtering_code[7][12:20]),
        )

        self.play(Create(indices_highlight))

        # SLIDE 34:  ===========================================================
        # LOCAL CONVOLUTION LINE WRITTEN
        self.next_slide(
            notes=
            '''For each i,j, we perform local convolution, calling the function
            we wrote before, and store the result in R, which remember, is
            row-2Xcols-2!
            '''
        )
        self.play(FadeOut(indices_highlight))
        self.play(im_filtering_code.TypeLetterbyLetter(lines=[8]))

        # SLIDE 35:  ===========================================================
        # HIGHLIGHT INDEX EXPRESSION
        self.next_slide(
            notes=
            '''Pay attention here: we store the value in position i-1, j-1,
            because indices start from zero!
            '''
        )
        ij_highlight = HighlightRectangle(im_filtering_code[8][2:9])
        self.play(Create(ij_highlight))

        # SLIDE 36:  ===========================================================
        # HIGHLIGHT RETURN R 
        self.next_slide(
            notes=
            '''We finally return the result.
            '''
        )
        return_highlight = HighlightRectangle(im_filtering_code[-1])
        self.play(ReplacementTransform(ij_highlight, return_highlight))

        # SLIDE 37:  ===========================================================
        # INTO COLAB, RUN CELL
        # CURSOR CLICKS +CODE, NEW CELL APPEARS, OUT OF COLAB
        # BLURRING KERNEL COMMENT WRITTEN
        self.next_slide(
            notes=
            '''We are almost ready to use this function, but first we have to
            declare a blurring kernel.
            '''
        )
        self.play(FadeOut(return_highlight))
        im_filtering_code.add_background_window(DSS.mainRect.suspend_updating())
        self.remove(DSS) # not needed anymore
        self.play(im_filtering_code.IntoColab(cl_env))
        self.play(cl_env.Run(cell=1, new_cursor=False))
        self.play(Succession(cl_env.cursor.animate.move_to(cl_env.PLUS_CODE_), cl_env.cursor.Click()))
        cl_env.add_cell()
        self.wait(0.3)
        self.play(cl_env.OutofColab(cl_env.get_cell(2)))
        
        blurring_code = ColabCode(
            r'''
            # Blurring kernel
            K = np.ones([3, 3]) / 9
            R = im_filtering(A, K)
            compare_images(A, R)
            '''
        )

        self.play(blurring_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 38:  ===========================================================
        # KERNEL DEFINTION WRITTEN
        self.next_slide(
            notes=
            '''We create a 3x3 matrix of ones, and then divide it by 9!.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 39:  ===========================================================
        # IM_FILTERING LINE WRITTEN
        self.next_slide(
            notes=
            '''Now we call the function im_filtering, to compute the resulting
            matrix.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 40:  ===========================================================
        # COMPARE_IMAGES LINE WRITTEN
        # INTO COLAB; RUN CELL, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''Finally, we use the function compare_images to visualize the
            original and the filtered images.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[3]))
        self.wait(0.5)
        blurring_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()
        self.play(blurring_code.IntoColab(cl_env))

        # Create the output of compare_images
        sample_A_image = ImageMobject(sample_A).scale_to_fit_height(0.22*FRAME_HEIGHT)
        sample_A_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        blur_result_image = ImageMobject(blurred_A[1:-1, 1:-1]).scale_to_fit_height(sample_A_image.height*10/12)
        blur_result_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        Group(sample_A_image, blur_result_image).arrange(buff=0.5)
        image_titles = VGroup(
            Text("Image 1", font=CODE_FONT, font_size=14, color=BLACK).next_to(sample_A_image, UP, buff=0.15),
            Text("Image 2", font=CODE_FONT, font_size=14, color=BLACK).next_to(blur_result_image, UP, buff=0.15),
        )
        image_titles[1].match_y(image_titles[0])
        compare_image_output = Group(sample_A_image, blur_result_image, image_titles)
        cl_env.get_cell(2).add_output(
            compare_image_output
        )

        self.play(cl_env.Run(cell=2, new_cursor=False))

        # Save last cell to be reused later
        cl_env.get_cell(2).save_state()
        for obj in compare_image_output:
            obj.save_state() 

        # SLIDE 41:  ===========================================================
        # FOCUS ON OUTPUT
        # HIGHLIGHT THAT THE SECOND IMAGE IS SMALLER 
        self.next_slide(
            notes=
            '''As we can see, the original image has been blurred. But it's also
            a bit smaller! Can we avoid this "side effect"?
            '''
        )
        self.play(cl_env.FocusOutput(cell=2))

        # Create the border highlight (square with square hole)
        size_difference_highlight = VMobject(fill_color=SATURATED_BLUE, fill_opacity=0.4, stroke_color=SATURATED_BLUE, stroke_width=4)
        size_difference_highlight.append_points(Square(sample_A_image.height).get_points()[::-1])
        size_difference_highlight.append_points(Square(blur_result_image.height).get_points())
        size_difference_highlight.move_to(sample_A_image)

        self.play(
            Succession(
                FadeIn(size_difference_highlight),
                Wait(1),
                ApplyMethod(size_difference_highlight.move_to, blur_result_image)
            )
        )

        # SLIDE 42:  ===========================================================
        # ORIGINAL IMAGE IS MOVED TO CENTER; SECOND ONE DISAPPEARS
        # BLACK SQUARES PADDING IS ADDED TO ORIGINAL IMAGE
        self.next_slide(
            notes=
            '''We can perform PADDING on the original image before filtering. We
            basically add pixels all around the original image so that every
            pixel has 8 neighbors around it! The easiest option is to add black
            pixels.
            '''
        )
        padding_title = Text('Padding', font_size=64, color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).to_edge(UP).shift(UP*0.5)
        self.play(
            AnimationGroup(
                FadeOut(size_difference_highlight, image_titles, blur_result_image),
                sample_A_image.animate.center().shift(DOWN*1),
                Write(padding_title),
                lag_ratio=0.5
            )
        )

        # Create padding pixels
        sample_A_image.set_z_index(1)
        padded_pixel_array = PixelArray(np.zeros((14, 14)), stroke_width=0.75).scale_to_fit_height(sample_A_image.height*14/12).move_to(sample_A_image)
        padding_pixels = VGroup()
        for i in range(14):
            padding_pixels.add(padded_pixel_array.pixel_array[0, i])
            padding_pixels.add(padded_pixel_array.pixel_array[i, 0])
        for i in range(14):
            padding_pixels.add(padded_pixel_array.pixel_array[i, -1])
            padding_pixels.add(padded_pixel_array.pixel_array[-1, i])
        padding_pixels.set_stroke(width=2)
        self.play(
            AnimationGroup(
                *[GrowFromCenter(pixel) for pixel in padding_pixels],
                lag_ratio=0.1,
                run_time=2
            )
        )

        # SLIDE 43:  ===========================================================
        # FADEOUT TO CODE
        # INITIALIZE A PADDED WRITTEN
        self.next_slide(
            notes=
            '''We can do it in this way: we create a larger matrix Ap, with 2
            rows and two columns more than A.
            '''
        )
        bp = FullScreenBackground(COLAB_LIGHTGRAY).set_z_index(2)
        self.play(FadeIn(bp))
        # cl_env.clear()
        sample_A_image.set_z_index(0)
        self.remove(sample_A_image, padding_title, padding_pixels, *padding_pixels.submobjects)
        bp.set_z_index(0)

        padding_code = ColabCode(  # Careful with the white spaces!
            r'''
            # Padding
            Ap = np.zeros((A.shape[0] + 2, A.shape[1] + 2))
            Ap[1:-1, 1:-1] = A

            Rp = im_filtering(Ap, K) 
            compare_images(A, Rp)
            '''
        )

        self.play(padding_code.TypeLetterbyLetter(lines=[0, 1]))

        # SLIDE 44:  ===========================================================
        # COPY A to A PADDED WRITTEN
        self.next_slide(
            notes=
            '''Then we copy A in Ap, starting from the second to the second to
            last row, and similarly for the columns.
            '''
        )
        self.play(padding_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 45:  ===========================================================
        # IMAGE FILTERING LINES WRITTEN
        # INTO COLAB, RUN CELL, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''And now, we can repeat image filtering giving Ap as an input, and
            obtain a blurred image which has the same size of the original.
            '''
        )
        self.play(padding_code.TypeLetterbyLetter(lines=[4,5]))
        self.wait(0.5)

        bp.set_z_index(-1)
        padding_code.add_background_window(bp)
        # "restore" (which is just "become") does not apply the z_index of hte saved state
        cl_env.get_cell(2).restore().set_z_index(-3)
        for obj in compare_image_output:
            obj.restore().set_z_index(-3)
        cl_env.remove_cell_from_top(n=2)
        cl_env.cursor.move_to(cl_env.get_cell(0).playButton)
        self.play(padding_code.IntoColab(cl_env))
        
        # Create updated output for compare_images
        second_compare_image_output = Group(sample_A_image.copy(), image_titles.copy())
        padded_blur_result_image = ImageMobject(blurred_A).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        padded_blur_result_image.match_height(sample_A_image).next_to(image_titles[1], DOWN).match_y(sample_A_image)
        second_compare_image_output.add(padded_blur_result_image)
        cl_env.get_cell(1).add_output(
            second_compare_image_output
        )

        self.play(cl_env.Run(cell=1, new_cursor=False))

        # SLIDE 46:  ===========================================================
        # FOCUS ON OUTPUT
        self.next_slide(
            notes=
            '''We can now replace our image with a version that respects
            privacy. [END]
            '''
        )
        self.play(cl_env.FocusOutput(cell=1))
