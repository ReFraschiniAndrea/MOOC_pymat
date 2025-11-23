import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from mooc_utils.colab import *
from W4Anim import DiscreteConvolutionPseudoCode


config.update(TEST_CONFIG)


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
        cl_env = ColabEnv(self, r'Assets\W3\colabGD.png')
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
        # SIDE BAR APPEARS
        self.next_slide(
            notes=
            '''on the folder icon on the left, [CLICK] a side bar appears
            showing the list of available files.
            '''
        )
        self.play(hand_cursor.animate.move_to(cl_env.MENU_))
        self.play(hand_cursor.Click())
        cl_env.set_image(r'Assets\W3\colabGD_sidemenu.png')

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
        self.play(hand_cursor.animate.move_to(cl_env.UPLOAD_))
        self.play(hand_cursor.Click())
        cl_env.set_image(r'Assets\W3\colabGD_uploaded.png')

        # SLIDE 05:  ===========================================================
        # OUT OF COLAB
        # IMPORT NUMPY AND MATPLOTLIB WRITTEN
        self.next_slide(
            notes=
            '''We need to import the numpy and matplotlib modules.
            '''
        )
        cl_env.set_image(r'Assets\W3\colabGD.png')
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
        # INTO COLAB, RUN CELL
        self.next_slide(
            notes=
            '''and with this syntax we import all the functions contained in
            helper_functions.py, to help us visualize the results.
            '''
        )
        self.play(import_code.TypeLetterbyLetter(lines=[3]))

        background_rectangle = cl_env.cells[0].window.copy().set_z_index(-1)
        import_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()
        self.play(import_code.IntoColab(cl_env))
        self.play(cl_env.Run(cell=0))

        # SLIDE 08:  ===========================================================
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

        # SLIDE 09:  ===========================================================
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

        # SLIDE 10:  ===========================================================
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

        # SLIDE 11:  ===========================================================
        # PRINT TYPE AND SHAPE LINE WRITTEN
        self.next_slide(
            notes=
            '''With the instructions "type" and "shape" we can extract some
            information on A:
            '''
        )
        self.play(load_image_code.TypeLetterbyLetter(lines=[4]))

        # SLIDE 12:  ===========================================================
        # INTO COLAB, RUN CELL, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''as we can see it is an n-dimensional numpy array of shape 12X12
            '''
        )
        import_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()
        self.play(load_image_code.IntoColab(cl_env))

        cl_env.cells[1].add_output(
            r"data type <class 'numpy.ndarray'> shape (12, 12)"
        )
        self.play(cl_env.Run(cell=1, new_cursor=False))
        
        # SLIDE 13:  ===========================================================
        # CURSOR CLICKS +CODE, NEW CELL APEARS, OUT OF COLAB
        # PRINT(A[1,2]) WRITTEN
        self.next_slide(
            notes=
            '''How does a 2D-numpy array work? To access to an element we use
            square brackets. For example the element in position 1, 2 can be
            accessed as A[1, 2].
            '''
        )
        bp = FullScreenBackground(COLAB_LIGHTGRAY)
        self.play(FadeIn(bp))
        cl_env.clear()
        # self.play(Succession(cl_env.cursor.animate.move_to(cl_env.PLUS_CODE_), cl_env.cursor.Click()))
        # cl_env.add_cell()
        # self.play(cl_env.OutofColab(cell=2))
        square_brackets_code = ColabCode(
            r'''
            print(A[1, 2])
            '''
        )
        self.play(square_brackets_code.TypeLetterbyLetter())
        square_brackets_code.add_background_window(bp)
        cl_env.remove_cell()
        self.play(square_brackets_code.IntoColab(cl_env))
        cl_env.get_cell(0).add_output(
            "250"
        )
        self.play(cl_env.Run(cell=0))

        # SLIDE 14:  ===========================================================
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
        self.play(print_matrix_cell.TypeLetterbyLetter())
       
        # self.play(square_brackets_code.TypeLetterbyLetter())
        # square_brackets_code.add_background_window(cl_env.get_cell(3).window.copy())
        # cl_env.remove_cell()
        # self.play(square_brackets_code.IntoColab())
        print_matrix_cell.add_output(
            "PLACEHOLDER"
        )
        self.play(cl_env.Run(cell=1, new_cursor=False))
    
        # SLIDE 15:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Instead, by using imshow from matplot lib we can display the
            matrix as an image; we just need to specify that we are working with
            a greyscale image.
            '''
        )
        self.play(Succession(cl_env.cursor.animate.move_to(cl_env.PLUS_CODE_), cl_env.cursor.Click()))
        cl_env.add_cell()
        self.play(cl_env.OutofColab(cell=2))
        imshow_code = ColabCode(
            r'''
            plt.imshow(A, "grey")
            '''
        )
        self.play(imshow_code.TypeLetterbyLetter())
        imshow_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()
        self.play(imshow_code.IntoColab(cl_env))
        # cl_env.get_cell(2).add_output(
        #     "250"
        # )
        self.play(cl_env.Run(cell=2))


        # SLIDE 16:  ===========================================================
        # 
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

        # SLIDE 17:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Let us create a function called local_convolution. This function
            takes as input the matrix, the kernel, and the indices of the pixel
            and returns the value of the filtered pixel.
            '''
        )
        DSS = DynamicSplitScreen(main_color=COLAB_LIGHTGRAY, side_color=WHITE)
        DSS.add_side_obj(pc.copy().scale(0.6))
        DSS.hard_bring_in()
        self.play(
            AnimationGroup(
                FadeIn(DSS.mainRect, DSS.secondaryRect),
                ReplacementTransform(pc, DSS.secondaryObj)
            )
        )

        local_convolution_code = ColabCode(
            r'''
            def local_convolution(M, K, i, j):
                v = 0
                for m in range(3):  # rows
                    for n in range(3):  # columns
                        v += M[i - 1 + m, j - 1 + n] * K[m, n]
                return v
            '''
        ).move_to(DSS.mainRect)
        input_highlight = HighlightRectangle(pc[1])

        self.play(local_convolution_code.TypeLetterbyLetter(lines=[0]), Create(input_highlight))

        # SLIDE 18:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''First of all, we declare the variable v and set it to zero.
            '''
        )
        v_highlight = HighlightRectangle(pc[2][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[1]), ReplacementTransform(input_highlight, v_highlight))

        # SLIDE 19:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Then, we introduce two for loops: The first one to access the
            rows, ...
            '''
        )
        first_loop_highlight = HighlightRectangle(pc[3][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[2]), ReplacementTransform(v_highlight, first_loop_highlight))

        # SLIDE 20:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''And the second to access the columns of the kernel. Notice that
            these two loops are NESTED, meaning, they are one inside the other.
            '''
        )
        second_loop_highlight = HighlightRectangle(pc[4][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[3]), ReplacementTransform(first_loop_highlight, second_loop_highlight))

        # SLIDE 21:  ===========================================================
        # CONVOLUTION LINE WRITTEN
        # 
        self.next_slide(
            notes=
            '''Now for each pixel i,j, and for each position in the kernel, we
            compute the product of the value in the kernel, and the
            corresponding pixel in the image... and we add this value to v.
            '''
        )
        convolution_highlight = HighlightRectangle(pc[5][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[4]), ReplacementTransform(second_loop_highlight, convolution_highlight))

        # SLIDE 22:  ===========================================================
        # RETURN LINE WRITTEN
        self.next_slide(
            notes=
            '''Finally, we return v, the value of a single filtered pixel.
            '''
        )
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[5]), FadeOut(convolution_highlight))

        local_convolution_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(
            DSS.bringOut(),
            local_convolution_code.IntoColab(cl_env)
        )
        self.play(cl_env.Run(cell=-1))

        # SLIDE 23:  ===========================================================
        # SLIDING KERNEL WHILE BLURRED IMAGE FILLS IN (3b1b ANIMATION)
        self.next_slide(
            notes=
            '''Now we need to loop over all the internal rows and columns of the
            original matrix to process the whole image!
            '''
        )
        self.play(cl_env.FadeOut())

        # SLIDE 24:  ===========================================================
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

        im_filtering_code = ColabCode(
            r'''
            # Image convolution
            def im_filtering(M, K):
                rows, cols = M.shape
                # Create an output matrix for the result
                R = np.zeros(shape=(rows - 2, cols - 2))

                for i in range(1, rows - 1):  # matrix internal rows
                    for j in range(1, cols - 1):  # matrix internal columns
                        R[i - 1, j - 1] = local_convolution(M, K, i, j)

                return R
            '''
        )

        self.play(DSS.bringOut())
        self.play(im_filtering_code.TypeLetterbyLetter(lines=[0]))

        fscheme = FunctionAbstraction(scale=0.7)
        DSS.add_side_obj(fscheme)
        DSS.add_main_obj(im_filtering_code[0], follow_obj=im_filtering_code[1:])
        self.play(DSS.bringIn())

        fscheme.add_inputs("A", "K")
       
        self.play(
            im_filtering_code.TypeLetterbyLetter(lines=[1]),
            FadeIn(fscheme.InputArrows, fscheme.InputLabels)
        )

        # SLIDE 25:  ===========================================================
        # OUTPUTS OF FUNCTION SCHEME APPEAR
        # RETURN LINE WRITTEN
        self.next_slide(
            notes=
            '''... returns
            '''
        )
        fscheme.add_outputs("R")
       
        self.play(
            im_filtering_code.TypeLetterbyLetter(lines=[-1]),
            FadeIn(fscheme.OutputArrows, fscheme.OutputLabels)
        )

        # SLIDE 26:  ===========================================================
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
        self.play(im_filtering_code.TypeLetterbyLetter(lines=[2,3,4]))

        # SLIDE 27:  ===========================================================
        # FOR LOOPS WRITTEN
        self.next_slide(
            notes=
            '''Now, we perform two loops: one for the rows, one for the columns.
            '''
        )
        self.play(im_filtering_code.TypeLetterbyLetter(lines=[6,7]))

        # SLIDE 28:  ===========================================================
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

        # SLIDE 29:  ===========================================================
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

        # SLIDE 30:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Pay attention here: we store the value in position i-1, j-1,
            because indices start from zero!
            '''
        )
        ij_highlight = HighlightRectangle(im_filtering_code[8][2:9])
        self.play(Create(ij_highlight))

        # SLIDE 31:  ===========================================================
        # HIGHLIGHT RETURN R 
        self.next_slide(
            notes=
            '''We finally return the result.
            '''
        )
        return_highlight = HighlightRectangle(im_filtering_code[-1])
        self.play(ReplacementTransform(ij_highlight, return_highlight))

        # SLIDE 32:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''We are almost ready to use this function, but first we have to
            declare a blurring kernel.
            '''
        )
        self.play(FadeOut(return_highlight))
        self.play(im_filtering_code.IntoColab(cl_env))
        self.play(cl_env.Run(cell=1))
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

        # SLIDE 33:  ===========================================================
        # KERNEL DEFINTION WRITTEN
        self.next_slide(
            notes=
            '''We create a 3x3 matrix of ones, and then divide it by 9!.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 34:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Now we call the function im_filtering, to compute the resulting
            matrix.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 35:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Finally, we use the function compare_images to visualize the
            original and the filtered images.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[3]))
        blurring_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()
        self.play(blurring_code.IntoColab(cl_env))
        cl_env.get_cell(2).add_output("PLACEHOLDER")
        self.play(cl_env.Run(cell=2))

        # SLIDE 36:  ===========================================================
        # HIGHLIGHT 
        self.next_slide(
            notes=
            '''As we can see, the original image has been blurred. But it's also
            a bit smaller! Can we avoid this "side effect"?
            '''
        )
        # SLIDE 37:  ===========================================================
        # PADDING IS ADDED TO THE ORIGINAL IMAGE
        self.next_slide(
            notes=
            '''We can perform PADDING on the original image before filtering. We
            basically add pixels all around the original image so that every
            pixel has 8 neighbors around it! The easiest option is to add black
            pixels.
            '''
        )
        # SLIDE 38:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''We can do it in this way: we create a larger matrix Ap, with 2
            rows and two columns more than A.
            '''
        )
        # SLIDE 39:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''Then we copy A in Ap, starting from the second to the second to
            last row, and simlarly for the columns.
            '''
        )
        # SLIDE 40:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''And now, we can repeat image filtering giving Ap as an input, and
            obtain a blurred image which has the same size of the original. We
            can now replace our image with a version that respects privacy.
            '''
        )
