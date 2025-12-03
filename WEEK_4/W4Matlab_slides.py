import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from mooc_utils.matlab import *
from W4Anim import DiscreteConvolutionPseudoCode, separable_box_blur, LiveConvolution, SATURATED_BLUE, Kernel3X3IndexAnimation
from PIL import Image
from io import StringIO

config.update(TEST_CONFIG)

WINDOW_BUFF = 0.25
PLOT_IMAGE_HEIGHT = 0.4*FRAME_HEIGHT

class W4Matlab_slides(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # EMPTY MATLAB ENVIRONMENT, PSEUDO-CODE APPEARS ON TOP
        self.next_slide(
            notes=
            '''Let's open MATLAB, and let's explore how to code the discrete
            convolution algorithm to compute filtered images.
            '''
        )
        mat_env = MatlabEnv(self, r'Assets\W4\matlab_empty.png')
        mat_env.RUN_BUTTON_ = MatlabEnv._pixel2p(1099, 67)
        mat_env.SAVE_PROMPT_BUTTON_ = MatlabEnv._pixel2p(877, 859)
        mat_env.OK_PROMPT_ = MatlabEnv._pixel2p(877, 816)

        pc = DiscreteConvolutionPseudoCode()
        pc.scale_to_fit_width(FRAME_WIDTH*0.65).center()
        pc.save_state()
        surrounding_rect = SurroundingRectangle(pc, fill_color=WHITE, fill_opacity=1, stroke_width=0.5,
                                              stroke_color=BLACK, corner_radius=0.2, buff=0.5).set_z_index(-0.5)
        
        self.play(FadeIn(mat_env.background))
        self.wait(1)
        self.play(FadeIn(surrounding_rect, pc))

        # SLIDE 02:  ===========================================================
        # PSEUDO-CODE FADES, CURSOR APPEARS
        self.next_slide(
            notes=
            '''First, we need to load an image and some additional functions. To
            do this, ...
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
        # CURSOR MOVES TO BROWSE FOLDER AND CLICKS IT
        self.next_slide(
            notes=
            '''...click on the "Browse for folder" button and select the folder
            containing the files compare_images.m and the image part.png.
            '''
        )
        self.play(Succession(hand_cursor.animate.move_to(mat_env.BROWSE_FOLDER_), hand_cursor.Click()))
        mat_env.set_image(r'Assets\W4\matlab_browsefolder.png')

        # SLIDE 04:  ===========================================================
        # CLICK OK, RETURN TO EMPTY ENVIRONMENT
        # CLCIK CURRENT FOLDER, SIDEMENU APPEARS
        # HIGHLIGHT FILES IN CURRENT FOLDER
        self.next_slide(
            notes=
            '''Then, check that the file appears in the "Current Folder" panel
            [CLICK]. This confirms that MATLAB can locate the file before we
            proceed with reading it. The function compare_images helps us to
            visualize the results with respect to the orginal image. While, some
            of the functions we use for the images comes from the Image
            Processing Toolbox, which is installed with matlab by defualt.
            '''
        )
        self.play(Succession(hand_cursor.animate.move_to(mat_env.OK_PROMPT_), hand_cursor.Click()))
        mat_env.set_image(r'Assets\W4\matlab_empty.png')
        self.play(Succession(hand_cursor.animate.move_to(mat_env.SIDEMENU_), hand_cursor.Click()))
        mat_env.set_image(r'Assets\W4\matlab_sidemenu.png')
        phony_files = Rectangle(width=180*mat_env.PIXEL, height=48*mat_env.PIXEL).move_to(MatlabEnv._pixel2p(58, 248), aligned_edge=UL).set_opacity(0)
        self.wait(0.5)
        self.play(Circumscribe(phony_files, color=BLUE, run_time=2))

        # SLIDE 05:  ===========================================================
        # CURSOR CLICKS NEW SCRIPT, SCRIPT IS SAVED WITH NAME
        self.next_slide(
            notes=
            '''Now we are ready to create a new script, named "week4.m".
            '''
        )
        self.play(Succession(hand_cursor.animate.move_to(mat_env.NEW_SCRIPT_), hand_cursor.Click()))
        mat_env.set_image(r'Assets\W4\matlab_newscript.png')
        self.play(Succession(hand_cursor.animate.move_to(mat_env.SAVE_), hand_cursor.Click()))
        mat_env.set_image(r'Assets\W4\matlab_saveW4.png')
        self.play(Succession(hand_cursor.animate.move_to(mat_env.SAVE_PROMPT_BUTTON_), hand_cursor.Click()))
        mat_env.set_image(r'Assets\W4\matlab_week4.png')

        # SLIDE 06:  ===========================================================
        # OUT OF MATLAB
        # IMREAD WRITTEN
        self.next_slide(
            notes=
            '''Let us load the image as a matrix: First, we use the function
            imread to load the image contained in "part.png".
            '''
        )
        mat_env.add_cell()
        self.play(mat_env.OutofMatlab(cell=0), FadeOut(hand_cursor))

        load_image_code = MatlabCode(
            r'''
            % Load the image
            file_name = "part.png";
            A_color = imread(file_name);
            A_g = rgb2gray(A_color);
            A = double(A_g);
            whos A
            '''
        )

        self.play(load_image_code.TypeLetterbyLetter(lines=[0]))
        self.play(load_image_code.TypeLetterbyLetter(lines=[1,2]))

        # SLIDE 07:  ===========================================================
        # CONVERT LINE WRITTEN
        self.next_slide(
            notes=
            '''Then, we convert A_color in greyscale using the function
            "rgb2gray".
            '''
        )
        self.play(load_image_code.TypeLetterbyLetter(lines=[3]))

        # SLIDE 08:  ===========================================================
        # DOUBLE() LINE WRITTEN
        self.next_slide(
            notes=
            '''And finally, we convert the values from integer to double with
            this syntax.
            '''
        )
        self.play(load_image_code.TypeLetterbyLetter(lines=[4]))

        # SLIDE 09:  ===========================================================
        # WHOS A LINE WRITTEN
        self.next_slide(
            notes=
            '''With the instruction "whos" we can extract some information on A:
            '''
        )
        self.play(load_image_code.TypeLetterbyLetter(lines=[5]))

        # SLIDE 10:  ===========================================================
        # INTO MATLAB, RUN CODE, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''as we can see it is a matrix of shape 12X12.
            '''
        )
        load_image_code.add_background_window(FullScreenBackground(WHITE))
        mat_env.remove_cell()
        self.play(load_image_code.IntoMatlab(mat_env))

        mat_env.add_output(
            "A\t\t12x12\t\t1152\tdouble"
        )
        self.play(mat_env.Run())
        
        # SLIDE 11:  ===========================================================
        # OUT OF MATLAB
        # DISP(A(2,3)) WRITTEN
        self.next_slide(
            notes=
            '''To access a matrix element, we use round brackets.
            '''
        )
        sample_A = np.array(Image.open(r'Assets\W4\part.png'), dtype=np.uint8)

        bp = FullScreenBackground(WHITE)
        self.play(FadeIn(bp))
        # Switch to wider command window for later
        mat_env.clear()
        mat_env.set_image(r"Assets\W4\matlab_week4Large.png")
        mat_env.OUTPUT_TOP_LEFT_CORNER_ = MatlabEnv._pixel2p(80, 715)

        square_brackets_code = MatlabCode(
            r'''
            disp(A(2, 3));
            '''
        )
        self.play(square_brackets_code.TypeLetterbyLetter())

        # SLIDE 12:  ===========================================================
        # INTO MATLAB, RUN CODE, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''For example, the element in position 2, 3 can be accessed as A(2,
            3).
            '''
        )
        square_brackets_code.add_background_window(bp)
        mat_env.remove_cell()
        self.play(square_brackets_code.IntoMatlab(mat_env))
        mat_env.add_output(
            str(sample_A[1,2])
        )
        self.play(mat_env.Run())

        # SLIDE 13:  ===========================================================
        # DISP(A) WRITTEN
        # RUN CODE, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''We can also display the matrix A. Notice the values 255 for the
            white parts of the image.
            '''
        )
        print_matrix_cell = MatlabCodeBlock(
            r'''
            disp(A);
            '''
        )
        mat_env.add_cell(print_matrix_cell)
        self.remove(print_matrix_cell.code)
       
        print_A_result_stream = StringIO()
        print(sample_A, file=print_A_result_stream)
        print_A_result = print_A_result_stream.getvalue()
        print_A_result_stream.close()
        print_A_result = print_A_result.replace('[', '').replace(']', '')
        mat_env.add_output(
            print_A_result
        )

        self.play(print_matrix_cell.TypeLetterbyLetter())
        self.wait(1)
        self.play(mat_env.Run(new_cursor=False))
    
        # SLIDE 14:  ===========================================================
        # OUT OF MATLAB
        # IMSHOW LINE WRITTEN
        self.next_slide(
            notes=
            '''Instead, by using imshow we can display the matrix as an image.
            '''
        )
        mat_env.add_cell()
        self.play(mat_env.OutofMatlab(cell=2))
        imshow_code = MatlabCode(
            r'''
            imshow(A, [0, 255], 'InitialMagnification', 6000);
            '''
        )
        self.play(imshow_code.TypeLetterbyLetter())

        # SLIDE 15:  ===========================================================
        # HIGHLIGHT 0-255 RANGE
        self.next_slide(
            notes=
            '''We need to specify the range 0, 255 for the greyscale.
            '''
        )
        range_0255_highlight = HighlightRectangle(imshow_code[0][10:15])
        self.play(Create(range_0255_highlight))

        # SLIDE 16:  ===========================================================
        # HIGHLIGHT INITIAL MAGNIFICATION
        self.next_slide(
            notes=
            '''Moreover, since it's very small image, we zoom in using the
            option "InitialMagnification",6000
            '''
        )
        magnification_highlight = HighlightRectangle(imshow_code[0][17:-2])
        self.play(ReplacementTransform(range_0255_highlight, magnification_highlight))

        # SLIDE 17:  ===========================================================
        # INTO MATLAB, RUN CODE, IMAGE APPEARS
        self.next_slide(
            notes=
            '''[...]
            '''
        )
        imshow_code.add_background_window(FullScreenBackground(WHITE))
        mat_env.remove_cell()
        mat_env.remove_cursor()
        image_plot = ImageMobject(sample_A).set_resampling_algorithm(RESAMPLING_ALGORITHMS['nearest']).scale_to_fit_height(PLOT_IMAGE_HEIGHT)

        self.play(FadeOut(magnification_highlight))
        self.play(imshow_code.IntoMatlab(mat_env))
        mat_env.add_output_plot(
            image_plot, window_buff=WINDOW_BUFF
        )
        self.play(mat_env.Run())

        # SLIDE 18:  ===========================================================
        # FADEOUT ALL
        # PSEUDO-CODE APPEARS
        self.next_slide(
            notes=
            '''Now we can implement the local convolution applied to a specific
            pixel in position i.j as in this pseudocode.
            '''
        )
        self.play(mat_env.FadeOut())
        self.clear()
        mat_env.clear()
        mat_env.set_image(r"Assets\W4\matlab_week4.png")
        pc.restore()

        self.play(FadeIn(pc))

        # SLIDE 19:  ===========================================================
        # PSEUDO-CODE FADES OUT
        # FUNCTION DEFINITION LINE WRITTEN
        self.next_slide(
            notes=
            '''Let us create a function called local_convolution.
            '''
        )
        DSS = DynamicSplitScreen(main_color=WHITE, side_color=MATLAB_LIGHTGRAY)

        local_convolution_code = MatlabCode(
            r'''
            function [v] = local_convolution(A, K, i, j):
                v = 0;
                for m = 1 : 3      % kernel rows
                    for n = 1 : 3  % kernel columns
                        v = v + A(i - 2 + m, j - 2 + n) * K(m, n);
                    end
                end
            end
            '''
        ).move_to(DSS.mainRect)
        pc.set_z_index(-2)
        self.play(FadeIn(DSS))
        self.remove(pc)
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[0,-1]))

        # SLIDE 20:  ===========================================================
        # FUNCTION SCHEME INPUTS APPEAR
        self.next_slide(
            notes=
            '''This function takes as input the matrix, the kernel, and the
            indices of the pixel ...
            '''
        )
        fscheme1 = FunctionAbstraction(scale=0.7)
        DSS.add_side_obj(fscheme1)
        DSS.add_main_obj(VGroup(local_convolution_code[0], local_convolution_code[-1]), local_convolution_code[1:-1]) 
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

        # SLIDE 21:  ===========================================================
        # FUNCTION SCHEME "V" OUTPUT APPEARS
        self.next_slide(
            notes=
            '''... and returns the value of the filtered pixel. Before
            continuing, we remark there are different ways to do that, for
            instance, one could use vectorized operations and SUM. However, we
            implement loops as in the pseudocode.
            '''
        )
        fscheme1.add_outputs("v", font_size=32)
        v_label = MathTex("v", color=BLACK).next_to(fscheme1.OutputLabels, RIGHT, buff=1)

        self.play(
            FadeIn(fscheme1.OutputLabels, fscheme1.OutputArrows, v_label),
        )

        # SLIDE 22:  ===========================================================
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

        # SLIDE 23:  ===========================================================
        # FIRST FOR LOOP WRITTEN
        self.next_slide(
            notes=
            '''Then, we introduce two for loops: The first one to access the
            rows, ...
            '''
        )
        first_loop_highlight = HighlightRectangle(pc[3][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[2,-2]), ReplacementTransform(v_highlight, first_loop_highlight))

        # SLIDE 24:  ===========================================================
        # SECOND FOR LOOP WRITTEN
        self.next_slide(
            notes=
            '''And the second to access the columns of the kernel. Notice that
            these two loops are NESTED, meaning, they are one inside the other.
            '''
        )
        second_loop_highlight = HighlightRectangle(pc[4][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[3,-3]), ReplacementTransform(first_loop_highlight, second_loop_highlight))

        # SLIDE 25:  ===========================================================
        # CONVOLUTION LINE WRITTEN
        # (i,j) (m,n) INDEX ANIMATION SHOWN ON THE SIDE
        self.next_slide(
            notes=
            '''Now for each pixel i,j, and for each position in the kernel, we
            compute the product of the value in the kernel, and the
            corresponding pixel in the image... and we add this value to v.
            '''
        )
        convolution_highlight = HighlightRectangle(pc[5][2:])
        self.play(local_convolution_code.TypeLetterbyLetter(lines=[4]), ReplacementTransform(second_loop_highlight, convolution_highlight))
        self.wait()
        # DSS.remove_main_obj()
        self.play(DSS.bringOut(), convolution_highlight.animate.shift(UP*DSS.secondaryRect.height))

        # Create the animation for the indices
        three_by_three: PixelArray = PixelArray(sample_A[:3,:3], stroke_width=2, stroke_color=WHITE).set_height(0.45*FRAME_HEIGHT)
        kernel_array, kernel_values = three_by_three.get_kernel_array(np.ones((3,3))/9, kernel_color=BLACK, kernel_stroke_width=4, add_values=True, kernel_tex = " 1 / 9", values_size_fator=0.5)
        kernel = VGroup(kernel_array, kernel_values).match_height(three_by_three)
        KIA = Kernel3X3IndexAnimation(three_by_three, kernel, highlight_color=SATURATED_BLUE, highlight_stroke_width=6)

        DSS.add_side_obj(KIA.scale(0.7))
        DSS.add_main_obj(local_convolution_code[:])
        self.play(DSS.bringIn())
        self.wait(0.75)
        KIA.setup()
        self.play(KIA.IndexAnimation(slide_dt=0.65, wait_dt=0.75))
        KIA.clear_updaters()

        # SLIDE 26:  ===========================================================
        # HIGHLIGHT MATRIX INDICES
        self.next_slide(
            notes=
            '''Note the indices. Since matlab does not allow negative and null
            indices, we start from m=1, n=1, which correspond to i-1, j-1 if we
            use a shift of -2.
            '''
        )
        matrix_indices_highlight =  HighlightRectangle(local_convolution_code[4][6:17])
        self.play(Create(matrix_indices_highlight))

        # SLIDE 27:  ===========================================================
        # HIGHLIGHT [V] IN FUNCTION DEFINTION
        self.next_slide(
            notes=
            '''Finally, we return v, the value of a single filtered pixel.
            '''
        )
        DSS.add_main_obj(VGroup(local_convolution_code, matrix_indices_highlight))
        self.play(DSS.bringOut())

        return_highlight = HighlightRectangle(local_convolution_code[0][8:11])
        self.play(ReplacementTransform(matrix_indices_highlight, return_highlight))

        # SLIDE 28:  ===========================================================
        # INTO MATLAB
        self.next_slide(
            notes=
            '''[...]
            '''
        )
        DSS.remove_main_obj()
        local_convolution_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(FadeOut(return_highlight))
        self.play(
            DSS.bringOut(),
            local_convolution_code.IntoMatlab(mat_env)
        )
        # self.play(mat_env.Run())

        # SLIDE 29:  ===========================================================
        # SLIDING KERNEL WHILE BLURRED IMAGE FILLS IN (3b1b ANIMATION)
        self.next_slide(
            notes=
            '''Now to blur the image we loop over all the internal rows and
            columns of the original matrix and for each internal pixel we apply
            the function local_convolution.
            '''
        )
        self.play(mat_env.FadeOut())
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

        # SLIDE 30:  ===========================================================
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

        im_filtering_code = MatlabCode(
            r'''
            % Image Convolution
            function [R] = im_filtering(A, K)
                [rows, cols] = size(A);
                % Create an output matrix for the result
                R = zeros(rows - 2, cols - 2);

                for i = 2 : rows - 1     % internal rows
                    for j = 2 : cols -1  % internal columns
                        R(i - 1, j - 1) = local_convolution(A, K, i, j);
                    end
                end
            end
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
                im_filtering_code.TypeLetterbyLetter(lines=[1,-1]),
                FadeIn(fscheme.InputArrows[0], fscheme.InputLabels[0], A_label),
                FadeIn(fscheme.InputArrows[1], fscheme.InputLabels[1], K_label)
            )
        )

        # SLIDE 31:  ===========================================================
        # OUTPUTS OF FUNCTION SCHEME APPEAR
        # RETURN LINE WRITTEN
        self.next_slide(
            notes=
            '''... returns the filtered image.
            '''
        )
        fscheme.add_outputs("R")
        R_label = MathTex("R", color=BLACK).next_to(fscheme.OutputLabels, RIGHT, buff=1)
       
        self.play(FadeIn(fscheme.OutputArrows, fscheme.OutputLabels, R_label))

        # SLIDE 32:  ===========================================================
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
        rows_cols_2_highlight = HighlightRectangle(im_filtering_code[4][8:21])
        self.play(Create(rows_cols_2_highlight))

        # SLIDE 33:  ===========================================================
        # FOR LOOPS WRITTEN
        self.next_slide(
            notes=
            '''Now, we perform two loops: one for the rows, one for the columns.
            '''
        )
        self.play(FadeOut(rows_cols_2_highlight))
        self.play(im_filtering_code.TypeLetterbyLetter(lines=[6,-2]))
        self.wait(0.5)
        self.play(im_filtering_code.TypeLetterbyLetter(lines=[7,-3]))

        # SLIDE 34:  ===========================================================
        # HIGHLIGHT THE INDICES IN THE CODE
        self.next_slide(
            notes=
            '''Notice that the index i starts from two and ends with rows-1, to
            skip the first and last rows of the original matrix, and similarly
            for the columns.
            '''
        )
        indices_highlight = VGroup(
            HighlightRectangle(im_filtering_code[6][5:13]),
            HighlightRectangle(im_filtering_code[7][5:13]),
        )

        self.play(Create(indices_highlight))

        # SLIDE 35:  ===========================================================
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

        # SLIDE 36:  ===========================================================
        # HIGHLIGHT INDEX EXPRESSION
        self.next_slide(
            notes=
            '''Pay attention here: we store the value in position i-1, j-1,
            because indices start from zero!
            '''
        )
        ij_highlight = HighlightRectangle(im_filtering_code[8][2:9])
        self.play(Create(ij_highlight))

        # SLIDE 37:  ===========================================================
        # HIGHLIGHT RETURN R 
        self.next_slide(
            notes=
            '''We finally return the result.
            '''
        )
        return_highlight = HighlightRectangle(im_filtering_code[1][8:11])
        self.play(ReplacementTransform(ij_highlight, return_highlight))

        # SLIDE 38:  ===========================================================
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
        self.play(im_filtering_code.IntoMatlab(mat_env))
        self.play(mat_env.Run(new_cursor=False))
        mat_env.add_cell()
        self.wait(0.3)
        self.play(mat_env.OutofMatlab(cell=2))
        
        blurring_code = MatlabCode(
            r'''
            % Blurring kernel
            K = ones(3, 3) / 9;
            R = im_filtering(A, K);
            R_g = uint8(R);
            compare_images(A, R)
            '''
        )

        self.play(blurring_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 39:  ===========================================================
        # KERNEL DEFINTION WRITTEN
        self.next_slide(
            notes=
            '''We create a 3x3 matrix of ones, and then divide it by 9!.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 40:  ===========================================================
        # IM_FILTERING LINE WRITTEN
        self.next_slide(
            notes=
            '''Now we call the function im_filtering, to compute the resulting
            matrix.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 41:  ===========================================================
        # IM_FILTERING LINE WRITTEN
        self.next_slide(
            notes=
            '''Then we convert the result into an image.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[3]))

        # SLIDE 42:  ===========================================================
        # COMPARE_IMAGES LINE WRITTEN
        # INTO COLAB; RUN CELL, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''Finally, we use the function compare_images to visualize the
            original and the filtered images.
            '''
        )
        self.play(blurring_code.TypeLetterbyLetter(lines=[4]))
        self.wait(0.5)
        blurring_code.add_background_window(FullScreenBackground(WHITE))
        mat_env.remove_cell()
        self.play(blurring_code.IntoMatlab(mat_env))

        # Create the output of compare_images
        sample_A_image = ImageMobject(sample_A).scale_to_fit_height(PLOT_IMAGE_HEIGHT)
        sample_A_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        blur_result_image = ImageMobject(blurred_A[1:-1, 1:-1]).scale_to_fit_height(sample_A_image.height*10/12)
        blur_result_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        Group(sample_A_image, blur_result_image).arrange(buff=0.8)  # also this is scaled up
        image_titles = VGroup(
            Text("Image 1", font=CODE_FONT, font_size=14, color=BLACK, weight=SEMIBOLD).next_to(sample_A_image, UP),
            Text("Image 2", font=CODE_FONT, font_size=14, color=BLACK, weight=SEMIBOLD).next_to(blur_result_image, UP),
        )
        image_titles[1].match_y(image_titles[0])
        compare_image_output = Group(sample_A_image, blur_result_image, image_titles)
        for obj in compare_image_output:
            obj.save_state() 
        mat_env.add_output_plot(
            image=compare_image_output, window_buff=WINDOW_BUFF
        )

        self.play(mat_env.Run(new_cursor=True))

        # SLIDE 43:  ===========================================================
        # FOCUS ON OUTPUT
        # HIGHLIGHT THAT THE SECOND IMAGE IS SMALLER 
        self.next_slide(
            notes=
            '''As we can see, the original image has been blurred. But, as said,
            it's also a bit smaller! Can we avoid this "side effect"?
            '''
        )
        self.play(mat_env.FocusPlot())

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

        # SLIDE 44:  ===========================================================
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
        padded_pixel_array = PixelArray(np.zeros((14, 14)), stroke_width=1.25).scale_to_fit_height(sample_A_image.height*14/12).move_to(sample_A_image)
        padding_pixels = VGroup()
        for i in range(14):
            padding_pixels.add(padded_pixel_array.pixel_array[0, i])
            padding_pixels.add(padded_pixel_array.pixel_array[i, 0])
        for i in range(14):
            padding_pixels.add(padded_pixel_array.pixel_array[i, -1])
            padding_pixels.add(padded_pixel_array.pixel_array[-1, i])
        
        self.play(
            AnimationGroup(
                *[GrowFromCenter(pixel) for pixel in padding_pixels],
                lag_ratio=0.1,
                run_time=2
            )
        )

        # SLIDE 45:  ===========================================================
        # FADEOUT TO CODE
        # INITIALIZE A PADDED WRITTEN
        self.next_slide(
            notes=
            '''We can do it in this way: we create a larger matrix Ap, with 2
            rows and two columns more than A.
            '''
        )
        bp = FullScreenBackground(WHITE).set_z_index(2)
        self.play(FadeIn(bp))
        mat_env.clear()
        self.remove(sample_A_image, padding_title, padding_pixels, *padding_pixels.submobjects)
        bp.set_z_index(-1)

        padding_code = MatlabCode(
            r'''
            % Padding
            Ap = zeros(size(A, 1) + 2, size(A, 2) + 2);
            Ap(2 : end - 1, 2 : end - 1) = A;

            Rp = im_filtering(Ap, K);
            Rp_g = uint8(Rp);  
            compare_images(A, Rp_g)
            '''
        )

        self.play(padding_code.TypeLetterbyLetter(lines=[0, 1]))

        # SLIDE 46:  ===========================================================
        # COPY A TO A PADDED WRITTEN
        self.next_slide(
            notes=
            '''Then we copy A in Ap, starting from the second to the second to
            last row, and similarly for the columns.
            '''
        )
        self.play(padding_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 47:  ===========================================================
        # IMAGE FILTERING LINES WRITTEN
        # INTO MATLAB, RUN CODE, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''And now, we can repeat image filtering giving Ap as an input, and
            obtain a blurred image which has the same size of the original.
            '''
        )
        self.play(padding_code.TypeLetterbyLetter(lines=[4,5,6]))
        padding_code.add_background_window(bp)
        self.wait(0.5)
        self.play(padding_code.IntoMatlab(mat_env))
        
        # Create updated output for compare_images
        for obj in compare_image_output:
            obj.restore() 
        compare_image_output.remove(blur_result_image)
        padded_blur_result_image = ImageMobject(blurred_A).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        padded_blur_result_image.match_height(sample_A_image).next_to(image_titles[1], DOWN).match_y(sample_A_image)
        compare_image_output.add(padded_blur_result_image)
        mat_env.add_output_plot(
            compare_image_output, window_buff=WINDOW_BUFF
        )

        self.play(mat_env.Run())

        # SLIDE 48:  ===========================================================
        # FOCUS ON OUTPUT
        self.next_slide(
            notes=
            '''We can now replace our image with a version that respects
            privacy. [END]
            '''
        )
        self.play(mat_env.FocusPlot())