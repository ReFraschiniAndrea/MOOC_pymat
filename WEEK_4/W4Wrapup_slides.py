import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from mooc_utils.colab import ColabCodeWithLogo, ColabCode
from mooc_utils.matlab import MatlabCodeWithLogo, MatlabCode
import skimage
from PIL import Image
from W4Anim import *


# Highly suggested to run with --disable_caching
config.update(RELEASE_CONFIG)

HIGHLIGHT_COLOR=SATURATED_BLUE

class W4Wrapup_slides(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # CAR PLATE IMAGE APPEARS AND BECOMES BLURRED
        self.next_slide(
            notes=
            '''In this project we have learned how to filter, and in particular
            how to blur images using an operation called discrete convolution.
            '''
        )
        # cp stands for car plate
        cp_original = np.array(Image.open(r'Assets\W4\car_plate.png'))  # uint8
        cp_blur = (skimage.filters.gaussian(cp_original, sigma=40, mode='reflect', channel_axis=-1)*255).astype(np.uint8)
        cp_gray = (skimage.color.rgb2gray(cp_original)*255).astype(np.uint8)
        por_xmin, por_ymin, por_L = 248, 108, 336  # L divisible by 12
        cp_portion = cp_gray[por_xmin:por_xmin+por_L, por_ymin:por_ymin+por_L]
        cp_downscaled_portion = (skimage.transform.resize(cp_portion, (12, 12))) # float64
        cp_downscaled_portion = ((cp_downscaled_portion - cp_downscaled_portion.min())/(cp_downscaled_portion.max()-cp_downscaled_portion.min())*255).astype(np.uint8)

        car_plate_original = ImageMobject(cp_original)
        car_plate_original.save_state()
        car_plate_blur = ImageMobject(cp_blur).shift(DOWN)
        DC_title = Text('Discrete Convolution', font_size=64, color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).to_edge(UP).shift(UP*0.5)

        self.play(FadeIn(car_plate_original))
        self.wait(1)
        self.play(
            AnimationGroup(
                ApplyMethod(car_plate_original.shift, DOWN),
                Write(DC_title),
                lag_ratio = 0.5
            )
        )
        self.play(ApplyMethod(car_plate_original.become, car_plate_blur, run_time=2))

        # SLIDE 02:  ===========================================================
        # GRAYSCALE VALUES ARE WRITTEN ON TOP OF THE PIXELS OF THE IMAGE
        # GRAYSCALE VALUES ARE MOVED TO THE SIDE AND ENCLOSED BY SQUARE BRACKETS
        self.next_slide(
            notes=
            '''First of all, we have learned that images are represented by
            matrices:
            '''
        )
        self.play(FadeOut(car_plate_original, DC_title))

        sample_A_PA: PixelArray = PixelArray(cp_downscaled_portion, stroke_width=0.75).set_height(0.75*FRAME_HEIGHT)
        sample_A_PA.add_pixel_values(color=SATURATED_RED)
        sample_A_PA.pixel_array.save_state()
        sample_A_PA.pixel_values.save_state()
        sample_A_PA.pixel_values.move_to(HALF_SCREEN_RIGHT).scale(0.6)  # so the brackets are already in place
        sample_A_PA.add_brackets(color=BLACK)
        sample_A_PA.pixel_values.restore()
       
        self.play(FadeIn(sample_A_PA.pixel_array))
        self.play(Create(sample_A_PA.pixel_values, lag_ratio=0.1, run_time=1))
        self.play(
            AnimationGroup(
                AnimationGroup(
                    sample_A_PA.pixel_array.animate.move_to(HALF_SCREEN_LEFT).scale(0.6),
                    sample_A_PA.pixel_values.animate.move_to(HALF_SCREEN_RIGHT).scale(0.6).set_color(BLACK),
                ),
                FadeIn(sample_A_PA.brackets),
                lag_ratio = 0.5
            )
        )
        
        # SLIDE 03:  ===========================================================
        # GRAYSCALE VALUES BACK ON TOP OF IMAGE
        # SINGLE PIXEL IS HIGHLIGHTED, ITS VALUE CHANGES TO 255-0-ORIGINAL
        self.next_slide(
            notes=
            '''in particular each pixel, in greyscale, corresponds to an entry
            of the matrix, and its value is in the range 0-255
            '''
        )
        self.play(
            AnimationGroup(
                FadeOut(sample_A_PA.brackets),
                AnimationGroup(
                    sample_A_PA.pixel_array.animate.restore(),
                    sample_A_PA.pixel_values.animate.restore()
                ),
                lag_ratio=0.5
            )
        )
        sample_A_PA.remove_brackets()

        pi, pj = 5, 8  # pixel used for examples
        pixel_to_highlight : Square = sample_A_PA.pixel_array[pi, pj]
        value_to_highlight : CustomDecimalNumber = sample_A_PA.pixel_values[pi, pj]
        tracker = ValueTracker(value_to_highlight.get_value())
        value_to_highlight.add_updater(
            lambda v: v.set_value(tracker.get_value()).move_to(pixel_to_highlight)
        )
        pixel_to_highlight.save_state()
        tracker.save_state()
        pixel_highlight = sample_A_PA.get_pixel_highlight(color=HIGHLIGHT_COLOR, stroke_width=4, position=(pi, pj))
        
        self.play(Create(pixel_highlight))
        self.wait(0.5)
        self.play(pixel_to_highlight.animate.set_color(WHITE), tracker.animate.set_value(255), run_time=0.5)
        self.wait(0.5)
        self.play(pixel_to_highlight.animate.set_color(BLACK), tracker.animate.set_value(0), run_time=0.5)
        self.wait(0.5)
        self.play(pixel_to_highlight.animate.restore(), tracker.animate.set_value(sample_A_PA.array[pi, pj]), run_time=0.5)
        
        value_to_highlight.clear_updaters()

        # SLIDE 04:  ===========================================================
        # LOAD IMAGE CODE APPEARS (PYTHON - MATLAB)
        self.next_slide(
            notes=
            '''In both languages we have import images and convert them in
            greyscale using existing methods and functions
            '''
        )
        self.play(FadeOut(sample_A_PA, pixel_highlight))

        load_image_python_code = ColabCodeWithLogo(
            r'''
            A_color = Image.open(file_name)
            A_g = A_color.convert('L')
            '''
        )
        load_image_matlab_code = MatlabCodeWithLogo(
            r'''
            A_color = imread(path);
            A_g = rgb2gray(A_color);
            '''
        )
        Group(load_image_python_code, load_image_matlab_code).arrange(buff=1)
        load_image_matlab_code.align_to(load_image_python_code, DOWN)

        self.play(FadeIn(load_image_python_code, load_image_matlab_code))
        
        # SLIDE 05:  ===========================================================
        # HIGHLIGHT PYTHON A[]
        self.next_slide(
            notes=
            '''In python matrices can be stored as 2-dimensional numpy arrays,
            and accessed with square brackets, and indices that start from ZERO,
            '''
        )
        self.play(Group(load_image_python_code, load_image_matlab_code).animate.shift(UP))

        matrix_indexing_python_code = ColabCode(
            r'''
            print(A[1, 2])
            '''
        )
        matrix_indexing_python_code.add_background_window()
        matrix_indexing_python_code.next_to(load_image_python_code, DOWN, buff=0.5).align_to(load_image_python_code, LEFT)
        matrix_indexing_matlab_code = MatlabCode(
            r'''
            disp(A(2, 3))
            '''
        )
        matrix_indexing_matlab_code.add_background_window()
        matrix_indexing_matlab_code.next_to(load_image_matlab_code, DOWN, buff=0.5).align_to(load_image_matlab_code, LEFT)
        brackets_python_highlight = HighlightRectangle(matrix_indexing_python_code[0][7:12])
        brackets_matlab_highlight = HighlightRectangle(matrix_indexing_matlab_code[0][6:11])

        self.play(FadeIn(matrix_indexing_python_code, matrix_indexing_matlab_code))
        self.wait(1)
        self.play(Create(brackets_python_highlight))

        # SLIDE 06:  ===========================================================
        # HIGHLIGHT MATLAB A()
        self.next_slide(
            notes=
            '''While in matlab matrices are accessed with round brackets, and
            indices start from ONE.
            '''
        )
        self.play(ReplacementTransform(brackets_python_highlight, brackets_matlab_highlight))

        # SLIDE 07:  ===========================================================
        # JIG SLIDES OVER ORIGINAL IMAGE AND KERNEL, MULTIPLICATION RESULT
        # APPEARS IN EMPTY KERNEL ON THE SIDE 
        self.next_slide(
            notes=
            '''To implement the operation called "convolution" we have used
            NESTED for loops, to cycle over the rows and columns of the image
            and the kernel.
            '''
        )
        self.play(FadeOut(load_image_python_code, load_image_matlab_code,matrix_indexing_python_code, matrix_indexing_matlab_code, brackets_matlab_highlight))

        # Move image into position adn kernel highlight
        sample_A_PA.set_height(0.6*FRAME_HEIGHT).move_to(HALF_SCREEN_LEFT).shift(2*DOWN)
        kernel_highlight = sample_A_PA.get_kernel_array(np.ones((3,3))/9, kernel_color=HIGHLIGHT_COLOR, kernel_stroke_width=4, add_values=False).move_to(sample_A_PA.pixel_array[pi, pj])
        kernel_highlight.set_z_index(1)

        # Kernel
        kernel_array, kernel_values = sample_A_PA.get_kernel_array(np.ones((3,3))/9, kernel_color=BLACK, kernel_stroke_width=4, add_values=True, kernel_tex = " 1 / 9", values_size_fator=0.5)
        VGroup(kernel_array, kernel_values).scale(2).center().to_edge(UP).shift(UP*1.5)
        kernel_values.set_z_index(2)

        # Empty 3 by 3 to store kernel multiplication result
        local_result_array = PixelArray(cp_downscaled_portion[pi-1:pi+2, pj-1:pj+2]/9, stroke_color=BLACK, stroke_width=4)
        local_result_array.scale_to_fit_height(kernel_array.height).move_to(HALF_SCREEN_RIGHT).shift(DOWN*2)
        local_result_array.pixel_array.set_fill(opacity=0)
        local_result_array.add_pixel_values(num_decimal_places=1, color=BLACK)
        local_result_array.pixel_values.set_opacity(0)

        # Create jig to illustrate procedure: 3 squares + 2 connecting rectangles
        for numbers in [sample_A_PA.pixel_values, kernel_values, local_result_array.pixel_values]:
            numbers.set_z_index(2)  # so that the jig is below
        high1 : Square = sample_A_PA.get_pixel_highlight(position=(pi-1, pj-1), color=HIGHLIGHT_COLOR).set_fill(color=HIGHLIGHT_COLOR, opacity=0.4)
        high3 : Square = local_result_array.get_pixel_highlight(position=(0,0), color=HIGHLIGHT_COLOR, stroke_width=6).set_fill(color=HIGHLIGHT_COLOR, opacity=0.4)
        high2 : Square = high3.copy().move_to(kernel_array[0,0])

        hv1, hv2, hv3 = high1.get_vertices(), high2.get_vertices(), high3.get_vertices()
        poly_1 = Polygon(hv1[1], hv1[0], hv1[3], hv2[3], hv2[2], hv2[1], fill_color=HIGHLIGHT_COLOR, fill_opacity=0.4, stroke_width=0)
        poly_2 = Polygon(hv2[0], hv2[3], hv2[2], hv3[2], hv3[1], hv3[0], fill_color=HIGHLIGHT_COLOR, fill_opacity=0.4, stroke_width=0)
        def poly_1_updater(m: Polygon):
            hv1, hv2 = high1.get_vertices(), high2.get_vertices()
            m.become(
                Polygon(
                hv1[1], hv1[0], hv1[3], hv2[3], hv2[2], hv2[1],
                fill_color = m.fill_color, fill_opacity=m.fill_opacity, stroke_width=m.stroke_width
                )
            )
        def poly_2_updater(m: Polygon):
            hv2, hv3 = high2.get_vertices(), high3.get_vertices()
            m.become(
                Polygon(
                hv2[0], hv2[3], hv2[2], hv3[2], hv3[1], hv3[0],
                fill_color = m.fill_color, fill_opacity=m.fill_opacity, stroke_width=m.stroke_width
                )
            )
        poly_1.add_updater(poly_1_updater)
        poly_2.add_updater(poly_2_updater)
        jig = VGroup(high1, high2, poly_1, high3, poly_2)
        
        self.play(FadeIn(sample_A_PA, kernel_highlight, kernel_array, kernel_values, local_result_array.pixel_array, jig))
        # self.play(FadeIn(high3, poly_2))
        self.play(
            Succession(
                Succession(
                    AnimationGroup(
                        high1.animate.move_to(sample_A_PA.pixel_array[pi-1+i, pj-1+j]),
                        high2.animate.move_to(kernel_array[i,j]),
                        high3.animate.move_to(local_result_array.pixel_array[i,j]),
                        run_time=0.3
                    ),
                    local_result_array.pixel_values[i,j].animate(run_time=0.1).set_opacity(1),
                    Wait(0.1)
                )
                for i in range(3) for j in range(3)
            )
        )

        # create result pixel
        result_pixel = VGroup(sample_A_PA.pixel_array[0, 0].copy(), sample_A_PA.pixel_values[0, 0].copy())
        cl = np.sum(local_result_array.array)
        result_pixel.set_fill(color=rgb_to_color((cl, cl, cl)), family=False)  # exlude the number
        result_pixel[1].set_value(cl).move_to(result_pixel).set_z_index(5)
        result_pixel.match_height(local_result_array.pixel_array[0, 0]).next_to(local_result_array, DOWN)
        self.play(FadeIn(result_pixel, shift=DOWN))

        # SLIDE 08:  ===========================================================
        # FOR LOOPS CODE APPEARS
        self.next_slide(
            notes=
            '''The syntax is similar in the two languages, ...
            '''
        )
        self.play(FadeOut(sample_A_PA, kernel_highlight, kernel_array, kernel_values, local_result_array, jig, result_pixel))

        for_loops_python_code = ColabCodeWithLogo(
            r'''
            for m in range(3):
                for n in range(3):
                    ...
            '''
        )
        for_loops_matlab_code = MatlabCodeWithLogo(
            r'''
            for m = 1 : 3
                for n = 1 : 3
                    ...
                end
            end
            '''
        )
        Group(for_loops_python_code, for_loops_matlab_code).arrange(buff=2)
        for_loops_python_code.shift(UP*(for_loops_matlab_code.codeMobject.window.get_top()[1]-for_loops_python_code.codeMobject.window.get_top()[1]))

        self.play(FadeIn(for_loops_python_code, for_loops_matlab_code))

        # SLIDE 09:  ===========================================================
        # END FOR (MATLAB) HIGHLITED
        self.next_slide(
            notes=
            '''... but while in matlab a for loop is closed by "end",
            '''
        )
        ends_matlab_highlight = VGroup(
            HighlightRectangle(for_loops_matlab_code[-2]),
            HighlightRectangle(for_loops_matlab_code[-1]),
        )

        self.play(Create(ends_matlab_highlight))

        # SLIDE 10:  ===========================================================
        # INDENTATION (PYTHON) HIGHLIGHTED
        self.next_slide(
            notes=
            '''in python we have to carefully use indentation.
            '''
        )
        width_8_tab = for_loops_python_code[0][:6]
        arrows_config = {'color': BLUE, 'buff': 0.05, 'stroke_width' : 15, 'max_tip_length_to_length_ratio':1, 'max_stroke_width_to_length_ratio':100}
        indentation_python_highlight = VGroup(
            Arrow(start = for_loops_python_code[0][0].get_left(), end=for_loops_python_code[0][3].get_left(),
                  **arrows_config).align_to(for_loops_python_code[0], LEFT).match_y(for_loops_python_code[1]),
            Arrow(start = width_8_tab.get_left(), end=width_8_tab.get_right(),
                  **arrows_config).align_to(for_loops_python_code[0], LEFT).match_y(for_loops_python_code[2]),
        )
        for arrow in indentation_python_highlight:
            arrow[1].set_stroke(width=1)
        self.play(
            Succession(
                GrowArrow(indent_arrow) for indent_arrow in indentation_python_highlight
            )
        )

        # SLIDE 11:  ===========================================================
        # SLIDING KERNEL WHILE BLURRED IMAGE FILLS IN (3b1b ANIMATION)
        self.next_slide(
            notes=
            '''By repeating local convolution on every internal pixel, we have
            obtained a blurred image, ...
            '''
        )
        self.play(FadeOut(for_loops_python_code, indentation_python_highlight, for_loops_matlab_code, ends_matlab_highlight))
        self.clear()

        sample_A_PA: PixelArray = PixelArray(cp_downscaled_portion, stroke_width=0.75).set_height(0.6*FRAME_HEIGHT)
        blurred_A = np.round(separable_box_blur(cp_downscaled_portion, 3, mode = "constant", cval=0)).astype(np.uint8)  # with zero padding for later
        blurred_A_PA: PixelArray = PixelArray(blurred_A[1:-1, 1:-1], stroke_color=WHITE, stroke_width=1).set_height(10/12*sample_A_PA.height)
        kernel_array, kernel_values = sample_A_PA.get_kernel_array(np.ones((3,3))/9, kernel_color=HIGHLIGHT_COLOR, kernel_stroke_width=4, add_values=True, kernel_tex = " 1 / 9", values_size_fator=0.5)
        kernel = VGroup(kernel_array, kernel_values)
        pixel_highlight = blurred_A_PA.get_pixel_highlight(color=HIGHLIGHT_COLOR, stroke_width=4)

        LC = LiveConvolution(sample_A_PA, blurred_A_PA, kernel, pixel_highlight, kernel_size=3,
                             background_color=LIGHTER_GRAY)
        self.play(FadeIn(sample_A_PA, blurred_A_PA, kernel, kernel, pixel_highlight))
        
        LC.setup()
        self.play(LC.SlideKernel(end=99, run_time=4))
        blurred_A_PA.pixel_array.set_fill(opacity=1)
        LC.clear_updaters()

        # SLIDE 12:  ===========================================================
        # HIGHLIGHT THAT THE SECOND IMAGE IS SMALLER 
        self.next_slide(
            notes=
            '''.. which however, is smaller than the original one.
            '''
        )
        self.play(FadeOut(kernel, pixel_highlight))

        # Create the border highlight (square with square hole)
        size_difference_highlight = VMobject(fill_color=HIGHLIGHT_COLOR, fill_opacity=0.4, stroke_color=HIGHLIGHT_COLOR, stroke_width=4)
        size_difference_highlight.append_points(Square(sample_A_PA.height).get_points()[::-1])
        size_difference_highlight.append_points(Square(blurred_A_PA.height).get_points())
        size_difference_highlight.move_to(sample_A_PA)

        self.play(
            Succession(
                FadeIn(size_difference_highlight),
                Wait(0.5),
                ApplyMethod(size_difference_highlight.move_to, blurred_A_PA)
            )
        )

        # SLIDE 13:  ===========================================================
        # ORIGINAL IMAGE IS MOVED TO CENTER; SECOND ONE DISAPPEARS
        # BLACK SQUARES PADDING IS ADDED TO ORIGINAL IMAGE
        self.next_slide(
            notes=
            '''To circumvent this problem we can perform PADDING, that is, we
            add pixels all around the image before filtering.
            '''
        )
        padding_title = Text('Padding', font_size=64, color=BLACK, font=SANS_SERIF_FONT, weight=LIGHT).to_edge(UP).shift(UP*0.5)
        self.play(
            AnimationGroup(
                FadeOut(size_difference_highlight, blurred_A_PA),
                sample_A_PA.animate.center().shift(DOWN*1),
                Write(padding_title),
                lag_ratio=0.5
            )
        )

        # Create padding pixels
        padded_pixel_array = PixelArray(np.zeros((14, 14)), stroke_width=0.75).scale_to_fit_height(sample_A_PA.height*14/12).move_to(sample_A_PA)
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

        # SLIDE 14:  ===========================================================
        # PADDING CODE APPEARS
        self.next_slide(
            notes=
            '''To add black pixels, we can simply create a matrix that is larger
            than the original, and copy the original one in the internal pixels.
            '''
        )
        self.play(FadeOut(padding_pixels, sample_A_PA, padding_title))

        padding_python_code = ColabCodeWithLogo(
            r'''
            Ap = np.zeros((A.shape[0] + 2, A.shape[1] + 2))
            Ap[1:-1, 1:-1] = A
            ''',
            logo_pos=UP
        )
        padding_matlab_code = MatlabCodeWithLogo(
            r'''
            Ap = zeros(size(A, 1) + 2, size(A, 2) + 2);
            Ap(2:end-1, 2:end-1) = A;
            ''',
            logo_pos=UP
        )

        Group(padding_python_code, padding_matlab_code).arrange_in_grid(2, 1, cell_alignment=LEFT, buff=1).center()

        self.play(FadeIn(padding_python_code, padding_matlab_code))

        # SLIDE 15:  ===========================================================
        # HIGHLIGHT PYTHON 1:-1 INDEXING
        self.next_slide(
            notes=
            '''To consider rows from the second to second to last, we use 1:-1
            in python, ...
            '''
        )
        one_to_one_python_highlight = HighlightRectangle(padding_python_code[1][3:12])
        self.play(Create(one_to_one_python_highlight))

        # SLIDE 16:  ===========================================================
        # HIGHLIGHT MATLAB 2-END-1 INDEXING
        self.next_slide(
            notes=
            '''... while in matlab we go from 2 to "end-1".
            '''
        )
        two_to_end_matlab_highlight = HighlightRectangle(padding_matlab_code[1][3:18])
        self.play(ReplacementTransform(one_to_one_python_highlight, two_to_end_matlab_highlight))

        # SLIDE 17:  ===========================================================
        # WHOLE CAR PLATE IMAGE REAPPEARS
        # IMAGE CONVERTED TO GRAYSCALED AND BLURRED
        self.next_slide(
            notes=
            '''Now it's your turn: try to load a larger image, for instance the
            whole car plate, convert it into greyscale and apply blurring. Can
            you still read the numbers? If so, write a code to repeat the
            blurring process many times.
            '''
        )
        self.play(FadeOut(padding_python_code, padding_matlab_code, two_to_end_matlab_highlight))
        
        car_plate_original.restore()
        car_plate_gray = ImageMobject(cp_gray)
        cp_gray_box_blurred = separable_box_blur(cp_gray, 3, mode='constant', cval=0).astype(np.uint8) 
        car_plate_gray_box_blurred = ImageMobject(cp_gray_box_blurred)
        
        self.play(FadeIn(car_plate_original))
        self.wait(2)
        self.play(ReplacementTransform(car_plate_original, car_plate_gray))
        self.wait(1)
        self.play(ReplacementTransform(car_plate_gray, car_plate_gray_box_blurred))

        # SLIDE 18:  ===========================================================
        # 5X5 BOX BLUR KERNEL APPEARS BESIDE THE IMAGE
        self.next_slide(
            notes=
            '''Another option for blurring is to use larger kernels, like this
            one, which is a 5X5 kernel. Try to modify the code to use this
            larger kernel. Is the blurring more pronounced?
            '''
        )
        box_blur_kernel_5x5 = VGroup(
            MathTex(r'K=\frac{1}{25}', color=BLACK),
            IntegerMatrix(np.ones((5,5))).set_color(BLACK)
        ).arrange()

        phony_rect = car_plate_gray_box_blurred.copy().set_height(0.45*FRAME_HEIGHT)
        
        Group(box_blur_kernel_5x5, phony_rect).arrange(DOWN, buff=0.5)

        self.play(
            AnimationGroup(
                car_plate_gray_box_blurred.animate.become(phony_rect),
                Write(box_blur_kernel_5x5),
                lag_ratio=0.5
            )
        )

        # SLIDE 19:  ===========================================================
        # BOX BLUR KERNEL AND IMAGE FADEOUT
        # EDGE DETECTION KERNEL APPEAR
        self.next_slide(
            notes=
            '''And finally, using convolution we can do much more than blurring!
            Try to apply for instance this kernel, ...
            '''
        )
        self.play(FadeOut(box_blur_kernel_5x5, car_plate_gray_box_blurred))

        vertical_edge_detection_kernel = VGroup(
            MathTex('K=', color=BLACK),
            Matrix([[-1, 2, -1] for _ in range(3)]).set_color(BLACK)
        ).arrange()
        sharpening_kernel = VGroup(
            MathTex('K=', color=BLACK),
            Matrix([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).set_color(BLACK)
        ).arrange()

        VGroup(vertical_edge_detection_kernel, sharpening_kernel).arrange(buff=1)

        self.play(Write(vertical_edge_detection_kernel))

        # SLIDE 20:  ===========================================================
        # SHARPENING KERNEL APPEARS
        self.next_slide(
            notes=
            '''... or this one, and see what happens to the image.
            '''
        )
        self.play(Write(sharpening_kernel))
