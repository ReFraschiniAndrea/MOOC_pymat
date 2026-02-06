import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from mooc_utils.matlab import *
from W5Anim import *
from PIL import Image
import matplotlib.pyplot as plt
from mooc_utils.colab import draw_plot

config.update(RELEASE_CONFIG)
config.max_files_cached = 250

        
class W5Matlab_slides(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # MATLAB ENV FADES IN
        # CNN SCHEME FADES IN ON TOP
        self.next_slide(
            notes=
            '''Let's open Matlab, and let's start to learn how to code an image
            digit classifier based on image processing and convolutional neural
            networks. This will be done in four steps:
            '''
        )
        mat_env = MatlabEnv(self, r'Assets\W4\matlab_empty.png')
        mat_env.RUN_BUTTON_ = MatlabEnv._pixel2p(1100, 67)
        mat_env.SAVE_PROMPT_BUTTON_ = MatlabEnv._pixel2p(877, 859)
        mat_env.OK_PROMPT_ = MatlabEnv._pixel2p(877, 816)

        # Create the full main scheme for later 
        digit = np.array(Image.open(r'Assets\W5\mnist8.png'))
        ms = CNNDigitRecognitionScheme(digit, input_label=8, n_filters=5, pooling_factor=7,
                                       pixel_size=(1.5/28, 1.5/28, 0.125, 0.125), horizontal_spacing=0.8,
                                       highlights_kwargs={'color': GOLD, 'stroke_width': 3},
                                       outline_kwargs={'color': DARK_BLUE, 'stroke_width': 6})
        ms.output_layer.activate(8)
        ms.save_state()
        ms.scale(0.75)

        surrounding_rect = SurroundingRectangle(ms, fill_color=WHITE, fill_opacity=1, stroke_width=0.5,
                                              stroke_color=BLACK, corner_radius=0.2, buff=0.5).set_z_index(-0.5)
        
        self.play(FadeIn(mat_env.background))
        self.play(
            Succession(
                Wait(1),
                FadeIn(surrounding_rect, ms)
            )
        )

        # SLIDE 02:  ===========================================================
        # CNN SCHEME FADES OUT; IN ITS PLACE, START LISTING THE STEPS
        self.next_slide(
            notes=
            '''First, we'll load and visualize the datasets;
            '''
        )
        self.play(FadeOut(ms))

        steps_paragraph = Paragraph(
            '1. Load datasets', '2. Define model architecture', '3. Training', '4. Predictions',
            color=BLACK, font=SANS_SERIF_FONT, font_size=48, alignment='left', weight=BOLD,
            line_spacing=1.5
        ).align_to(surrounding_rect, LEFT).shift(RIGHT*0.5)

        self.play(FadeIn(steps_paragraph[0]))

        # SLIDE 03:  ===========================================================
        # SECOND STEP WRITTEN
        self.next_slide(
            notes=
            '''Second, we'll define the model architecture;
            '''
        )
        self.play(FadeIn(steps_paragraph[1]))

        # SLIDE 04:  ===========================================================
        # THIRD STEP WRITTEN
        self.next_slide(
            notes=
            '''Third, we'll train the model;
            '''
        )
        self.play(FadeIn(steps_paragraph[2]))

        # SLIDE 05:  ===========================================================
        # FOURTH STEP WRITTEN
        self.next_slide(
            notes=
            '''Fourth, we'll use the trained CNN to recognize digits of new
            images, in other words, to make predictions.
            '''
        )
        self.play(FadeIn(steps_paragraph[3]))

        # SLIDE 06:  ===========================================================
        # BROWSE FOLDER
        # OPEN SIDEMENU
        # HIGHLIGHT UPLOADED FILES
        self.next_slide(
            notes=
            '''Let's upload the digits_dataset.zip file, containing the datasets
            we will need, and the load_data.m, show_random_samples.m,
            visualize_digit_prediction.m Matlab files. These tools will be
            useful to read, visualize, and store the images in our datasets.
            '''
        )
        hand_cursor = mat_env.cursor.center()

        self.play(
            Succession(
                FadeOut(steps_paragraph, surrounding_rect),
                Wait(0.3)
            )
        )
        self.play(GrowFromCenter(hand_cursor))
        self.play(hand_cursor.MoveAndClick(mat_env.BROWSE_FOLDER_))
        mat_env.set_image(r'Assets\W4\matlab_browsefolder.png')
        self.play(hand_cursor.MoveAndClick(mat_env.OK_PROMPT_))
        mat_env.set_image(r'Assets\W4\matlab_empty.png')
        self.play(hand_cursor.MoveAndClick(mat_env.SIDEMENU_))
        mat_env.set_image(r'Assets\W5\matlab_uploaded.png')
        phony_files = Rectangle(width=220*mat_env.PIXEL, height=96*mat_env.PIXEL).move_to(MatlabEnv._pixel2p(58, 248), aligned_edge=UL).set_opacity(0)
        self.wait(0.5)
        self.play(Circumscribe(phony_files, color=BLUE, run_time=2))


        # SLIDE 07:  ===========================================================
        # OUT OF MATLAB (FROM COMMAND WINDOW)
        # UNZIP CODE WRITTEN
        # INTO MATLAB, RUN, UNZIPPED FILES APPEAR IN SIDEMENU
        # HIGHLIGHT UNZIPPED FILES
        self.next_slide(
            notes=
            '''We extract the contents of the digits_dataset.zip file using the
            unzip command, passing the file name as input, and pressing Enter.
            '''
        )
        unzip_code = MatlabCode(
            r'''
            unzip("digits_dataset.zip")
            '''
        )
        SIDEMENU_WIDTH = (459-43)*mat_env.PIXEL
        mat_env.add_cell()
        mat_env.get_cell(0).move_to(mat_env.TOP_LEFT_CORNER_NOSCRIPT_, aligned_edge=UL).shift(RIGHT*SIDEMENU_WIDTH)
        self.play(mat_env.OutofMatlab(cell=0))
        self.play(unzip_code.TypeLetterbyLetter())
        self.wait(1)
        # Setup target cell in command window
        mat_env.remove_cell()
        mat_env.add_cell(MatlabCodeBlock(unzip_code.code_string))
        target_unzip_code = mat_env.get_cell(0).move_to(mat_env.TOP_LEFT_CORNER_NOSCRIPT_, aligned_edge=UL).shift(RIGHT*SIDEMENU_WIDTH)

        unzip_code.add_background_window(FullScreenBackground(WHITE))
        self.play(unzip_code.IntoMatlab(mat_env, target_cell=0))
        self.wait(0.5)
        mat_env.set_image(r'Assets\W5\matlab_unzipped.png')
        phony_files = Rectangle(width=180*mat_env.PIXEL, height=48*mat_env.PIXEL).move_to(MatlabEnv._pixel2p(58, 248), aligned_edge=UL).set_opacity(0)
        self.wait(0.5)
        self.play(Circumscribe(phony_files, color=BLUE, run_time=2))

        # SLIDE 08:  ===========================================================
        # MATLAB ENV FADES OUT
        # MNIST DATASET ILLUSTRATION: IMAGES APPEAR
        self.next_slide(
            notes=
            '''The folder we extracted contains images from the open-source
            MNIST dataset,
            '''
        )
        self.play(mat_env.FadeOut())

        mnist_title = SlideTitle('MNIST Dataset')
        training_sample = Group(
            *[PixelImage(rf'Assets\W5\mnist{i}.png').scale_to_fit_height(0.25*FRAME_HEIGHT) for i in (3,5,8,6,1,9,)]
        ).arrange_in_grid(2,3, buff=1).move_to(TITLED_CENTER)

        self.play(
            FadeIn(training_sample, lag_ratio=0.2, run_time=2),
            Write(mnist_title)
        )

        # SLIDE 09:  ===========================================================
        # DIMENSIONS WITH BRACES APPEAR
        self.next_slide(
            notes=
            '''..., which consists of 28x28 pixel images, for a total of 784
            pixels per image.
            '''
        )

        self.play(
            AnimationGroup(
                FadeOut(training_sample[1:]),
                training_sample[0].animate.scale_to_fit_height(0.45*FRAME_HEIGHT).move_to(TITLED_CENTER),
                lag_ratio=0.5
            )
        )

        grid_28_highlight = VGroup(
            training_sample[0].get_pixel_highlight(color=WHITE, stroke_width=2)
            for _ in range(28*28)
        ).arrange_in_grid(28, 28, buff=0).move_to(training_sample)
        image_size_braces = get_labeled_braces(training_sample[0], LEFT, '28', DOWN, '28')

        self.play(
            Succession(
                Create(grid_28_highlight, run_time=0.5),
                FadeIn(image_size_braces)
            )
        )

        # SLIDE 10:  ===========================================================
        # LABEL WRITTEN UNDER THE IMAGES
        self.next_slide(
            notes=
            '''Each example in the dataset includes an image and its label,
            indicating which digit it represents.
            '''
        )
        phony_rects = VGroup(
            Square(0.25*FRAME_HEIGHT, stroke_width=0) for _ in range(6)
        ).arrange_in_grid(2,3, buff=(1, 0.5)).move_to(TITLED_CENTER)
        for i in range(1,3):
            training_sample[i].match_height(phony_rects[i]).move_to(phony_rects[i])
        ground_truth_config = {'stroke_color':GREEN_D,'fill_color':WHITE, 'text_kwargs':{'fill_color': GREEN_D, 'stroke_color':GREEN_D}}
        training_labels = VGroup(
            DigitRecognitionOutputCircle(i, stroke_width=12, **ground_truth_config).match_height(rect).move_to(rect)
            for i, rect in zip((3,5,8), phony_rects[3:])
        )
        sample_rects = VGroup(
            SurroundingRectangle(im, lab, color=BLUE, stroke_width=4, buff=0.25, corner_radius=0.25)
            for im, lab in zip(phony_rects[:3], phony_rects[3:])
        )
        image_label = Text('Image', font=SANS_SERIF_FONT, weight=BOLD, font_size=32, color=BLACK).next_to(phony_rects[0], LEFT, buff=0.5)
        label_label = Text('Label', font=SANS_SERIF_FONT, weight=BOLD, font_size=32, color=BLACK).next_to(phony_rects[3], LEFT, buff=0.5).match_x(image_label)

        self.play(
            Succession(
                AnimationGroup(
                    FadeOut(grid_28_highlight, image_size_braces),
                    training_sample[0].animate.match_height(phony_rects[0]).move_to(phony_rects[0]),
                    FadeIn(training_sample[1:3]),
                    lag_ratio=0.5
                ),
                FadeIn(training_labels),
                Create(sample_rects),
                FadeIn(image_label),
                FadeIn(label_label)
            )
        )
        # SLIDE 11:  ===========================================================
        # GRAPHIC ILLUSTRATING TRAINING AND VALIDATION DATASET APPEARS
        self.next_slide(
            notes=
            ''' We will use two separate datasets: one for training the network,
            called training dataset,
            '''
        )
        self.play(FadeOut(*[mob for mob in self.mobjects]))

        hex1 = RegularPolygon(n=6, stroke_color=GRAY, stroke_width=12, fill_color=BLUE, fill_opacity=0.4).round_corners(0.1)
        hex2 = RegularPolygon(n=6, stroke_color=GRAY, stroke_width=12, fill_color=ORANGE, fill_opacity=0.4).round_corners(0.1)
        VGroup(hex1, hex2).arrange(buff=0.2).scale_to_fit_width(0.8*FRAME_WIDTH)
        dataset_text_config = {'color':BLACK, 'font':SANS_SERIF_FONT, 'weight': BOLD, 'alignment':'center'}
        train_dat_label = Paragraph('Training\nDataset',   **dataset_text_config, font_size=48).move_to(hex1)
        hex1.add(train_dat_label)
        valid_dat_label = Paragraph('Validation\nDataset', **dataset_text_config, font_size=48).move_to(hex2)
        hex2.add(valid_dat_label)
        train_des_label = Paragraph('To train\nthe model', **dataset_text_config, font_size=40).next_to(hex1, DOWN, buff=0.35)
        valid_des_label = Paragraph('To evaluate\nmodel performance', **dataset_text_config, font_size=40).next_to(hex2, DOWN, buff=0.35)
        
        self.play(
            Succession(
                FadeIn(hex1),
                FadeIn(train_des_label)
            )
        )

        # SLIDE 12:  ===========================================================
        # VALIDATION DATASET ILLUSTRATION
        self.next_slide(
            notes=
            '''and one for evaluating its performance, called validation
            dataset.
            '''
        )
        self.play(
            Succession(
                FadeIn(hex2),
                FadeIn(valid_des_label)
            )
        )

        # SLIDE 13:  ===========================================================
        # CLOSE SIDE MENU
        # CREATE AND SAVE NEW SCRIPT
        self.next_slide(
            notes=
            '''Let's start with the training dataset. To load the training
            dataset, we first create a new script and
            '''
        )
        self.play(FadeOut(hex1, hex2, train_des_label, valid_des_label))
        self.play(mat_env.FadeIn())

        self.play(mat_env.cursor.MoveAndClick(mat_env.SIDEMENU_))
        mat_env.set_image(r'Assets\W4\matlab_empty.png')
        target_unzip_code.shift(LEFT*SIDEMENU_WIDTH)
        self.play(mat_env.cursor.MoveAndClick(mat_env.NEW_SCRIPT_))
        mat_env.set_image(r'Assets\W4\matlab_newscript.png')
        target_unzip_code.move_to(mat_env.OUTPUT_TOP_LEFT_CORNER_, aligned_edge=UL)
        self.play(mat_env.cursor.MoveAndClick(mat_env.SAVE_))
        mat_env.set_image(r'Assets\W5\matlab_saveW5.png')
        alpha_rect = target_unzip_code.window.copy().set_opacity(0.5).set_z_index(1)
        self.add(alpha_rect)
        self.play(mat_env.cursor.MoveAndClick(mat_env.SAVE_PROMPT_BUTTON_))
        mat_env.set_image(r'Assets\W5\matlab_week5.png')
        self.remove(alpha_rect)

        # SLIDE 14:  ===========================================================
        # TRAINING PATH WRITTEN
        self.next_slide(
            notes=
            '''specify the path to the folder containing the images and their
            labels.
            '''
        )
        load_training_code = MatlabCode(
            r'''
            % Define paths
            training_dir_path = 'training_set';

            % Training Data
            [X_train, y_train] = load_data(training_dir_path);

            % Set random seed for reproducibility
            rng(1);

            % Visualize 4 random sample of the digit 5
            show_random_samples(training_dir_path, 5);
            '''
        )
        mat_env.remove_cell()
        mat_env.add_cell()
        self.play(mat_env.OutofMatlab(cell=0), FadeOut(target_unzip_code))
        self.play(load_training_code.TypeLetterbyLetter(lines=[0,1]))

        # SLIDE 15:  ===========================================================
        # LAD DATA LINE WRITTEN
        self.next_slide(
            notes=
            '''We then use the load_data function to store the matrix
            representation of images in X_train and the corresponding labels in
            y_train.
            '''
        )
        self.play(load_training_code.TypeLetterbyLetter(lines=[3,4]))

        # SLIDE 16:  ===========================================================
        # SHOW_RANDOM_SAMPLES LINE WRITTEN
        self.next_slide(
            notes=
            '''Let's start to explore our dataset, using the function
            show_random_samples. This function picks randomly four images from
            the sample we have just loaded, and shows them.
            '''
        )
        show_random_samples_lines = load_training_code[9:11]
        show_random_samples_lines.save_state()
        show_random_samples_lines.align_to(load_training_code[6:8], UP)

        self.play(load_training_code.TypeLetterbyLetter(lines=[9,10]))

        # SLIDE 17:  ===========================================================
        # BASE_DIR PARAM HIGHLIGHTED
        self.next_slide(
            notes=
            '''The first parameter specifies the path from which picking the
            images, ...
            '''
        )
        base_dir_highlight = HighlightRectangle(show_random_samples_lines[1][20:37])
        self.play(Create(base_dir_highlight))

        # SLIDE 18:  ===========================================================
        # DIGIT=5 PARAM HIGHLIGHTED
        self.next_slide(
            notes=
            '''..., while setting the second one tells the function to display
            the images which represent of the digit 5. We recall that this
            information is stored in the variable y_train.
            '''
        )
        digit_5_highlight = HighlightRectangle(show_random_samples_lines[1][-3])
        self.play(ReplacementTransform(base_dir_highlight, digit_5_highlight))

        # SLIDE 19:  ===========================================================
        # SHOW_RANDOM_SAMPLES SHIFT DOWN, SEED LINE WRITTEN.
        self.next_slide(
            notes=
            '''It's convenient to first set the random seed to 1 for
            reproducibility. This forces the random number generator to produce
            a fixed sequence of numbers each time the code is run.
            '''
        )
        self.play(
            Succession(
                FadeOut(digit_5_highlight),
                show_random_samples_lines.animate.restore(),
                load_training_code.TypeLetterbyLetter(lines=[6,7])
            )
        )

        # SLIDE 20:  ===========================================================
        # INTO MATLAB, RUN CELL, 4 IMAGES OF DIGIT '5' APPEAR
        self.next_slide(
            notes=
            '''
            '''
        )
        load_training_code.add_background_window(FullScreenBackground(WHITE))
        mat_env.clear()
        self.play(load_training_code.IntoMatlab(mat_env))

        # Create the output plot
        random_samples_plot = Group()
        for i in range(1,5):
            img = PixelImage(rf'Assets\W5\ex5{i}.png').scale_to_fit_height(0.25*FRAME_HEIGHT)
            lab = Text(f'Sample {i}', color=BLACK, weight=BOLD, font=SANS_SERIF_FONT, font_size=18).next_to(img, UP, buff=0.1)
            img.add(lab)
            random_samples_plot.add(img)
        random_samples_plot.arrange_in_grid(2,2, buff=(1, 0.5))
        mat_env.add_output_plot(random_samples_plot, buff=0.5)

        self.play(mat_env.Run(new_cursor=True))

        # SLIDE 21:  ===========================================================
        # NEW CELL, OUT OF COLAB
        # LOAD VALIDATION DATASET CODE WRITTEN
        self.next_slide(
            notes=
            '''Next, we prepare the validation dataset by calling the same
            loading function.
            '''
        )
        load_validation_code = MatlabCode(
            r'''
            % Validation Data
            validation_dir_path = 'validation_set';
            [X_validation, y_validation] = load_data(validation_dir_path);
            '''
        )

        mat_env.add_cell()
        self.play(mat_env.OutofMatlab(cell=-1))
        self.play(load_validation_code.TypeLetterbyLetter(lines=[0]))
        mat_env.remove_plot(); mat_env.remove_cursor()

        # SLIDE 22:  ===========================================================
        # VALIDATION PATH HIGHLIGHT
        self.next_slide(
            notes=
            '''This time we use a different path to the folder containing
            additional images that will not be used during training, defined by
            the string validation_set.
            '''
        )
        self.play(load_validation_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 23:  ===========================================================
        # LOAD_DATA HIGHLIGHT
        # INTO MATLAB, RUN CELL
        self.next_slide(
            notes=
            '''The function load_data stores the matrix representation of images
            in X_valid and the corresponding labels in y_valid.
            '''
        )
        self.play(load_validation_code.TypeLetterbyLetter(lines=[2]))
        load_validation_code.add_background_window(FullScreenBackground(WHITE))
        mat_env.remove_cell()
        self.play(load_validation_code.IntoMatlab(mat_env))

        # SLIDE 24:  ===========================================================
        # NEW CELL, OUT OF MATLAB
        # DATSET DIMENSIONS COMMENT WRITTEN
        self.next_slide(
            notes=
            '''Let's see how many images are contained in the training and
            validation datasets.
            '''
        )
        dataset_sizes_code = MatlabCode(
            r'''
            % Size of the dataset
            fprintf('Amount of training data: %d\n', size(X_train, 4))
            fprintf('Amount of validation data: %d\n', size(X_validation, 4))

            % Size of a single image
            [img_height, img_width, ~, ~] = size(X_train);
            fprintf('Image shape: %dx%d\n', img_width, img_height)
            '''
        )

        mat_env.add_cell()
        self.play(mat_env.OutofMatlab(cell=-1))
        self.play(dataset_sizes_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 25:  ===========================================================
        # DISP(SIZE) LINES WRITTEN
        self.next_slide(
            notes=
            '''Using the "size" function on X_train and X_validation, we can see
            how many samples each dataset contains.
            '''
        )
        self.play(dataset_sizes_code.TypeLetterbyLetter(lines=[1]))
        self.play(dataset_sizes_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 26:  ===========================================================
        # SIZE(.,4) HIGHLIGGHT
        self.next_slide(
            notes=
            '''Since these are 4D arrays, to check the number of images, we look
            at the fourth dimension of the array by passing the argument 4.
            '''
        )
        size_4_highlights = VGroup(
            HighlightRectangle(dataset_sizes_code[1][-3]),
            HighlightRectangle(dataset_sizes_code[2][-3])
        )
        self.play(Create(size_4_highlights))

        # SLIDE 27:  ===========================================================
        # IMAGE SHAPE LINES WRITTEN
        self.next_slide(
            notes=
            '''To inspect the dimension of each single image, we again use the
            size function applied to X_train. This time we focus only the first
            two dimensions: the output img_height and img_width.
            '''
        )
        self.play(FadeOut(size_4_highlights))
        self.play(dataset_sizes_code.TypeLetterbyLetter(lines=[4,5,6]))

        # SLIDE 28:  ===========================================================
        # INTO COLAB, RUN CELL, OUTPUT APPEARS
        # FOCUS ON OUTPUT
        self.next_slide(
            notes=
            '''we see that the training dataset contains 9,990 images made of
            28x28 pixels, while the validation dataset contains 1000 images.
            '''
        )
        dataset_sizes_code.add_background_window(FullScreenBackground(WHITE))
        mat_env.remove_cell()
        self.play(dataset_sizes_code.IntoMatlab(mat_env))

        mat_env.add_output(
            'Amount of training data: 9990\nAmount of validation data: 1000\nImage shape: 28x28'
        )
        self.play(mat_env.Run(new_cursor=True))
        self.play(mat_env.FocusOutput())

        # SLIDE 29:  ===========================================================
        # FULL CNN SCHEME REAPPEARS ON TOP (SAME AS THE BEGINNING)
        self.next_slide(
            notes=
            '''Now, let's address the second step: the definition of the
            architecture of our classifier.
            '''
        )
        dummy_gray_rect = FullScreenBackground(MATLAB_LIGHTGRAY).set_z_index(1)
        self.play(FadeIn(dummy_gray_rect))

        # reset everything
        self.clear()
        mat_env.clear()
        ms.restore()
        DSS = DynamicSplitScreen(WHITE, MATLAB_LIGHTGRAY)  # cover the clear with the DSS we need alter
        DSS.add_empty_side_obj(FRAME_HEIGHT)
        DSS.hard_bring_in()
        self.add(DSS)

        self.play(FadeIn(ms))

        # SLIDE 30:  ===========================================================
        # CNN SCHEME GOES TO TOP
        # LAYERS LINE WRITTEN
        self.next_slide(
            notes=
            '''The idea is to concatenate all the layers defined by specific
            Matlab commands to create the final architecture.
            '''
        )
        # Setup splitscreen
        self.play(
            DSS.secondaryRect.animate.stretch_to_fit_height(ms.height*0.75+2*DSS.buff_).move_to(UP*FRAME_HEIGHT/2, aligned_edge=UP),
            ms.animate.scale(0.75).move_to(UP*FRAME_HEIGHT/2+DOWN*(DSS.buff_+ms.height*0.75/2))
        )

        cnn_architecture_code = MatlabCode(
            r'''
            % Define the model
            layers = [
                imageInputLayer([img_height img_width 1])
                convolution2dLayer(3, 32, 'Padding', 'same')
                reluLayer
                maxPooling2dLayer(7, 'Stride', 7)
                flattenLayer
                fullyConnectedLayer(10)
                softmaxLayer
                classificationLayer
            ];
            '''
        )
        cnn_architecture_code.move_to(DSS.mainRect)

        layer_highlight_config = {'height':5.5, 'v_buff':0.25, 'h_buff':0.15}
        full_layers_highlight = ms.get_layer_highlight('all', **layer_highlight_config)

        self.play(cnn_architecture_code.TypeLetterbyLetter(lines=[0,1,10]))
        self.play(ShowPassingFlash(full_layers_highlight, run_time=2, time_width=0.3))

        # SLIDE 31:  ===========================================================
        # INPUTLAYER LINE WRITTEN
        self.next_slide(
            notes=
            '''First, imageInputLayer informs the CNN about the size of the
            input image.
            '''
        )
        self.play(cnn_architecture_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 32:  ===========================================================
        # KEARS.INPUT WRITTEN
        self.next_slide(
            notes=
            '''Here the dimensions are: 28x28 pixel times 1 channel.
            '''
        )
        brace_label_config = {'font_size': 24}
        input_braces = get_labeled_braces(
            ms.input, LEFT, '28', DOWN, '28', label_config=brace_label_config
        )

        self.play(FadeIn(input_braces))
        
        # SLIDE 33:  ===========================================================
        # HIGHLIGHT THE 4 LAYERS
        self.next_slide(
            notes=
            '''The following CNN layers are are defined using functions provided
            by the Matlab Deep Learning Toolbox:
            '''
        )
        dummy_architecture_code = MatlabCode(
            r'''
            convolution2dLayer(...)
            reluLayer
            maxPooling2dLayer(...)
            flattenLayer
            fullyConnectedLayer(...)
            softmaxLayer
            classificationLayer
            '''
        ).align_to(cnn_architecture_code[3], UL).set_color(BLACK)  # coloring is broken

        self.play(FadeOut(input_braces))

        # SLIDE 34:  ===========================================================
        # LAYER CONV2D WRITTEN, CONV LAYER HIGHLIGHTED ABOVE
        self.next_slide(
            notes=
            '''Convolution2dlayer defines the convolutional layer.
            '''
        )
        conv_highlight = ms.get_layer_highlight('conv', **layer_highlight_config)
        self.play(
            dummy_architecture_code.TypeLetterbyLetter(lines=[0]),
            Create(conv_highlight)
        )
        # SLIDE 35:  ===========================================================
        # LAYER CONV2D WRITTEN, CONV LAYER HIGHLIGHTED ABOVE
        self.next_slide(
            notes=
            '''Its activation function is applied separately by the function
            reluLayer, which follows the convolution.
            '''
        )
        self.play(dummy_architecture_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 36:  ===========================================================
        # LAYER MAXPOOL WRITTEN, POOLING LAYER HIGHLIGHTED ABOVE
        self.next_slide(
            notes=
            '''Next, MaxPooling2DLayer implements the pooling layer.
            '''
        )
        pool_highlight = ms.get_layer_highlight('pool', **layer_highlight_config)
        self.play(
            dummy_architecture_code.TypeLetterbyLetter(lines=[2]),
            ReplacementTransform(conv_highlight, pool_highlight)
        )

        # SLIDE 37:  ===========================================================
        # LAYER FLATTEN WRITTEN, FLATTEN LAYER HIGHLIGHTED ABOVE
        self.next_slide(
            notes=
            '''The output of the pooling layer is re-organized into a vector
            thanks to FlattenLayer.
            '''
        )
        flat_highlight = ms.get_layer_highlight('flat', **layer_highlight_config)
        self.play(
            dummy_architecture_code.TypeLetterbyLetter(lines=[3]),
            ReplacementTransform(pool_highlight, flat_highlight)
        )

        # SLIDE 38:  ===========================================================
        # LAYER DENSE WRITTEN, DENSE LAYER HIGHLIGHTED ABOVE
        self.next_slide(
            notes=
            '''And finally, fullyConnectedLayer defines a dense neural network.
            '''
        )
        dense_highlight = ms.get_layer_highlight('dense', **layer_highlight_config)
        self.play(
            dummy_architecture_code.TypeLetterbyLetter(lines=[4]),
            ReplacementTransform(flat_highlight, dense_highlight)
        )

        # SLIDE 39:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''The classification task is completed by calling the softmax
            activation function in softmaxLayer,
            '''
        )
        self.play(dummy_architecture_code.TypeLetterbyLetter(lines=[5]))

        # SLIDE 40:  ===========================================================
        # 
        self.next_slide(
            notes=
            '''and converting the resulting probabilities into classes with the
            classificationLayer. It's important to note that the layers'
            arguments are design choices that depend on the problem. They are
            usually fine-tuned through trial and error. In this example, we
            present one specific configuration, but we encourage you to explore
            different ones and see how the results change.
            '''
        )
        self.play(dummy_architecture_code.TypeLetterbyLetter(lines=[6]))

        # SLIDE 41:  ===========================================================
        # CODE GLIMPSE DISAPPEARS
        # HIGHLIGHT CONV LAYER AND FOCUS ON IT
        self.next_slide(
            notes=
            '''Let's see in detail how to define these layers, starting from the
            convolutional one.
            '''
        )

        new_highlight_config = {'color':GOLD, 'stroke_width':4}
        new_input = ms.input.copy()
        new_conv = ms.conv_layer.copy()
        new_input_highlight = ms.input_highlight.copy().set(**new_highlight_config)
        new_conv_highlight = ms.conv_highlight_1.copy().set(**new_highlight_config)
        Group(new_input, new_input_highlight).scale(2).move_to(HALF_SCREEN_LEFT + 3*UP)
        Group(new_conv, new_conv_highlight).scale(2).move_to(HALF_SCREEN_RIGHT + 3*UP)
        new_input.save_state()
        new_conv.save_state()
        phony_conv_rect=SurroundingRectangle(new_conv, buff=0, stroke_width=0)  # will need for later
        phony_input_rect=SurroundingRectangle(new_input, buff=0, stroke_width=0)  # will need for later


        side=0.5
        filter_kernel = VGroup(
            Square(side, fill_opacity=0, **new_highlight_config).add(
                MathTex(f'l_{i}', color=GOLD).scale_to_fit_height(side*0.5)
            ) for i in range(9)
        ).arrange_in_grid(3,3, buff=0).align_to(DSS.secondaryRect.get_bottom()+1*UP, DOWN)

        phony_filter_square = Square(filter_kernel.width, **new_highlight_config).set_opacity(0).move_to(filter_kernel)
        new_giz1 = create_gizmo(new_input_highlight, phony_filter_square)
        new_giz2 = create_gizmo(phony_filter_square, new_conv_highlight)

        self.play(
            FadeOut(dummy_architecture_code, dense_highlight),
            FadeOut(*[mob for mob in ms.submobjects if mob not in (ms.input, ms.input_highlight, ms.conv_layer, ms.conv_highlight_1)]),
            # FadeOut(ms.conv_layer[1:])
        )
        self.play(
            ReplacementTransform(ms.input, new_input),  # Easier for later
            ms.input_highlight.animate.become(new_input_highlight),
            ReplacementTransform(ms.conv_layer, new_conv),   # For conv layer stack it will be easier to work with a copy
            ms.conv_highlight_1.animate.become(new_conv_highlight),
        )
        self.play(
            Create(filter_kernel, run_time=1),
            Succession(Wait(0.3),  Create(new_giz1, lag_ratio=0, run_time=0.5)),
            Succession(Wait(1),  Create(new_giz2, lag_ratio=0, run_time=0.5)),
            Succession(Wait(1), cnn_architecture_code.TypeLetterbyLetter(lines=[3]))
        )

        # SLIDE 42:  ===========================================================
        # CONV2D LAYER LINE WRITTEN
        # '3' HIGHLIGHTED, BRACE WITH 3 APPEARS ON FILTER KERNEL
        self.next_slide(
            notes=
            '''The first argument to the convolution2dlayer layer is the
            dimension of the filter. Here we set both height and width to 3.
            '''
        )
        filter_3x3_highlight = HighlightRectangle(cnn_architecture_code[3][19])
        kernel_braces = get_labeled_braces(filter_kernel, DOWN, '3', LEFT, '3', brace_config={'buff':0.1}, label_config=brace_label_config)
        kernel_braces.set_z_index(50)

        self.play(
            Create(filter_3x3_highlight),
            FadeIn(kernel_braces)
        )

        # SLIDE 43:  ===========================================================
        # 32 HIGHLIGHTED, SMALL KERNEL APPEARS
        self.next_slide(
            notes=
            '''The second argument, 32, sets the number of filters.
            '''
        )
        filter_32_highlight = HighlightRectangle(cnn_architecture_code[3][21:23])

        # Create the many filters and convolution
        filter_colors = [GOLD, RED, GREEN, ORANGE, PURPLE]
        star_symbol = MathTex('*', color=BLACK, font_size=48)
        equal_symbol = MathTex('=', color=BLACK, font_size=48)
        im_side_length=0.75
        grid_of_convolutions = Group( 
            *[m for i in range(5) for m in [
                new_input.copy().scale_to_fit_height(im_side_length),
                star_symbol.copy(),
                filter_kernel.copy().set_color(filter_colors[i]).scale_to_fit_height(im_side_length),
                equal_symbol.copy(),
                ms.conv_layer[i].copy().scale_to_fit_height(im_side_length)
            ]]
        ).arrange_in_grid(5,5, (1.25, 0.15))
        vdots = VGroup(
            MathTex(r'\vdots', color=BLACK, font_size=32).next_to(grid_of_convolutions[j], DOWN, buff=0.25)
            for j in [-1, -3, -5]
        )
        grid_of_convolutions.add(vdots)
        grid_of_convolutions.move_to(DSS.secondaryRect)

        self.play(
            AnimationGroup(
                FadeOut(kernel_braces, ms.conv_highlight_1, ms.input_highlight, new_giz1, new_giz2),
                AnimationGroup(
                    *[new_conv[i].animate.become(grid_of_convolutions[4+5*i]) for i in range(5)],
                    new_input.animate.become(grid_of_convolutions[0]),
                    ReplacementTransform(filter_kernel, grid_of_convolutions[2]),
                ),
                FadeIn(*[grid_of_convolutions[i] for i in range(len(grid_of_convolutions)) if i not in [0,2, *[4+5*j for j in range(5)]]]),
                lag_ratio=0.5
            ),
        )
        self.play(
            ReplacementTransform(filter_3x3_highlight, filter_32_highlight),
        )

        # SLIDE 44:  ===========================================================
        # 'PADDING=SAME' HIGHLIGHT
        self.next_slide(
            notes=
            '''The option padding equal to 'same' ensures that the output have
            the same dimension of the input image.
            '''
        )
        padding_highlight = HighlightRectangle(cnn_architecture_code[3][24:-1])

        VGroup(phony_input_rect, phony_conv_rect).arrange(buff=2).move_to(DSS.secondaryRect)
        self.play(
            Succession(
                AnimationGroup(
                    FadeOut(  # FadeOut input images, *,=
                        *[grid_of_convolutions[i] for i in range(len(grid_of_convolutions)) if i not in [4+5*j for j in range(5)]]
                    ),
                    AnimationGroup(
                        new_input.animate.restore().move_to(phony_input_rect),
                        new_conv.animate.restore().move_to(phony_conv_rect), # move conv layer into position,
                    ),
                    lag_ratio=0.5
                ),
            )
        )
        self.play(ReplacementTransform(filter_32_highlight, padding_highlight))



        # SLIDE 45:  ===========================================================
        # RELU DISAPPEARS, RETURN REST TO CENTER
        # BRACES WITH 28 APPEAR ON INPUT AND CONV LAYER
        self.next_slide(
            notes=
            '''As a result, the input of the convolutional layer is a single
            28x28 image, while the output is a set of 32 28x28 matrices. It is
            worth noting that Since each filter consists of 9 learnable
            parameters, we have 9*32 = 288 parameters in total to be optimized.
            But we will come back to this later.
            '''
        )

        input_size_braces = get_labeled_braces(new_input, LEFT, '28', DOWN, '28', label_config=brace_label_config)
        output_size_braces = get_labeled_braces(new_conv.top(), RIGHT, '28', DOWN, '28', label_config=brace_label_config)
        conv_braces = new_conv.get_stack_brace(DL, '32', label_config=brace_label_config)
        arrow_config = {'buff':0.5, 'color':DARK_BLUE, 'stroke_width':4, 'max_stroke_width_to_length_ratio':20}
        conv_arrow = Arrow(new_input.get_right(), new_conv.get_left(), **arrow_config)

        self.play(
            Succession(
                FadeIn(input_size_braces),
                FadeIn(conv_arrow),
                FadeIn(conv_braces),
                FadeIn(output_size_braces)
            )
        )

        # SLIDE 46:  ===========================================================
        # RELULAYER WRITTEN; GRAPH OF RELU APPEARS ABOVE
        self.next_slide(
            notes=
            '''We then set the activation function to relu to keep the output
            positive.
            '''
        )
        relu_plot = ReLUPlot().scale_to_fit_width(4)
        VGroup(phony_conv_rect, relu_plot).arrange(buff=2).move_to(DSS.secondaryRect)
        relu_arrow = Arrow(phony_conv_rect.get_right(), relu_plot.get_left(), **arrow_config)

        self.play(
            Succession(
                AnimationGroup(
                    FadeOut(padding_highlight, conv_arrow, new_input, input_size_braces, output_size_braces, conv_braces),
                    new_conv.animate.move_to(phony_conv_rect),
                    lag_ratio=0.5
                ),
                AnimationGroup(
                    FadeIn(relu_plot.axes, relu_plot.labels, shift=RIGHT),
                    GrowArrow(relu_arrow),
                    cnn_architecture_code.TypeLetterbyLetter(lines=[4])
                ),
                Create(relu_plot.relu, run_time=2)
            )
        )

        # SLIDE 47:  ===========================================================
        # SCHEME REPLACED WITH POOLING LAYER
        # POOLING LAYER LINE WRITTEN
        self.next_slide(
            notes=
            '''Next, we apply max pooling.
            '''
        )
        ms.restore()  # full screen, not 0.75
        pooling_example_g = Group(ms.conv_layer, ms.conv_highlight_2, ms.pooling_layer, ms.pooling_highlight_1, ms.conv_pool_gizmo)
        pooling_example_g.scale(1.5).move_to(DSS.secondaryRect)
        
        self.play(
            AnimationGroup(
                FadeOut(relu_plot, relu_arrow),
                ReplacementTransform(new_conv, ms.conv_layer),
                FadeIn(*[mob for mob in pooling_example_g if mob not in [ms.conv_layer]]),
                lag_ratio=0.5
            )
        )

        self.play(cnn_architecture_code.TypeLetterbyLetter(lines=[5]))

        # SLIDE 48:  ===========================================================
        # GRID HIGHLIGHTING REDUCTION FACTOR IS CREATED
        # POOL_SIZE=(7,7) HIGHLIGHTED 
        self.next_slide(
            notes=
            '''In the argument we define the dimensions of the pool size: for
            instance 7 by 7.
            '''
        )
        conv_top = pooling_example_g[0][0]
        pool_factor_grid_highlight = VGroup(
            ms.conv_highlight_2.copy() for _ in range(16)
        ).set_stroke(WHITE, 3).arrange_in_grid(4,4, buff=0).move_to(conv_top).set_z_index(conv_top.z_index + 0.5)
        
        pool_factor_braces = get_labeled_braces(ms.conv_highlight_2, RIGHT, '7', DOWN, '7', label_config=brace_label_config).set_z_index(ms.conv_highlight_2.z_index)
        pool_factor_braces[0].align_to(conv_top.get_right(), LEFT).shift(RIGHT*0.1)
        pool_factor_braces[1].align_to(conv_top.get_bottom(), UP).shift(DOWN*0.1)

        pool_f, pool_x, pool_y = ms.pooling_factor, ms.POOLING_HIGHLIGHT_POS[0], ms.POOLING_HIGHLIGHT_POS[1]
        to_pool = PixelArray(ms.filtered_[0][pool_f*pool_x:pool_f*(pool_x+1), pool_f*pool_y:pool_f*(pool_y+1)], stroke_width=2, stroke_color=WHITE)
        to_pool.match_height(ms.conv_highlight_2).move_to(ms.conv_highlight_2)
        to_pool.add_pixel_values(color=SATURATED_RED)
        to_pool.set_z_index(ms.conv_highlight_2.z_index + 1)

        self.play(
            Succession(
                Create(pool_factor_grid_highlight),
                FadeIn(pool_factor_braces)
            )
        )

        # SLIDE 49:  ===========================================================
        # BRACES WITH 7 APPEAR
        self.next_slide(
            notes=
            '''Here, we also add a stride of the same dimension, ...
            '''
        )
        pool_size_highlight = HighlightRectangle(cnn_architecture_code[5][20:-1])
        self.play(Create(pool_size_highlight))

        # SLIDE 50:  ===========================================================
        # BRACES WITH 7 APPEAR
        self.next_slide(
            notes=
            '''..., meaning that this layer retains the maximum value within
            each 7x7 window.
            '''
        )
        # updaters on gizmo for later
        ms.conv_pool_gizmo[0].add_updater(
            lambda mob: mob.put_start_and_end_on(
                ms.conv_highlight_2.get_corner(UL), ms.pooling_highlight_1.get_corner(UL)
            )
        )
        ms.conv_pool_gizmo[1].add_updater(
            lambda mob: mob.put_start_and_end_on(
                ms.conv_highlight_2.get_corner(DR), ms.pooling_highlight_1.get_corner(DR)
            )
        )
        self.play(Create(to_pool))
        self.play(
            Group(ms.conv_layer, ms.conv_highlight_2, pool_factor_grid_highlight,pool_factor_braces).animate.scale(0.75).shift(1.25*UP+2*LEFT),
            Group(ms.pooling_layer, ms.pooling_highlight_1).animate.scale(0.75).shift(1.25*UP+0.5*RIGHT),
            to_pool.animate.scale_to_fit_height(1.8).set_x(0).align_to(DSS.secondaryRect.get_bottom()+0.5*UP, DOWN),
        )
        ms.conv_pool_gizmo.clear_updaters()

        to_pool.add_brackets(left=r"(", right=r")", color=BLACK)
        max_label = MathTex(r'\max', color=BLACK, font_size=64).next_to(to_pool.brackets, LEFT)
        max_id = np.argmax(to_pool.array)  # index into the flattened array
        max_result = to_pool.pixel_array[max_id].copy().set_stroke(GOLD,3).set_fill(opacity=0)
        
        self.play(
            Succession(
                FadeIn(max_label, to_pool.brackets),
                Create(max_result),
            )
        )

        max_result.set_fill(opacity=1).add(to_pool.pixel_values[max_id].copy())
        def put_result_into_position(mob):
            mob.match_height(ms.pooling_highlight_1).move_to(ms.pooling_highlight_1)
            return mob
        
        self.play(
            Succession(
                Wait(0.5),
                ApplyFunction(put_result_into_position, max_result)
            )
        )

        # SLIDE 51:  ===========================================================
        # FADE OUT EVERYTHING BUT CONVOLUTION AND POOLING MATRICES
        # BRACES WITH 32 AND 4 APPEAR
        self.next_slide(
            notes=
            '''This reduces the dimensions significantly: the input consisting
            in 32 matrices of size 28x28 is reduced to a set of 32 matrices of
            reduced size 4x4.
            '''
        )
        self.play(
            AnimationGroup(
                FadeOut(max_label, to_pool, max_result,*[mob for mob in pooling_example_g if mob not in [ms.conv_layer, ms.pooling_layer]],
                        pool_factor_grid_highlight, pool_factor_braces),
                Group(ms.conv_layer, ms.pooling_layer).animate.scale(4/3).arrange(buff=3).match_y(DSS.secondaryRect),
                lag_ratio=0.5
            )
        )

        input_size_braces = get_labeled_braces(conv_top, DOWN, '28', RIGHT, '28', label_config=brace_label_config)
        conv_braces = ms.conv_layer.get_stack_brace(DL, '32', label_config=brace_label_config)
        output_size_braces = get_labeled_braces(ms.pooling_layer[0], DOWN, '4', RIGHT, '4', label_config=brace_label_config)
        pool_braces = ms.pooling_layer.get_stack_brace(DL, '32', label_config=brace_label_config)
        pool_arrow = Arrow(ms.conv_layer.get_right(), ms.pooling_layer.get_left(), buff=0.75, color=DARK_BLUE, stroke_width=4, max_stroke_width_to_length_ratio=20)

        self.play(
            Succession(
                FadeIn(input_size_braces, conv_braces),
                FadeIn(pool_arrow),
                FadeIn(output_size_braces, pool_braces)
            )
        )

        # SLIDE 52:  ===========================================================
        # SCHEME REPLACED WITH FLATTEN LAYER
        # FLATTEN LAYER LINE WRITTEN
        self.next_slide(
            notes=
            '''At this point, we convert the set of matrices into a vector
            format that a dense neural network can process using the Flatten
            layer. Note that this layer does not perform any computation on the
            data:
            '''
        )
        self.play(
            FadeOut(
                input_size_braces, conv_braces, output_size_braces,
                pool_braces, pool_arrow, ms.conv_layer,
                pool_size_highlight
            )
        )
        self.play(
            ms.pooling_layer.animate.arrange(UL, buff=0.1).scale_to_fit_height(DSS.secondaryRect.height*0.8).move_to(DSS.secondaryRect),
            cnn_architecture_code.TypeLetterbyLetter(lines=[6])
        )
        pooled_PA = VGroup(
            PixelArray(
                ms.pooled_[i], stroke_color=None, stroke_width=0
            ).match_height(ms.pooling_layer[i]).move_to(ms.pooling_layer[i]).set_z_index(ms.pooling_layer[i].z_index)
            for i in range(5)
        )
        self.add(pooled_PA)

        # SLIDE 53:  ===========================================================
        # FALTTENING ANIMATION
        self.next_slide(
            notes=
            '''It simply reshapes it, transforming the set of 32 matrices into a
            one dimensional vector with 512 elements.
            '''
        )
        self.play(
            AnimationGroup(
                FadeOut(ms.pooling_layer),
                AnimationGroup(
                    p.pixel_array.animate.arrange(RIGHT, buff=0).move_to(p)
                    for p in pooled_PA
                ),
                Wait(0.5),
                lag_ratio=1
            )
        )
        self.play(pooled_PA.animate.arrange(LEFT, buff=0).scale_to_fit_width(0.9*FRAME_WIDTH).move_to(DSS.secondaryRect))

        pooled_PA.set_z_index(1)
        pooled_PA_outline = SurroundingRectangle(pooled_PA, color=DARK_BLUE, buff=0, stroke_width=6).set_z_index(pooled_PA.z_index-1)
        brace_tot = get_labeled_brace(pooled_PA, UP, '512', label_config=brace_label_config)

        self.play(Create(pooled_PA_outline), FadeIn(brace_tot))
        pooled_PA.add(pooled_PA_outline)

        # SLIDE 54:  ===========================================================
        # SCHEME REPLACED WITH DENSE LAYER
        self.next_slide(
            notes=
            '''The Dense layer is the last layer in our CNN architecture.
            '''
        )
        self.play(FadeOut(pooled_PA, brace_tot))

        ms.restore()
        dense_layer_vg = VGroup(ms.dense_layer, ms.softmax, ms.output_layer, ms.output_arrow)
        dense_layer_vg.scale(0.75).move_to(DSS.secondaryRect)

        self.play(
            cnn_architecture_code.TypeLetterbyLetter(lines=[7]),
            FadeIn(dense_layer_vg)
        )

        # SLIDE 55:  ===========================================================
        # '10' HIGHLIGHTED, OUTPUT DIGITS BRACE APPEARS
        self.next_slide(
            notes=
            '''The argument, 10, defines the number of output neurons,
            corresponding to the 10 possible classes in our classification task,
            that is the digits from 0 to 9.
            '''
        )
        dense_10_highlight = HighlightRectangle(cnn_architecture_code[7][-3:-1])
        classific_10_brace = get_labeled_brace(ms.output_layer, RIGHT, '10', label_config=brace_label_config)

        self.play(
            Succession(
                Create(dense_10_highlight),
                FadeIn(classific_10_brace)
            )
        )

        # SLIDE 56:  ===========================================================
        # 'ACTIVATION=SOFTMAX' HIGHLIGHTED, SOFTMAX APPEARS IN SCHEME
        self.next_slide(
            notes=
            '''Then we add a SoftMax Layer, that converts the output scores of
            the network in classification probabilities.
            '''
        )
        softmax_scheme_highlight = SurroundingRectangle(ms.softmax, color=GOLD, stroke_width=4, corner_radius=0.25, buff=0.1)
        self.play(
            Succession(
                FadeOut(classific_10_brace, dense_10_highlight),
                cnn_architecture_code.TypeLetterbyLetter(lines=[8]),
                ShowPassingFlash(softmax_scheme_highlight, run_time=2, time_width=0.3)
            )
        )

        # SLIDE 57:  ===========================================================
        # FULL SCHEME REAPPEARS ON TOP
        self.next_slide(
            notes=
            '''Finally, the classificationLayer computes the cross-entropy loss
            function for classification tasks.
            '''
        )
        self.play(cnn_architecture_code.TypeLetterbyLetter(lines=[9]))

        # SLIDE 58:  ===========================================================
        # FULL SCHEME REAPPEARS ON TOP
        self.next_slide(
            notes=
            '''We have now finished defining the CNN architecture.
            '''
        )
        DSS.add_side_obj(dense_layer_vg)
        DSS.remove_main_obj()
        mat_env.clear()
        cnn_architecture_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(
            AnimationGroup(
            cnn_architecture_code.IntoMatlab(mat_env),
            DSS.bringOut(),
            )
        )
        self.play(mat_env.Run(new_cursor=True))

        # SLIDE 59:  ===========================================================
        # LEARNABLE COEFFICIENT SCHEME APPEAR
        self.next_slide(
            notes=
            '''Remember! Along our way we have many unknowns to determine.
            '''
        )
        LCNNscheme = LearnableCoefficientsScheme()
        dummy_gray_rect = FullScreenBackground(MATLAB_GRAY).set_z_index(50)

        self.play(FadeIn(dummy_gray_rect))
        self.clear()
        # mat_env.clear()

        DSS.reset()
        DSS.add_empty_side_obj(FRAME_HEIGHT)
        DSS.hard_bring_in()
        self.add(DSS.secondaryRect)

        self.play(FadeIn(LCNNscheme))

        # SLIDE 60:  ===========================================================
        # TRAINING TITLE APPEARS
        self.next_slide(
            notes=
            ''' This is the task for the training process. Indeed, the training
            process corresponds to the solution of an optimization problem in
            which we minimize the loss function L.
            '''
        )
        training_title = SlideTitle('Training')

        self.play(
            AnimationGroup(
                LCNNscheme.animate.move_to(TITLED_CENTER),
                Write(training_title),
                lag_ratio=0.5
            )
        )

        # SLIDE 61:  ===========================================================
        # LEARNABLE COEFFICIENT BROGHT OUT
        # KEARS OPTMIZER LINE WRITTEN
        self.next_slide(
            notes=
            '''We begin by setting the training options using the
            trainingOptions function.
            '''
        )
       
        DSS.add_side_obj(VGroup(LCNNscheme, training_title))
        self.play(DSS.bringOut())
        self.remove(training_title, LCNNscheme)

        training_code = MatlabCode(
            r'''
            % Set up training parameters
            options = trainingOptions(...
                'sgdm', ...
                'InitialLearnRate', 0.01, ...
                'MaxEpochs', 20, ...
                'MiniBatchSize', 16, ...
                'Shuffle', 'every-epoch', ...
                'ValidationData', {X_validation, y_validation}, ...
                'Plots', 'training-progress', ...
                'Verbose', false ...
            );
            '''
        )

        self.play(training_code.TypeLetterbyLetter(lines=[0,1,10]))

        # SLIDE 62:  ===========================================================
        # SGDM LINE WRITTEN
        self.next_slide(
            notes=
            '''The first argument selects the optimizer. Here we use stochastic
            gradient descent method, abbreviated sdgm.
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 63:  ===========================================================
        # LEARNING RATE LINE WRITTEN
        self.next_slide(
            notes=
            '''Next, the second argument defines the learning rate. Here it is
            set to 0.01. We opt to perform training using a minibatches
            approach. At each optmization step, the CNN processes only a small
            subset of the dataset, called a batch. Once the optimizer has seen
            all the dataset, an epoch is completed.
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[3]))

        # SLIDE 64:  ===========================================================
        # NUMBER OF EPOCHS LINE WRITTEN
        self.next_slide(
            notes=
            '''In our case, we set the number of epochs to 20;
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[4]))

        # SLIDE 65:  ===========================================================
        # BATCH SIZE LINE WRITTEN
        self.next_slide(
            notes=
            '''The batch size is instead set in the foruth argument to 16,
            meaning the model updates its learnable parameters by processing 16
            examples at the time.
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[5]))

        # SLIDE 66:  ===========================================================
        # SHUFFLE LINE WRITTEN
        self.next_slide(
            notes=
            '''We also allow shuffling at every epoch, meaning that the
            minibatches change during the training.
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[6]))

        # SLIDE 67:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The sixth argument is used to associate the validation data to
            the training,
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[7]))

        # SLIDE 68:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''and the seventh enables a live visualization of the loss function
            during training.
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[8]))

        # SLIDE 69:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''Finally, verbose argument is set to false to limit the amount of
            text printed.
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[9]))

        # SLIDE 70:  ===========================================================
        # INTO MATLAB
        # NEW CELL, OUT OF MATLAB
        self.next_slide(
            notes=
            '''We're ready to start the training process.
            '''
        )
        train_net_code = MatlabCode(
            r'''
            % Train the network
            CNN = trainNetwork(...
                X_train, categorical(y_train), ...
                layers, ...
                options ...
            );
            '''
        )

        training_code.add_background_window(FullScreenBackground(WHITE))
        self.wait(0.5)
        self.play(training_code.IntoMatlab(mat_env))
        self.wait(0.5)
        mat_env.add_cell()
        self.play(mat_env.OutofMatlab(cell=-1))
        self.play(train_net_code.TypeLetterbyLetter(lines=[0]))
        
        # SLIDE 71:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The trainNetwork function starts the training process. The
            arguments are:
            '''
        )
        self.play(train_net_code.TypeLetterbyLetter(lines=[1,5]))

        # SLIDE 72:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The training data, consisting of images X_train and labels
            y_train converted into categorical variables with the function
            "categorical";
            '''
        )
        self.play(train_net_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 73:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The architecture of the CNN defined in "layers";
            '''
        )
        self.play(train_net_code.TypeLetterbyLetter(lines=[3]))

        # SLIDE 74:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''the training options defined in "options".
            '''
        )
        self.play(train_net_code.TypeLetterbyLetter(lines=[4]))

        # SLIDE 75:  ===========================================================
        # INTO MATLAB, RUN CODE, TRAINING PLOT APPEARS
        self.next_slide(
            notes=
            '''Here's the output of this code.
            '''
        )
        train_net_code.add_background_window(FullScreenBackground(WHITE))
        mat_env.remove_cell()
        mat_env.remove_cell_from_top(n=1)  # remove # define the model code  
        self.play(train_net_code.IntoMatlab(mat_env))
        # Create training plot
        TRAIN_PLOT_COLOR = ManimColor("#F5F5F5")
        loss_plot = ImageMobject(r'Assets\W5\mat_loss_plot.png')
        acc_plot = ImageMobject(r'Assets\W5\mat_acc_plot.png').next_to(loss_plot, UP, buff=0).align_to(loss_plot, LEFT)
        train_log = ImageMobject(r'Assets\W5\mat_train_log.png').next_to(loss_plot, RIGHT, buff=0).align_to(loss_plot, DOWN)
        training_progress = Text('Training progress', color=BLACK, weight=BOLD, font=SANS_SERIF_FONT, font_size=18).next_to(acc_plot, UP, buff=0.1)
        training_plot = Group(acc_plot, loss_plot, train_log, training_progress).scale_to_fit_width(0.75*FRAME_WIDTH).center()
        mat_env.add_output_plot(training_plot, color=TRAIN_PLOT_COLOR)

        self.play(mat_env.Run())
        self.wait(0.5)
        self.play(mat_env.FocusPlot(scale=0.99, background_color=TRAIN_PLOT_COLOR))
        
        # SLIDE 76:  ===========================================================
        # HIGHLIGHT ACCURACY PLOT
        self.next_slide(
            notes=
            '''In the top figure, we can visualize the accuracy of the model on
            the training set (blue line) and on the validation set (black line).
            '''
        )
        plot_highlight_config = {'color':BLUE, 'stroke_width':4, 'buff':0, 'corner_radius':0.25}
        accuracy_highlight = SurroundingRectangle(acc_plot, **plot_highlight_config)
        self.play(Create(accuracy_highlight))

        # SLIDE 77:  ===========================================================
        # HIGHLIGHT LOSS PLOT
        self.next_slide(
            notes=
            '''In the bottom figure, at each optimizer iteration the code
            displays the loss value on the training set in red the loss value on
            the validation set in black.
            '''
        )
        loss_highlight = SurroundingRectangle(loss_plot, **plot_highlight_config)
        self.play(ReplacementTransform(accuracy_highlight, loss_highlight))

        # SLIDE 78:  ===========================================================
        # FINAL ACCURACY HIGHLIGHT
        self.next_slide(
            notes=
            '''By the final iteration, we can see that the model reaches very
            satisfactory performances: about 97% accuracy on the validation
            dataset.
            '''
        )
        valid_acc_highlight = loss_highlight.copy().stretch_to_fit_height(21*mat_env.PIXEL).stretch_to_fit_width(62*mat_env.PIXEL)
        valid_acc_highlight.move_to(MatlabEnv._pixel2p(1182, 68), aligned_edge=UL)
        self.play(ReplacementTransform(loss_highlight, valid_acc_highlight))

        # SLIDE 79:  ===========================================================
        # FINAL ACCURACY HIGHLIGHT
        self.next_slide(
            notes=
            '''We can also display this information by performing the
            classification task on the validation set using the function
            classify with arguments the CNN. The predicted categories are store
            into the array Ypred.
            '''
        )
        dummy_rect = FullScreenBackground(WHITE).set_z_index(0)
        self.play(FadeIn(dummy_rect))
        mat_env.remove_plot()
        self.remove(valid_acc_highlight)

        test_valid_acc_code = MatlabCode(
            r'''
            % Evaluate model on validation dataset
            [YPred, scores] = classify(CNN, X_validation);
            validation_acc = sum(YPred == categorical(y_validation)) ...
                             / numel(y_validation);
            fprintf('Validation Accuracy: %f\n%', validation_acc * 100)
            '''
        )

        self.play(test_valid_acc_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 80:  ===========================================================
        # FINAL ACCURACY HIGHLIGHT
        self.next_slide(
            notes=
            '''The validation accuracy is then computed by checking how many
            categories match with the ground truth.
            '''
        )
        self.play(test_valid_acc_code.TypeLetterbyLetter(lines=[1,2,3,4]))

        # SLIDE 81:  ===========================================================
        # FINAL ACCURACY HIGHLIGHT
        self.next_slide(
            notes=
            '''Specifically, we count the number of matches with the sum
            function applied to the boolean vector resulting from checking if
            Ypred is equal to the validation labels. This quantity is divided by
            the number of validation samples computed by the numel function.
            '''
        )
        match_category_highlight = HighlightRectangle(test_valid_acc_code[2][19:51])
        self.play(FadeIn(match_category_highlight))
        self.wait(2)
        test_valid_acc_code.add_background_window(dummy_rect.set_z_index(-1))
        mat_env.remove_output()
        mat_env.remove_plot()
        mat_env.remove_cursor()
        self.play(FadeOut(match_category_highlight))
        self.play(test_valid_acc_code.IntoMatlab(mat_env))
        mat_env.add_output_command_window('Final Validation Accuracy: 96.700000')
        self.play(mat_env.Run())

        # SLIDE 82:  ===========================================================
        # PREDICTION CODE WRITTEN
        self.next_slide(
            notes=
            ''' Now we are at the fourth step! Let's look at some results using
            the function visualize_digit_prediction. This function visualizes
            one randomly selected image of the digit 3 from the validation
            set...
            '''
        )
        first_prediction_code = MatlabCode(
            r'''
            % Prediction sample
            visualize_digit_prediction(CNN, validation_dir_path, 3);
            '''
        )

        mat_env.add_cell()
        self.play(mat_env.OutofMatlab(cell=-1))
        self.play(first_prediction_code.TypeLetterbyLetter())

        # SLIDE 83:  ===========================================================
        # INTO COLAB, RUN CELL, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''and displays the prediction probabilities for each digit made by
            the model.
            '''
        )
        MATLAB_HIST_COLOR = "#0072bd"
        def visualize_probabilites(image: str,predictions: np.ndarray) -> ImageMobject:
            fig = plt.figure(figsize=(15, 5), dpi=300)
            ax0 = plt.subplot2grid(shape=(1,3), loc=(0,0), rowspan=1, colspan=1)
            ax1 = plt.subplot2grid(shape=(1,3), loc=(0,1), rowspan=1, colspan=2)
            # Display the original image
            ax0.imshow(np.array(Image.open(image)), cmap='gray')
            ax0.set_title('Input Image')
            ax0.axis('off')
            # Probabilities
            digits = range(10)
            ax1.bar(digits, predictions * 100, color = MATLAB_HIST_COLOR, edgecolor="black")
            ax1.set_xlabel('Digit Prediction')
            ax1.set_ylabel('Prediction Probability (%)')
            ax1.set_xticks(digits)
            ax1.set_ylim(0, 100)
            ax1.tick_params(bottom=True, top=True, left=True, right=True)
            ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
            ax1.tick_params(axis="x", direction="in", length=6)
            ax1.tick_params(axis="y", direction="in", length=6)
            # Add percentage labels on top of each bar
            for i, v in enumerate(predictions):
                ax1.text(i, v * 100 + 1, f'{v * 100:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            fig.patch.set_alpha(0)
            return draw_plot(fig).scale_to_fit_height(0.25*FRAME_HEIGHT)

        predict_3_prob = np.array([0,0, 0.008, 0.99,0,0,0,0, 0.003,0])
        predict_3_output = visualize_probabilites(r'Assets\W5\validation3.png', predict_3_prob)

        first_prediction_code.add_background_window(FullScreenBackground(WHITE))
        mat_env.clear()
        self.play(first_prediction_code.IntoMatlab(mat_env))
        mat_env.add_output_plot(predict_3_output)
        self.play(mat_env.Run())

        # SLIDE 84:  ===========================================================
        # FOCUS OUTPUT
        self.next_slide(
            notes=
            '''For this validation sample, the model correctly classified the
            image as a 3 with 99% confidence.
            '''
        )
        self.play(mat_env.FocusPlot(scale=0.9))

        # SLIDE 85:  ===========================================================
        # UNFOCUS OUTPUT
        # NEW CELL, SECOND PREDICTION CODE WRITTEN
        self.next_slide(
            notes=
            '''Let's try with another image with the digit 5.
            '''
        )
        mat_env.remove_cursor()
        self.play(FadeOut(mat_env.plot))

        second_prediction_code_cell = MatlabCodeBlock(
            r'''
            visualize_digit_prediction(CNN, validation_dir_path, 5);
            '''
        )
        mat_env.add_cell(second_prediction_code_cell)
        self.remove(second_prediction_code_cell.code)
        self.wait(0.3)
        self.play(second_prediction_code_cell.TypeLetterbyLetter())

        # SLIDE 86:  ===========================================================
        # RUN CODE, OUTPUT APPEARS
        # FOCUS OUTPUT
        self.next_slide(
            notes=
            '''For this other validation example, the model correctly classifies
            the image as a 5, with a confidence of 56.9%. Interestingly, it also
            assigns a 37.3% probability to the digit 3, indicating some
            uncertainty between the two classes.
            '''
        )
        predict_5_prob = np.array([0,0,0,0.373, 0.003, 0.569, 0.001, 0,0.048,0.005])
        predict_5_output = visualize_probabilites(r'Assets\W5\validation5.png', predict_5_prob)

        mat_env.add_output_plot(predict_5_output)
        self.play(mat_env.Run())
        self.play(mat_env.FocusPlot(scale=0.9))
        self.wait(0.05)
