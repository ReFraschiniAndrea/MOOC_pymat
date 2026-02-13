import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from manim import *
from mooc_utils import *
from mooc_utils.colab import *
from W5Anim import *
from PIL import Image
import matplotlib.pyplot as plt

config.update(RELEASE_CONFIG)
config.max_files_cached = 250


class W5Python_slides(MOOCSlide):
    def construct(self):
        # SLIDE 01:  ===========================================================
        # COLAB NOTEBOOK FADES IN
        # CNN SCHEME FADES IN ON TOP
        self.next_slide(
            notes=
            '''Let's open a notebook, and let's start to learn how to code an
            image digit classifier based on image processing and convolutional
            neural networks. This will be done in four steps:
            '''
        )
        cl_env = ColabEnv(self, r'Assets\W5\colabCNN.png')

        # Create the full main scheme for later 
        digit = np.array(Image.open(r'Assets\W5\mnist\mnist80.png'))
        ms = CNNDigitRecognitionScheme(digit, input_label=8, n_filters=5, pooling_factor=7,
                                       pixel_size=(1.5/28, 1.5/28, 0.125, 0.125), horizontal_spacing=0.8,
                                       highlights_kwargs={'color': GOLD, 'stroke_width': 3},
                                       outline_kwargs={'color': DARK_BLUE, 'stroke_width': 6})
        ms.output_layer.activate(8)
        ms.save_state()
        ms.scale(0.75)

        surrounding_rect = SurroundingRectangle(ms, fill_color=WHITE, fill_opacity=1, stroke_width=0.5,
                                              stroke_color=BLACK, corner_radius=0.2, buff=0.5).set_z_index(-0.5)
        
        self.play(FadeIn(cl_env.background))
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
        # STEP EXPLANATION FADES OUT
        # OUT OF COLAB, WRITE IMPORT LINES
        self.next_slide(
            notes=
            '''Let's start by uploading the library numpy and the file
            helper_functions.py.
            '''
        )
        import_code = ColabCode(
            r'''
            import numpy as np
            import helper_functions as hf
            '''
        )
        hand_cursor = cl_env.cursor.center()

        self.play(
            Succession(
                FadeOut(steps_paragraph, surrounding_rect),
                Wait(0.3)
            )
        )
        self.play(
             Succession(
                GrowFromCenter(hand_cursor),
                ApplyMethod(hand_cursor.move_to, cl_env.PLUS_CODE_),
                hand_cursor.Click()
            )
        )
        cl_env.add_cell()
        self.wait(0.3)
        self.play(cl_env.OutofColab(cell=0), FadeOut(hand_cursor))
        self.play(import_code.TypeLetterbyLetter(lines=[0,1]))

        # SLIDE 07:  ===========================================================
        # INTO COLAB, OPEN SIDE MENU
        # CLICK UPLOAD BUTTON, HELPER FUNCTIONS FILE APPEARS
        # RUN CELL
        self.next_slide(
            notes=
            '''For this latter, we need to click on Files on the left tab,
            select upload, and pick the python file from your disk. These tools
            will be useful to read, visualize, and store the images in our
            datasets in matrix form.
            '''
        )
        import_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()
        self.play(import_code.IntoColab(cl_env))
        # Open side menu and shift cell accordingly
        self.play(cl_env.cursor.MoveAndClick(cl_env.MENU_))
        cl_env.set_image(r'Assets\W5\colabCNN_sidemenu.png')
        cl_env.get_cell(0).shift(RIGHT*cl_env.SIDE_MENU_WIDTH_)
        # Upload the file
        self.play(
            Succession(
                Wait(0.2),
                ApplyMethod(hand_cursor.move_to, cl_env.UPLOAD_),
                hand_cursor.Click()
            )
        )
        cl_env.add_file_to_sidemenu('helper_functions.py')

        # Run cell
        self.play(cl_env.Run(cell=0, new_cursor=False))

        # SLIDE 08:  ===========================================================
        # MNIST DATASET ILLUSTRATION: IMAGES APPEAR
        self.next_slide(
            notes=
            '''The dataset we will use is the open-source MNIST dataset,
            '''
        )
        self.play(cl_env.FadeOut())

        mnist_title = SlideTitle('MNIST Dataset')
        training_sample = [
            Group(
                *[PixelImage(rf'Assets\W5\mnist\mnist{i}{j}.png').scale_to_fit_height(0.24*FRAME_HEIGHT) for i in range(10)]
            ).arrange_in_grid(2, 5, buff=(0.1, 0.1)).move_to(TITLED_CENTER)
            for j in range(3)
        ]

        self.play(
            FadeIn(training_sample[0], run_time=1, lag_ratio=1),
            Write(mnist_title, run_time=1)
        )
        self.wait(0.5)
        self.play(
            Succession(
                Succession(
                    training_sample[0][i].animate(run_time=0).become(training_sample[1][i]),
                    Wait(0.1)
                )
                for i in range(10)
            )
        )
        self.wait(0.5)
        self.play(
            Succession(
                Succession(
                    training_sample[0][i].animate(run_time=0).become(training_sample[2][i]),
                    Wait(0.1)
                )
                for i in range(10)
            )
        )

        # SLIDE 09:  ===========================================================
        # DIMENSIONS WITH BRACES APPEAR
        self.next_slide(
            notes=
            '''..., which consists of 28x28 pixel images, for a total of 784
            pixels per image.
            '''
        )
        training_sample: Group = training_sample[0]
        ex_digit: PixelImage = training_sample[2]
        self.play(
            AnimationGroup(
                FadeOut(training_sample[:2], training_sample[3:]),
                ex_digit.animate.scale_to_fit_height(0.45*FRAME_HEIGHT).move_to(TITLED_CENTER),
                lag_ratio=0.5
            )
        )

        grid_28_highlight = VGroup(
            ex_digit.get_pixel_highlight(color=WHITE, stroke_width=2)
            for _ in range(28*28)
        ).arrange_in_grid(28, 28, buff=0).move_to(ex_digit)
        image_size_braces = get_labeled_braces(ex_digit, LEFT, '28', DOWN, '28')

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
            Square(0.22*FRAME_HEIGHT, stroke_width=0) for _ in range(8)
        ).arrange_in_grid(2,4, buff=(0.4, 0.4)).move_to(TITLED_CENTER)
        for i in [0,1,3]:
            training_sample[i].match_height(phony_rects[i]).move_to(phony_rects[i])
        ground_truth_config = {'stroke_color':GREEN_D,'fill_color':WHITE, 'text_kwargs':{'fill_color': GREEN_D, 'stroke_color':GREEN_D}}
        training_labels = VGroup(
            DigitRecognitionOutputCircle(i, stroke_width=12, **ground_truth_config).match_height(rect).move_to(rect)
            for i, rect in zip(range(4), phony_rects[4:])
        )
        sample_rects = VGroup(
            SurroundingRectangle(im, lab, color=BLUE, stroke_width=4, buff=0.15, corner_radius=0.25)
            for im, lab in zip(phony_rects[:4], phony_rects[4:])
        )

        self.play(
            Succession(
                AnimationGroup(
                    FadeOut(grid_28_highlight, image_size_braces),
                    ex_digit.animate.match_height(phony_rects[2]).move_to(phony_rects[2]),
                    FadeIn(*[training_sample[i] for i in [0,1,3]]),
                    lag_ratio=0.5
                ),
                FadeIn(training_labels),
                Create(sample_rects),
            )
        )
        image_label = Text('Image', font=SANS_SERIF_FONT, weight=BOLD, font_size=32, color=BLACK).next_to(phony_rects[0], LEFT, buff=0.5)
        label_label = Text('Label', font=SANS_SERIF_FONT, weight=BOLD, font_size=32, color=BLACK).next_to(phony_rects[4], LEFT, buff=0.5).match_x(image_label)
        VGroup(image_label, label_label, phony_rects).move_to(TITLED_CENTER)
        self.play(
            AnimationGroup(
                Group(training_sample[:4], training_labels, sample_rects).animate.move_to(phony_rects),
                Succession(
                    FadeIn(image_label),
                    FadeIn(label_label)
                ),
                lag_ratio=0.5
            )
        )

        # SLIDE 11:  ===========================================================
        # COLAB ENV FADES OUT
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
        # BACK TO COLAB, OPENS SIDE MENU, CLICK UPLOAD BUTTON, ZIP FILE APPEARS
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
        # BACK TO COLAB, OPENS SIDE MENU, CLICK UPLOAD BUTTON, ZIP FILE APPEARS
        self.next_slide(
            notes=
            '''We need to upload the file digits_dataset.zip containing the
            MNIST dataset in the colab folder. To do so, we click Files on the
            left tab, select upload, and pick the zip file from your disk.
            '''
        )
        [mob.set_z_index(-10) for mob in self.mobjects]
        self.play(cl_env.FadeIn())
        self.remove(hex1, hex2, train_des_label, valid_des_label)

        self.play(
            Succession(
                Wait(0.2),
                ApplyMethod(hand_cursor.move_to, cl_env.UPLOAD_),
                hand_cursor.Click()
            )
        )
        cl_env.add_file_to_sidemenu('digits_dataset.zip', type='folder')

        # SLIDE 14:  ===========================================================
        # CLICK NEW CELL, OUT OF COLAB, WRITE UNZIP COMMAND
        # INTO COLAB, RUN CELL
        # UNZIPPED FILES APPEAR IN SIDEMENU
        self.next_slide(
            notes=
            '''Once done, we extract it with the command unzip by running this
            cell.
            '''
        )
        unzip_code = ColabCode(
            r'''
            !unzip -q "/content/digits_dataset.zip" -d /content/digits_dataset
            '''
        )
        # New cell
        self.play(cl_env.cursor.MoveAndClick(cl_env.PLUS_CODE_))
        cl_env.add_cell()
        self.wait(0.3)
        self.play(cl_env.OutofColab(cell=1))
        # Unzip code
        self.play(unzip_code.TypeLetterbyLetter())
        self.wait(1)
        # into colab: target cell already in the right place
        cl_env.remove_cell()
        unzip_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        self.play(unzip_code.IntoColab(cl_env))
        self.play(cl_env.Run(cell=1, new_cursor=False))
        cl_env.add_file_to_sidemenu('training_set', type='folder')
        cl_env.add_file_to_sidemenu('validation_set', type='folder')

        # SLIDE 15:  ===========================================================
        # CLOSE SIDEMENU; NEW CELL, OUT OF COLAB
        # TRAINING PATH WRITTEN
        self.next_slide(
            notes=
            '''To load the training dataset, we first specify the path to the
            folder containing the images and their labels.
            '''
        )
        # Close sidemenu
        self.play(cl_env.cursor.MoveAndClick(cl_env.MENU_))
        cl_env.set_image(r'Assets\W5\colabCNN.png')
        cl_env.get_cells().shift(LEFT*cl_env.SIDE_MENU_WIDTH_)
        cl_env.clear_sidemenu()
        # New cell, out of colab
        self.play(cl_env.cursor.MoveAndClick(cl_env.PLUS_CODE_))
        cl_env.add_cell()
        self.wait(0.3)
        self.play(cl_env.OutofColab(cell=-1))

        load_training_code = ColabCode(
            r'''
            # Define paths
            training_dir_path = 'digits_dataset/training_set'

            # Training Data
            X_train, y_train = hf.load_data(training_dir_path)

            # Set random seed for reproducibility
            np.random.seed(1)

            # Visualize 4 random sample of the digit 5
            hf.show_random_samples(base_dir=training_dir_path, digit=5)
            '''
        )
        self.play(load_training_code.TypeLetterbyLetter(lines=[0,1]))

        # SLIDE 16:  ===========================================================
        # LAD DATA LINE WRITTEN
        self.next_slide(
            notes=
            '''We then use the load_data function, defined in the
            helper_function file, to store the matrix representation of images
            in X_train and the corresponding labels in y_train.
            '''
        )
        self.play(load_training_code.TypeLetterbyLetter(lines=[3,4]))

        # SLIDE 17:  ===========================================================
        # SHOW_RANDOM_SAMPLES LINE WRITTEN
        self.next_slide(
            notes=
            '''Let's start to explore our dataset, using the function
            show_random_samples, which is defined in helper_functions.py. This
            function picks randomly four images from the sample we have just
            loaded, and shows them.
            '''
        )
        show_random_samples_lines = load_training_code[9:11]
        show_random_samples_lines.save_state()
        show_random_samples_lines.align_to(load_training_code[6:8], UP)

        self.play(load_training_code.TypeLetterbyLetter(lines=[9,10]))

        # SLIDE 18:  ===========================================================
        # BASE_DIR PARAM HIGHLIGHTED
        self.next_slide(
            notes=
            '''The parameter "base_dir" specifies the path from which picking
            the images, ...
            '''
        )
        base_dir_highlight = HighlightRectangle(show_random_samples_lines[1][23:49])
        self.play(Create(base_dir_highlight))

        # SLIDE 19:  ===========================================================
        # DIGIT=5 PARAM HIGHLIGHTED
        self.next_slide(
            notes=
            '''..., while setting "digit=5" tells the function to display four
            randomly selected the images which represent of the digit 5. We
            recall that this information is stored in the variable y_train.
            '''
        )
        digit_5_highlight = HighlightRectangle(show_random_samples_lines[1][50:57])
        self.play(ReplacementTransform(base_dir_highlight, digit_5_highlight))

        # SLIDE 20:  ===========================================================
        # SHOW_RANDOM_SAMPLES SHIFT DOWN, SEED LINE WRITTEN.
        self.next_slide(
            notes=
            '''It's convenient to first set the random seed to 42 for
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

        # SLIDE 21:  ===========================================================
        # INTO COLAB, RUN CELL, 4 IMAGES OF DIGIT '5' APPEAR
        self.next_slide(
            notes=
            '''
            '''
        )
        load_training_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.clear()
        self.play(load_training_code.IntoColab(cl_env))

        # Create the output plot
        fig, ax = plt.subplots(2,2,figsize=(8, 8), dpi=300)
        ax: list[Axes] = ax.flatten()
        for i in range(4):
            img = Image.open(rf'Assets\W5\mnist\ex5{i+1}.png')
            ax[i].imshow(img, cmap='grey')
            ax[i].set_axis_off()
            ax[i].set_title(f"Sample {i + 1}")
        plt.tight_layout()
        random_samples_plot = draw_plot(fig).scale_to_fit_height(0.45*FRAME_HEIGHT)
        cl_env.get_cell(-1).add_output(random_samples_plot)

        self.play(cl_env.Run(cell=-1, new_cursor=True))

        # SLIDE 22:  ===========================================================
        # NEW CELL, OUT OF COLAB
        # LOAD VALIDATION DATASET CODE WRITTEN
        self.next_slide(
            notes=
            '''Next, we prepare the validation dataset by calling the same
            loading function.
            '''
        )
        load_validation_code = ColabCode(
            r'''
            # Validation Data
            validation_dir_path = 'digits_dataset/validation_set'
            X_validation, y_validation = hf.load_data(validation_dir_path)
            '''
        )

        self.play(cl_env.cursor.MoveAndClick(cl_env.PLUS_CODE_))
        cl_env.add_cell()
        self.wait(0.3)
        self.play(cl_env.OutofColab(cell=-1))
        self.play(load_validation_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 23:  ===========================================================
        # VALIDATION PATH HIGHLIGHT
        self.next_slide(
            notes=
            '''This time we use a different path to the folder containing
            additional images that will not be used during training, defined by
            the string digits_dataset/validation_set.
            '''
        )
        self.play(load_validation_code.TypeLetterbyLetter(lines=[1]))

        # SLIDE 24:  ===========================================================
        # LOAD_DATA HIGHLIGHT
        # INTO COLAB, RUN CELL
        self.next_slide(
            notes=
            '''The function load_data stores the matrix representation of images
            in X_valid and the corresponding labels in y_valid.
            '''
        )
        self.play(load_validation_code.TypeLetterbyLetter(lines=[2]))
        
        cl_env.clear()
        load_validation_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        self.play(load_validation_code.IntoColab(cl_env))
        self.play(cl_env.Run(cell=0))

        # SLIDE 25:  ===========================================================
        # NEW CELL, OUT OF COLAB
        # DATSET DIMENSIONS COMMENT WRITTEN
        self.next_slide(
            notes=
            '''Let's see how many images are contained in the training and
            validation datasets.
            '''
        )
        dataset_sizes_code = ColabCode(
            r'''
            # Size of the dataset
            print('Amount of training data: ', len(X_train))
            print('Amount of validation data: ', len(X_validation))

            # Size of a single image
            img_height, img_width = X_train[0].shape
            print(f'Image shape: {img_height}x{img_width}')
            '''
        )

        self.play(cl_env.cursor.MoveAndClick(cl_env.PLUS_CODE_))
        cl_env.add_cell()
        self.wait(0.3)
        self.play(cl_env.OutofColab(cell=-1))
        self.play(dataset_sizes_code.TypeLetterbyLetter(lines=[0]))

        # SLIDE 26:  ===========================================================
        # PRINT(LEN) LINES WRITTEN
        self.next_slide(
            notes=
            '''Using the "len" function on X_train and X_validation, we can see
            how many samples each dataset contains.
            '''
        )
        self.play(dataset_sizes_code.TypeLetterbyLetter(lines=[1]))
        self.play(dataset_sizes_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 27:  ===========================================================
        # IMAGE SHAPE LINES WRITTEN
        self.next_slide(
            notes=
            '''To check the size of a single image, we use the "shape" property
            of the first image, which we select by accessing the first element
            of the X_train array: the one with index 0.
            '''
        )
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
        dataset_sizes_code.add_background_window(FullScreenBackground(COLAB_LIGHTGRAY))
        cl_env.remove_cell()
        self.play(dataset_sizes_code.IntoColab(cl_env))

        cl_env.get_cell(-1).add_output(
            'Amount of training data: 9990\nAmount of validation data: 1000\nImage shape: 28x28'
        )
        cl_env.get_cell(-1).output.save_state(); cl_env.get_cell(-1).outputWindow.save_state()
        self.play(cl_env.Run(cell=-1, new_cursor=False))
        self.play(cl_env.FocusOutput(cell=-1))

        # SLIDE 29:  ===========================================================
        # FULL CNN SCHEME REAPPEARS ON TOP (SAME AS THE BEGINNING)
        self.next_slide(
            notes=
            '''Now, let's address the second step: the definition of the
            architecture of our classifier. In particular, we will use Keras,
            based on the library TensorFlow, to define an architecture with the
            main components of a CNN.
            '''
        )
        dummy_white_rect = FullScreenBackground(WHITE).set_z_index(1)
        self.play(FadeIn(dummy_white_rect))

        # reset everything
        self.clear()
        cl_env.get_cell(-1).output.restore().set_z_index(-3); cl_env.get_cell(-1).outputWindow.restore().set_z_index(-3)
        ms.restore()

        self.play(FadeIn(ms))

        # SLIDE 30:  ===========================================================
        # IMPORT KERAS CODE WRITTEN
        self.next_slide(
            notes=
            '''We start by loading the libraries that provide the tools needed
            to define and train neural networks. These libraries are coherent
            with the mathematical descritption of the previous lesson and
            greatly simplifies the development process.
            '''
        )
        import_keras_code = ColabCode(
            r'''
            import tensorflow as tf
            from tensorflow import keras
            import tensorflow.keras.layers as layersModule

            # Set random seed for reproducibility
            tf.random.set_seed(1)
            '''
        )

        DSS = DynamicSplitScreen(COLAB_LIGHTGRAY, WHITE)
        DSS.add_empty_side_obj(FRAME_HEIGHT)
        DSS.hard_bring_in()
        self.add(DSS)
        DSS.add_side_obj(ms)
       
        self.play(DSS.bringOut())
        self.play(import_keras_code.TypeLetterbyLetter(lines=[0,1,2]))
        self.remove(ms)

        # SLIDE 31:  ===========================================================
        # TENSORFLOW RANDOM SEED WRITTEN
        # INTO COLAB, RUN CELL
        self.next_slide(
            notes=
            '''For reproducibility of the initialization and training process,
            we use the command tf.random.set_seed(1) to fix the random seed used
            by tensorflow.
            '''
        )
        self.play(import_keras_code.TypeLetterbyLetter(lines=[4,5]))
        self.wait(2)
        import_keras_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(import_keras_code.IntoColab(cl_env))
        self.play(cl_env.Run(cell=-1, new_cursor=False))

        # SLIDE 32:  ===========================================================
        # NEW CELL, OUT OF COLAB
        # WRITE # DEFINE ARCHITECTURE
        # FULL CNN SCHEME BROUGHT IN FROM TOP
        self.next_slide(
            notes=
            '''Now we can proceed to build our Convolutional Neural Network
            Architecture.
            '''
        )
        cnn_architecture_code = ColabCode(
            r'''
            # Define the model
            CNN = keras.Sequential([
                keras.Input(shape=(img_height, img_width, 1)),
                layersModule.Conv2D(32, (3, 3),
                    activation='relu', padding='same'),
                layersModule.MaxPooling2D(pool_size=(7, 7)),
                layersModule.Flatten(),
                layersModule.Dense(10, activation='softmax')
            ])
            '''
        )

        self.play(cl_env.cursor.MoveAndClick(cl_env.PLUS_CODE_))
        cl_env.add_cell()
        self.wait(0.3)
        self.play(cl_env.OutofColab(cell=-1))
        self.play(cnn_architecture_code.TypeLetterbyLetter(lines=[0]))

        # Setup splitscreen
        DSS.reset()
        self.add(DSS)
        DSS.add_main_obj(cnn_architecture_code[0], cnn_architecture_code[1:])
        ms.scale(0.75)
        DSS.add_side_obj(ms)

        self.play(DSS.bringIn())

        # SLIDE 33:  ===========================================================
        # KEARS.INPUT WRITTEN
        self.next_slide(
            notes=
            '''To do this, we use the keras.Sequential class, which is used to
            define a model as a plain sequence of layers.
            '''
        )
        layer_highlight_config = {'height':5.5, 'v_buff':0.25, 'h_buff':0.15}
        full_layers_highlight = ms.get_layer_highlight('all', **layer_highlight_config)
        self.play(cnn_architecture_code.TypeLetterbyLetter(lines=[1,8]))
        self.play(ShowPassingFlash(full_layers_highlight, rate_func=smoothstep, run_time=3, time_width=0.3))

        # SLIDE 34:  ===========================================================
        # KEARS.INPUT WRITTEN
        self.next_slide(
            notes=
            '''First, keras.Input is used to define the shape of the input
            image. Here the sizes are 28x28 pixels times 1 channel.
            '''
        )
        brace_label_config = {'font_size': 24}
        input_braces = get_labeled_braces(
            ms.input, LEFT, '28', DOWN, '28', label_config=brace_label_config
        )

        self.play(
            Succession(
                cnn_architecture_code.TypeLetterbyLetter(lines=[2]),
                FadeIn(input_braces)
            )
        )
        
        # SLIDE 35:  ===========================================================
        # HIGHLIGHT THE 4 LAYERS
        self.next_slide(
            notes=
            '''The following CNN layers are provided by the layersModule through
            specific classes.
            '''
        )
        dummy_architecture_code = ColabCode(
            r'''
            layersModule.Conv2D(...),
            layersModule.MaxPooling2D(...),
            layersModule.Flatten(...),
            layersModule.Dense(...)
            '''
        ).align_to(cnn_architecture_code[3], UL)

        self.play(FadeOut(input_braces))

        # SLIDE 36:  ===========================================================
        # LAYER CONV2D WRITTEN, CONV LAYER HIGHLIGHTED ABOVE
        self.next_slide(
            notes=
            '''"Conv2D" for the convolutional layer,
            '''
        )
        conv_highlight = ms.get_layer_highlight('conv', **layer_highlight_config)
        self.play(
            dummy_architecture_code.TypeLetterbyLetter(lines=[0]),
            Create(conv_highlight)
        )
        # SLIDE 37:  ===========================================================
        # LAYER MAXPOOL WRITTEN, POOLING LAYER HIGHLIGHTED ABOVE
        self.next_slide(
            notes=
            '''"MaxPooling2D" for the pooling layer,
            '''
        )
        pool_highlight = ms.get_layer_highlight('pool', **layer_highlight_config)
        self.play(
            dummy_architecture_code.TypeLetterbyLetter(lines=[1]),
            ReplacementTransform(conv_highlight, pool_highlight)
        )

        # SLIDE 38:  ===========================================================
        # LAYER FLATTEN WRITTEN, FLATTEN LAYER HIGHLIGHTED ABOVE
        self.next_slide(
            notes=
            '''"Flatten" for the flatten layer,
            '''
        )
        flat_highlight = ms.get_layer_highlight('flat', **layer_highlight_config)
        self.play(
            dummy_architecture_code.TypeLetterbyLetter(lines=[2]),
            ReplacementTransform(pool_highlight, flat_highlight)
        )

        # SLIDE 39:  ===========================================================
        # LAYER DENSE WRITTEN, DENSE LAYER HIGHLIGHTED ABOVE
        self.next_slide(
            notes=
            '''And finally, "Dense" for the dense layer that completes the
            classification task. It's important to note that the layers' inputs
            are design choices that depend on the problem. They are usually
            fine-tuned through trial and error. In this example, we present one
            specific configuration, but we encourage you to explore different
            ones and see how the results change.
            '''
        )
        dense_highlight = ms.get_layer_highlight('dense', **layer_highlight_config)
        self.play(
            dummy_architecture_code.TypeLetterbyLetter(lines=[3]),
            ReplacementTransform(flat_highlight, dense_highlight)
        )

        # SLIDE 40:  ===========================================================
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


        side=0.5
        filter_kernel = VGroup(
            Square(side, fill_opacity=0, **new_highlight_config).add(
                MathTex(f'l_{i}', color=GOLD).scale_to_fit_height(side*0.5)
            ) for i in range(9)
        ).arrange_in_grid(3,3, buff=0).align_to(DSS.secondaryRect.get_bottom()+0.5*UP, DOWN)

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
            Succession(Wait(1), cnn_architecture_code.TypeLetterbyLetter(lines=[3, 4]))
        )

        # SLIDE 41:  ===========================================================
        # CONV2D LAYER LINE WRITTEN
        # '32' HIGHLIGHTED, BRACE WITH 32 APPEARS ON CONV LAYER
        self.next_slide(
            notes=
            '''The first argument to the conv2D layer, 32, sets the number of
            filters.
            '''
        )
        filter_32_highlight = HighlightRectangle(cnn_architecture_code[3][20:22])
        # Create the many filters and convolution
        goc = GridOfConvolutions(
            input_image=new_input, filtered_images=ms.conv_layer, kernel=filter_kernel, n_filters=5,
            im_side_length=0.75, filter_colors=[GOLD, RED, GREEN, ORANGE, PURPLE]
        ).move_to(DSS.secondaryRect)

        self.play(
            AnimationGroup(
                FadeOut(ms.conv_highlight_1, ms.input_highlight, new_giz1, new_giz2),
                AnimationGroup(
                    *[new_conv[i].animate.become(goc.filtered[i]) for i in range(5)],
                    ReplacementTransform(new_input, goc.inputs[0]),
                    ReplacementTransform(filter_kernel, goc.kernels[0]),
                ),
                FadeIn(goc.inputs[1:], goc.stars, goc.kernels[1:], goc.equals, goc.vdots, goc.numbers),
                lag_ratio=0.5
            ),
        )
        self.play(Create(filter_32_highlight))

        # SLIDE 42:  ===========================================================
        # (3,3) HIGHLIGHTED, SMALL KERNEL APPEARS
        self.next_slide(
            notes=
            '''The second one, (3,3), iS the size of each filter, making each
            filter a small 3x3 matrix with learnable entries.
            '''
        )
        self.play(
            AnimationGroup(*[mob.animate.shift(UP*0.25) for mob in goc if mob not in goc.filtered], run_time=0.5),
            new_conv.animate(run_time=0.5).shift(UP*0.25)
        )

        filter_3x3_highlight = HighlightRectangle(cnn_architecture_code[3][23:28])
        kernel_braces = get_labeled_braces(goc.kernels[-1], DOWN, '3', LEFT, '3', brace_config={'buff':0.05}, label_config=brace_label_config, buff=0.1)
        self.play(
            ReplacementTransform(filter_32_highlight, filter_3x3_highlight),
            FadeIn(kernel_braces)
        )

        # SLIDE 43:  ===========================================================
        # 'RELU' HIGHLIGHTED; GRAPH OF RELU APPEARS ABOVE
        self.next_slide(
            notes=
            '''We then set the activation function to relu to keep the output
            positive.
            '''
        )
        relu_highlight = HighlightRectangle(cnn_architecture_code[4][:17])
        relu_plot = ReLUPlot().scale_to_fit_width(4)
        VGroup(phony_conv_rect, relu_plot).arrange(buff=2).move_to(DSS.secondaryRect)
        arrow_config = {'buff':0.5, 'color':DARK_BLUE, 'stroke_width':4, 'max_stroke_width_to_length_ratio':20}
        relu_arrow = Arrow(phony_conv_rect.get_right(), relu_plot.get_left(), **arrow_config)

        self.play(
            Succession(
                AnimationGroup(
                    FadeOut(  # FadeOut input images, *,=
                        kernel_braces, 
                        *[mob for mob in goc.submobjects if mob not in goc.filtered]
                    ),
                    new_conv.animate.restore().move_to(phony_conv_rect), # move conv layer into position,
                    lag_ratio=0.5
                ),
                AnimationGroup(
                    FadeIn(relu_plot.axes, relu_plot.labels, shift=RIGHT),
                    GrowArrow(relu_arrow),
                    ReplacementTransform(filter_3x3_highlight, relu_highlight),
                ),
                Create(relu_plot.relu, run_time=2)
            )
        )

        # SLIDE 44:  ===========================================================
        # 'PADDING=SAME' HIGHLIGHT
        self.next_slide(
            notes=
            '''The 'same' padding option ensures that the outputs have the same
            size of the input.
            '''
        )
        padding_highlight = HighlightRectangle(cnn_architecture_code[4][18:-2])
        new_input.restore().move_to(new_conv)
        self.play(
            AnimationGroup(
                FadeOut(relu_plot, relu_arrow),
                new_conv.animate.move_to(relu_plot),
                FadeIn(new_input),
                lag_ratio=0.5
            )
        )
        self.play(ReplacementTransform(relu_highlight, padding_highlight))

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
                FadeOut(new_input, input_size_braces, output_size_braces, conv_braces, conv_arrow,
                        padding_highlight),
                ReplacementTransform(new_conv, ms.conv_layer),
                FadeIn(*[mob for mob in pooling_example_g if mob not in [ms.conv_layer]]),
                lag_ratio=0.5
            )
        )

        self.play(cnn_architecture_code.TypeLetterbyLetter(lines=[5]))

        # SLIDE 47:  ===========================================================
        # GRID HIGHLIGHTING REDUCTION FACTOR IS CREATED
        # POOL_SIZE=(7,7) HIGHLIGHTED 
        self.next_slide(
            notes=
            '''In the argument we define the dimensions of the pool size: for
            instance 7 by 7.
            '''
        )
        pool_size_highlight = HighlightRectangle(cnn_architecture_code[5][26:-2])

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
                Create(pool_size_highlight),
                Create(pool_factor_grid_highlight),
                FadeIn(pool_factor_braces)
            )
        )

        # SLIDE 48:  ===========================================================
        # BRACES WITH 7 APPEAR
        self.next_slide(
            notes=
            '''This means that this layer retains the maximum value within each
            7x7 window.
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

        # SLIDE 49:  ===========================================================
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

        # SLIDE 50:  ===========================================================
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

        # SLIDE 51:  ===========================================================
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

        # SLIDE 52:  ===========================================================
        # SCHEME REPLACED WITH DENSE LAYER
        self.next_slide(
            notes=
            '''The Dense layer is the final layer in our CNN architecture.
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

        # SLIDE 53:  ===========================================================
        # '10' HIGHLIGHTED, OUTPUT DIGITS BRACE APPEARS
        self.next_slide(
            notes=
            '''The first argument, 10, defines the number of output neurons,
            corresponding to the 10 possible classes in our classification task,
            that is the digits from 0 to 9.
            '''
        )
        dense_10_highlight = HighlightRectangle(cnn_architecture_code[7][19:21])
        classific_10_brace = get_labeled_brace(ms.output_layer, RIGHT, '10', label_config=brace_label_config)

        self.play(
            Succession(
                Create(dense_10_highlight),
                FadeIn(classific_10_brace)
            )
        )

        # SLIDE 54:  ===========================================================
        # 'ACTIVATION=SOFTMAX' HIGHLIGHTED, SOFTMAX APPEARS IN SCHEME
        self.next_slide(
            notes=
            '''The second argument is the SoftMax activation function.
            '''
        )
        softmax_code_highlight = HighlightRectangle(cnn_architecture_code[7][22:-1])
        softmax_scheme_highlight = SurroundingRectangle(ms.softmax, color=GOLD, stroke_width=4, corner_radius=0.25, buff=0.1)
        self.play(
            Succession(
                FadeOut(classific_10_brace),
                ReplacementTransform(dense_10_highlight, softmax_code_highlight),
                ShowPassingFlash(softmax_scheme_highlight, run_time=2)
            )
        )

        # SLIDE 55:  ===========================================================
        # FULL SCHEME REAPPEARS ON TOP
        self.next_slide(
            notes=
            '''We have now finished defining the CNN architecture.
            '''
        )
        DSS.add_side_obj(dense_layer_vg)
        DSS.remove_main_obj()
        cnn_architecture_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(FadeOut(softmax_code_highlight))
        cl_env.remove_cell()
        self.play(
            DSS.bringOut(),
            cnn_architecture_code.IntoColab(cl_env)
        )
        self.play(cl_env.Run(cell=-1, new_cursor=False))

        # SLIDE 56:  ===========================================================
        # LEARNABLE COEFFICIENT SCHEME APPEAR
        self.next_slide(
            notes=
            '''Remember! Along our way we have many unknowns to determine.
            '''
        )
        LCNNscheme = LearnableCoefficientsScheme()
        dummy_white_rect = FullScreenBackground(WHITE).set_z_index(50)

        self.play(FadeIn(dummy_white_rect))
        self.clear()
        # remove 3 cell and leave 1 (CNN architecture) for when CNN summary appears
        cl_env.remove_cell_from_top(n=3)
        self.play(FadeIn(LCNNscheme))

        # SLIDE 57:  ===========================================================
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

        # SLIDE 58:  ===========================================================
        # LEARNABLE COEFFICIENT BROUGHT OUT
        # KEARS OPTMIZER LINE WRITTEN
        self.next_slide(
            notes=
            '''First, we choose an optimizer provided by Keras. In this case, we
            use SGD - the stochastic gradient descent - with an arbitrary
            learning rate of 0.01.
            '''
        )
        DSS.reset()
        DSS.add_empty_side_obj(FRAME_HEIGHT)
        DSS.hard_bring_in()
        self.add(DSS)
        DSS.add_side_obj(VGroup(LCNNscheme, training_title))
        self.play(DSS.bringOut())
        self.remove(training_title, LCNNscheme)

        compile_cnn_code = ColabCode(
            r'''
            # Select the optmizer
            optimizer = keras.optimizers.SGD(learning_rate=0.01)

            # Compile the CNN
            CNN.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

            # Print CNN summary
            CNN.summary()
            '''
        )

        self.play(compile_cnn_code.TypeLetterbyLetter(lines=[0,1]))

        # SLIDE 59:  ===========================================================
        # COMPILE CNN LINES WRITTEN
        self.next_slide(
            notes=
            '''Next, we call the method compile on our model. This step prepares
            our CNN for training by specifying 3 arguments: the optimizer, the
            loss function and a performance metric.
            '''
        )
        self.play(compile_cnn_code.TypeLetterbyLetter(lines=[3,4,5,6]))
        
        # SLIDE 60:  ===========================================================
        # CROSSENTROPY HIGHLIGHT
        self.next_slide(
            notes=
            '''Since this is a multi-class classification problem and our labels
            are integers from 0 to 9, we use sparse categorical crossentropy as
            loss function. This quantifies how far the CNN's predictions are
            from the true labels.
            '''
        )
        crossentropy_highlight = HighlightRectangle(compile_cnn_code[5][5:-1])
        self.play(Create(crossentropy_highlight))

        # SLIDE 61:  ===========================================================
        # 'ACCURACY' HIGHLIGHT
        self.next_slide(
            notes=
            '''The string accuracy tells the CNNto report the number of
            correctly classified images, both on the training and validation
            sets.
            '''
        )
        accuracy_highlight = HighlightRectangle(compile_cnn_code[6][9:-2])
        self.play(ReplacementTransform(crossentropy_highlight, accuracy_highlight))

        # SLIDE 62:  ===========================================================
        # CNN SUMMARY LINE WRITTEN
        # INTO COLAB, RUN CELL, OUTPUT APPEARS
        # FOCUS ON OUTPUT
        self.next_slide(
            notes=
            '''Finally, by calling the summary method, we can see that this
            architecture has a total of 5450 learnable parameters. We're ready
            to start the training process. We use an approach based on
            minibatches. At each optmization step, the CNN processes only a
            small subset of the dataset, called a batch. Once the optimizer has
            seen all the dataset, an epoch is completed.
            '''
        )
        self.play(
            Succession(
                FadeOut(accuracy_highlight),
                compile_cnn_code.TypeLetterbyLetter(lines=[8,9]),
                Wait(1)
            )
        )

        cl_env.clear_cursor()
        compile_cnn_code.add_background_window(DSS.mainRect.suspend_updating())
        self.play(compile_cnn_code.IntoColab(cl_env))
        cl_env.get_cell(-1).add_output(
            KerasCNNSummary().scale_to_fit_height(3)
        )
        self.play(cl_env.Run(cell=-1, new_cursor=True))
        self.play(cl_env.FocusOutput(cell=-1))

        # SLIDE 63:  ===========================================================
        # NUMBER OF EPOCHS LINE WRITTEN
        self.next_slide(
            notes=
            '''In our case, we set the number of epochs to 20
            '''
        )
        training_code = ColabCode(
            r'''
            # Set up training parameters
            num_epochs = 20
            batch_size = 16

            # Tracker of the loss and accuracy during training
            loss_results = hf.LossTracker()

            history = CNN.fit(
                X_train, y_train,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(X_validation, y_validation),
                callbacks=[loss_results],
                verbose=False,
            )
            '''
        )
        dummy_rect = FullScreenBackground(COLAB_LIGHTGRAY).set_z_index(0)

        self.play(
            Succession(
                FadeIn(dummy_rect),
                training_code.TypeLetterbyLetter(lines=[0,1])
            )
        )
        cl_env.clear()

        # SLIDE 64:  ===========================================================
        # NATCH SIZE LINE WRITTEN
        self.next_slide(
            notes=
            '''The batch size is instead 16, meaning the model updates its
            learnable parameters by processing 16 examples at the time.
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[2]))

        # SLIDE 65:  ===========================================================
        # TRACKER LINES WRITTEN
        self.next_slide(
            notes=
            '''We also use a tracking function from the helper function file to
            monitor how the CNN performance evolves during training.
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[4,5]))

        # SLIDE 66:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The fit() function starts the training process. The arguments
            are:
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[7]))

        # SLIDE 67:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The training data, consisting of images X_train and labels
            y_train
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[8]))

        # SLIDE 68:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The maximum number of epochs for the optimizer;
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[9]))

        # SLIDE 69:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The batch size;
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[10]))
        
        # SLIDE 70:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The validation data, given by images X_validation and labels
            y_validation
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[11]))

        # SLIDE 71:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''The tracker;
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[12]))

        # SLIDE 72:  ===========================================================
        # LINE WRITTEN
        self.next_slide(
            notes=
            '''And a flag that limits the number of information displayed.
            '''
        )
        self.play(training_code.TypeLetterbyLetter(lines=[13,14]))

        # SLIDE 73:  ===========================================================
        # INITIAL LOSS AND ACCURACY HIGHLIGHT
        self.next_slide(
            notes=
            '''Here's the output of this code. At each epoch, the code displays
            the loss value on both the training and validation datasets and the
            accuracy metric.
            '''
        )
        training_code.add_background_window(dummy_rect.set_z_index(-1))
        self.play(training_code.IntoColab(cl_env))
        training_output = ColabBlockOutputText(
            '''\
            Epoch 1:
            Training   - loss: 1.9768, accuracy: 0.4160
            Validation - loss: 1.4017, accuracy: 0.6310

            Epoch 2:
            Training   - loss: 1.1178, accuracy: 0.6776
            Validation - loss: 0.8826, accuracy: 0.7410
            ...

            Epoch 20:
            Training   - loss: 0.1767, accuracy: 0.9496
            Validation - loss: 0.1920, accuracy: 0.9450'''
        )
        cl_env.get_cell(-1).add_output(training_output)
        self.play(cl_env.Run(cell=-1, new_cursor=True))
        self.play(cl_env.FocusOutput(cell=-1))

        loss_disp_highlight = VGroup(
            HighlightRectangle(training_output[1][14:20], color=BLUE),
            HighlightRectangle(training_output[2][16:22], color=ORANGE),
        )
        accuracy_disp_highlight = VGroup(
            HighlightRectangle(training_output[1][-6:], color=BLUE),
            HighlightRectangle(training_output[2][-6:], color=ORANGE),
        )

        self.play(
            Succession(
                Wait(),
                Create(loss_disp_highlight),
                Wait(),
                Create(accuracy_disp_highlight)
            )
        )
        
        # SLIDE 74:  ===========================================================
        # FINAL ACCURACIES HIGHLIGHT
        self.next_slide(
            notes=
            '''By the final iteration, we can see that the model reaches very
            satisfactory performances: about 98% accuracy on the training
            dataset and a 96% accuracy on the validation dataset.
            '''
        )
        self.play(FadeOut(loss_disp_highlight, accuracy_disp_highlight))

        final_accuracy_disp_highlight = VGroup(
            HighlightRectangle(training_output[-2][-6:], color=BLUE),
            HighlightRectangle(training_output[-1][-6:], color=ORANGE),
        )

        self.play(
            Succession(
                Create(final_accuracy_disp_highlight[0]),
                Create(final_accuracy_disp_highlight[1])
            )
        )

        # SLIDE 75:  ===========================================================
        # PREDICTION CODE WRITTEN
        self.next_slide(
            notes=
            ''' Now we are at the fourth step! Let's look at some results using
            the function visualize_digit_prediction from the file
            helper_functions.py. This function visualizes one randomly selected
            image of the digit 3 from the validation set...
            '''
        )
        dummy_rect = FullScreenBackground(COLAB_LIGHTGRAY).set_z_index(50)

        first_prediction_code = ColabCode(
            r'''
            # Prediction sample
            hf.visualize_digit_prediction(
                CNN, directory=validation_dir_path, digit=3)
            '''
        )

        self.play(FadeIn(dummy_rect))
        cl_env.clear()
        self.remove(*final_accuracy_disp_highlight)
        dummy_rect.set_z_index(-1)
        self.play(first_prediction_code.TypeLetterbyLetter())

        # SLIDE 76:  ===========================================================
        # INTO COLAB, RUN CELL, OUTPUT APPEARS
        self.next_slide(
            notes=
            '''and displays the prediction probabilities for each digit made by
            the model.
            '''
        )

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
            ax1.bar(digits, predictions * 100)
            ax1.set_xlabel('Digit Prediction')
            ax1.set_ylabel('Prediction Probability (%)')
            ax1.set_xticks(digits)
            ax1.set_ylim(0, 100)
            # Add percentage labels on top of each bar
            for i, v in enumerate(predictions):
                ax1.text(i, v * 100 + 1, f'{v * 100:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            return draw_plot(fig).scale_to_fit_height(0.25*FRAME_HEIGHT)

        predict_3_prob = np.array([0,0, 0.008, 0.99,0,0,0,0, 0.003,0])
        predict_3_output = visualize_probabilites(r'Assets\W5\mnist\validation3.png', predict_3_prob)

        first_prediction_code.add_background_window(dummy_rect)
        self.play(first_prediction_code.IntoColab(cl_env))
        cl_env.get_cell(-1).add_output(predict_3_output)
        self.play(cl_env.Run(-1, new_cursor=True))

        # save state for restoring later
        for mob in cl_env.get_cell(-1).output:
            mob.save_state()
        cl_env.get_cell(-1).outputWindow.save_state()

        # SLIDE 77:  ===========================================================
        # FOCUS OUTPUT
        self.next_slide(
            notes=
            '''For this validation sample, the model correctly classified the
            image as a 3 with 99% confidence.
            '''
        )
        self.play(cl_env.FocusOutput(cell=-1, scale=0.8))

        # SLIDE 78:  ===========================================================
        # UNFOCUS OUTPUT
        # NEW CELL, SECOND PREDICTION CODE WRITTEN
        self.next_slide(
            notes=
            '''Let's try with another image with the digit 5.
            '''
        )
        self.play(
            *[mob.animate.restore() for mob in cl_env.get_cell(-1).output],
            cl_env.get_cell(-1).outputWindow.animate.restore()
        )
        
        self.play(cl_env.cursor.MoveAndClick(cl_env.PLUS_CODE_))
        second_prediction_code_cell = ColabCodeBlock(
            r'''
            hf.visualize_digit_prediction(
                CNN, directory=validation_dir_path, digit=5)
            '''
        )
        cl_env.add_cell(second_prediction_code_cell)
        self.remove(second_prediction_code_cell.code)
        self.wait(0.3)
        self.play(second_prediction_code_cell.TypeLetterbyLetter())

        # SLIDE 79:  ===========================================================
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
        predict_5_output = visualize_probabilites(r'Assets\W5\mnist\validation5.png', predict_5_prob)

        cl_env.get_cell(-1).add_output(predict_5_output)
        self.play(cl_env.Run(-1, new_cursor=False))
        self.play(cl_env.FocusOutput(cell=-1, scale=0.8))
        self.wait(0.05)
