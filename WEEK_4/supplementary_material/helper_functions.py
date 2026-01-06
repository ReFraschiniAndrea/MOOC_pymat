import matplotlib.pyplot as plt

def compare_images(
    np_matrix1, np_matrix2, title1="Image 1", title2="Image 2"
):
    # Create a figure with two subplots
    ratio = np_matrix2.shape[1] / np_matrix1.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, ratio]})

    # Display the first image
    axes[0].imshow(np_matrix1, cmap="gray" if np_matrix1.ndim == 2 else None, vmin=0, vmax=255)
    axes[0].set_title(title1)
    axes[0].axis("off")

    # Display the second image
    axes[1].imshow(np_matrix2, cmap="gray" if np_matrix2.ndim == 2 else None, vmin=0, vmax=255)
    axes[1].set_title(title2)
    axes[1].axis("off")

    # Show the figure
    plt.show()
