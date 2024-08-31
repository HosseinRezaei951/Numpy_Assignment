# Vector Analysis and Image Processing with Numpy

## Project Overview

The **Vector Analysis and Image Processing with Numpy** project is designed to deepen your understanding and proficiency with the Numpy library in Python. The project includes several tasks that involve vector manipulation, trigonometric functions, image processing, and matrix operations. The tasks challenge you to utilize Numpy for efficient and concise coding, leveraging its capabilities for handling arrays, performing mathematical operations, and processing images.

## Project Structure

- **main.py**: The main Python script containing the code for all tasks.
- **Data/**: Directory containing sub-directories for storing output images from the image processing tasks.
  - **Part3/**: Contains images generated from the third task (basic image processing).
  - **Part4/**: Contains images generated from the fourth task (chessboard similarity).
  - **Part5/**: Contains images generated from the fifth task (advanced image processing).

## Tasks

### 1. Vector Distance (Euclidean Distance)

This task calculates the Euclidean distance between multiple user-defined vectors and a randomly generated vector. The script determines the minimum distance between the user vectors and the random vector.

- **Implementation**:
  - The user inputs `n` vectors, each containing `m` elements.
  - The Euclidean distance between each vector and a random vector (of length `m`) is calculated.
  - The minimum distance is identified and displayed.

- **Example Usage**:
  ```python
  # The program will prompt the user to enter vectors.
  # After processing, it will output the minimum Euclidean distance.
  ```

### 2. Trigonometric Functions and Cosine Similarity

This task involves plotting trigonometric functions, calculating cosine similarity between pairs of functions, and finding the difference between corresponding points on the functions.

#### 2-1: Plotting Trigonometric Functions

The script plots the following trigonometric functions:
- `sin(x)`
- `cos(x)`
- `tan(x)`
- `cotan(x)`

- **Implementation**:
  - `x` values range from `0` to `2π` for `sin(x)` and `cos(x)`.
  - `x` values range from `-π/2` to `π/2` for `tan(x)` and `cotan(x)`.
  - The plots are displayed using Matplotlib.

- **Example Usage**:
  ```python
  # The script generates and displays plots for the four trigonometric functions.
  ```

#### 2-2: Cosine Similarity Calculation

The script calculates the cosine similarity between:
- `sin(x)` and `cos(x)`
- `tan(x)` and `cotan(x)`

- **Implementation**:
  - Cosine similarity is calculated using the formula:
    \[
    \text{cosine\_similarity} = \frac{\text{A} \cdot \text{B}}{\|\text{A}\| \|\text{B}\|}
    \]
  - The similarity values are printed.

- **Example Usage**:
  ```python
  # The program outputs the cosine similarity between sin(x) and cos(x), as well as tan(x) and cotan(x).
  ```

#### 2-3: Difference Between Trigonometric Function Points

The script calculates the difference between corresponding points on the following function pairs:
- `sin(x)` and `cos(x)`
- `tan(x)` and `cotan(x)`

- **Implementation**:
  - The difference between the points is calculated and displayed.

- **Example Usage**:
  ```python
  # The program outputs the differences between points on the sin and cos plots, and tan and cotan plots.
  ```

### 3. Basic Image Processing

This task involves performing several image processing operations on a user-provided image.

- **Operations**:
  - Convert the image to `float64`.
  - Change the image color to blue.
  - Split the image into four equal parts.
  - Convert one of the four parts to a negative image.
  - Crop a portion of the negative image and save it.
  - Print the dimensions and format of the cropped image.

- **Example Output**:
  - `blueImage.jpg`: Image with all pixels turned blue.
  - `image1.jpg`, `image2.jpg`, `image3.jpg`, `image4.jpg`: The four parts of the original image.
  - `negetiveImage1.jpg`: Negative of one part of the image.
  - `trimNegetiveImage1.jpg`: Cropped portion of the negative image.

### 4. Chess Board Similarity

This task generates two chessboards with different dimensions and calculates their similarity.

- **Implementation**:
  - Two matrices representing chessboards are generated (8x8 and 8x18).
  - A threshold matrix (8x8) is used to compare the sampled sections of the larger chessboard.
  - The similarity results are saved as images.

- **Example Output**:
  - Images representing the original and sampled chessboards, along with their similarity matrices, are saved in the `Part4/` directory.

### 5. Advanced Image Processing

This task performs advanced image processing operations on a user-provided image.

- **Operations**:
  - Reduce the color depth of the image.
  - Rotate the image by 45 degrees.
  - Flip the image horizontally.
  - Crop a portion of the flipped image and save it.
  - Combine the cropped portion with the original image.

- **Example Output**:
  - `lowColorImage.jpg`: Image with reduced color depth.
  - `rotatedImage.jpg`: Image rotated by 45 degrees.
  - `flipedImage.jpg`: Horizontally flipped image.
  - `cropedImage.jpg`: Cropped portion of the flipped image.
  - `combinedImage.jpg`: Final image combining the cropped portion with the original image.

## How to Run the Scripts

1. **Ensure Required Libraries Are Installed**:
   Make sure you have the following Python libraries installed:
   - `numpy`
   - `Pillow` (PIL)
   - `matplotlib`
   - `opencv-python`

   You can install them using pip:
   ```bash
   pip install numpy pillow matplotlib opencv-python
   ```

2. **Run the Script**:
   Execute the `main.py` script to access the various tasks. Follow the on-screen instructions to navigate through the different tasks and see the results.

   ```bash
   python main.py
   ```

## Conclusion

This assignment illustrates the power and versatility of Numpy in handling array-based operations, performing mathematical calculations, and processing images. Through the tasks, you will gain hands-on experience in using Numpy for a wide range of applications, from vector manipulation to advanced image processing.
