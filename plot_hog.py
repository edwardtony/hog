# Reference
# https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html
# https://www.tensorflow.org/tutorials/layers
import numpy as np
import cv2
from math import sqrt
from skimage.feature import hog
from skimage import data, exposure


class HOG:

    def __init__(self, img_path, pdf_path, bins):
        self.img_path = img_path
        self.pdf_path = pdf_path
        self.bins = bins
        self.squares = []

    def run(self):
        # Load and create the histogram of the image to find
        self.hog_image()

        # Load pdf and get shape
        pdf = cv2.imread(self.pdf_path, cv2.IMREAD_GRAYSCALE)
        self.pdf_h, self.pdf_w = pdf.shape


        # The sliding window starts being half of the original size and will finish being 3 times the original size, in order to find the image in different dimensions
        for z in range(round(self.image_w / 2), self.image_w*3, min(round(self.image_w*0.5), 30)):

            # If we have already found the image, stop searching
            if len(self.squares): return


            # Usefull variables to iterate
            x, y, i = (0, 0, 0)

            # Define the horizontal stride: https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
            step = z // 8

            # Slide over the whole image
            while x + z < self.pdf_w or y + z*self.proportion < self.pdf_h:
                copy = np.array(pdf)

                # Segment Sliding Window
                window = pdf[y:y + z*self.proportion,x: x + z]

                # Skip white zones
                if window.mean() < 230:

                    window = cv2.resize(window, (self.image_w, self.image_h))

                    # Get Hog form Image
                    # fd1: Hog features
                    # hot_image1: Hog Image
                    fd1, hog_image1 = hog(window, orientations=self.bins, pixels_per_cell=(self.image_h/20, self.image_w/20), cells_per_block=(2, 2), visualize=True, feature_vector=False)
                    (a,b,c,d,e) = fd1.shape

                    # Reduce all values to 8-length array (Histogram)
                    histogram = np.sum(fd1.reshape(a*b*c*d,self.bins), axis=0)
                    print("Distance: {}".format(((self.histo1 - histogram) ** 2).sum()), self.image_h, self.image_w)

                    # If the distance between the histogram values is not too different, consider it as a match
                    if ((self.histo1 - histogram) ** 2).sum() < round(sqrt(self.image_h*self.image_w)*0.6) :
                        self.squares.append((x,y,x+z,y+z*self.proportion))

                    # Print all matches already found
                    for (x1, y1, w1, h1) in self.squares:
                        cv2.rectangle(copy,(x1,y1),(w1,h1),(0,0,0),1)

                    # Print the Sliding Window
                    cv2.rectangle(copy,(x,y),(x+z,y+z*self.proportion),(0,0,0),1)

                    # Show Image
                    cv2.imshow("Original", copy)

                # Calculate new x and y values
                x = (i + z) % self.pdf_w
                y = ((i + z) // self.pdf_w)*step*3
                i += step

                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break


    # Show image with the matches found
    def showImage(self):
        final = cv2.imread(self.pdf_path, cv2.IMREAD_GRAYSCALE)
        (x1, y1, w1, h1) = self.maxBox()
        cv2.rectangle(final,(x1,y1),(w1,h1),(0,0,0),1)
        while True:
            cv2.imshow("FINAL", final)
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break

    # Calculate a box which contains all matches found
    def maxBox(self):
        np_squares = np.array(self.squares)
        x1 = np_squares[:,:1].min()
        y1 = np_squares[:,1:2].min()
        w1 = np_squares[:,2:3].max()
        h1 = np_squares[:,3:4].max()
        return (x1, y1, w1, h1)


    def hog_image(self):
        gray = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        self.image_h, self.image_w = gray.shape
        self.proportion = round(self.image_h / self.image_w)

        window = cv2.resize(gray, (self.image_w,self.image_h))
        fd1, hog_image1 = hog(window, orientations=self.bins, pixels_per_cell=(self.image_h/20, self.image_w/20), cells_per_block=(2, 2), visualize=True, feature_vector=False)
        (a,b,c,d,e) = fd1.shape

        histogram = np.sum(fd1.reshape(a*b*c*d,self.bins), axis=0)
        self.histo1 = histogram


def main():
    hog = HOG("images/img1.jpg", "images/background1.png", 8)
    hog.run()
    hog.showImage()


if __name__ == '__main__':
    main()
