import cv2
import numpy as np


# def main():
#     img = cv2.imread("getperspective_transform_03.jpg")
#     # points for test.jpg
#     cnt = np.array([
#         [[63, 242]],
#         [[291, 110]],
#         [[361, 252]],
#         [[78, 386]]
#     ])
#     print("shape of cnt: {}".format(cnt.shape))
#     rect = cv2.minAreaRect(cnt)
#     print("rect: {}".format(rect))

#     # the order of the box points: bottom left, top left, top right,
#     # bottom right
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)

#     print("bounding box: {}".format(box))
#     cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

#     # get width and height of the detected rectangle
#     width = int(rect[1][0])
#     height = int(rect[1][1])

#     src_pts = box.astype("float32")
#     # coordinate of the points in box points after the rectangle has been
#     # straightened
#     dst_pts = np.array([
#                         [0, 0],
#                         [width-1, 0],
#                         [width-1, height-1], [0, height-1]], dtype="float32")

#     # the perspective transformation matrix
#     M = cv2.getPerspectiveTransform(src_pts, dst_pts)

#     # directly warp the rotated rectangle to get the straightened rectangle
#     warped = cv2.warpPerspective(img, M, (width, height))

#     cv2.imwrite("crop_img.jpg", warped)
#     cv2.waitKey(0)


# if __name__ == "__main__":
#     main()

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


image = cv2.imread("enhanced_images/inverse_law/test_dataset.mp4/frame_0.jpg")
pts = np.array([(140, 202), (202, 202), (202, 230), (140, 230)], dtype="float32")
# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)
# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
