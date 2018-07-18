import argparse
import cv2
 
ref_point = []
cropping = False
 
def shape_selection(event, x, y, flags, param):
	global ref_point, cropping
 
	if event == cv2.EVENT_LBUTTONDOWN:
		ref_point = [(x, y)]
		cropping = True
 
	elif event == cv2.EVENT_LBUTTONUP:
 
		ref_point.append((x, y))
		cropping = False
 
		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)
 
 
while True:
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("r"):
		image = clone.copy()
 
 
	elif key == ord("c"):
		break
 
if len(ref_point) == 2:
	crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
	cv2.imshow("crop_img", crop_img)
	cv2.waitKey(0)
 
cv2.destroyAllWindows()