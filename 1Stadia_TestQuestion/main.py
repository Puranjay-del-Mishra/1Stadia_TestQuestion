import cv2

image1_rgb = cv2.imread('coca-cola.png')
image2_rgb = cv2.imread('pepsi.png')
combined_image_rgb = cv2.imread('Pepsi-vs-Coca-Cola.jpeg')
image1_gray= cv2.cvtColor(image1_rgb, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2_rgb, cv2.COLOR_BGR2GRAY)
combined_image_gray = cv2.cvtColor(combined_image_rgb, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(image1_gray, None)

keypoints2, descriptors2 = sift.detectAndCompute(image2_gray, None)
keypoints_combined, descriptors_combined = sift.detectAndCompute(combined_image_gray, None)

bf = cv2.BFMatcher()

matches1 = bf.match(descriptors_combined, descriptors1)
matches2 = bf.match(descriptors_combined, descriptors2)

matches1 = sorted(matches1, key=lambda x: x.distance)
matches2 = sorted(matches2, key=lambda x: x.distance)

# first 20 matches for image1
matching_result1 = cv2.drawMatches(combined_image_rgb, keypoints_combined, image1_rgb, keypoints1, matches1[:20], None)

#first 20 matches for image2
matching_result2 = cv2.drawMatches(combined_image_rgb, keypoints_combined, image2_rgb, keypoints2, matches2[:20], None)
matching_result1_resized = cv2.resize(matching_result1, (matching_result2.shape[1], matching_result2.shape[0]))

final_result = cv2.hconcat([matching_result1_resized, matching_result2])

cv2.imshow('Final Matching Result', final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('final_result.jpg', final_result)