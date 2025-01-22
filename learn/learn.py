# import numpy as np
# import cv2

# image = cv2.imread("path_to_image.jpg")
# print("Original shape:", image.shape)  # 輸出可能是 (720, 1280, 3)

# height, width = image.shape[:2]
# print("Height:", height)  # 720
# print("Width:", width)  # 1280

# blank_image = np.zeros((height, width, 3), dtype=np.uint8)

# array = np.array([100, 200, 255], dtype=np.uint8)

# print("Original array:", array)

# result = array + 60
# print("Array after adding 60:", result)

# array1 = np.array([True, False, True, False])
# array2 = np.array([True, True, False, False])

# result = np.logical_or(array1, array2)
# print(result)  # [ True  True  True False]

# array3d = np.array(
#     [
#         [[True, False], [False, True], [True, False]],
#         [[False, False], [True, False], [True, True]],
#         [[True, False], [False, True], [False, False]],
#     ]
# )

# result_reduce = np.logical_or.reduce(array3d, axis=0)
# print(result_reduce)
# # [[ True  False]
# #  [ True  True]
# #  [ True  True]]

# uint8_result = result_reduce.astype(np.uint8) * 255
# print(uint8_result)
# # [[255 255]
# #  [255 255]
# #  [255 255]]
