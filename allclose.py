import numpy as np

box = [43.501, 359.22, 141.1, 660.56]
pos = (42.10124206542969, 359.086669921875, 142.75177001953125, 662.58349609375)
tol = 5

result = np.allclose(box, pos, atol=tol)
print(result)  # 這將打印 True 或 False，取決於 box 和 pos 是否在誤差範圍內相等
