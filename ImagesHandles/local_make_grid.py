import cv2
import numpy as np

# 加载图像
image_path = './local_img_0831/0831.png'
out_path = './local_img_0831/0831_grid.png'
image = cv2.imread(image_path)

# 获取图像的高度和宽度
height, width, _ = image.shape

# 设置网格的大小
grid_size = 32

# 创建一个副本以绘制网格
image_with_grid = image.copy()

# 绘制垂直线
for x in range(0, width, grid_size):
    cv2.line(image_with_grid, (x, 0), (x, height), (0, 0, 255), 1)

# 绘制水平线
for y in range(0, height, grid_size):
    cv2.line(image_with_grid, (0, y), (width, y), (0, 0, 255), 1)

# 显示带网格的图像
cv2.imshow('Image with Grid', image_with_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存带网格的图像
cv2.imwrite(out_path, image_with_grid)