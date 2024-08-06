x, y = 592, 818  # 给定坐标
grid_size = 32
image_width = 2016  # 示例图像宽度
image_height = 1344  # 示例图像高度

row_index = y // grid_size
col_index = x // grid_size
print(row_index-1, col_index-2)
print(row_index+1, col_index+1)


