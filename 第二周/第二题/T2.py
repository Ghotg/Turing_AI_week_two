import numpy as np
import cv2 as cv

def conv():
    number = int(input("请输入特征图矩阵大小的的平方根："))
    lis = [list(map(int, input(f"请输入{number}个数字，用空格隔开：").split())) for _ in range(number)]
    number_2 = int(input("请输入卷积核矩阵大小的的平方根："))
    lis_2 = [list(map(int, input(f"请输入{number_2}个数字，用空格隔开：").split())) for _ in range(number_2)]
    lit = np.array(lis,dtype=np.uint8)
    kernel = np.array(lis_2)
    result = cv.filter2D(lit,-1,kernel)
    pad = kernel.shape[0] // 2
    dst_valid = result[pad:-pad, pad:-pad]
    print(dst_valid)
def main():
    conv()

if __name__ == '__main__':
    main()

