#include "im2col.h"
#include <stdio.h>
/*
输入：im      输入，所有数据存成一个一维数组，例如对于3通道的二维图像而言，每一通道按行存储（每一通道所有行并成一行），三通道依次再并成一行
     height  每一通道的高度（即输入图像的真正的高度，补0之前）
     width   每一通道的宽度（即输入图像的宽度，补0之前）
     channels 输入im的通道数，比如彩色图为3通道，之后每一卷积层的输入的通道数等于上一卷积层卷积核的个数
     row     要提取的元素所在的行（二维图像补0之后的行数）
     col     要提取的元素所在的列（二维图像补0之后的列数）
     channel 要提取的元素所在的通道
     pad     图像左右上下各补0的长度（四边补0的长度一样）
功能：从输入的多通道数组im（存储图像数据）中获取指定行、列、、通道数处的元素值
输出：float类型数据，为im中channel通道，row-pad行，col-pad列处的元素值
*/
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    // 减去补0长度，获取元素真实的行列数，这里之所以减去一个pad，而不是两个，推测是两边补零和上下补零
    // 补零后的位置与原位置坐标变换公式
    row -= pad;
    col -= pad;

    // 如果行列数小于0,则返回0（刚好是补0的效果）
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;

    /*im存储多通道二维图像的数据的格式为：先将各通道所有行并成一行，再将多通道依次并成一行，
    因此width*height*channel首先移位到所在通道的起点位置，加上width*row移位到
    所在指定通道所在行，再加上col移位到所在列*/
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
/*
输入：输入数据 data_im，输入数据的通道数channels、高度height、宽度width，卷积核大小ksize，步长stride，补零个数pad，输出data_col
功能：依次取data_im中与卷积核中每个元素点乘的所有位置排列成一维数组，大小为输出宽×输出高×size×size×输入channels。
     
     具体排列方式为：首先将输入数据中要和卷积核中第一个参数进行点乘的数据提取出来组成一行；然后将输入数据中要和卷积核中第二个参数进行点乘的数据提取出来组成一行；
     依次提取，直到将输入数据中要和卷积核中第size×size×输入channels个参数进行点乘的数据提取出来组成一行。每次能提取出的数据大小为输出宽×输出高。

     例如：数据有两个维度，第一维为，   第二维为
                        0 1 2      9  10 11
                        3 4 5      12 13 14
                        6 7 8      15 16 17
          卷积核大小为2×2，步长为1，通道数为2，padding为0，则输出大小为2×2
          得到的数据为，数据为为2×2×2×2×2=32，可以看成是size×size×输入channels个输出宽×输出高的矩阵：
          0  1  3  4    1  2  4  5    3  4  6  7     4  5  7  8
          9  10 12 13   10 11 13 14   12 13 15 16    13 14 16 17

          或者另外一种理解方式，因此输出的float *data_col均是一维向量，因此上面也可以表示成下式，原矩阵按照卷积划窗的大小与顺序 依次表示为下面中的每一列
          0  1  3  4
          1  2  4  5
          3  4  6  7
          4  5  7  8

          9  10 12 13
          10 11 13 14
          12 13 15 16
          13 14 16 17
输出：float *data_col
返回：无
*/
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1; // 上下补零，height_col输出的高度
    int width_col = (width + 2*pad - ksize) / stride + 1;  // 两边补零，width_col输出的宽度

    // 每个卷积核的参数总数
    int channels_col = channels * ksize * ksize;
    // 每次c增加一，意味着列偏移增加一或者行偏移增加一，或者通道偏移加一
    for (c = 0; c < channels_col; ++c) {
        /* 计算列偏移(取值范围为0～ksize-1)，卷积核是一个二维矩阵，但按行存储在一维数组中，利用求余运算获取对应在卷积核中的列数，比如对于
        3*3的卷积核（3通道），当c=0时，显然在第一列，当c=5时，显然在第2列，当c=9时，在第二通道上的卷积核的第一列，
        当c=26时，在第三列（第三通道上）*/
        int w_offset = c % ksize;
        /* 计算行偏移(取值范围为0～ksize-1)，卷积核是一个多维的矩阵，先将各通道所有行并成一行，再将多通道依次并成一行，存储在一维数组中的，
           每当c为ksize的倍数，就意味着卷积核换了一行 */
        int h_offset = (c / ksize) % ksize;
        /*计算通道偏移，每当c为ksize x ksize大小的时候，就意味着换了一行*/
        int c_im = c / ksize / ksize;
        // 这里height_col和width_col可以看做是在图像的行维度和列维度上可以进行多少次卷积
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;   // 计算行的位置
                int im_col = w_offset + w * stride;   // 计算列的位置
                // col_index为重排后图像中的像素索引，等于c * height_col * width_col + h * width_col +w（还是按行存储，所有通道再并成一行），
                // 对应第c通道，h行，w列的元素(理解的时候，可以将下式展开)
                int col_index = (c * height_col + h) * width_col + w;

                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

