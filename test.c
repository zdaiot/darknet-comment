#include <stdio.h>
#include <stdlib.h>

int conv_out_height(int h, int pad, int size, int stride) {
        return (h + 2*pad - size) / stride + 1;
}

int conv_out_width(int w, int pad, int size, int stride) {
        return (w + 2*pad - size) / stride + 1;
}

int im2col_get_pixel(int *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(int* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, int* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) { //卷积核参数个数
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

int main(int argc, char* argv[]) {
        int *data_im=NULL;
        int *data_col=NULL;
        int channels=3,height=4,width=4;
        int ksize=2,stride=2,pad=0;
        int out_w,out_h;
        int workspace_size;

        int inputs = height * width * channels;
        data_im = (int*)malloc(inputs * sizeof(int));
        if (!data_im) {
                printf("malloc error\n");
                exit(EXIT_FAILURE);
        }

        out_w = conv_out_width(width, pad, ksize, stride);
        out_h = conv_out_width(height, pad, ksize, stride);
        workspace_size = out_h * out_w * ksize * ksize * channels;

        data_col = (int*)malloc(workspace_size * sizeof(int));
        if (!data_col) {
                printf("malloc error\n");
                exit(EXIT_FAILURE);
        }

        //init image
        for (int i=0; i<inputs; i++) data_im[i] = i;

        im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);

        printf("data_im:\n");
        for (int i=0; i<inputs; i++) {
                printf("%-3d", data_im[i]);
                //if( (i+1) % 4 == 0) printf("\n");
        }

        printf("\ndata_col:\n");
        for (int i=0; i<workspace_size; i++) {
                printf("%-3d", data_col[i]);
                //if( (i+1) % 4 == 0) printf("\n");
        }
        printf("\n");

        free(data_im);
        free(data_col);

        exit(EXIT_SUCCESS);
}
