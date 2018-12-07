#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include <stdio.h>

layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer l = {0};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.output = calloc(h * w * c * batch, sizeof(float));
    l.delta  = calloc(h * w * c * batch, sizeof(float));
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.scales = calloc(c, sizeof(float));
    l.scale_updates = calloc(c, sizeof(float));
    l.biases = calloc(c, sizeof(float));
    l.bias_updates = calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        l.scales[i] = 1;
    }

    l.mean = calloc(c, sizeof(float));
    l.variance = calloc(c, sizeof(float));

    l.rolling_mean = calloc(c, sizeof(float));
    l.rolling_variance = calloc(c, sizeof(float));

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;
#ifdef GPU
    l.forward_gpu = forward_batchnorm_layer_gpu;
    l.backward_gpu = backward_batchnorm_layer_gpu;

    l.output_gpu =  cuda_make_array(l.output, h * w * c * batch);
    l.delta_gpu =   cuda_make_array(l.delta, h * w * c * batch);

    l.biases_gpu = cuda_make_array(l.biases, c);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

    l.scales_gpu = cuda_make_array(l.scales, c);
    l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

    l.mean_gpu = cuda_make_array(l.mean, c);
    l.variance_gpu = cuda_make_array(l.variance, c);

    l.rolling_mean_gpu = cuda_make_array(l.mean, c);
    l.rolling_variance_gpu = cuda_make_array(l.variance, c);

    l.mean_delta_gpu = cuda_make_array(l.mean, c);
    l.variance_delta_gpu = cuda_make_array(l.variance, c);

    l.x_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    #ifdef CUDNN
    cudnnCreateTensorDescriptor(&l.normTensorDesc);
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
    cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 

    #endif
#endif
    return l;
}

/*
输入：卷积操作的输出(不经过scale与shift)    float *x_norm
      损失函数关于 z 的导数               float *delta
      batch 的大小                       int batch
      卷积核的数目                       n
      每一个卷积核输出的大小              size
      存放卷积层的输出关于 gamma 的偏导数  float *scale_updates
功能：对卷积层的输出关于batchnormal中的gamma求导数，并放到 float *scale_updates
输出：卷积层的输出关于batchnormal中的gamma的导数 float *scale_updates
返回：无
*/
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];   // 点乘并相加
            }
        }
        scale_updates[f] += sum;   // 每一个卷积核对应一个 gamma 系数
    }
}

/*
输入：当前的 delta                       float *delta
      batch 的大小                       int batch
      卷积核或神经元的数量                 filter
      卷积核或神经元的输出大小             spatial          对于CNN 等于输出的长和宽乘积，对于DNN 等于1，保证代码具有统一性
      损失函数关于均值的偏导              float *mean_delta
功能：求损失函数关于均值的偏导，并放到 float *mean_delta中
输出：损失函数关于均值的偏导  float *mean_delta
返回：无   
*/
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];   // 一个卷积核或者神经元对应一个均值，所以只和 i 有关
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));  // 因为这个式子里面只有 i，放到外面可以减少计算量，这里应该省略了求偏导公式的第二项
    }
}

/*
输入：batchnorm 的输入                    float *x
      当前的 delta                       float *delta
      均值                               float *mean
      方差                               float *variance
      batch 的大小                       int batch
      卷积核或神经元的数量                 filter
      卷积核或神经元的输出大小             spatial          对于CNN 等于输出的长和宽乘积，对于DNN 等于1，保证代码具有统一性
      损失函数关于方差的偏导  float *variance_delta
功能：求损失函数关于方差的偏导，并放到 float *variance_delta中
输出：损失函数关于方差的偏导  float *variance_delta
返回：无
*/
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);   // 一个卷积核或者神经元对应一个方差，所以只和 i 有关
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));  // 因为这个式子里面只有 i，放到外面可以减少计算量
    }
}

/*
输入：batchnorm 的输入                    float *x
      均值                               float *mean
      方差                               float *variance
      损失函数关于均值的偏导               float *mean_delta
      损失函数关于方差的偏导               float *variance_delta
      batch 的大小                       int batch
      卷积核或神经元的数量                 filter
      卷积核或神经元的输出大小             spatial          对于CNN 等于输出的长和宽乘积，对于DNN 等于1，保证代码具有统一性
      当前的 delta                       float *delta
功能：求损失函数关于x的偏导，并放到 float *delta中
输出：损失函数关于x的偏导  float *delta中
返回：无
*/
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void resize_batchnorm_layer(layer *layer, int w, int h)
{
    fprintf(stderr, "Not implemented\n");
}

/*
输入：要batch的层 l，网络参数 net
功能：对 l 层的输出所有元素进行batchnormal操作
输出：无
*/
void forward_batchnorm_layer(layer l, network net)
{
    // l.type 等于 BATCHNORM 的时候，为 BATCHNORM 层。将 net.input 的值赋值为 l.output，也就是说该层的输入就是batchnorm的输入，使用l.output 是为了与下面非 BATCHNORM 层的时候统一变量名
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);  
    copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);  // batchnorm的输入为当前层输入与神经元作用后得到的l.output，所以这里将 l.output 的值赋值给 l.x
    if(net.train){
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        // l.out_c 为输出的通道数，等于卷积核的数目总数
        scal_cpu(l.out_c, .99, l.rolling_mean, 1);    // 滚动平均值的方法，并简化 1/m = 0.01
        axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);  
        scal_cpu(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);   // 对l.output中每一个元素进行batchnormal操作并返回
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);  // 将l.output的值赋值给l.x_norm，在反向传播中需要使用到该函数
    } else {   // 网络处于非训练状态时，使用滚动平均的方法估计出来的均值和方差
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    // 实现论文 http://arxiv.org/pdf/1502.03167.pdf 中的gamma和beta参数的前向计算，即scale and shift
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer(layer l, network net)
{
    if(!net.train){   // 在推理阶段，使用从所有训练实例中获得的统计量代替全局统计量
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h); // 求损失函数关于 beta 的偏导
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates); // 求损失函数关于 gamma 的偏导

    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w); // 计算损失函数关于\hat{x}的偏导，并放到 l.delta 中

    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta); //计算损失函数关于均值的偏导
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta); // 计算损失函数关于方差的偏导
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta); // 计算损失函数关于 x 的偏导
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);  // 如果是 BATCHNORM 层的话，将 l.delta 赋值给 net.delta，完成损失函数关于 a^{l-1}的偏导，损失函数关于 z^{l-1}的偏导在上一层的backword中完成
}

#ifdef GPU

void pull_batchnorm_layer(layer l)
{
    cuda_pull_array(l.scales_gpu, l.scales, l.c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
    cuda_push_array(l.scales_gpu, l.scales, l.c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}

void forward_batchnorm_layer_gpu(layer l, network net)
{
    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
    if (net.train) {
#ifdef CUDNN
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &one,
                &zero,
                l.dstTensorDesc,
                l.x_gpu,
                l.dstTensorDesc,
                l.output_gpu,
                l.normTensorDesc,
                l.scales_gpu,
                l.biases_gpu,
                .01,
                l.rolling_mean_gpu,
                l.rolling_variance_gpu,
                .00001,
                l.mean_gpu,
                l.variance_gpu);
#else
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

        scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
        axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
        axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);

        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
#endif
    } else {
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    }

}

void backward_batchnorm_layer_gpu(layer l, network net)
{
    if(!net.train){
        l.mean_gpu = l.rolling_mean_gpu;
        l.variance_gpu = l.rolling_variance_gpu;
    }
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    cudnnBatchNormalizationBackward(cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            &one,
            &one,
            l.dstTensorDesc,
            l.x_gpu,
            l.dstTensorDesc,
            l.delta_gpu,
            l.dstTensorDesc,
            l.x_norm_gpu,
            l.normTensorDesc,
            l.scales_gpu,
            l.scale_updates_gpu,
            l.bias_updates_gpu,
            .00001,
            l.mean_gpu,
            l.variance_gpu);
    copy_gpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
#else
    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);

    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);

    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
#endif
    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
