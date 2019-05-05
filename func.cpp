#include <math.h>
#define POSITIVE(v) ((v)>=0 ? (v) : 0)

cv_errcode RedCorrection(float *data, float* correct_matrix, const int height, const int width, const int matrix_height, const int matrix_width) {
    // matrix: matrix_height * matrix_width * 4
    if ((matrix_height - matrix_width) * (height - width) < 0) {
        fprintf(stderr, "invalid h and w order\n");
        return -1;
    }
    float *resize_matrix = (float*)malloc(sizeof(height*width/4));
    for (int i=0; i<4; ++i) {
        int x = i / 2;
        int y = i % 2;
        float single_matrix[matrix_height*matrix_width];
        for (int h=0; h<matrix_height; ++h) {
            int h_offset = h*matrix_width;
            for (int w=0; w<matrix_width; ++w) {
                single_matrix[h_offset+w] = correct_matrix[h_offset*4+w*4+i];
            }
        }
        //resize
        
        for (int h=0; h<height/2; ++h) {
            int h_offset1 = (2*h+x)*width;
            int h_offset2 = h*width/2;
            for (int w=0; w<width/2; ++w) {
                data[h_offset1 + 2*w + y] *= resize_matrix[h_offset2 + w];
            }
        }
    }
    free(resize_matrix);
    return SOFT_ISP_OK;
}

template<typename T>
cv_errcode GetEstimateMean(T* data, const int height, const int width, float& estimate_mean) {
    const float factor[4] = {0.2126, 0.7152/2, 0.7152/2, 0.0722};
    const float delta = 0.01;
    float sum = 0.;
    float total_size = height*width/4;
    for (int h=0; h<height/2; ++h) {
        for (int w=0; w<width/2; ++w) {
            float val = 0.;
            for (int i=0; i<4; ++i) {
                int x = i / 2;
                int y = i % 2;
                val += factor[i] * data[2*(h+x)*width + 2*w + y]; 
            }
            sum += log(delta + POSITIVE(val));
        }
    }
    estimate_mean = exp(sum / total_size);
    return SOFT_ISP_OK;
} 

template<typename T>
cv_errcode ColorJitter(T* data, const int height, const int width, const int channel, const color_jitter_param_t& param, const float target_medium_num, const float estimate_mean, bool channel_first=true) {
    float key_scale = target_medium_num / estimate_mean;
    if (channel_first) {
        for (int c=0; c<channel; ++c) {
            int c_offset = c*height*width;
            for (int h=0; h<height; ++h) {
                int h_offset = h*width;
                for (int w=0; w<width; ++w) {
                    T val = data[c_offset + h_offset + w] * key_scale;
                    if (val <= 0) {
                        val = val;
                    } else {
                        val = (val*(param.A*val+param.B)) / (val*(param.C*val+param.D) + param.E);
                    }
                    data[c_offset + h_offset + w] = val;
                }
            }
        }
    } else {
        for (int h=0; h<height; ++h) {
            int h_offset = h*width*channel;
            for (int w=0; w<width; ++w) {
                int w_offset = w*channel;
                for (int c=0; c<channel; ++c) {
                    T val = data[h_offset + w_offset + c] * key_scale;
                    if (val <= 0) {
                        val = val;
                    } else {
                        val = (val*(param.A*val+param.B)) / (val*(param.C*val+param.D) + param.E);
                    }
                    data[h_offset + w_offset + c] = val;
                }
            }
        }
    }
    return SOFT_ISP_OK;
}

template<typename T>
cv_errcode Killer(const T *ref, T *alt, T *mask, const int height, const int width, const float sigma1=3.0, const float winear_factor=0.01, const float threshold=0.5, const float k_sigmoid=25., const float a_sigmoid=0.45) {
    const int kernel_len1 = 5;
    const int kernel_len2 = 11;
    const float sigma2 = 10.0;
    int half_size = height*width/2;
    T *ref_resize = (T*)malloc(sizeof(T)*half_size);
    T *alt_resize = (T*)malloc(sizeof(T)*half_size);
    RESIZE_BILIEAR(ref, ref_resize);
    RESIZE_BILIEAR(alt, alt_resize);
    
    T *ref_blur = (T*)malloc(sizeof(T)*half_size);
    T *alt_blur = (T*)malloc(sizeof(T)*half_size);
    GAUSS_BLUR(ref_resize, ref_blur, kernel_len1, sigma1);
    GAUSS_BLUR(alt_resize, alt_blur, kernel_len1, sigma1);

    T *mask_dist = (T*)malloc(sizeof(T)*half_size);
    for (int h=0; h<height/2; ++h) {
        int h_offset = h*width/2;
        for (int w=0; w<width/2; ++w) {
            T ref_val = ref_blur[h_offset + w];
            T alt_val = alt_blur[h_offset + w];
            T dist = ref_val - alt_val;
            dist *= dist;
            dist -= threshold;
            dist = POSITIVE(dist);
            dist = (dist) / (dist+winear_factor);
            dist = k_sigmoid*(a_sigmoid - dist);
            dist = Clamp<T>(dist, 0.0, 1.0);
            mask_dist[h_offset + w];
        }
    }

    T *mask_blur = (T*)malloc(sizeof(T)*half_size);
    GAUSS_BLUR(mask_dist, mask_blur, kernel_len2, sigma2);
    RESIZE_NEAREST(mask_blur, mask);

    for (int h=0; h<height; ++h) {
        int h_offset = h*width;
        for (int w=0; w<width; ++w) {
            int index = h_offset + w;
            T mask_val = mask[index];
            alt[index] = mask_val*ref[index] + (1-mask_val)*alt[index];
        }
    }
    free(mask_blur);
    free(mask_dist);
    free(ref_blur);
    free(alt_blur);
    free(ref_resize);
    free(alt_resize);
    return SOFT_ISP_OK;
}
