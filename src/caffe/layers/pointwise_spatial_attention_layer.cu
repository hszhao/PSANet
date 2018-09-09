#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pointwise_spatial_attention_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PSAForward_buffer_mask_collect_gpu(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* mask_data, Dtype* buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        buffer_data[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w] =
            mask_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
      }
    }
  }
}

template <typename Dtype>
__global__ void PSAForward_buffer_mask_distribute_gpu(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* mask_data, Dtype* buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        buffer_data[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)] =
            mask_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
      }
    }
  }
}

template <typename Dtype>
void PointwiseSpatialAttentionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // set mask buffer
  caffe_gpu_set(mask_buffer_.count(), Dtype(0), mask_buffer_.mutable_gpu_data());
  int nthreads = num_ * feature_H_ * feature_W_;
  switch (this->layer_param_.pointwise_spatial_attention_param().psa_type()) {
  case PointwiseSpatialAttentionParameter_PSAType_COLLECT:
    PSAForward_buffer_mask_collect_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
        bottom[1]->gpu_data(), mask_buffer_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    break;
  case PointwiseSpatialAttentionParameter_PSAType_DISTRIBUTE:
    PSAForward_buffer_mask_distribute_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
        bottom[1]->gpu_data(), mask_buffer_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    break;
  default:
    LOG(FATAL) << "Unknown PSA type.";
  }
  // normalize by softmax.
  if(is_softmax_) {
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  }
  // aggregate
  const Dtype* this_mask_data_root = NULL;
  if(is_softmax_) {
    this_mask_data_root = mask_buffer_prob_.gpu_data();
  }
  else {
    this_mask_data_root = mask_buffer_.gpu_data();
  }
  for(int n = 0; n < num_; n++) {
    const Dtype* this_bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(n);
    const Dtype* this_mask_data = this_mask_data_root + mask_buffer_.offset(n);
    Dtype* this_top_data = top[0]->mutable_gpu_data() + top[0]->offset(n);
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
                   channels_, feature_H_ * feature_W_, feature_H_ * feature_W_,
                   Dtype(1.0/normalization_factor_), this_bottom_data, this_mask_data, Dtype(0), this_top_data);
  }
}

template <typename Dtype>
__global__ void PSABackward_buffer_mask_collect_gpu(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* buffer_diff, Dtype* mask_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
      	mask_diff[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] =
            buffer_diff[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w];
      }
    }
  }
}

template <typename Dtype>
__global__ void PSABackward_buffer_mask_distribute_gpu(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* buffer_diff, Dtype* mask_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
      	mask_diff[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] =
            buffer_diff[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)];
      }
    }
  }
}

template <typename Dtype>
void PointwiseSpatialAttentionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // BP to feature
  if (propagate_down[0]) {
    const Dtype* this_mask_data_root = NULL;
    if(is_softmax_) {
      this_mask_data_root = mask_buffer_prob_.gpu_data();
    }
    else {
      this_mask_data_root = mask_buffer_.gpu_data();
    }
    for(int n = 0; n < num_; n++) {
      const Dtype* this_top_diff = top[0]->gpu_diff() + top[0]->offset(n);
      const Dtype* this_mask_data = this_mask_data_root + mask_buffer_.offset(n);
      Dtype* this_bottom_diff = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n);
      caffe_gpu_gemm(CblasNoTrans, CblasTrans,
                     channels_, feature_H_ * feature_W_, feature_H_ * feature_W_,
                     Dtype(1.0/normalization_factor_), this_top_diff, this_mask_data, Dtype(0), this_bottom_diff);
    }
  }

  // BP to attention
  if (propagate_down[1]) {
    Dtype* this_mask_diff_root = NULL;
    if(is_softmax_) {
      this_mask_diff_root = mask_buffer_prob_.mutable_gpu_diff();
    }
    else {
      this_mask_diff_root = mask_buffer_.mutable_gpu_diff();
    }
    for(int n = 0; n < num_; n++) {
      const Dtype* this_top_diff = top[0]->gpu_diff() + top[0]->offset(n);
      const Dtype* this_bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(n);
      Dtype* this_mask_diff = this_mask_diff_root + mask_buffer_.offset(n);
      caffe_gpu_gemm(CblasTrans, CblasNoTrans,
                     feature_H_ * feature_W_, feature_H_ * feature_W_, channels_,
                     Dtype(1.0/normalization_factor_), this_bottom_data, this_top_diff, Dtype(0), this_mask_diff);
    }
    // BP of softmax.
    if(is_softmax_) {
      softmax_layer_->Backward(softmax_top_vec_, softmax_propagate_down_, softmax_bottom_vec_);
    }
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    int nthreads = num_ * feature_H_ * feature_W_;
    switch (this->layer_param_.pointwise_spatial_attention_param().psa_type()) {
    case PointwiseSpatialAttentionParameter_PSAType_COLLECT:
      PSABackward_buffer_mask_collect_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
          mask_buffer_.gpu_diff(), bottom[1]->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      break;
    case PointwiseSpatialAttentionParameter_PSAType_DISTRIBUTE:
      PSABackward_buffer_mask_distribute_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
          nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
          mask_buffer_.gpu_diff(), bottom[1]->mutable_gpu_diff());
      CUDA_POST_KERNEL_CHECK;
      break;
    default:
      LOG(FATAL) << "Unknown PSA type.";
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PointwiseSpatialAttentionLayer);
}  // namespace caffe
