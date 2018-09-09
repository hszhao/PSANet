#include <vector>
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pointwise_spatial_attention_layer.hpp"

namespace caffe {

template <typename Dtype>
void PointwiseSpatialAttentionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PointwiseSpatialAttentionParameter pointwise_spatial_attention_param = this->layer_param_.pointwise_spatial_attention_param();

  if (pointwise_spatial_attention_param.has_attention_mask_height() &&
      pointwise_spatial_attention_param.has_attention_mask_width()) {
    mask_H_ = pointwise_spatial_attention_param.attention_mask_height();
    mask_W_ = pointwise_spatial_attention_param.attention_mask_width();
    CHECK((mask_H_ > 1) && (mask_H_ % 2 == 1)) << "attention_mask_height must be an odd number and larger than 1.";
    CHECK((mask_W_ > 1) && (mask_W_ % 2 == 1)) << "attention_mask_width must be an odd number and larger than 1.";
  }
  else if (!pointwise_spatial_attention_param.has_attention_mask_height() &&
           !pointwise_spatial_attention_param.has_attention_mask_width()){
  	feature_H_ = bottom[0]->height();
    feature_W_ = bottom[0]->width();
    mask_H_ = 2 * feature_H_ - 1;
    mask_W_ = 2 * feature_W_ - 1;
  }
  else {
    LOG(FATAL) << "attention_mask_height and attention_mask_widht both must be specified.";
  }
  half_mask_H_ = (mask_H_ - 1) / 2;
  half_mask_W_ = (mask_W_ - 1) / 2;

  // row-major in channel of bottom[1]
  CHECK_EQ(bottom[1]->channels(), mask_H_ * mask_W_)
    << "Channel of bottom[1] should be equal to mask_H_ * mask_W_.";

  if (pointwise_spatial_attention_param.has_normalization_factor()) {
    normalization_factor_ = pointwise_spatial_attention_param.normalization_factor();
  }
  else {
    normalization_factor_ = Dtype(mask_H_ * mask_W_);
  }

  if (pointwise_spatial_attention_param.has_is_softmax()) {
    is_softmax_ = pointwise_spatial_attention_param.is_softmax();
  }
  else {
    is_softmax_ = false;
  }
  LOG(INFO) << "PSA module is_softmax: " << is_softmax_;

  // Internal Softmax Layer
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(&mask_buffer_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&mask_buffer_prob_);
  softmax_propagate_down_.resize(1, true);
  if (is_softmax_) {
    num_ = bottom[0]->num();
    feature_H_ = bottom[0]->height();
    feature_W_ = bottom[0]->width();
    mask_buffer_.Reshape(num_, feature_H_*feature_W_, feature_H_, feature_W_);
    softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  }

  // PSA type
  CHECK(pointwise_spatial_attention_param.psa_type()
      == PointwiseSpatialAttentionParameter_PSAType_COLLECT
      || pointwise_spatial_attention_param.psa_type()
      == PointwiseSpatialAttentionParameter_PSAType_DISTRIBUTE)
      << "PSA implemented only for COLLECT and DISTRIBUTE mode.";
  if(pointwise_spatial_attention_param.psa_type()
      == PointwiseSpatialAttentionParameter_PSAType_COLLECT) {
    LOG(INFO) << "PSA mode: COLLECT";
  }
  else {
    LOG(INFO) << "PSA mode: DISTRIBUTE";
  }
}

template <typename Dtype>
void PointwiseSpatialAttentionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "#Samples of the two bottom must be the same.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "bottom[0] and bottom[1] should have the same width.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "bottom[0] and bottom[1] should have the same height.";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  feature_H_ = bottom[0]->height();
  feature_W_ = bottom[0]->width();
  top[0]->ReshapeLike(*bottom[0]);
  // buffer for fast CPU/GPU implementation.
  mask_buffer_.Reshape(num_, feature_H_*feature_W_, feature_H_, feature_W_);
  if(is_softmax_) {
    softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  }
}

template <typename Dtype>
void PSAForward_buffer_mask_collect_cpu(const int num_,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* mask_data, Dtype* buffer_data) {
  for(int n = 0; n < num_; n++) {
    for(int h = 0; h < feature_H_; h++) {
      for(int w = 0; w < feature_W_; w++) {
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
  }
}

template <typename Dtype>
void PSAForward_buffer_mask_distribute_cpu(const int num_,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* mask_data, Dtype* buffer_data) {
  for(int n = 0; n < num_; n++) {
    for(int h = 0; h < feature_H_; h++) {
      for(int w = 0; w < feature_W_; w++) {
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
  }
}

template <typename Dtype>
void PointwiseSpatialAttentionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_set(mask_buffer_.count(), Dtype(0), mask_buffer_.mutable_cpu_data());
  // set mask buffer
  switch (this->layer_param_.pointwise_spatial_attention_param().psa_type()) {
  case PointwiseSpatialAttentionParameter_PSAType_COLLECT:
    PSAForward_buffer_mask_collect_cpu<Dtype>(num_, feature_H_, feature_W_,
        mask_H_, mask_W_, half_mask_H_, half_mask_W_,
        bottom[1]->cpu_data(), mask_buffer_.mutable_cpu_data());
    break;
  case PointwiseSpatialAttentionParameter_PSAType_DISTRIBUTE:
    PSAForward_buffer_mask_distribute_cpu<Dtype>(num_, feature_H_, feature_W_,
        mask_H_, mask_W_, half_mask_H_, half_mask_W_,
        bottom[1]->cpu_data(), mask_buffer_.mutable_cpu_data());
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
    this_mask_data_root = mask_buffer_prob_.cpu_data();
  }
  else {
    this_mask_data_root = mask_buffer_.cpu_data();
  }
  for(int n = 0; n < num_; n++) {
    const Dtype* this_bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(n);
    const Dtype* this_mask_data = this_mask_data_root + mask_buffer_.offset(n);
    Dtype* this_top_data = top[0]->mutable_cpu_data() + top[0]->offset(n);
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans,
                   channels_, feature_H_ * feature_W_, feature_H_ * feature_W_,
                   Dtype(1.0/normalization_factor_), this_bottom_data, this_mask_data, Dtype(0), this_top_data);
  }
}

template <typename Dtype>
void PSABackward_buffer_mask_collect_cpu(const int num_,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* buffer_diff, Dtype* mask_diff) {
  for(int n = 0; n < num_; n++) {
    for(int h = 0; h < feature_H_; h++) {
      for(int w = 0; w < feature_W_; w++) {
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
  }
}

template <typename Dtype>
void PSABackward_buffer_mask_distribute_cpu(const int num_,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* buffer_diff, Dtype* mask_diff) {
  for(int n = 0; n < num_; n++) {
    for(int h = 0; h < feature_H_; h++) {
      for(int w = 0; w < feature_W_; w++) {
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
  }
}

template <typename Dtype>
void PointwiseSpatialAttentionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // BP to feature
  if (propagate_down[0]) {
    const Dtype* this_mask_data_root = NULL;
    if(is_softmax_) {
      this_mask_data_root = mask_buffer_prob_.cpu_data();
    }
    else {
      this_mask_data_root = mask_buffer_.cpu_data();
    }
    for(int n = 0; n < num_; n++) {
      const Dtype* this_top_diff = top[0]->cpu_diff() + top[0]->offset(n);
      const Dtype* this_mask_data = this_mask_data_root + mask_buffer_.offset(n);
      Dtype* this_bottom_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(n);
      caffe_cpu_gemm(CblasNoTrans, CblasTrans,
                     channels_, feature_H_ * feature_W_, feature_H_ * feature_W_,
                     Dtype(1.0/normalization_factor_), this_top_diff, this_mask_data, Dtype(0), this_bottom_diff);
    }
  }
  // BP to attention
  if (propagate_down[1]) {
    Dtype* this_mask_diff_root = NULL;
    if(is_softmax_) {
      this_mask_diff_root = mask_buffer_prob_.mutable_cpu_diff();
    }
    else {
      this_mask_diff_root = mask_buffer_.mutable_cpu_diff();
    }
    for(int n = 0; n < num_; n++) {
      const Dtype* this_top_diff = top[0]->cpu_diff() + top[0]->offset(n);
      const Dtype* this_bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(n);
      Dtype* this_mask_diff = this_mask_diff_root + mask_buffer_.offset(n);
      caffe_cpu_gemm(CblasTrans, CblasNoTrans,
                     feature_H_ * feature_W_, feature_H_ * feature_W_, channels_,
                     Dtype(1.0/normalization_factor_), this_bottom_data, this_top_diff, Dtype(0), this_mask_diff);
    }
    // BP of softmax.
    if(is_softmax_) {
      softmax_layer_->Backward(softmax_top_vec_, softmax_propagate_down_, softmax_bottom_vec_);
    }
    caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
    switch (this->layer_param_.pointwise_spatial_attention_param().psa_type()) {
    case PointwiseSpatialAttentionParameter_PSAType_COLLECT:
      PSABackward_buffer_mask_collect_cpu<Dtype>(num_, feature_H_,
          feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
          mask_buffer_.cpu_diff(), bottom[1]->mutable_cpu_diff());
      break;
    case PointwiseSpatialAttentionParameter_PSAType_DISTRIBUTE:
      PSABackward_buffer_mask_distribute_cpu<Dtype>(num_, feature_H_,
          feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
          mask_buffer_.cpu_diff(), bottom[1]->mutable_cpu_diff());
      break;
    default:
      LOG(FATAL) << "Unknown PSA type.";
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PointwiseSpatialAttentionLayer);
#endif

INSTANTIATE_CLASS(PointwiseSpatialAttentionLayer);
REGISTER_LAYER_CLASS(PointwiseSpatialAttention);

}  // namespace caffe
