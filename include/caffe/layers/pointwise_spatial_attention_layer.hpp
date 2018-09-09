#ifndef CAFFE_POINTWISE_SPATIAL_ATTENTION_HPP_
#define CAFFE_POINTWISE_SPATIAL_ATTENTION_HPP_

#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class PointwiseSpatialAttentionLayer : public Layer<Dtype> {
 public:
  explicit PointwiseSpatialAttentionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "PointwiseSpatialAttention"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int channels_;
  int feature_H_, feature_W_;
  int mask_H_, mask_W_;
  int half_mask_H_, half_mask_W_;
  Dtype normalization_factor_;
  Blob<Dtype> mask_buffer_;
  bool is_softmax_;
  // The internal SoftmaxLayer used to normalize the mask value in buffer.
  shared_ptr<Layer<Dtype> > softmax_layer_;
  // For softmax layer backward
  vector<bool> softmax_propagate_down_;
  // bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  // top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  // prob stores the output probability predictions from the SoftmaxLayer
  Blob<Dtype> mask_buffer_prob_;
};

}  // namespace caffe

#endif  // CAFFE_POINTWISE_SPATIAL_ATTENTION_HPP_
