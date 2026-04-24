//! Candle backend implementation for fast-umap.
//!
//! Implements the custom [`Backend`](crate::backend::Backend) trait for
//! [`burn_candle::Candle`] by delegating to the generic tensor-based
//! implementations in [`generic_backend`](crate::generic_backend).
//!
//! Candle supports CPU, CUDA, and Metal devices via Hugging Face's
//! [candle](https://github.com/huggingface/candle) framework.

use burn::tensor::ops::{FloatTensor, IntTensor};

use crate::backend::Backend;

impl<F: burn_candle::element::FloatCandleElement, I: burn_candle::element::IntCandleElement>
    Backend for burn_candle::Candle<F, I>
{
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        crate::generic_backend::euclidean_pairwise_distance::<Self>(x)
    }

    fn euclidean_pairwise_distance_backward(
        grad_pairwise: FloatTensor<Self>,
        x: FloatTensor<Self>,
        pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::generic_backend::euclidean_pairwise_distance_backward::<Self>(
            grad_pairwise, x, pairwise,
        )
    }

    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>) {
        crate::generic_backend::knn::<Self>(pairwise_distances, k)
    }

    fn knn_backward(
        pairwise_distances: FloatTensor<Self>,
        k: u32,
        grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::generic_backend::knn_backward::<Self>(pairwise_distances, k, grad_output)
    }
}
