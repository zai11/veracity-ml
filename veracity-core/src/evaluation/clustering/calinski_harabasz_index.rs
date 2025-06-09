use std::collections::BTreeMap;

use ndarray::{Array1, Array2, ArrayBase, Dim, ViewRepr};
use veracity_data::{data_matrix::{DataMatrix, TDataMatrix}, data_vector::{DataVector, TDataVectorExt}};

use crate::enums::errors::metric_errors::EvaluationMetricError;


pub fn _calinski_harabasz_index<T, U>(data: &Array2<T>, labels: &Array1<U>) -> Result<f64, EvaluationMetricError>
where 
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Eq + std::hash::Hash + Clone + Ord
{
    assert_eq!(data.nrows(), labels.len(), "Each point must have a label.");

    let n_samples: f64 = data.nrows() as f64;
    let n_features: usize = data.ncols();
    let mut label_map: BTreeMap<U, Vec<usize>> = BTreeMap::new();

    for (i, label) in labels.iter().enumerate() {
        label_map.entry(label.clone()).or_default().push(i);
    }

    let n_clusters: usize = label_map.len();
    if n_clusters <= 1 {
        return Ok(0.0);
    }

    let mut overall_mean: Vec<f64> = vec![0.0; n_features];
    for row in data.outer_iter() {
        for (i, val) in row.iter().enumerate() {
            overall_mean[i] += (*val).into();
        }
    }
    for val in &mut overall_mean {
        *val /= n_samples;
    }

    let mut between_dispersion: f64 = 0.0;
    let mut within_dispersion: f64 = 0.0;

    for (_label, indices) in label_map {
        let cluster_size: f64 = indices.len() as f64;

        let mut cluster_mean: Vec<f64> = vec![0.0; n_features];
        for &i in &indices {
            let row: ArrayBase<ndarray::ViewRepr<&T>, Dim<[usize; 1]>> = data.row(i);
            for (j, val) in row.iter().enumerate() {
                cluster_mean[j] += (*val).into();
            }
        }
        for val in &mut cluster_mean {
            *val /= cluster_size;
        }

        let mut b_dist: f64 = 0.0;
        for j in 0..n_features {
            let diff: f64 = cluster_mean[j] - overall_mean[j];
            b_dist += diff * diff;
        }
        between_dispersion += cluster_size * b_dist;

        for &i in &indices {
            let row: ArrayBase<ViewRepr<&T>, Dim<[usize; 1]>> = data.row(i);
            let mut w_dist: f64 = 0.0;
            for (j, val) in row.iter().enumerate() {
                let diff = (*val).into() - cluster_mean[j];
                w_dist += diff * diff;
            }
            within_dispersion += w_dist;
        }
    }

    Ok((between_dispersion / (n_clusters as f64 - 1.0))
        / (within_dispersion / (n_samples - n_clusters as f64)))
}

pub fn calinski_harabasz_index<T, U>(data: &DataMatrix, labels: &DataVector<U>) -> Result<f64, EvaluationMetricError> 
where 
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Eq + std::hash::Hash + Clone + Ord + Send + Sync + 'static
{
    _calinski_harabasz_index(&data.to_ndarray::<T>().unwrap(), &labels.to_ndarray().unwrap())
}