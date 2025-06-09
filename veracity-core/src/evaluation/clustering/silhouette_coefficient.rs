use std::collections::BTreeMap;

use ndarray::{Array1, Array2};
use veracity_data::{data_matrix::{DataMatrix, TDataMatrix}, data_vector::{DataVector, TDataVectorExt}};

use crate::{enums::errors::metric_errors::EvaluationMetricError, utility::distance::find_distance_euclidean};


pub fn _silhouette_score<T, U>(data: &Array2<T>, labels: &Array1<U>) -> Result<f64, EvaluationMetricError>
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Ord + Clone + Send + Sync + 'static,
{
    assert_eq!(data.nrows(), labels.len(), "Data and labels must be the same length.");

    let mut clusters: BTreeMap<U, Vec<usize>> = BTreeMap::new();
    for (i, label) in labels.iter().enumerate() {
        clusters.entry(label.clone()).or_default().push(i);
    }

    let mut silhouette_sum: f64 = 0.0;
    let mut count: i32 = 0;

    for (i, point_row) in data.outer_iter().enumerate() {
        let point: Array1<f64> = point_row.map(|x| (*x).into());
        let label: &U = &labels[i];
        let own_cluster: &Vec<usize> = &clusters[label];

        // a(i): Mean distance to points in same cluster
        let a_i: f64 = own_cluster
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| find_distance_euclidean::<f64>(&point.view(), &data.row(j).map(|x: &T| (*x).into()).view()))
            .sum::<f64>()
            / ((own_cluster.len() - 1).max(1) as f64);

        // b(i): Min mean distance to other clusters
        let mut b_i: f64 = f64::MAX;
        for (other_label, indices) in &clusters {
            if other_label == label {
                continue;
            }

            let mean_dist: f64 = indices
                .iter()
                .map(|&j| find_distance_euclidean(&point.view(), &data.row(j).map(|x: &T| (*x).into()).view()))
                .sum::<f64>()
                / (indices.len() as f64);

            if mean_dist < b_i {
                b_i = mean_dist;
            }
        }

        let s_i: f64 = if a_i < b_i {
            (b_i - a_i) / b_i
        } else if a_i > b_i {
            (b_i - a_i) / a_i
        } else {
            0.0
        };

        silhouette_sum += s_i;
        count += 1;
    }

    Ok(silhouette_sum / (count as f64))
}

pub fn silhouette_score<T, U>(data: &DataMatrix, labels: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where 
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Eq + std::hash::Hash + Clone + Ord + Send + Sync + 'static
{
    _silhouette_score(&data.to_ndarray::<T>().unwrap(), &labels.to_ndarray().unwrap())
}