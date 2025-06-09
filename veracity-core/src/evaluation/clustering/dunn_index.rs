use std::collections::BTreeMap;

use ndarray::{Array1, Array2};
use veracity_data::{data_matrix::{DataMatrix, TDataMatrix}, data_vector::{DataVector, TDataVectorExt}};

use crate::{enums::errors::metric_errors::EvaluationMetricError, utility::distance::find_distance_euclidean};


pub fn _dunn_index<T, U>(data: &Array2<T>, labels: &Array1<U>) -> Result<f64, EvaluationMetricError>
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Ord + Clone + Send + Sync + 'static,
{
    assert_eq!(data.nrows(), labels.len(), "Data and labels must match in length.");

    let mut clusters: BTreeMap<U, Vec<usize>> = BTreeMap::new();
    for (i, label) in labels.iter().enumerate() {
        clusters.entry(label.clone()).or_default().push(i);
    }

    let mut centroids: Vec<Array1<f64>> = Vec::new();
    let mut max_diameter: f64 = 0.0;

    for indices in clusters.values() {
        let cluster_points: Vec<Array1<f64>> = indices
            .iter()
            .map(|&i| data.row(i).map(|x: &T| (*x).into()).to_owned())
            .collect();

        let centroid = mean_point(&cluster_points);
        let diameter: f64 = cluster_points
            .iter()
            .map(|p: &Array1<f64>| find_distance_euclidean(&p.view(), &centroid.view()))
            .fold(0.0, f64::max);

        centroids.push(centroid);
        max_diameter = max_diameter.max(diameter);
    }

    let mut min_intercluster: f64 = f64::MAX;
    for i in 0..centroids.len() {
        for j in i + 1..centroids.len() {
            let dist: f64 = find_distance_euclidean(&centroids[i].view(), &centroids[j].view());
            min_intercluster = min_intercluster.min(dist);
        }
    }

    if max_diameter == 0.0 {
        return Ok(0.0);
    }

    Ok(min_intercluster / max_diameter)
}

pub fn dunn_index<T, U>(data: &DataMatrix, labels: &DataVector<U>) -> Result<f64, EvaluationMetricError>
where 
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Eq + std::hash::Hash + Clone + Ord + Send + Sync + 'static
{
    _dunn_index(&data.to_ndarray::<T>().unwrap(), &labels.to_ndarray().unwrap())
}

fn mean_point(points: &[Array1<f64>]) -> Array1<f64> {
    let dim: usize = points[0].len();
    let mut sum: Array1<f64> = Array1::<f64>::zeros(dim);
    for p in points {
        sum = &sum + p;
    }
    sum / (points.len() as f64)
}