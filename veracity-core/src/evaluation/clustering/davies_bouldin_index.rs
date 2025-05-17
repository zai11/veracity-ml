use std::collections::BTreeMap;

use ndarray::{Array1, Array2};
use veracity_data::{data_matrix::DataMatrix, data_vector::DataVector};

use crate::utility::distance::find_distance_euclidean;


pub fn _davies_bouldin_index<T, U>(data: &Array2<T>, labels: &Array1<U>) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Ord + Clone + Send + Sync + 'static,
{
    assert_eq!(data.nrows(), labels.len(), "Data and labels must match in length.");

    let n_features: usize = data.ncols();
    let mut clusters: BTreeMap<U, Vec<usize>> = BTreeMap::new();

    for (i, label) in labels.iter().enumerate() {
        clusters.entry(label.clone()).or_default().push(i);
    }

    let n_clusters: usize = clusters.len();
    if n_clusters <= 1 {
        return 0.0;
    }

    let mut centroids: Vec<Vec<f64>> = Vec::new();
    let mut scatters: Vec<f64> = Vec::new();

    for indices in clusters.values() {
        let cluster_size: f64 = indices.len() as f64;
        let mut centroid: Vec<f64> = vec![0.0; n_features];

        for &i in indices {
            for (j, val) in data.row(i).iter().enumerate() {
                centroid[j] += (*val).into();
            }
        }
        for val in &mut centroid {
            *val /= cluster_size;
        }
        centroids.push(centroid.clone());

        let mut scatter: f64 = 0.0;
        for &i in indices {
            for (j, val) in data.row(i).iter().enumerate() {
                let diff = (*val).into() - centroid[j];
                scatter += diff * diff;
            }
        }
        scatters.push((scatter / cluster_size).sqrt());
    }

    let mut dbi_sum: f64 = 0.0;

    for i in 0..n_clusters {
        let mut max_ratio: f64 = f64::MIN;
        for j in 0..n_clusters {
            if i == j {
                continue;
            }

            let dist: f64 = find_distance_euclidean(&Array1::from_vec(centroids[i].clone()).view(), &Array1::from_vec(centroids[j].clone()).view());
            if dist == 0.0 {
                continue;
            }
            let ratio: f64 = (scatters[i] + scatters[j]) / dist;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        dbi_sum += max_ratio;
    }

    dbi_sum / n_clusters as f64
}

pub fn davies_bouldin_index<T, U>(data: &DataMatrix, labels: &DataVector) -> f64 
where 
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Eq + std::hash::Hash + Clone + Ord + Send + Sync + 'static
{
    _davies_bouldin_index(&data.to_ndarray::<T>().unwrap(), &labels.to_ndarray::<U>().unwrap())
}