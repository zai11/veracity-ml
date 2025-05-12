use ndarray::ArrayView1;
use num_traits::{Num, ToPrimitive};

pub fn find_distance_cosine<T: Num + ToPrimitive + Copy>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| {
        x.to_f64().unwrap() * y.to_f64().unwrap()
    }).sum::<f64>();

    let norm_a = a.iter().map(|x| {
        let v = x.to_f64().unwrap();
        v * v
    }).sum::<f64>().sqrt();

    let norm_b = b.iter().map(|x| {
        let v = x.to_f64().unwrap();
        v * v
    }).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return f64::NAN;
    }

    1.0 - (dot / (norm_a * norm_b))
}

pub fn find_distance_euclidean<T: Num + ToPrimitive + Copy>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
    let sum_sq_diff: f64 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let dx = x.to_f64().unwrap();
            let dy = y.to_f64().unwrap();
            (dx - dy).powi(2)
        })
        .sum();

    sum_sq_diff
}

pub fn find_distance_manhatten<T: Num + ToPrimitive + Copy>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f64().unwrap() - y.to_f64().unwrap()).abs())
        .sum()
}

pub fn find_distance_minkowski<T: Num + ToPrimitive + Copy>(a: &ArrayView1<T>, b: &ArrayView1<T>, p: i64) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f64().unwrap() - y.to_f64().unwrap()).abs().powf(p as f64))
        .sum::<f64>()
        .powf(1.0 / p as f64)
}

pub fn find_distance_nan_euclidean<T: Num + ToPrimitive + Copy>(a: &ArrayView1<T>, b: &ArrayView1<T>) -> f64 {
    a.iter()
        .zip(b.iter())
        .filter_map(|(x, y)| {
            let x_f64 = x.to_f64().unwrap();
            let y_f64 = y.to_f64().unwrap();
            if x_f64.is_nan() || y_f64.is_nan() {
                None
            } else {
                Some((x_f64 - y_f64).powi(2))
            }
        })
        .sum::<f64>()
}