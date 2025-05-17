use ndarray::Array1;
use veracity_data::data_vector::DataVector;

pub fn _adjusted_r2<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>, num_features: &usize) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static,
{
    assert_eq!(
        y_pred.len(),
        y_actual.len(),
        "Predicted and actual arrays must be the same length"
    );

    let num_features: f64 = *num_features as f64;

    let n: f64 = y_actual.len() as f64;
    assert!(
        n > num_features + 1.0,
        "Number of samples must be greater than number of features + 1"
    );

    let y_mean: f64 = y_actual.iter().map(|y: &U| (*y).into()).sum::<f64>() / (n as f64);

    let ss_total: f64 = y_actual
        .iter()
        .map(|y| {
            let val = (*y).into();
            (val - y_mean).powi(2)
        })
        .sum::<f64>();

    let ss_res: f64 = y_pred
        .iter()
        .zip(y_actual.iter())
        .map(|(yp, ya)| {
            let yp_val = (*yp).into();
            let ya_val = (*ya).into();
            (ya_val - yp_val).powi(2)
        })
        .sum::<f64>();

    let r2: f64 = 1.0 - (ss_res / ss_total);

    1.0 - (1.0 - r2) * ((n - 1.0) as f64) / ((n - num_features - 1.0) as f64)
}

pub fn adjusted_r2<T, U>(y_pred: &DataVector, y_actual: &DataVector, num_features: &usize) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<f64> + Copy + Send + Sync + 'static
{
    _adjusted_r2(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap(), num_features)
}