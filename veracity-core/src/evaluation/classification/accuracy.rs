use ndarray::Array1;
use veracity_data::data_vector::DataVector;

pub fn _accuracy<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> f64
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: Clone + Send + Sync + 'static
{
    assert_eq!(y_pred.len(), y_actual.len(), "Arrays must be the same length.");

    let correct: usize = y_pred
        .iter()
        .zip(y_actual.iter())
        .filter(|(pred, actual)| pred == actual)
        .count();

    correct as f64 / y_pred.len() as f64
}

pub fn accuracy<T, U>(y_pred: &DataVector, y_actual: &DataVector) -> f64
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: Clone + Send + Sync + 'static
{
    _accuracy(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap())
}