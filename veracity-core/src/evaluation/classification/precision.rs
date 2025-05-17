use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _precision<T, U>(y_pred: &Array1<T>, y_actual: &Array1<U>) -> f64
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: Clone + Send + Sync + 'static
{
    assert_eq!(y_pred.len(), y_actual.len(), "Arrays must be the same length.");

    if y_actual.len() == 0 {
        return 0.0;
    }

    let positive_class: &U = match y_actual.first() {
        Some(val) => val,
        None => return 0.0,
    };

    let mut true_positives: f64 = 0.0;
    let mut false_positives: f64 = 0.0;

    for (pred, actual) in y_pred.iter().zip(y_actual.iter()) {
        if pred == positive_class {
            if pred == actual {
                true_positives += 1.0;
            } else {
                false_positives += 1.0;
            }
        }
    }

    if true_positives + false_positives == 0.0 {
        0.0
    } else {
        true_positives / (true_positives + false_positives)
    }
}

pub fn precision<T, U>(y_pred: &DataVector, y_actual: &DataVector) -> f64
where
    T: PartialEq<U> + Clone + Send + Sync + 'static,
    U: Clone + Send + Sync + 'static
{
    _precision(&y_pred.to_ndarray::<T>().unwrap(), &y_actual.to_ndarray::<U>().unwrap())
}