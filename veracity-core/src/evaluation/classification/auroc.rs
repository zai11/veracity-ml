use ndarray::Array1;
use veracity_data::data_vector::DataVector;

pub fn _auroc<T, U>(proba_pred: &Array1<T>, actual: &Array1<U>, positive_class: &U) -> f64
where
    T: PartialEq<U> + Into<f64> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    assert_eq!(
        proba_pred.len(),
        actual.len(),
        "Inputs must have the same length"
    );

    let mut paired: Vec<(f64, bool)> = proba_pred
        .iter()
        .cloned()
        .zip(actual.iter())
        .map(|(score, label)| (score.into(), label == positive_class))
        .collect();

    paired.sort_by(|a: &(f64, bool), b: &(f64, bool)| b.0.partial_cmp(&a.0).unwrap());

    let total_positives: f64 = paired.iter().filter(|(_, is_pos)| *is_pos).count() as f64;
    let total_negatives: f64 = paired.len() as f64 - total_positives;

    if total_positives == 0.0 || total_negatives == 0.0 {
        return 0.0;
    }

    let mut tp: f64 = 0.0;
    let mut fp: f64 = 0.0;

    let mut tpr: Vec<f64> = vec![0.0];
    let mut fpr: Vec<f64> = vec![0.0];

    for &(_, is_positive) in &paired {
        if is_positive {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        tpr.push(tp / total_positives);
        fpr.push(fp / total_negatives);
    }

    // Trapezoidal integration over FPR (x-axis) to compute area under ROC curve
    let mut auroc: f64 = 0.0;
    for i in 1..fpr.len() {
        let dx: f64 = fpr[i] - fpr[i - 1];
        let avg_y: f64 = (tpr[i] + tpr[i - 1]) / 2.0;
        auroc += dx * avg_y;
    }

    auroc
}

pub fn auroc<T, U>(proba_pred: &DataVector, actual: &DataVector, positive_class: &U) -> f64
where
    T: PartialEq<U> + Into<f64> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    _auroc(&proba_pred.to_ndarray::<T>().unwrap(), &actual.to_ndarray::<U>().unwrap(), positive_class)
}