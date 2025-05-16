use ndarray::Array1;
use veracity_data::data_vector::DataVector;


pub fn _auprc<T, U>(proba_pred: &Array1<T>, actual: &Array1<U>, positive_class: &U) -> f64
where
    T: PartialEq<U> + Into<f64> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    assert_eq!(proba_pred.len(), actual.len(), "Inputs must be the same length");

    let mut paired: Vec<(f64, u8)> = proba_pred
        .iter()
        .cloned()
        .zip(actual.iter().cloned())
        .map(|(pred, label)| {
            let score: f64 = pred.into();
            let binary: u8 = if &label == positive_class { 1 } else { 0 };
            (score, binary)
        })
        .collect();

    paired.sort_by(|a: &(f64, u8), b: &(f64, u8)| b.0.partial_cmp(&a.0).unwrap());

    let total_positives: f64 = paired.iter().filter(|&&(_, y)| y == 1).count() as f64;
    if total_positives == 0.0 {
        return 0.0;
    }

    let mut tp: f64 = 0.0;
    let mut fp: f64 = 0.0;

    let mut precision: Vec<f64> = vec![];
    let mut recall: Vec<f64> = vec![];

    for &(_, label) in &paired {
        if label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        precision.push(tp / (tp + fp));
        recall.push(tp / total_positives);
    }


    // Trapezoidal integration over FPR (x-axis) to compute area under ROC curve
    let mut auprc: f64 = 0.0;
    for i in 1..recall.len() {
        let delta_recall: f64 = recall[i] - recall[i - 1];
        let avg_precision: f64 = (precision[i] + precision[i - 1]) / 2.0;
        auprc += delta_recall * avg_precision;
    }

    auprc
}

pub fn auprc<T, U>(proba_pred: &DataVector, actual: &DataVector, positive_class: &U) -> f64
where
    T: PartialEq<U> + Into<f64> + Clone + Send + Sync + 'static,
    U: PartialEq + Clone + Send + Sync + 'static
{
    _auprc(&proba_pred.to_ndarray::<T>().unwrap(), &actual.to_ndarray::<U>().unwrap(), positive_class)
}