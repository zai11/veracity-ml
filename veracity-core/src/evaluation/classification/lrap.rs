use ndarray::{Array2, Data};
use veracity_data::data_matrix::DataMatrix;

pub fn _lrap<T, U>(y_pred: &Array2<T>, y_true: &Array2<U>) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<bool> + Copy + Send + Sync + 'static,
{
    assert_eq!(y_pred.dim(), y_true.dim(), "Predicted and actual arrays must have the same shape.");

    let (n_samples, _) = y_pred.dim();
    let mut total_lrap: f64 = 0.0;

    for (pred_row, true_row) in y_pred.outer_iter().zip(y_true.outer_iter()) {
        let mut scored_labels: Vec<(usize, f64, bool)> = pred_row
            .iter()
            .zip(true_row.iter())
            .enumerate()
            .map(|(i, (p, t))| (i, (*p).into(), (*t).into()))
            .collect();

        scored_labels.sort_by(|a: &(usize, f64, bool), b: &(usize, f64, bool)| b.1.partial_cmp(&a.1).unwrap());

        let mut num_relevant: i32 = 0;
        let mut score: f64 = 0.0;

        for (rank, &(_, _, is_relevant)) in scored_labels.iter().enumerate() {
            if is_relevant {
                num_relevant += 1;

                let num_relevant_up_to_rank: usize = scored_labels
                    .iter()
                    .take(rank + 1)
                    .filter(|(_, _, rel)| *rel)
                    .count();

                score += num_relevant_up_to_rank as f64 / (rank + 1) as f64;
            }
        }

        if num_relevant > 0 {
            total_lrap += score / num_relevant as f64;
        }
    }

    total_lrap / n_samples as f64
}

pub fn lrap<T, U>(proba_pred: DataMatrix, actual: DataMatrix) -> f64
where
    T: Into<f64> + Copy + Send + Sync + 'static,
    U: Into<bool> + Copy + Send + Sync + 'static,
{
    _lrap(&proba_pred.to_ndarray::<T>().unwrap(), &actual.to_ndarray::<U>().unwrap())
}