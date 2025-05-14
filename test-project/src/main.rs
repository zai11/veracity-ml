use std::time::Instant;

use veracity_core::enums::distance_metrics::DistanceMetrics;
use veracity_core::neighbors::k_neighbors_classifier::{KNeighborsClassifier, KNeighborsClassifierSettings};
use veracity_core::base::regressor_base::RegressorBase;
use veracity_core::neighbors::k_neighbors_regressor::{KNeighborsRegressor, KNeighborsRegressorSettings};
use veracity_core::neighbors::k_neighbors_weights::KNeighborsWeights;
use veracity_data::{data_loader::{csv_loader::{CSVLoader, CSVLoaderSettings}, DataLoader}, data_matrix::DataMatrix, data_vector::DataVector};

#[tokio::main]
async fn main() {
    println!("Running...");
    let csv_loader_settings: CSVLoaderSettings = CSVLoaderSettings {
        skip_rows: 5,
        ..Default::default()
    };
    let data_loader: CSVLoader = CSVLoader::new(csv_loader_settings);

    let training_data: DataMatrix = data_loader.load_from("/Users/zai/data/training.csv").await.unwrap();
    let x_train: DataMatrix = training_data.exclude_columns(vec!["class", "b4"]).unwrap();
    let y_train: DataVector = training_data.get_column("b4").unwrap();

    let testing_data: DataMatrix = data_loader.load_from("/Users/zai/data/testing.csv").await.unwrap();
    let x_test: DataMatrix = testing_data.exclude_columns(vec!["class", "b4"]).unwrap();
    let y_test: DataVector = testing_data.get_column("b4").unwrap();

    let regressor_settings: KNeighborsRegressorSettings = KNeighborsRegressorSettings {
        ..Default::default()
    };

    let mut regressor: KNeighborsRegressor<f64, f64> = KNeighborsRegressor::<f64, f64>::new();
    regressor.add_settings(regressor_settings).unwrap();

    regressor.fit(&x_train, &y_train).unwrap();

    let start = Instant::now();

    for _ in 0..100 {
        regressor.score(&x_test, &y_test).unwrap();
    }

    let duration = start.elapsed();

    println!("Execution took: {}ms", duration.as_millis());
    println!("Average: {}ms", duration.as_millis() / 100);

}