use std::time::Instant;

use veracity_core::enums::distance_metrics::DistanceMetrics;
use veracity_core::neighbors::k_neighbors_classifier::{KNeighborsClassifier, KNeighborsClassifierSettings, KNeighborsWeights};
use veracity_core::base::classifier_base::Classifier;
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

    println!("training_data rows: {}", training_data.columns["class"].len);

    let x_train: DataMatrix = training_data.exclude_column("class").unwrap();
    let y_train: DataVector = training_data.get_column("class").unwrap();

    println!("X_train rows: {}", x_train.columns["b2"].len);
    println!("y_train rows: {}", y_train.len);

    let testing_data: DataMatrix = data_loader.load_from("/Users/zai/data/testing.csv").await.unwrap();
    let x_test: DataMatrix = testing_data.exclude_column("class").unwrap();
    let y_test: DataVector = testing_data.get_column("class").unwrap();

    let classifier_settings: KNeighborsClassifierSettings = KNeighborsClassifierSettings {
        metric: DistanceMetrics::Euclidean,
        weights: KNeighborsWeights::Distance,
        ..Default::default()
    };

    let mut classifier: KNeighborsClassifier<f64, String> = KNeighborsClassifier::<f64, String>::new();
    classifier.add_settings(classifier_settings).unwrap();

    classifier.fit(&x_train, &y_train).unwrap();

    println!("{}", classifier.score(&x_test, &y_test).unwrap());

}