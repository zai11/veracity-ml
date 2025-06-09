use std::time::Instant;

use veracity_core::base::classifier_base::ClassifierBase;
use veracity_core::base::regressor_base::RegressorBase;
use veracity_core::enums::evaluation_metrics::EvaluationMetrics;
use veracity_core::neighbors::k_neighbors_classifier::{KNeighborsClassifier, KNeighborsClassifierSettings};
use veracity_core::neighbors::k_neighbors_regressor::{KNeighborsRegressor, KNeighborsRegressorSettings};
use veracity_data::data_matrix::TDataMatrix;
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
    let x_train: DataMatrix = training_data.exclude_columns(vec!["class"]).unwrap();
    let y_train: DataVector<String> = training_data.get_column("class").unwrap();

    let testing_data: DataMatrix = data_loader.load_from("/Users/zai/data/testing.csv").await.unwrap();
    let x_test: DataMatrix = testing_data.exclude_columns(vec!["class"]).unwrap();
    let y_test: DataVector<String> = testing_data.get_column("class").unwrap();

    let classifier_settings: KNeighborsClassifierSettings = KNeighborsClassifierSettings {
        //score_metric: EvaluationMetrics::AUPRC,
        ..Default::default()
    };

    // TODO: Make settings a macro for models.
    let mut classifier: KNeighborsClassifier<String> = KNeighborsClassifier::<String>::new();
    classifier.add_settings(classifier_settings).unwrap();

    classifier.fit(&x_train, &y_train).unwrap();

    //println!("{:#}", classifier.predict_proba(&x_test).unwrap());

    println!("{}", classifier.score(&x_test, &y_test).unwrap());

    let start = Instant::now();

    for _ in 0..100 {
        classifier.score(&x_test, &y_test).unwrap();
    }

    let duration = start.elapsed();

    println!("Execution took: {}ms", duration.as_millis());
    println!("Average: {}ms", duration.as_millis() / 100);

}