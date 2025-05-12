use std::{collections::BTreeMap, io::{BufRead, BufReader}, usize};

use async_trait::async_trait;

use crate::{data_matrix::DataMatrix, data_vector::DataVector, enums::error_types::DataLoaderError};

use super::{data_loader_settings::DataLoaderSettings, DataLoader};

pub struct CSVLoader {
    pub settings: CSVLoaderSettings
}

pub struct CSVLoaderSettings {
    pub separator: char,
    pub header_names: Vec<String>,
    pub header_indices: Vec<usize>,
    pub index_col: Option<usize>,
    pub skip_initial_space: bool,
    pub skip_rows: usize,
    pub skip_footer: usize,
    pub n_rows: usize,
    pub skip_blank_lines: bool
}

impl Default for CSVLoaderSettings {
    fn default() -> Self {
        Self {
            separator: ',',
            header_names: Vec::new(),
            header_indices: Vec::new(),
            index_col: None,
            skip_initial_space: true,
            skip_rows: 0,
            skip_footer: 0,
            n_rows: usize::MAX,
            skip_blank_lines: true,
        }
    }
}

impl DataLoaderSettings for CSVLoaderSettings {}

impl CSVLoader {
    pub fn new(settings: CSVLoaderSettings) -> Self {
        CSVLoader { 
            settings
        }
    }

    fn get_headers(&self, lines: &Vec<String>) -> Result<Vec<String>, DataLoaderError> {
        if self.settings.header_names.len() >= 1 {
            let first_line: &str = lines.get(0).ok_or(DataLoaderError::GenericError("CSV file contains no data".to_string()))?;
            let column_count: usize = first_line.split(self.settings.separator).count();
            if column_count != self.settings.header_names.len() {
                return Err(DataLoaderError::ColumnCountMismatch(format!("header_names property had {} values and the csv file had {} columns", self.settings.header_names.len(), column_count)));
            }
            Ok(self.settings.header_names.clone())
        }
        else if self.settings.header_indices.len() >= 1 {
            let first_line: &str = lines.get(0).ok_or(DataLoaderError::GenericError("CSV file contains no data".to_string()))?;
            let headers: Vec<&str> = first_line.split(self.settings.separator).collect::<Vec<&str>>();
            let header_count = headers.len();
            if header_count != self.settings.header_indices.len() {
                return Err(DataLoaderError::ColumnCountMismatch(format!("header_indices property had {} values and the csv file had {} columns", self.settings.header_names.len(), header_count)));
            }

            let mut reordered_headers: Vec<&str> = headers.clone();
            for (header_index, header_value) in self.settings.header_indices.iter().zip(headers) {
                reordered_headers.insert(*header_index, header_value);
            }

            Ok(reordered_headers.iter().map(|&h| h.to_owned()).collect::<Vec<String>>())
        }
        else {
            let first_line: &str = lines.get(0).ok_or(DataLoaderError::GenericError("CSV file contains no data".to_string()))?;
            let headers: Vec<&str> = first_line.split(self.settings.separator).collect::<Vec<&str>>();
            Ok(headers.iter().map(|&h| h.to_owned()).collect::<Vec<String>>())
        }
    }

    fn try_parse<T: std::str::FromStr>(&self, values: &[String]) -> Option<Vec<T>> {
        let mut out = Vec::with_capacity(values.len());
        for v in values {
            match v.parse::<T>() {
                Ok(parsed) => out.push(parsed),
                Err(_) => return None,
            }
        }
        Some(out)
    }
}

#[async_trait]
impl DataLoader for CSVLoader {
    async fn load_from<'a>(&'a self, path: &'a str) -> Result<DataMatrix, DataLoaderError> {
        let file: std::fs::File = std::fs::File::open(path).map_err(|e| DataLoaderError::FileRead(e.to_string()))?;
        let reader: BufReader<std::fs::File> = BufReader::new(file);
        let mut lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>().map_err(|e| DataLoaderError::GenericError(e.to_string()))?;

        let headers: Vec<String> = self.get_headers(&lines)?;

        if self.settings.header_names.len() == 0 {
            lines.remove(0);
        }

        let mut raw_columns: BTreeMap<String, Vec<String>> = headers
            .iter()
            .map(|h| (h.clone(), Vec::new()))
            .collect();

        let mut index = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            let line = if line.trim().is_empty() {
                if self.settings.skip_blank_lines {
                    continue;
                } else {
                    format!("NA{}", self.settings.separator).repeat(headers.len())
                }
            } else {
                line.to_string()
            };
            let fields: Vec<&str> = line.split(self.settings.separator).map(|s| s.trim()).collect();
            if fields.len() != headers.len() {
                return Err(DataLoaderError::ColumnCountMismatch("There is a different number of columns in two or more rows.".to_string()));
            }

            index.push(format!("{}", i));

            for (header, field) in headers.iter().zip(fields.iter()) {
                raw_columns.get_mut(header).unwrap().push(field.to_string());
            }
        }

        let mut columns = BTreeMap::new();

        for (header, values) in raw_columns {
            let values = values[..values.len().saturating_sub(self.settings.skip_rows)].iter()
                .take(values.len().saturating_sub(self.settings.skip_footer))
                .take(self.settings.n_rows)
                .map(|s| s.clone())
                .collect::<Vec<String>>();

            let row_count = values.len();

            if values.iter().all(|v| v == "true" || v == "false") {
                let parsed: Vec<bool> = values.iter().map(|v| v == "true").collect();
                columns.insert(
                    header.clone(),
                    DataVector {
                        label: Some(header),
                        data: Box::new(parsed),
                        len: row_count,
                        dtype: Some("bool"),
                    },
                );
                continue;
            }

            if let Some(parsed) = self.try_parse::<f64>(&values) {
                columns.insert(
                    header.clone(),
                    DataVector {
                        label: Some(header),
                        data: Box::new(parsed),
                        len: row_count,
                        dtype: Some("f64"),
                    },
                );
                continue;
            }

            columns.insert(
                header.clone(),
                DataVector {
                    label: Some(header),
                    data: Box::new(values),
                    len: row_count,
                    dtype: Some("String"),
                },
            );
        }

        Ok(DataMatrix { columns, index })
    }
}