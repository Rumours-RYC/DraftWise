PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    columns_json TEXT NOT NULL,
    dtypes_json TEXT NOT NULL,
    import_note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS dataset_rows (
    dataset_id TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    row_json TEXT NOT NULL,
    PRIMARY KEY (dataset_id, row_index),
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_dataset_rows_dataset_id
    ON dataset_rows(dataset_id, row_index);
