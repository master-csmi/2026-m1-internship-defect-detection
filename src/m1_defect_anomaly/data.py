import os
import wfdb

DEFAULT_DATA_DIR = "data/raw"

# download the full MIT-BIH Arrhythmia Database
def download_mitbih(dest_dir: str = DEFAULT_DATA_DIR) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir = dest_dir)
    print(f"Downloaded MIT-BIH Arrhythmia Database to '{dest_dir}'")


# function that helps us read the record
def load_record(record_name: str, data_dir: str = DEFAULT_DATA_DIR):
    record_path = os.path.join(data_dir, record_name) # building the path
    record = wfdb.rdrecord(record_path) # read the raw ECG signal
    annotation = wfdb.rdann(record_path,'atr') # read labels
    return record, annotation

if __name__ == "__main__":
    download_mitbih()