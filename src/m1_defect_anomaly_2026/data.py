import os
import wfdb
import time

DEFAULT_DATA_DIR = "data/raw"

# download the full MIT-BIH Arrhythmia Database
def download_mitbih(dest_dir=DEFAULT_DATA_DIR, records=None, retries=3, delay=10):
    os.makedirs(dest_dir, exist_ok=True)
    for attempt in range(retries):
        try:
            wfdb.dl_database("mitdb", dl_dir=dest_dir, records=records)
            print(f"Downloaded MIT-BIH Arrhythmia Database to '{dest_dir}'")
            return
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"Download failed ({e}), retrying in {delay}s...")
            time.sleep(delay)


# function that helps us read the record
def load_record(record_name: str, data_dir: str = DEFAULT_DATA_DIR):
    record_path = os.path.join(data_dir, record_name) # building the path
    record = wfdb.rdrecord(record_path) # read the raw ECG signal
    annotation = wfdb.rdann(record_path,'atr') # read labels
    return record, annotation

if __name__ == "__main__":
    download_mitbih()