from m1_defect_anomaly_2026.data import load_record

def test_load_record_returns_signal_and_annotation():
    record, annotation = load_record("100")
    assert record.fs == 360
    assert len(annotation.symbol)>0


def test_record_has_expected_signal_shape():
    record, _ = load_record("100")
    assert record.sig_len>0
    assert record.n_sig>=1