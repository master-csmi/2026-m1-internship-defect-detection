DS1_train = ["101", "106", "108", "109", "112","114", "115", "116", "118", "119", "122",
             "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2_test = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", 
            "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"] 


PACED_EXCLUDED = ["102","104","107","217"]


BEAT_SYMBOLS = set("NLRejAaJSVEFP/fQ")

AAMI_MAP = {# N: normal heart beat
            "N": "N", "L": "N","R":"N","e":"N","j":"N",
            # S: supraventicular ectopic
            "A":"S","a":"S","J":"S", "S":"S",
            # V: ventricular ectopic
            "V":"V", "E":"V",
            # F: fusion
            "F":"F",
            # Q: unknown/paced/ unclassifiable
            "/":"Q", "f":"Q","Q":"Q", "P":"Q"
}
AAMI_CLASSES = ["N","S","V","F","Q"]

def check_split():
    assert not set(DS1_train) & set(DS2_test), " a record is in both sets!"
    assert not (set(DS1_train) | set(DS2_test)) & set(PACED_EXCLUDED), "a paced record leaked in!"
    assert len(DS1_train)==22 and len(DS2_test)==22
    assert not (BEAT_SYMBOLS - set(AAMI_MAP)), "a beat symbol has no AAMI mapping"
    assert set(AAMI_MAP.values()) <= set(AAMI_CLASSES), "AAMI_MAP produces an unknown class"