
import numpy as np

from ecg_chagas_embeddings.analysis.embeddings_probe import (
    add_ptb_rbbb_flags,
    build_probe_index,
    l2_normalize_np,
    parse_ptb_scp_codes,
)


def test_parse_ptb_scp_codes_valid_dict_string():
    s = "{'CRBBB': 100.0, 'LAFB': 100.0, 'SR': 0.0}"
    d = parse_ptb_scp_codes(s)
    assert isinstance(d, dict)
    assert "CRBBB" in d


def test_parse_ptb_scp_codes_invalid_returns_empty():
    assert parse_ptb_scp_codes("not a dict") == {}
    assert parse_ptb_scp_codes("") == {}
    assert parse_ptb_scp_codes(None) == {}


def test_add_ptb_rbbb_flags_from_scp_codes():
    import pandas as pd

    df = pd.DataFrame(
        {
            "scp_codes": [
                "{'CRBBB': 100.0}",
                "{'IRBBB': 100.0}",
                "{'NORM': 100.0}",
                "{}",
            ]
        }
    )
    out = add_ptb_rbbb_flags(df)
    assert out["ptb_crbbb"].tolist() == [True, False, False, False]
    assert out["ptb_irbbb"].tolist() == [False, True, False, False]
    assert out["ptb_any_rbbb"].tolist() == [True, True, False, False]
    assert out["ptb_normal_ecg"].tolist() == [False, False, True, False]


def test_l2_normalize_np_is_unit_norm_and_safe_for_zeros():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    x_u = l2_normalize_np(x, axis=1, eps=1e-12)
    n0 = float(np.linalg.norm(x_u[0]))
    n1 = float(np.linalg.norm(x_u[1]))
    assert abs(n0 - 1.0) < 1e-6
    assert n1 == 0.0


def test_build_probe_index_respects_negative_source_fracs_when_available():
    import pandas as pd

    rows = []
    # 10 positives, all CODE15.
    for i in range(10):
        rows.append(
            {
                "exam_id": f"p{i}",
                "dataset_source": "CODE15",
                "chagas": 1,
                "has_rbbb": False,
                "normal_ecg": True,
                "age_bin": "30-50",
            }
        )
    # Negatives: 100 CODE15 + 100 PTBXL.
    for i in range(100):
        rows.append(
            {
                "exam_id": f"c{i}",
                "dataset_source": "CODE15",
                "chagas": 0,
                "has_rbbb": False,
                "normal_ecg": True,
                "age_bin": "30-50",
            }
        )
        rows.append(
            {
                "exam_id": f"t{i}",
                "dataset_source": "PTBXL",
                "chagas": 0,
                "has_rbbb": False,
                "normal_ecg": True,
                "age_bin": "30-50",
            }
        )
    df = pd.DataFrame(rows)

    probe = build_probe_index(
        df,
        seed=0,
        neg_multiplier=2,
        neg_source_fracs={"CODE15": 0.5, "PTBXL": 0.5, "SAMITROP": 0.0},
    )
    # 10 pos + 20 neg.
    assert len(probe) == 30
    neg = probe[probe["chagas"] == 0]
    # Expect an even split for negatives (within rounding).
    counts = neg["dataset_source"].value_counts().to_dict()
    assert counts.get("CODE15", 0) == 10
    assert counts.get("PTBXL", 0) == 10
