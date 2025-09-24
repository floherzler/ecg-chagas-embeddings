def test_import_package():
    import ecg_chagas_embeddings

    assert hasattr(ecg_chagas_embeddings, "__version__")
