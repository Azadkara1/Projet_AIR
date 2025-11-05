import numpy as np
from services.analytics.pca import acp_temperature

def test_acp_shapes_and_explained(multi_year_df):
    start = "2023-06-01"; end = "2023-08-31"
    df_pcs, loadings, explained = acp_temperature(multi_year_df, start, end)
    assert "temperature_2m_mean" in df_pcs.columns and "date" in df_pcs.columns
    assert loadings.shape[0] >= 4
    assert np.isclose(np.sum(explained), 1.0, atol=1e-6)
