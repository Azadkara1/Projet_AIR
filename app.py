import os
import io
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import statistiques as stats_mod  
except Exception:
    stats_mod = None

try:
    import main_stats as main_stats_mod  
except Exception:
    main_stats_mod = None

st.set_page_config(page_title="Projet_AIR — Dashboard", layout="wide")
st.title("🌬️ Projet_AIR — Interface interactive")
st.caption("MVP Streamlit pour explorer les données, calculer des statistiques et visualiser les résultats.")

# -----------------------------
# Sidebar: Data source
# -----------------------------
st.sidebar.header("📦 Source de données")
repo_root = Path.cwd()

# Option 1: Upload CSV
uploaded = st.sidebar.file_uploader("Téléverse un fichier CSV", type=["csv"]) 

# Option 2: Pick a local CSV from DataCollecte if the folder exists
candidate_paths = []
data_folder = repo_root / "DataCollecte"
if data_folder.exists() and data_folder.is_dir():
    for p in sorted(data_folder.rglob("*.csv")):
        candidate_paths.append(p)

selected_local = None
if candidate_paths:
    selected_label = st.sidebar.selectbox(
        "…ou choisis un CSV existant (DataCollecte)",
        ["(aucun)"] + [str(p.relative_to(repo_root)) for p in candidate_paths],
        index=0,
    )
    if selected_label != "(aucun)":
        selected_local = repo_root / selected_label

# Load df
@st.cache_data(show_spinner=False)
def load_csv(_file: io.BytesIO | str | Path) -> pd.DataFrame:
    return pd.read_csv(_file)

_df = None
load_note = None
if uploaded is not None:
    _df = load_csv(uploaded)
    load_note = "Chargé depuis l'upload."
elif selected_local is not None:
    _df = load_csv(selected_local)
    load_note = f"Chargé depuis {selected_local}."
else:
    st.info("➡️ Téléverse un CSV à gauche ou place des fichiers dans `DataCollecte/`.")

# -----------------------------
# Data preview
# -----------------------------
if _df is not None:
    st.subheader("👀 Aperçu des données")
    st.write(load_note)
    st.dataframe(_df.head(50), use_container_width=True)

    # Basic info
    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.metric("Lignes", len(_df))
    with c2:
        st.metric("Colonnes", _df.shape[1])
    with c3:
        st.metric("Valeurs manquantes", int(_df.isna().sum().sum()))

    st.markdown("---")
    st.subheader("📊 Statistiques descriptives")
    with st.expander("Voir les stats numériques"):
        st.dataframe(_df.describe(include=[np.number]).T, use_container_width=True)
    with st.expander("Voir les stats catégorielles"):
        cat_cols = _df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            summary = {
                col: {
                    "n_unique": _df[col].nunique(),
                    "top": _df[col].value_counts(dropna=False).head(5).to_dict(),
                }
                for col in cat_cols
            }
            st.json(summary)
        else:
            st.info("Aucune colonne catégorielle détectée.")

    # -----------------------------
    # Visualisations
    # -----------------------------
    st.markdown("---")
    st.subheader("📈 Visualisations rapides")

    num_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        # Histogram for a selected numeric column
        col_choice = st.selectbox("Choisis une colonne numérique", num_cols)
        bins = st.slider("Nombre de bacs (histogramme)", 5, 100, 30)
        fig, ax = plt.subplots()
        _df[col_choice].dropna().hist(bins=bins, ax=ax)
        ax.set_title(f"Histogramme — {col_choice}")
        st.pyplot(fig, use_container_width=True)

        # Correlation heatmap
        if len(num_cols) >= 2:
            fig2, ax2 = plt.subplots()
            corr = _df[num_cols].corr(numeric_only=True)
            cax = ax2.imshow(corr, aspect='auto')
            ax2.set_xticks(range(len(num_cols)))
            ax2.set_xticklabels(num_cols, rotation=45, ha='right')
            ax2.set_yticks(range(len(num_cols)))
            ax2.set_yticklabels(num_cols)
            ax2.set_title("Matrice de corrélation")
            fig2.colorbar(cax)
            st.pyplot(fig2, use_container_width=True)
    else:
        st.info("Aucune colonne numérique détectée pour les graphes.")

    # -----------------------------
    # Hooks vers vos scripts du repo
    # -----------------------------
    st.markdown("---")
    st.subheader("🧩 Intégration avec les scripts du dépôt")

    if stats_mod is not None:
        st.success("Module `statistiques.py` détecté.")
        # Si votre fichier expose une fonction haut-niveau, on essaie de la trouver :
        candidate_fn_names = [
            "compute_statistics",
            "run",
            "main",
        ]
        picked = None
        for name in candidate_fn_names:
            if hasattr(stats_mod, name):
                picked = getattr(stats_mod, name)
                break
        if picked is not None:
            if st.button(f"Exécuter statistiques.py → {picked.__name__}()"):
                try:
                    res = picked()  # idéalement, rendez vos fonctions pures et paramétrables
                    st.success("Exécution terminée.")
                    if res is not None:
                        st.write(res)
                except Exception as e:
                    st.error(f"Erreur pendant l'exécution de {picked.__name__}: {e}")
        else:
            st.info("Aucune fonction publique trouvée automatiquement dans `statistiques.py`. Exposez une fonction (ex: `def compute_statistics(df): ...`).")
    else:
        st.warning("Module `statistiques.py` introuvable/import impossible.")

    if main_stats_mod is not None:
        st.success("Module `main_stats.py` détecté.")
        candidate_fn_names = ["main", "run"]
        picked2 = None
        for name in candidate_fn_names:
            if hasattr(main_stats_mod, name):
                picked2 = getattr(main_stats_mod, name)
                break
        if picked2 is not None:
            if st.button(f"Exécuter main_stats.py → {picked2.__name__}()"):
                try:
                    res2 = picked2()
                    st.success("Exécution terminée.")
                    if res2 is not None:
                        st.write(res2)
                except Exception as e:
                    st.error(f"Erreur pendant l'exécution de {picked2.__name__}: {e}")
        else:
            st.info("Exposez une fonction `main()` ou `run()` dans `main_stats.py` pour l'appeler depuis l'interface.")
    else:
        st.warning("Module `main_stats.py` introuvable/import impossible.")

    # -----------------------------
    # Export
    # -----------------------------
    st.markdown("---")
    st.subheader("💾 Export")
    csv_name = st.text_input("Nom du CSV exporté", value="export_projet_air.csv")
    if st.button("Exporter le DataFrame visible en CSV"):
        try:
            st.download_button(
                label="Télécharger le CSV",
                data=_df.to_csv(index=False).encode("utf-8"),
                file_name=csv_name,
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Export impossible : {e}")
else:
    st.stop()

st.caption("Astuce : refactorisez vos scripts pour exposer des fonctions réutilisables (ex: `compute_statistics(df)`) afin d'enrichir l'interface.")
