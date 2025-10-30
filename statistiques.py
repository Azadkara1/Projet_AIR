import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class AnalyseMeteo:
    def __init__(self, df_today, df_last_year):
        self.df_today = df_today
        self.df_last_year = df_last_year

    # --- 1. STATISTIQUES DESCRIPTIVES ---
    def stats_descriptives(self):
        print("\n=== Statistiques descriptives ===")
        desc_today = self.df_today.describe()
        desc_last = self.df_last_year.describe()
        print("Données du jour :\n", desc_today)
        print("\nDonnées de l’an passé :\n", desc_last)
        return desc_today, desc_last

    # --- 2. STATISTIQUES INFÉRENTIELLES ---
    def stats_inferentielles(self):
        print("\n=== Statistiques inférentielles ===")
        # Exemple : test t pour comparer les températures entre les deux jours
        t_stat, p_value = stats.ttest_ind(self.df_today['temperature_2m'],
                                          self.df_last_year['temperature_2m'],
                                          equal_var=False)
        print(f"Test t température → t={t_stat:.3f}, p={p_value:.3f}")
        return t_stat, p_value

    # --- 3. ANALYSE MULTIVARIÉE ---
    def analyse_multivariee(self):
        print("\n=== Analyse multivariée (ACP) ===")
        variables = ['temperature_2m', 'pressure_msl', 'relative_humidity_2m', 'wind_speed_10m']
        df_combined = pd.concat([self.df_today[variables], self.df_last_year[variables]], axis=0)
        df_scaled = StandardScaler().fit_transform(df_combined)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(df_scaled)
        explained = pca.explained_variance_ratio_
        print(f"Variance expliquée : {explained}")
        return components, explained
    

