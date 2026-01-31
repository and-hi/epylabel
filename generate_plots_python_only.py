"""
Vereinfachtes Script zur Generierung von Plots ohne R-Abhängigkeit.
Verwendet nur Python-Algorithmen (Shapelet, WaveFinder).
"""
import matplotlib
matplotlib.use('Agg')  # Headless backend - keine GUI nötig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from epylabel.labeler import (
    Ensemble,
    ParquetWriter,
    Shapelet,
    SummaryWriter,
    WaveFinder,
)
from epylabel.pipeline import Pipeline
from epylabel.utils import to_wide
from epylabel.metrics import summary


class StandardForm:
    """Transformiert RKI-Daten in das Standard-Format."""
    def __init__(self, col='Inzidenz_7-Tage'):
        self.col = col

    def transform(self, data):
        if 'Bundesland_id' in data.columns:
            data = data.rename(columns={'Bundesland_id': 'location'})
        elif 'Landkreis_id' in data.columns:
            data = data.rename(columns={'Landkreis_id': 'location'})
        else:
            data['location'] = 0
        data = data.rename(columns={'Meldedatum': 'target', self.col: 'value'})
        if 'Altersgruppe' in data.columns:
            data = data.query("Altersgruppe == '00+'")
        data = data[['target', 'location', 'value']]
        data = data.sort_values(['location', 'target'])
        data['target'] = pd.to_datetime(data['target'])
        return to_wide(data)


def load_and_process_data():
    """Lädt COVID-Daten von GitHub und generiert Labels."""
    data_map = {
        "DE": "https://raw.githubusercontent.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/main/COVID-19-Faelle_7-Tage-Inzidenz_Deutschland.csv",
        "BL": "https://raw.githubusercontent.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/main/COVID-19-Faelle_7-Tage-Inzidenz_Bundeslaender.csv",
    }

    # Algorithmen
    wv = WaveFinder(abs_prominence_threshold=5, prominence_height_threshold=0.01, t_sep_a=35)
    sp = Shapelet(n_days=7, x_max=3, thresh=0.8)
    ens = Ensemble(n_min=2)

    results = {}

    for region, url in data_map.items():
        print(f"\nVerarbeite {region}...")
        output_dir = Path(f'output/{region}')
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels').mkdir(exist_ok=True)
        (output_dir / 'summary').mkdir(exist_ok=True)

        # Lade Daten
        data_rki = pd.read_csv(url)
        data_wide = StandardForm().transform(data_rki)

        # Speichere Inzidenz
        incidence_long = data_wide.reset_index().melt(id_vars='target', var_name='location', value_name='value')
        incidence_long.to_parquet(output_dir / 'incidence.parquet')

        # Generiere Labels
        sp_labels = sp.transform(data_wide)
        wv_labels = wv.transform(data_wide)
        ens_labels = ens.transform(sp_labels, wv_labels)

        # Speichere Labels
        for name, labels in [('sp', sp_labels), ('wv', wv_labels), ('sp_wv', ens_labels)]:
            labels_long = labels.reset_index().melt(id_vars='target', var_name='location', value_name='label')
            labels_long.to_parquet(output_dir / f'labels/{name}.parquet')

            # Summary
            summary_df = summary(labels)
            summary_df.to_parquet(output_dir / f'summary/{name}.parquet')

        results[region] = {
            'incidence': incidence_long,
            'sp': sp_labels,
            'wv': wv_labels,
            'sp_wv': ens_labels,
        }

        print(f"  Shapelet: {sp_labels.sum().sum()} Labels")
        print(f"  WaveFinder: {wv_labels.sum().sum()} Labels")
        print(f"  Ensemble: {ens_labels.sum().sum()} Labels")

    return results


def plot_incidence_with_labels(incidence, labels_dict, location, title, output_path):
    """Plottet Inzidenz mit Labels als farbige Bänder."""
    fig, ax1 = plt.subplots(figsize=(12, 4))

    # Inzidenz-Linie
    loc_data = incidence[incidence['location'] == location]
    ax1.plot(loc_data['target'], loc_data['value'], 'k-', linewidth=1)
    ax1.set_ylabel('7-Tage-Inzidenz')
    ax1.set_xlabel('')

    # Labels als Bänder
    algs = list(labels_dict.keys())
    n_algs = len(algs)
    segments = np.linspace(0, 1, n_algs + 1)

    colors = {'True': 'firebrick', 'False': '#dae6f0'}

    for i, (alg_name, labels_df) in enumerate(labels_dict.items()):
        if location in labels_df.columns:
            for date, is_label in labels_df[location].items():
                color = colors['True'] if is_label else colors['False']
                ax1.axvspan(date, date + pd.Timedelta(days=1),
                           ymin=segments[i], ymax=segments[i+1] - 0.01,
                           facecolor=color, alpha=0.8)

    # Rechte Y-Achse für Algorithmen-Labels
    ax2 = ax1.twinx()
    ytick_pos = [(segments[i] + segments[i+1]) / 2 for i in range(n_algs)]
    ax2.set_yticks(ytick_pos)
    ax2.set_yticklabels(algs)
    ax2.set_ylim(0, 1)

    ax1.set_title(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot gespeichert: {output_path}")


def main():
    print("=" * 60)
    print("EPYLABEL - Python-Only Plot Generation")
    print("(Ohne R/BCP-Abhängigkeit)")
    print("=" * 60)

    # Daten laden und verarbeiten
    results = load_and_process_data()

    output_dir = Path('output')

    # Plots generieren
    print("\nGeneriere Plots...")

    # Deutschland-Plot
    plot_incidence_with_labels(
        results['DE']['incidence'],
        {
            'Shapelet': results['DE']['sp'],
            'WaveFinder': results['DE']['wv'],
            'Ensemble': results['DE']['sp_wv'],
        },
        location=0,
        title='COVID-19 Inzidenz Deutschland mit Outbreak-Labels',
        output_path=output_dir / 'plot_DE.png'
    )

    # Bundesländer-Beispiel (Bayern = 9)
    plot_incidence_with_labels(
        results['BL']['incidence'],
        {
            'Shapelet': results['BL']['sp'],
            'WaveFinder': results['BL']['wv'],
            'Ensemble': results['BL']['sp_wv'],
        },
        location=9,
        title='COVID-19 Inzidenz Bayern mit Outbreak-Labels',
        output_path=output_dir / 'plot_Bayern.png'
    )

    # Summary-Statistiken Plot
    print("\nGeneriere Summary-Plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, region in enumerate(['DE', 'BL']):
        ax = axes[idx]
        summary_data = pd.read_parquet(f'output/{region}/summary/sp_wv.parquet')

        if len(summary_data) > 1:
            ax.bar(range(len(summary_data)), summary_data['n_labels'])
            ax.set_xlabel('Location')
            ax.set_ylabel('Anzahl Label-Perioden')
            ax.set_title(f'{region}: Ensemble (SP+WV) Labels pro Region')
        else:
            ax.text(0.5, 0.5, f'{region}: {summary_data["n_labels"].iloc[0]} Label-Perioden',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{region}: Ensemble Summary')

    fig.tight_layout()
    plt.savefig(output_dir / 'summary_plot.png', dpi=150)
    plt.close()
    print(f"  Plot gespeichert: {output_dir / 'summary_plot.png'}")

    print("\n" + "=" * 60)
    print("FERTIG!")
    print(f"Plots gespeichert in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
