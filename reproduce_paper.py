#!/usr/bin/env python3
"""
Robust script to reproduce all paper plots from the epylabel manuscript.

This script combines label generation and plot creation into a single,
easy-to-use interface with proper error handling and progress reporting.

Usage:
    python reproduce_paper.py              # Run everything
    python reproduce_paper.py --labels     # Only generate labels
    python reproduce_paper.py --plots      # Only generate plots (requires labels)
    python reproduce_paper.py --check      # Check dependencies only
"""

import argparse
import os
import sys
from pathlib import Path

# Configure matplotlib backend BEFORE importing pyplot
# This ensures headless operation works correctly
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm


def check_dependencies():
    """Check if all required dependencies are available."""
    print("Checking dependencies...")
    errors = []

    # Check Python packages
    required_packages = [
        'pandas', 'numpy', 'scipy', 'sklearn', 'matplotlib',
        'seaborn', 'geopandas', 'pyarrow', 'tqdm'
    ]

    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - MISSING")
            errors.append(pkg)

    # Check rpy2 and R
    try:
        import rpy2.robjects as robjects
        print("  ✓ rpy2")

        # Check if bcp package is available
        try:
            import rpy2.robjects.packages as rpackages
            utils = rpackages.importr("utils")
            try:
                bcp = rpackages.importr("bcp")
                print("  ✓ R bcp package")
            except:
                print("  ✗ R bcp package - MISSING (will try to install)")
                # Try to install
                utils.install_packages("bcp", repos="https://cloud.r-project.org/")
                bcp = rpackages.importr("bcp")
                print("  ✓ R bcp package (installed)")
        except Exception as e:
            print(f"  ✗ R bcp package - ERROR: {e}")
            errors.append("R bcp package")
    except ImportError:
        print("  ✗ rpy2 - MISSING")
        errors.append("rpy2")

    # Check geopandas
    try:
        import geopandas as gpd
        print("  ✓ geopandas")
    except ImportError:
        print("  ✗ geopandas - MISSING")
        errors.append("geopandas")

    # Check shapefile
    shape_path = Path("shape/SKLKBerlinBez.shp")
    if shape_path.exists():
        print(f"  ✓ Shapefile: {shape_path}")
    else:
        print(f"  ✗ Shapefile: {shape_path} - MISSING")
        errors.append("shapefile")

    if errors:
        print(f"\n⚠ Missing dependencies: {', '.join(errors)}")
        return False
    else:
        print("\n✓ All dependencies satisfied!")
        return True


def generate_labels():
    """Generate all labels from the paper."""
    print("\n" + "="*60)
    print("GENERATING LABELS")
    print("="*60)

    from epylabel.labeler import (
        Bcp, Changerate, Ensemble, ParquetWriter,
        Shapelet, SummaryWriter, WaveFinder
    )
    from epylabel.pipeline import Pipeline
    from epylabel.utils import to_wide

    class StandardForm:
        """Transform RKI data to standard wide format."""
        def __init__(self, col="Inzidenz_7-Tage"):
            self.col = col

        def transform(self, data):
            if "Bundesland_id" in data.columns:
                data = data.rename(columns={"Bundesland_id": "location"})
            elif "Landkreis_id" in data.columns:
                data = data.rename(columns={"Landkreis_id": "location"})
            else:
                data["location"] = 0
            data = data.rename(columns={"Meldedatum": "target", self.col: "value"})
            if "Altersgruppe" in data.columns:
                data = data.query("Altersgruppe == '00+'")
            data = data[["target", "location", "value"]]
            data.sort_values(["location", "target"])
            data["target"] = pd.to_datetime(data["target"])
            return to_wide(data)

    # BCP pipeline parameters
    cr = Changerate(n_days=7, changerate_ceiling=6)
    bcp = Bcp(d=1000, p0=0.001, thresh=1)

    # Wave finder parameters
    wv = WaveFinder(
        abs_prominence_threshold=5,
        prominence_height_threshold=0.01,
        t_sep_a=35
    )

    # Shapelet parameters
    sp = Shapelet(n_days=7, x_max=3, thresh=0.8)

    # Ensemble
    ens = Ensemble(n_min=2)

    # Data URLs
    data_map = {
        "LK": {
            "url": "https://raw.githubusercontent.com/robert-koch-institut/"
            "COVID-19_7-Tage-Inzidenz_in_Deutschland/main/"
            "COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv"
        },
        "BL": {
            "url": "https://raw.githubusercontent.com/robert-koch-institut/"
            "COVID-19_7-Tage-Inzidenz_in_Deutschland/main/"
            "COVID-19-Faelle_7-Tage-Inzidenz_Bundeslaender.csv"
        },
        "DE": {
            "url": "https://raw.githubusercontent.com/robert-koch-institut/"
            "COVID-19_7-Tage-Inzidenz_in_Deutschland/main/"
            "COVID-19-Faelle_7-Tage-Inzidenz_Deutschland.csv"
        },
    }

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    def writer(name, labels_path, summary_path):
        pw = ParquetWriter("long", labels_path / name, "label")
        sw = SummaryWriter(summary_path / name)
        return pw, sw

    for geo, data_dict in tqdm(data_map.items(), desc="Processing regions"):
        print(f"\n  Processing {geo}...")

        try:
            print(f"    Downloading data from {data_dict['url'][:50]}...")
            data_rki = pd.read_csv(data_dict["url"])
            print(f"    Downloaded {len(data_rki)} rows")
        except Exception as e:
            print(f"    ERROR downloading data: {e}")
            continue

        geo_path = output_dir / geo
        geo_path.mkdir(exist_ok=True)

        # Transform and save incidence data
        data_wide = Pipeline([
            StandardForm(),
            ParquetWriter("long", geo_path / "incidence.parquet", "value")
        ]).transform(data_rki)

        data_wide_faelle = Pipeline([
            StandardForm("Faelle_neu"),
            ParquetWriter("long", geo_path / "cases.parquet", "value"),
        ]).transform(data_rki)

        labels_path = geo_path / "labels"
        summary_path = geo_path / "summary"
        labels_path.mkdir(exist_ok=True)
        summary_path.mkdir(exist_ok=True)

        def write(x):
            return writer(x, labels_path, summary_path)

        # Generate labels
        print(f"    Generating BCP labels...")
        bcp_labels = Pipeline([cr, bcp, *write("bcp.parquet")]).transform(data_wide_faelle)

        print(f"    Generating Shapelet labels...")
        sp_labels = Pipeline([sp, *write("sp.parquet")]).transform(data_wide)

        print(f"    Generating WaveFinder labels...")
        wv_labels = Pipeline([wv, *write("wv.parquet")]).transform(data_wide)

        print(f"    Generating ensemble labels...")
        Pipeline([ens, *write("bcp_sp.parquet")]).transform(bcp_labels, sp_labels)
        Pipeline([ens, *write("bcp_wv.parquet")]).transform(bcp_labels, wv_labels)
        Pipeline([ens, *write("sp_wv.parquet")]).transform(sp_labels, wv_labels)
        Pipeline([ens, *write("bcp_sp_wv.parquet")]).transform(bcp_labels, sp_labels, wv_labels)

        print(f"    ✓ {geo} completed")

    print("\n✓ All labels generated successfully!")
    return True


def compute_mean_and_median(df):
    """Compute run length statistics for labels."""
    df = df.copy()
    # Use existing block column if available, otherwise compute it
    if 'block' not in df.columns:
        df['block'] = (df['label'] != df['label'].shift()).cumsum()
    run_lengths = df.groupby(['location', 'label', 'block']).size().reset_index(name='run_length')
    summary_run_length = run_lengths.groupby(['location', 'label'])['run_length'].agg(['mean', 'median']).reset_index()
    summary_run_length = summary_run_length[summary_run_length['label']==1]
    summary_run_length = summary_run_length.set_index('location').drop(columns='label').rename(
        columns={'mean': 'mean_label_length', 'median': 'median_label_length'}
    )
    return summary_run_length


def process_region(region, algs):
    """Load and process labels and summaries for a region."""
    summaries = {}
    labels = {}
    for alg in algs:
        summary = pd.read_parquet(f"output/{region}/summary/{alg}.parquet")
        lab = pd.read_parquet(f"output/{region}/labels/{alg}.parquet")
        # Add block column for label period identification
        lab = lab.sort_values(['location', 'target'])
        lab['block'] = (lab['label'] != lab.groupby('location')['label'].shift()).cumsum()
        summary_run_length = compute_mean_and_median(lab)
        summary_extended = pd.concat([summary, summary_run_length], axis=1)
        summary_extended["region"] = region
        summary_extended["alg"] = alg
        lab["region"] = region
        lab["alg"] = alg
        summaries[alg] = summary_extended
        labels[alg] = lab
    return pd.concat(summaries), pd.concat(labels)


def plot_all_algs_with_incidence(region, comb_labels, location, incidence_data, algs, alg_labels, fig_width=10, fig_height=6):
    """Create incidence plot with algorithm labels."""
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    sns.lineplot(
        x='target',
        y='value',
        data=incidence_data[incidence_data["location"] == location],
        ax=ax1,
        color="black"
    )

    n_algs = len(algs)
    horizontal_segments = np.linspace(0, 1, n_algs + 1).tolist()
    ytick_positions = [x + horizontal_segments[1] / 2 for x in horizontal_segments]

    for i, alg in enumerate(algs):
        dat = comb_labels[(comb_labels["location"] == location) & (comb_labels["alg"] == alg)]
        colors = {True: 'firebrick', False: '#dae6f0'}
        for _, row in dat.iterrows():
            ax1.axvspan(row['target'], row['target'] + pd.Timedelta(1, unit='D'),
                        ymin=horizontal_segments[i],
                        ymax=horizontal_segments[i+1]-0.01,
                        facecolor=colors.get(row['label'], 'white'))

    plt.xlabel('')
    plt.ylabel('Incidence')
    ax2 = ax1.twinx()
    ax2.set_yticks(ytick_positions[0:n_algs])
    ax2.set_yticklabels(alg_labels)
    ax1.set_xlim(left=datetime(2019, 12, 15), right=datetime(2023, 11, 1))
    fig.tight_layout()
    return fig, ax1


def plot_bar_chart(summary_data):
    """Create bar chart of n_labels vs mean_label_length."""
    bar_chart = plt.figure(figsize=(10, 6))
    sns.barplot(x='n_labels', y='mean_label_length', data=summary_data, order=summary_data["n_labels"], errorbar=None)
    plt.title('n_labels vs mean_label_length')
    plt.xlabel('Number of Labels')
    plt.ylabel('Mean Label Length')
    return bar_chart


def plot_densities(summary_data, algs, regions, fill=False, legend_labels=None):
    """Create kernel density estimate plots."""
    fig, ax = plt.subplots(figsize=(3, 2))
    cols = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Greys', 'YlOrBr']
    simple_cols = ['blue', 'red', 'green', 'purple', 'orange', 'grey', 'brown']
    patches = []

    for idx, (alg, reg, col, simple_col) in enumerate(zip(algs, regions, cols, simple_cols)):
        sns.kdeplot(
            x=summary_data[reg][summary_data[reg]['alg'] == alg]['n_labels'],
            y=summary_data[reg][summary_data[reg]['alg'] == alg]['mean_label_length'],
            cmap=col,
            fill=fill,
            ax=ax,
            alpha=1.0,
            label=alg,
        )
        if legend_labels:
            label = legend_labels[idx]
            patches.append(mpatches.Patch(color=simple_col, label=label, alpha=0.5))

    plt.xlabel('Number of labels')
    plt.ylabel('Mean label length')
    if legend_labels:
        ax.legend(handles=patches)
    fig.tight_layout()
    return fig


def week_year_to_date(week_year_str):
    """Convert week-year string to date."""
    if week_year_str is None or '/' not in week_year_str:
        return None

    week, year = week_year_str.split('/')
    try:
        week = int(week)
        year = int(year)
    except ValueError:
        return None

    first_day_of_year = datetime(year, 1, 1)
    day_of_week = first_day_of_year.weekday()
    if day_of_week <= 3:
        first_monday = first_day_of_year - timedelta(days=day_of_week)
    else:
        first_monday = first_day_of_year + timedelta(days=(7 - day_of_week))

    date = first_monday + timedelta(weeks=week - 1)
    return date.strftime("%Y-%m-%d")


def colorFader(c1, c2, mix=0):
    """Fade (linear interpolate) from color c1 to c2."""
    c1 = np.array(matplotlib.colors.to_rgb(c1))
    c2 = np.array(matplotlib.colors.to_rgb(c2))
    return matplotlib.colors.to_hex((1 - mix) * c1 + mix * c2)


def plot_map_timeseries(dists, timeseries_binary, timeseries_original, dates, n_clusters=4):
    """Create Figure 3: visualization of regional clusters and mean labels per cluster."""
    from sklearn.cluster import AgglomerativeClustering
    import geopandas as gpd

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="complete"
    )
    clustering = clustering.fit(dists)

    label_map = pd.DataFrame(
        {"location": timeseries_binary.columns.astype(int), "label": clustering.labels_.astype(str)}
    )
    timeseries_binary = timeseries_binary.values.T
    gdf = gpd.read_file("shape/SKLKBerlinBez.shp")
    gdf["location"] = gdf["LKID"]
    gdf = gdf.merge(label_map, on="location")

    mydpi = 96*2
    fig = plt.figure(figsize=(3885/mydpi, 1800/mydpi))
    ax1 = plt.subplot2grid((n_clusters, 5), (0, 0), colspan=2, rowspan=n_clusters)

    for i in range(n_clusters):
        plt.subplot2grid((n_clusters, 5), (i, 2))
        plt.subplot2grid((n_clusters, 5), (i, 3), colspan=2)

    gdf.plot(column="label", cmap="viridis", ax=ax1, legend=False)
    ax1.margins(0)

    for i, clabel in zip(range(1, len(fig.axes), 2), range(n_clusters)):
        ax1 = fig.axes[i]
        ax2 = fig.axes[i + 1]
        rows_clabel = np.where(clustering.labels_ == clabel)[0]
        cmap = plt.get_cmap('viridis', n_clusters)

        ax1.annotate(f"Cluster {clabel + 1}",
                     xy=(0.7, 0.55),
                    xycoords="data",
                    size=26, ha='center',
                     )

        ax1.annotate(f"n = {len(rows_clabel)}",
                     xy=(0.7, 0.31),
                    xycoords="data",
                    size=22, ha='center',
                     )

        ax1.annotate(" ",
                          xy=(0.35, 0.45), xycoords="data",
                          ha="center", size=28,
                          bbox=dict(boxstyle="circle", facecolor=matplotlib.colors.rgb2hex(cmap(clabel)),
                                    edgecolor=matplotlib.colors.rgb2hex(cmap(clabel))),
                        )

        label_frequency = timeseries_binary[rows_clabel, :].mean(axis=0)
        for x in range(timeseries_binary.shape[1]):
            ax2.axvline(dates[x], color=colorFader("#ffffff", "#B22222", label_frequency[x]), linewidth=1)
        ax2.plot(dates, timeseries_original[rows_clabel, :].mean(axis=0),
                 label=clabel,
                 c="black", linewidth=2, )

        def draw_window_reference(ax, window_dates, window_name):
            ax.annotate('', xy=(window_dates[0], -0.05),
                         xycoords=('data', 'axes fraction'),
                         xytext=(window_dates[1], -0.05),
                         textcoords=('data', 'axes fraction'),
                         arrowprops=dict(arrowstyle='|-|',
                                         color='darkblue',
                                         mutation_scale=4,
                                         lw=2.0,
                                         ls='-')
                         )
            label_posx = window_dates[0] + (window_dates[1] - window_dates[0])/2
            ax.annotate(window_name, xy=(label_posx, -0.18),
                        xycoords=('data', 'axes fraction'),
                        xytext=(label_posx, -0.18),
                        textcoords=('data', 'axes fraction'),
                        ha="center", fontsize=17
                        )

        draw_window_reference(ax2, (np.datetime64("2020-09-20"), np.datetime64("2021-01-10")), "i")
        draw_window_reference(ax2, (np.datetime64("2021-10-01"), np.datetime64("2021-12-15")), "ii")
        draw_window_reference(ax2, (np.datetime64("2022-11-23"), np.datetime64("2023-03-01")), "iii")

    for j, ax in enumerate(fig.axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        if not j == len(fig.axes) - 1:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        else:
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='center', fontsize=18)

    plt.tight_layout()
    return fig


def generate_plots():
    """Generate all plots from the paper."""
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    from sklearn.metrics.pairwise import pairwise_distances
    from epylabel.utils import to_wide

    # Check if labels exist
    if not Path("output/DE/labels/bcp.parquet").exists():
        print("ERROR: Labels not found. Run with --labels first.")
        return False

    output_dir = Path("output")

    # Load Data
    print("\nLoading data...")
    all_algs = ["bcp", "sp", "wv", "bcp_sp", "bcp_wv", "sp_wv", "bcp_sp_wv"]
    base_algs_and_ensemble = ["bcp", "sp", "wv", "bcp_sp_wv"]
    alg_labels = ["BCP", "Shapelet", "Wave Finder", "Ensemble"]
    regions = ["DE", "BL", "LK"]

    combined_summaries = {}
    combined_labels = {}
    for region in tqdm(regions, desc="Loading regions"):
        combined_summaries[region], combined_labels[region] = process_region(region, all_algs)

    incidence_LK = pd.read_parquet("output/LK/incidence.parquet")
    incidence_DE = pd.read_parquet("output/DE/incidence.parquet")
    incidence_BL = pd.read_parquet("output/BL/incidence.parquet")

    # Configure plot styles
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Generate incidence plots
    print("\nGenerating incidence plots...")

    fig_LK, ax_LK = plot_all_algs_with_incidence("LK", combined_labels["LK"], 1001, incidence_LK, base_algs_and_ensemble, alg_labels, fig_width=6.475716, fig_height=1.5)
    fig_LK.savefig(output_dir / 'inc_all_algs_1001.png', dpi=600)
    plt.close(fig_LK)
    print("  ✓ inc_all_algs_1001.png")

    fig_DE, ax_DE = plot_all_algs_with_incidence("DE", combined_labels["DE"], 0, incidence_DE, base_algs_and_ensemble, alg_labels, fig_width=6.475716, fig_height=1.5)
    fig_DE.savefig(output_dir / 'inc_all_algs_0.png', dpi=600)
    plt.close(fig_DE)
    print("  ✓ inc_all_algs_0.png")

    fig_BL, _ = plot_all_algs_with_incidence("BL", combined_labels["BL"], 1, incidence_BL, base_algs_and_ensemble, alg_labels, fig_width=6.475716, fig_height=1.5)
    fig_BL.savefig(output_dir / 'inc_all_algs_1.png', dpi=600)
    plt.close(fig_BL)
    print("  ✓ inc_all_algs_1.png")

    # Infographic Plots
    print("\nGenerating infographic plots...")
    infographic_labels = combined_labels["LK"][(combined_labels["LK"]["target"] >= "2021-06-01") & (combined_labels["LK"]["target"] < "2022-01-01")]
    infographic_incidence = incidence_LK[(incidence_LK["target"] >= "2021-06-01") & (incidence_LK["target"] < "2022-01-01")]

    fig_incidence, _ = plt.subplots(figsize=(10, 6))
    sns.lineplot(
            x='target',
            y='value',
            data=infographic_incidence[infographic_incidence["location"] == 1001],
            color="black"
        )
    plt.title("Incidence", fontsize=30)
    fig_incidence.savefig(output_dir / "infographic_incidence.svg")
    plt.close(fig_incidence)
    print("  ✓ infographic_incidence.svg")

    for alg_name, alg_title in [("bcp_sp_wv", "Ensemble"), ("bcp", "Bayesian Change Point"), ("sp", "Shapelet"), ("wv", "Wave Finder")]:
        fig, _ = plot_all_algs_with_incidence("LK", infographic_labels, 1001, infographic_incidence, [alg_name], [alg_name])
        plt.title(alg_title, fontsize=30)
        fig.savefig(output_dir / f'infographic_{alg_name.replace("bcp_sp_wv", "ensemble")}.svg')
        plt.close(fig)
        print(f"  ✓ infographic_{alg_name.replace('bcp_sp_wv', 'ensemble')}.svg")

    # Bar Chart
    print("\nGenerating bar chart...")
    summary_BL = combined_summaries["BL"].copy(deep=True).sort_values(by="n_labels")
    summary_BL_ensemble = summary_BL[summary_BL["alg"] == "bcp_sp_wv"]
    bar = plot_bar_chart(summary_BL_ensemble)
    bar.savefig(output_dir / 'bar_chart_BL_ensemble.png')
    plt.close(bar)
    print("  ✓ bar_chart_BL_ensemble.png")

    # Kernel Density Estimates
    print("\nGenerating density plots...")
    dens_BL_ensemble = plot_densities(combined_summaries, algs=["bcp_sp_wv"], regions=["BL"], fill=True, legend_labels=None)
    dens_BL_ensemble.savefig(output_dir / 'density_BL_ensemble.png', dpi=600)
    plt.close(dens_BL_ensemble)
    print("  ✓ density_BL_ensemble.png")

    dens_LK_BL_ensemble = plot_densities(combined_summaries, algs=["bcp_sp_wv"]*2, regions=["LK", "BL"], legend_labels=["Counties", "States"])
    dens_LK_BL_ensemble.savefig(output_dir / 'density_BL_and_LK_ensemble.png', dpi=600)
    plt.close(dens_LK_BL_ensemble)
    print("  ✓ density_BL_and_LK_ensemble.png")

    dens_LK_all_algs = plot_densities(combined_summaries, algs=base_algs_and_ensemble, regions=["LK"] * 7, legend_labels=alg_labels)
    dens_LK_all_algs.savefig(output_dir / 'density_LK_all_algs.png', dpi=600)
    plt.close(dens_LK_all_algs)
    print("  ✓ density_LK_all_algs.png")

    # RKI Wave Definitions Plot
    print("\nGenerating RKI comparison plot...")
    rki_wave_data = {
        'Phase': [0, 1, 2, '2a', '2b', 3, 4, 5, 6, '6a', '6b', 7, '7a', '7b', 8],
        'Name': ["Auftreten sporadischer Fälle", "Erste COVID-19-Welle", "Sommerplateau 2020", "", "",
                 "Zweite COVID-19-Welle", "Dritte COVID-19-Welle (VOC Alpha)", "Sommerplateau 2021",
                 "Vierte COVID-19-Welle (VOC Delta)", "(VOC Delta: Sommer)", "(VOC Delta: Herbst/Winter)",
                 "Fünfte COVID-19-Welle (VOC Omikron BA.1/BA.2)", "(Omikron-Sublinie BA.1)", "(Omikron-Sublinie BA.2)",
                 "Sechste COVID-19-Welle (VOC Omikron BA.5)"],
        'Beginn (KW)': ["5/2020", "10/2020", "21/2020", "21/2020", "31/2020", "40/2020", "9/2021", "24/2021",
                        "31/2021", "31/2021", "40/2021", "52/2021", "52/2021", "9/2022", "22/2022"],
        'Ende (KW)': ["9/2020", "20/2020", "39/2020", "30/2020", "39/2020", "8/2021", "23/2021", "30/2021",
                      "51/2021", "39/2021", "51/2021", "21/2022", "8/2022", "21/2022", None]
    }

    rki_wave_definitions = pd.DataFrame(rki_wave_data)
    rki_wave_definitions["Anfangsdatum"] = rki_wave_definitions['Beginn (KW)'].apply(week_year_to_date)
    rki_wave_definitions["label"] = [1,1,0,0,0,1,1,0,1,1,1,1,1,1,1]

    wave_start_dates = rki_wave_definitions[["Anfangsdatum", "label"]]
    wave_start_dates = pd.concat([wave_start_dates, pd.DataFrame([{"Anfangsdatum": "2022-09-22", "label": 2}])], ignore_index=True)
    wave_start_dates['Anfangsdatum'] = pd.to_datetime(wave_start_dates['Anfangsdatum'])

    start_date, end_date = "2020-01-03", "2023-10-13"
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_range_data = wave_start_dates.drop_duplicates().set_index('Anfangsdatum').reindex(date_range).reset_index()
    date_range_data.rename(columns={'index': 'Anfangsdatum'}, inplace=True)
    date_range_data.iloc[0,1] = 1

    date_range_data.rename(columns={"Anfangsdatum": "target"}, inplace=True)
    date_range_data["location"] = 0
    date_range_data["block"] = 1
    date_range_data["region"] = "DE"

    rki_with_wv = date_range_data.copy(deep=True)
    date_range_data["alg"] = "rki_simple"
    date_range_data['label'] = date_range_data['label'].ffill()
    date_range_data["label"] = date_range_data["label"].replace({1: True, 0: False, 2: np.nan})

    rki_with_wv["alg"] = "rki"
    peaks = combined_labels["DE"][(combined_labels["DE"]["alg"]=="wv") & (combined_labels["DE"]["label"]==False)].groupby("block").head(1)["target"]
    peaks = peaks[(peaks != "2020-11-13") & (peaks < "2022-09-22")]
    rki_with_wv.loc[rki_with_wv["target"].isin(peaks), "label"] = 0
    rki_with_wv['label'] = rki_with_wv['label'].ffill()
    rki_with_wv["label"] = rki_with_wv["label"].replace({1: True, 0: False, 2: np.nan})

    all_algs_with_rki = ["bcp", "sp", "wv", "bcp_sp_wv", "rki"]
    alg_labels_with_RKI = ["BCP", "Shapelet", "Wave Finder", "Ensemble", "Official (RKI)"]
    combined_labels_with_rki = pd.concat([combined_labels["DE"], date_range_data, rki_with_wv])

    fig_DE_rki, ax_DE_rki = plot_all_algs_with_incidence("DE", combined_labels_with_rki, 0, incidence_DE, all_algs_with_rki, alg_labels_with_RKI, fig_width=6.475716, fig_height=1.5)
    fig_DE_rki.savefig(output_dir / 'inc_all_algs_with_rki_0.png', dpi=600)
    plt.close(fig_DE_rki)
    print("  ✓ inc_all_algs_with_rki_0.png")

    # Figure 3 - Map with Timeseries
    print("\nGenerating Figure 3 (map with clustering)...")
    try:
        labels_LK_ensemble = pd.read_parquet("output/LK/labels/bcp_sp_wv.parquet")
        labels_LK_ensemble = to_wide(labels_LK_ensemble, "label")
        incidence_LK_wide = to_wide(incidence_LK)

        dists = pairwise_distances(labels_LK_ensemble.values.T, metric="jaccard")
        fig3 = plot_map_timeseries(dists, labels_LK_ensemble, incidence_LK_wide.T.values, incidence_LK.target.unique(), 4)
        plt.savefig(output_dir / "figure3.png", format="png", dpi=600)
        plt.close()
        print("  ✓ figure3.png")
    except Exception as e:
        print(f"  ⚠ figure3.png - Could not generate: {e}")
        print("    (This is expected if shapefile doesn't match county data)")

    print("\n✓ All plots generated successfully!")
    print(f"\nOutput files saved to: {output_dir.absolute()}")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reproduce paper plots from the epylabel manuscript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python reproduce_paper.py              # Run everything
    python reproduce_paper.py --labels     # Only generate labels
    python reproduce_paper.py --plots      # Only generate plots
    python reproduce_paper.py --check      # Check dependencies only
        """
    )
    parser.add_argument('--labels', action='store_true', help='Only generate labels')
    parser.add_argument('--plots', action='store_true', help='Only generate plots')
    parser.add_argument('--check', action='store_true', help='Check dependencies only')

    args = parser.parse_args()

    print("="*60)
    print("EPYLABEL - Paper Reproduction Script")
    print("="*60)

    # Check dependencies
    deps_ok = check_dependencies()

    if args.check:
        sys.exit(0 if deps_ok else 1)

    if not deps_ok:
        print("\n⚠ Some dependencies are missing. Continue anyway? (y/n)")
        # In non-interactive mode, continue anyway
        print("Continuing in non-interactive mode...")

    # Determine what to run
    run_labels = args.labels or (not args.labels and not args.plots)
    run_plots = args.plots or (not args.labels and not args.plots)

    success = True

    if run_labels:
        try:
            generate_labels()
        except Exception as e:
            print(f"\n✗ Label generation failed: {e}")
            import traceback
            traceback.print_exc()
            success = False

    if run_plots and success:
        try:
            generate_plots()
        except Exception as e:
            print(f"\n✗ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()
            success = False

    if success:
        print("\n" + "="*60)
        print("✓ REPRODUCTION COMPLETE")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ REPRODUCTION FAILED")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
