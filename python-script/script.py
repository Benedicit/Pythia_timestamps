from abc import abstractmethod
import argparse
import csv
from dataclasses import dataclass
from typing import Dict, Type
import matplotlib.pyplot as plt
from statistics import geometric_mean

import numpy as np


@dataclass
class Rollup:
    Trace: str
    Exp: str
    Filter: int

    @property
    def CAT(self) -> str:
        cat = metadata.get(self.Trace, None)
        if cat is None:
            raise ValueError(f"Unknown trace '{self.Trace}'")
        return cat

    @property
    @abstractmethod
    def Mean_IPC(self) -> float:
        pass


@dataclass
class Rollup4C(Rollup):
    Core_0_IPC: float
    Core_1_IPC: float
    Core_2_IPC: float
    Core_3_IPC: float

    @property
    def Mean_IPC(self) -> float:
        return geometric_mean(
            [
                self.Core_0_IPC,
                self.Core_1_IPC,
                self.Core_2_IPC,
                self.Core_3_IPC,
            ]
        )


@dataclass
class Rollup1C(Rollup):
    Core_0_IPC: float

    @property
    def Mean_IPC(self) -> float:
        return self.Core_0_IPC


@dataclass
class Rollup1CVaryingDram(Rollup):
    Core_0_IPC: float

    @property
    def Mean_IPC(self) -> float:
        return self.Core_0_IPC


metadata: Dict[str, str] = {}


def extract_cases_from_file(file: str, chosen_class_type: type):
    cases: list[Rollup] = []

    with open(file, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            if "--- IGNORE ---" in row["Filter"]:
                continue
            # Convert comma decimal to dot decimal
            for key in row:
                row[key] = row[key].replace(",", ".")
                if key == "Exp" and row[key] == "spp":
                    row[key] = "spp_dev"
            case = chosen_class_type(
                Trace=row["Trace"],
                Exp=row["Exp"],
                Filter=int(row["Filter"].split(" ")[0]),
                Core_0_IPC=float(row["Core_0_IPC"]),
                **(
                    {
                        "Core_1_IPC": float(row["Core_1_IPC"]),
                        "Core_2_IPC": float(row["Core_2_IPC"]),
                        "Core_3_IPC": float(row["Core_3_IPC"]),
                    }
                    if chosen_class_type is Rollup4C
                    else {}
                ),
            )
            cases.append(case)
    return cases


def calc_product_of_mean_ipc(cases_subset):
    product = 1.0
    for case in cases_subset:
        product *= case.Mean_IPC
    return product


def calc_geometric_mean(
    cases: list[Rollup], categories: list[str], prefetchers: list[str]
):
    gmeans: Dict[str, Dict[str, float]] = {}

    for cat in categories:
        cat_gmean = {}
        for pref in prefetchers:
            subset = [
                case
                for case in cases
                if case.CAT == cat and case.Exp == pref and case.Filter == 1
            ]
            gmean = calc_product_of_mean_ipc(subset) ** (1 / len(subset))
            cat_gmean[pref] = gmean
        gmeans[cat] = cat_gmean

    for cat in categories:
        baseline = gmeans[cat]["nopref"]
        for pref in prefetchers:
            gmeans[cat][pref] /= baseline
        gmeans[cat].pop("nopref", None)

    return gmeans


def create_plot(
    categories: list[str],
    prefetchers: list[str],
    base_cases: list[Rollup],
    test_cases: list[Rollup],
):
    filtered_test_cases = [case for case in test_cases if case.Exp == "pythia"]
    for case in filtered_test_cases:
        case.Exp = "pythia_new"

    merged_cases = base_cases + filtered_test_cases

    base_gmeans = calc_geometric_mean(
        merged_cases, categories, prefetchers + ["pythia_new"]
    )

    # Categories on the x-axis (keep GEOMEAN as last entry)
    base_categories = [
        "SPEC06",
        "SPEC17",
        "PARSEC",
        "Ligra",
        "Cloudsuite",
        "Mix",
        "GEOMEAN",
    ]
    categories = [cat for cat in base_categories if cat in base_gmeans] + ["GEOMEAN"]

    # Build series from computed `gmeans` (normalized to `nopref` earlier in the script)
    # Labels/order must match the prefetcher names used in the data
    labels = ["spp_dev", "bingo", "mlop", "pythia", "pythia_new"]

    def build_series_for(pref_name: str, gmeans: Dict[str, Dict[str, float]]):
        # Collect values for each category except the GEOMEAN placeholder
        vals = []
        for cat in categories[:-1]:
            # default to 1.0 if a value is missing for this prefetcher/category
            val = gmeans.get(cat, {}).get(pref_name, 1.0)
            if cat == "SPEC06":
                print(f"{pref_name} - {cat}: {val}")
            vals.append(val)

        # Compute geometric mean across the categories for the GEOMEAN column
        geo = geometric_mean(vals) if vals else 1.0
        return vals + [geo]

    data = [build_series_for(pref, base_gmeans) for pref in labels]

    spacing = 1.25  # space between categories
    x = np.arange(len(categories)) * spacing  # x positions
    width = 0.2  # width of each bar

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot each series next to each other
    for i, y in enumerate(data):
        ax.bar(x + i * width, y, width, label=labels[i])

    # Decoration
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(categories)
    ax.set_ylim(1.0, 1.7)
    ax.set_ylabel("Speedup")
    ax.set_title("Figure 11(a)")
    ax.legend()


def get_paths_for_class(chosen_class: Type[Rollup]):
    if chosen_class is Rollup4C:
        return "metadata/metadata-trace-4t.csv", "data/rollup_4C_full.csv"
    if chosen_class is Rollup1C:
        return "metadata/metadata-trace.csv", "data/rollup_1C_base_config.csv"
    if chosen_class is Rollup1CVaryingDram:
        return "metadata/metadata-exp", "data/rollup_1C_varying_DRAM_bw.csv"
    raise ValueError("Unknown chosen_class")


def load_metadata(metadata_file: str):
    with open(metadata_file, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        next(reader)  # Skip header
        return {row[0]: row[1] for row in reader}


def plot_figure(file: str, chosen_class: Type[Rollup], base: str):
    base_cases_unfiltered = extract_cases_from_file(base, chosen_class)
    base_cases_traces = set(
        case.Trace for case in base_cases_unfiltered if case.Filter == 1
    )

    test_cases_unfiltered = extract_cases_from_file(file, chosen_class)
    test_cases_traces = set(
        case.Trace for case in test_cases_unfiltered if case.Filter == 1
    )

    base_cases = [
        case
        for case in base_cases_unfiltered
        if case.Trace in test_cases_traces and case.Filter == 1
    ]
    test_cases = [
        case
        for case in test_cases_unfiltered
        if case.Trace in base_cases_traces and case.Filter == 1
    ]

    for case in base_cases:
        print(f"Base case: {case.Trace} - {case.Exp} - IPC: {case.Mean_IPC}")

    base_categories = set(case.CAT for case in base_cases)
    test_categories = set(case.CAT for case in test_cases)
    common_categories = sorted(base_categories.intersection(test_categories))

    base_prefetchers = set(case.Exp for case in base_cases)
    test_prefetchers = set(case.Exp for case in test_cases)
    common_prefetchers = sorted(base_prefetchers.intersection(test_prefetchers))

    create_plot(common_categories, sorted(base_prefetchers), base_cases, test_cases)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot normalized speedups for the given rollup file",
    )
    parser.add_argument(
        "rollup_class",
        choices=["Rollup4C", "Rollup1C", "Rollup1CVaryingDram"],
        help="Rollup class to use",
    )
    parser.add_argument("file", help="CSV file to plot")
    return parser.parse_args()


def main():
    args = parse_args()
    class_map: Dict[str, Type[Rollup]] = {
        "Rollup4C": Rollup4C,
        "Rollup1C": Rollup1C,
        "Rollup1CVaryingDram": Rollup1CVaryingDram,
    }
    chosen_class = class_map[args.rollup_class]

    metadata_file, base = get_paths_for_class(chosen_class)
    global metadata
    metadata = load_metadata(metadata_file)

    plot_figure(args.file, chosen_class, base)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
