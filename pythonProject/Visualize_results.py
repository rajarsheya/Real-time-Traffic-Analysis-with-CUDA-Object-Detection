#--------------------------------------------------------------------------------------------------------------------------------------------------
# Project : Real-time Traffic Analysis with CUDA Object Detection
#
# Implemented CUDA-accelerated object detection (YOLO) to analyze a sample image dataset. Performed vehicle counting and simulated speed estimation
# to demonstrate real-time traffic analysis capabilities.
#
# Author: Arsheya Raj
# Date: 11th April 2025
#--------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict

sns.set(style="whitegrid")


def parse_speed_data(file_path):
    vehicle_speeds = defaultdict(list)
    pattern = re.compile(r"Vehicle (\d+) Speed: ([\d.]+)")

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                vehicle_id = int(match.group(1))
                speed = float(match.group(2))
                vehicle_speeds[vehicle_id].append(speed)

    return vehicle_speeds


def plot_histogram(avg_speeds):
    plt.figure(figsize=(12, 6))
    sns.histplot(avg_speeds, bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.title("Distribution of Average Vehicle Speeds")
    plt.xlabel("Speed (px/frame)")
    plt.ylabel("Number of Vehicles")
    plt.tight_layout()
    plt.show()


def plot_speed_variation(vehicle_speeds):
    plt.figure(figsize=(14, 6))
    for vid in sorted(vehicle_speeds.keys())[:5]:  # Show only first 5 vehicles
        plt.plot(vehicle_speeds[vid], label=f'Vehicle {vid}')
    plt.title("Speed Variation Over Time (Sample Vehicles)")
    plt.xlabel("Frame Number")
    plt.ylabel("Speed (px/frame)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_boxplot(avg_speeds):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=avg_speeds, orient='h', color='lightgreen')
    plt.title("Box Plot of Average Speeds")
    plt.xlabel("Speed (px/frame)")
    plt.tight_layout()
    plt.show()


def plot_top_vehicles(avg_speeds_dict, top_n=10):
    sorted_speeds = sorted(avg_speeds_dict.items(), key=lambda x: x[1])
    slowest = sorted_speeds[:top_n]
    fastest = sorted_speeds[-top_n:]

    def plot_bar(data, title, color):
        ids = [f"V{vid}" for vid, _ in data]
        speeds = [s for _, s in data]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=ids, y=speeds, palette=color)
        plt.title(title)
        plt.ylabel("Speed (px/frame)")
        plt.xlabel("Vehicle ID")
        plt.tight_layout()
        plt.show()

    plot_bar(slowest, "Top 10 Slowest Vehicles", "Blues_d")
    plot_bar(fastest, "Top 10 Fastest Vehicles", "Reds_d")


def plot_violin(vehicle_speeds):
    all_speeds = []
    ids = []
    for vid, speeds in vehicle_speeds.items():
        all_speeds.extend(speeds)
        ids.extend([vid] * len(speeds))

    plt.figure(figsize=(12, 6))
    sns.violinplot(x=ids, y=all_speeds, scale='width', inner='quartile')
    plt.title("Vehicle Speed Distributions by Vehicle")
    plt.xlabel("Vehicle ID")
    plt.ylabel("Speed (px/frame)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# ------------------
# Run everything
# ------------------
if __name__ == "__main__":
    data_path = "output_speeds.txt"
    vehicle_speeds = parse_speed_data(data_path)
    avg_speeds_dict = {vid: sum(s) / len(s) for vid, s in vehicle_speeds.items()}
    avg_speeds = list(avg_speeds_dict.values())

    plot_histogram(avg_speeds)
    plot_speed_variation(vehicle_speeds)
    plot_boxplot(avg_speeds)
    plot_top_vehicles(avg_speeds_dict)
    plot_violin(vehicle_speeds)
