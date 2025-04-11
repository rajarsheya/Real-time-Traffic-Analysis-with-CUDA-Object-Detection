﻿//--------------------------------------------------------------------------------------------------------------------------------------------------
// Project : Real-time Traffic Analysis with CUDA Object Detection
// 
// Implemented CUDA-accelerated object detection (YOLO) to analyze a sample image dataset. Performed vehicle counting and simulated speed estimation 
// to demonstrate real-time traffic analysis capabilities.
// 
// Author: Arsheya Raj
// Date: 11th April 2025
//--------------------------------------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

struct Vehicle {
    int id;
    float y_center;
    float delta_time;
};

// CUDA kernel to calculate vehicle speeds based on their positions
__global__ void calculate_speeds_kernel(Vehicle* vehicles, int num_vehicles, float* speeds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vehicles - 1) {
        float distance = fabs(vehicles[idx + 1].y_center - vehicles[idx].y_center);
        float speed = distance / vehicles[idx + 1].delta_time;
        speeds[idx] = speed;
    }
}

// CUDA kernel to match vehicles between frames (simple nearest-neighbor matching based on y_center)
__global__ void match_vehicles_kernel(Vehicle* prev_vehicles, int num_prev, Vehicle* curr_vehicles, int num_curr, int* matched_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_prev) {
        int closest_id = -1;
        float min_distance = FLT_MAX;
        for (int j = 0; j < num_curr; j++) {
            float distance = fabs(prev_vehicles[idx].y_center - curr_vehicles[j].y_center);
            if (distance < min_distance) {
                min_distance = distance;
                closest_id = j;
            }
        }
        matched_ids[idx] = closest_id;
    }
}

// Host function to process positions and calculate speeds
void process_positions(const std::vector<Vehicle>& vehicles, float delta_time, const std::string& output_filename) {
    // Allocate memory on the device (GPU)
    Vehicle* d_vehicles;
    float* d_speeds;
    int* d_matched_ids;

    int num_vehicles = vehicles.size();

    // Allocate memory for the arrays on the device
    cudaMalloc((void**)&d_vehicles, num_vehicles * sizeof(Vehicle));
    cudaMalloc((void**)&d_speeds, num_vehicles * sizeof(float));
    cudaMalloc((void**)&d_matched_ids, num_vehicles * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_vehicles, vehicles.data(), num_vehicles * sizeof(Vehicle), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel to calculate speeds
    int blockSize = 256;
    int numBlocks = (num_vehicles + blockSize - 1) / blockSize;
    calculate_speeds_kernel << <numBlocks, blockSize >> > (d_vehicles, num_vehicles, d_speeds);

    // Allocate space to store the results
    std::vector<float> speeds(num_vehicles);
    cudaMemcpy(speeds.data(), d_speeds, num_vehicles * sizeof(float), cudaMemcpyDeviceToHost);

    // Open the output file to write the speeds
    std::ofstream output_file(output_filename);

    if (!output_file.is_open()) {
        std::cerr << "[ERROR] Could not open output file!" << std::endl;
        cudaFree(d_vehicles);
        cudaFree(d_speeds);
        cudaFree(d_matched_ids);
        return;
    }

    // Write the vehicle speeds to the output file
    for (int i = 0; i < num_vehicles; ++i) {
        if (speeds[i] != FLT_MAX) {
            output_file << "Vehicle " << vehicles[i].id << " Speed: " << speeds[i] << " units/s" << std::endl;
        }
    }

    // Close the output file
    output_file.close();

    // Free device memory
    cudaFree(d_vehicles);
    cudaFree(d_speeds);
    cudaFree(d_matched_ids);

    std::cout << "[INFO] Speed calculation complete. Results saved to " << output_filename << std::endl;
}

// Function to read vehicle positions from positions.txt
std::vector<Vehicle> read_positions(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<Vehicle> vehicles;

    if (file.is_open()) {
        int id;
        float y_center, delta_time;
        while (file >> id >> y_center >> delta_time) {
            vehicles.push_back({ id, y_center, delta_time });
        }
        file.close();
    }
    else {
        std::cerr << "[ERROR] Could not open positions.txt!" << std::endl;
    }

    return vehicles;
}

int main() {
    // Read positions from file
    std::string filename = "positions.txt";
    std::vector<Vehicle> vehicles = read_positions(filename);

    // Output filename
    std::string output_filename = "output_speeds.txt";

    // Process the positions and calculate speeds
    float delta_time = 0.0333333f;  // Adjust delta_time as necessary (e.g., for 30 FPS video)
    process_positions(vehicles, delta_time, output_filename);

    return 0;
}