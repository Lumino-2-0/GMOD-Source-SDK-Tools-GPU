#pragma once

#include <mutex>
#include <string>
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdint>

// Structures partagees avec le kernel OpenCL
struct cl_plane_t {
    float normal[3];
    float dist;
};

struct cl_winding_t {
    int numpoints;
    float points[16][3]; // MAX_POINTS_ON_FIXED_WINDING = 16
};

struct cl_portal_t {
    cl_plane_t plane;
    int leaf;
    float origin[3];
    float radius;
    int winding_idx;
};

struct cl_leaf_t {
    int first_portal;
    int num_portals;
};

struct OpenCLManager {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel floodfill_kernel = nullptr;
    cl_kernel countbits_kernel = nullptr;
    bool ok = false;
    std::mutex init_mutex;
    void log(const std::string& s) { std::cout << "[OpenCL|GPU-Mod] " << s << std::endl; }
    void init_once();
    void cleanup();


};



extern OpenCLManager g_clManager;

void MassiveFloodFillGPU();
