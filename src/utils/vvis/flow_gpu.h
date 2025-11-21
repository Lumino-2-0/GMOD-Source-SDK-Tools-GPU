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

struct cl_leafportal_t {
    int leaf;
    int portal;
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

    // prune kernels
    cl_kernel tiny_kernel = nullptr;
    cl_kernel backface_kernel = nullptr;
    cl_kernel angle_kernel = nullptr;
    cl_kernel convexity_kernel = nullptr;
    cl_kernel frustum_kernel = nullptr;
    cl_kernel distance_prune_kernel = nullptr;
    cl_kernel opposite_prune_kernel = nullptr;
    cl_kernel sector_prune_kernel = nullptr;
    cl_kernel z_occlusion_kernel = nullptr;
    cl_kernel solid_angle_kernel = nullptr;
    cl_kernel pyramid_sector_kernel = nullptr;
    // prune buffers
    cl_mem buf_portal_origin = nullptr;
    cl_mem buf_portal_normal = nullptr;
    cl_mem buf_portal_radius = nullptr;
    cl_mem buf_portalvis = nullptr;

    bool ok = false;
    std::mutex init_mutex;
    void log(const std::string& s) { std::cout << "[OpenCL|GPU-Mod] " << s << std::endl; }
    void init_once();
    void cleanup();

    // NEW: leaf-based propagation kernel
    cl_kernel leaf_kernel = nullptr;

    // NEW: GPU buffers for leaf propagation
    cl_mem buf_leaf_first = nullptr;
    cl_mem buf_leaf_count = nullptr;
    cl_mem buf_leaf_portals = nullptr;
    cl_mem buf_portal_leaf = nullptr;
    cl_mem buf_leafvis = nullptr;


};



extern int g_gpuPreset;

extern OpenCLManager g_clManager;

void MassiveFloodFillGPU();
