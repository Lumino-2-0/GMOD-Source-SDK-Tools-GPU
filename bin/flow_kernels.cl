// =============================================================
// VVIS_GPU — flow_kernels.cl (Version 0.5 FINAL)
// Compatible avec FlowGPU.cpp v0.5
// =============================================================


// =============================================================
// PRESET 1 — HIGH PRECISION RAYCAST
// =============================================================
kernel void RaycastPreset(
    global const float3* centers,
    global uchar* outBits,
    int leafCount,
    int leafBytes)
{
    int L = get_global_id(0);
    if (L >= leafCount) return;

    uchar* myVis = outBits + L * leafBytes;

    // Self visibility
    myVis[L >> 3] |= (1 << (L & 7));

    float3 origin = centers[L];

    // Simple distance-based ray test
    for (int T = 0; T < leafCount; T++)
    {
        if (T == L) continue;

        float3 delta = centers[T] - origin;
        float dist = length(delta);

        // VERY permissive (Kindercity friendly)
        if (dist < 20000.0f)
        {
            myVis[T >> 3] |= (1 << (T & 7));
        }
    }
}



// =============================================================
// PRESET 2 — CONE FIELD OCCLUSION (simple)
// =============================================================
kernel void ConePreset(
    global const float3* centers,
    global uchar* outBits,
    int leafCount,
    int leafBytes)
{
    int L = get_global_id(0);
    if (L >= leafCount) return;

    uchar* myVis = outBits + L * leafBytes;

    // Always see itself
    myVis[L >> 3] |= (1 << (L & 7));

    float3 origin = centers[L];

    for (int T = 0; T < leafCount; T++)
    {
        float3 delta = centers[T] - origin;
        float dist = length(delta);

        // MORE permissive than preset 1 (for open world)
        if (dist < 32000.0f)
        {
            myVis[T >> 3] |= (1 << (T & 7));
        }
    }
}



// =============================================================
// PRESET 3 — HLOD VOLUMETRIC GRID (AABB overlap)
// =============================================================
kernel void HLODPreset(
    global const float3* mins,
    global const float3* maxs,
    global uchar* outBits,
    int leafCount,
    int grid,
    float3 gmin,
    float3 gmax)
{
    int L = get_global_id(0);
    if (L >= leafCount) return;

    int leafBytes = (leafCount + 7) / 8;
    uchar* myVis = outBits + L * leafBytes;

    // Self
    myVis[L >> 3] |= (1 << (L & 7));

    // Current leaf AABB
    float3 A_min = mins[L];
    float3 A_max = maxs[L];

    // Simple volumetric AABB overlap test
    for (int T = 0; T < leafCount; T++)
    {
        float3 B_min = mins[T];
        float3 B_max = maxs[T];

        bool overlap =
            (A_max.x >= B_min.x && A_min.x <= B_max.x) &&
            (A_max.y >= B_min.y && A_min.y <= B_max.y) &&
            (A_max.z >= B_min.z && A_min.z <= B_max.z);

        if (overlap)
        {
            myVis[T >> 3] |= (1 << (T & 7));
        }
    }
}



// =============================================================
// BFS PROPAGATION (leaf graph spreading)
// =============================================================
kernel void BFS_Step(
    global uchar* pvs,
    global int* changed,
    int leafCount,
    int leafBytes,
    global int* neighborIndex,
    global int* neighborStart,
    global int* neighborCount)
{
    int L = get_global_id(0);
    if (L >= leafCount) return;

    uchar* myVis = pvs + L * leafBytes;

    int start = neighborStart[L];
    int count = neighborCount[L];

    // Spread visibility from neighbors
    for (int i = 0; i < count; i++)
    {
        int N = neighborIndex[start + i];
        uchar* visN = pvs + N * leafBytes;

        for (int b = 0; b < leafBytes; b++)
        {
            uchar old = myVis[b];
            uchar neu = old | visN[b];

            if (neu != old)
            {
                myVis[b] = neu;
                *changed = 1;
            }
        }
    }
}
