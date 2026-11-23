// flow_cpu_ray_avx2.cpp
//
// AVX2 build of the CPU raycast engine (compiled with /arch:AVX2 and /fp:precise, see
// the vcxproj). Only called when the CPU and the OS support AVX2
// (FlowCPU_GetRaycastRange). FP contraction stays off: AVX2 CPUs have FMA, but using
// it would change the rounding and break the bit-for-bit match with the GPUs.

#pragma float_control( precise, on )
#pragma fp_contract( off )

#include <intrin.h>
#include <math.h>
#include <string.h>

#include "flow_cpu_ray.h"

namespace FlowCpuAVX2
{
#include "flow_cpu_ray_impl.h"
}