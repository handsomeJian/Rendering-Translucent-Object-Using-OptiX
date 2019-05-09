
#pragma once

#include <optixu/optixu_math_namespace.h>                                        

struct ParallelogramLight                                                        
{                                                                                
    optix::float3 corner;                                                          
    optix::float3 v1, v2;                                                          
    optix::float3 normal;                                                          
    optix::float3 emission;                                                        
};                                                                               

struct PerRayData_pathtrace
{
	optix::float3 result;
	optix::float3 radiance;
	optix::float3 attenuation;
	optix::float3 origin;
	optix::float3 direction;
	unsigned int seed;
	int depth;
	int countEmitted;
	int done;
};

struct PerRayData_pathtrace_insect
{
	optix::float3 hitpoint;
	optix::float3 normal;
	optix::float3 radiance;
};

struct PerRayData_pathtrace_shadow
{
	bool inShadow;
};
