#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "optixPathTracer.h"
#include "random.h"
#include <optix_math.h>

using namespace optix;

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );
rtDeclareVariable(float3, diffuse_color, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(unsigned int, pathtrace_ray_type, , );
rtDeclareVariable(unsigned int, pathtrace_shadow_ray_type, , );
rtDeclareVariable(unsigned int, pathtrace_insect_ray_type, , );
rtDeclareVariable(float3, bg_color, , );

rtBuffer<ParallelogramLight>     lights;



RT_PROGRAM void diffuse()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hitpoint = ray.origin + t_hit * ray.direction;


	current_prd.attenuation = current_prd.attenuation * diffuse_color;
	current_prd.countEmitted = false;


	unsigned int num_lights = lights.size();
	float3 result = make_float3(0.0f);

	for (int i = 0; i < num_lights; ++i)
	{
		// Choose random point on light
		ParallelogramLight light = lights[i];
		const float z1 = rnd(current_prd.seed);
		const float z2 = rnd(current_prd.seed);
		const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;
		//const float3 light_pos = light.corner + light.v1 * 0.5 + light.v2 * 0.5;

		// Calculate properties of light sample (for area based pdf)
		const float  Ldist = length(light_pos - hitpoint);
		const float3 L = normalize(light_pos - hitpoint);
		const float  nDl = dot(ffnormal, L);
		const float  LnDl = dot(light.normal, L);

		// cast shadow ray
		if (nDl > 0.0f && LnDl > 0.0f)
		{
			PerRayData_pathtrace_shadow shadow_prd;
			shadow_prd.inShadow = false;
			Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
			rtTrace(top_shadower, shadow_ray, shadow_prd);

			if (!shadow_prd.inShadow)
			{
				const float A = length(cross(light.v1, light.v2));
				// convert area based pdf to solid angle
				const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
				result += light.emission * weight;
			}
		}
		/*PerRayData_pathtrace_insect insect_prd;
		Ray insect_ray = make_Ray(light_pos, -L, pathtrace_insect_ray_type, scene_epsilon, Ldist - scene_epsilon);
		rtTrace(top_shadower, insect_ray, insect_prd);
		double reducenum = exp(1 + length(insect_prd.hitpoint - hitpoint) / 100);
		result += insect_prd.radiance / reducenum;*/
	}

	int scatternum = 15;
	for (int i = 0; i < scatternum; ++i)
	{
		float z1 = rnd(current_prd.seed);
		float z2 = rnd(current_prd.seed);
		float3 p1;
		cosine_sample_hemisphere(z1, z2, p1);
		optix::Onb onb1(-ffnormal);
		onb1.inverse_transform(p1);

		PerRayData_pathtrace_insect insect_prd;
		Ray insect_ray = make_Ray(hitpoint, p1, pathtrace_insect_ray_type, scene_epsilon, 0x7f7f7f7f);
		rtTrace(top_shadower, insect_ray, insect_prd);
		double reducenum = exp(1 + length(insect_prd.hitpoint - hitpoint) / 75);
		result += insect_prd.radiance / reducenum;
	}


	current_prd.radiance = result;


	/*double sigmaT = 10.01f;
	double bias = 0.99;
	double ul = bias * rnd(current_prd.seed);
	double r_max = -(1.0f / sigmaT) * log(1.0f - bias);
	double r = -(1.0f / sigmaT) * log(1.0f - ul);
	double l = sqrtf(r_max*r_max - r*r);

	float ztmp = rnd(current_prd.seed);
	float3 p = make_float3(r*cos(ztmp), r*sin(ztmp), -l);
	optix::Onb onb(ffnormal);
	onb.inverse_transform(p);

	float3 base_pos = hitpoint + p;
	float3 pTarget = base_pos - l * ffnormal;
	
	PerRayData_pathtrace_insect insect_prd;
	Ray insect_ray = make_Ray(base_pos, -ffnormal, pathtrace_insect_ray_type, scene_epsilon, r_max);
	rtTrace(top_shadower, insect_ray, insect_prd);
	
	
	current_prd.origin = insect_prd.hitpoint;

	float z1 = rnd(current_prd.seed);
	float z2 = rnd(current_prd.seed);
	float3 p1;
	cosine_sample_hemisphere(z1, z2, p1);
	optix::Onb onb1(insect_prd.normal);
	onb1.inverse_transform(p1);
	current_prd.direction = p1;

	

	bool into = dot(ffnormal, world_geometric_normal) > 0.0;
	double nc = 1.0;
	double nt = 1.3;
	double nnt = into ? nc / nt : nt / nc;
	double ddn = dot(ray.direction, ffnormal);
	double a = nt - nc, b = nt + nc;
	double R0 = (a * a) / (b * b);
	double c = 1.0 + ddn;
	double Re_in = R0 + (1.0 - R0) * pow(c, 5.0);
	double Tr_in = 1.0 - Re_in;

	nnt = 1 / nnt;
	c = 1.0 - dot(current_prd.direction, ffnormal);
	const double Re_out = R0 + (1.0 - R0) * pow(c, 5.0);
	const double Tr_out = 1.0 - Re_out;

	double albed_dush = 0.999001;
	double sigma_tr = 0.547996;
	double zr = 0.0999;
	double zv = -0.446566;
	double r2 = r * r;
	double dr = sqrtf(r2 + zr * zr);
	double dv = sqrtf(r2 + zv * zv);
	double phi_r = zr * (dr * sigma_tr + 1) * exp(-sigma_tr * dr) / (dr * dr * dr);
	double phi_v = zv * (dv * sigma_tr + 1) * exp(-sigma_tr * dv) / (dv * dv * dv);
	double Rd = (albed_dush / (4.0 * 3.1415926)) * (phi_r - phi_v);
	float Sd = Tr_out*(1.0 / 3.1415926)*Tr_in;
	current_prd.radiance = result + insect_prd.radiance;*/
}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
	current_prd_shadow.inShadow = true;
	rtTerminateRay();
}



// Insect close hit

rtDeclareVariable(PerRayData_pathtrace_insect, current_prd_insect, rtPayload, );

RT_PROGRAM void find_insect()
{
	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = world_geometric_normal;

	float3 hitpoint = ray.origin + t_hit * ray.direction;
	current_prd_insect.hitpoint = hitpoint;
	current_prd_insect.normal = ffnormal;

	unsigned int num_lights = lights.size();
	float3 result = make_float3(0.0f);

	for (int i = 0; i < num_lights; ++i)
	{
		// Choose random point on light
		ParallelogramLight light = lights[i];
		const float z1 = rnd(current_prd.seed);
		const float z2 = rnd(current_prd.seed);
		const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

		// Calculate properties of light sample (for area based pdf)
		const float  Ldist = length(light_pos - hitpoint);
		const float3 L = normalize(light_pos - hitpoint);
		const float  nDl = dot(ffnormal, L);
		const float  LnDl = dot(light.normal, L);

		// cast shadow ray
		//if (nDl > 0.0f && LnDl > 0.0f)
		//{
			PerRayData_pathtrace_shadow shadow_prd;
			shadow_prd.inShadow = false;
			Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
			rtTrace(top_shadower, shadow_ray, shadow_prd);

			if (!shadow_prd.inShadow)
			{
				const float A = length(cross(light.v1, light.v2));
				// convert area based pdf to solid angle
				const float weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
				result += light.emission * fabs(weight);
			}
		//}
	}
	current_prd_insect.radiance = result;
}
