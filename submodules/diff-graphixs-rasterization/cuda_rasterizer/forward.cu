/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This file is a derivative work of the original software.
 * Modifications and additions:
 * 2026, Doga Yilmaz (doga.yilmaz@ucl.ac.uk)
 * Virtual Environments and Computer Graphics Lab, UCL
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For original inquiries contact  george.drettakis@inria.fr
 * For modification inquiries contact doga.yilmaz@ucl.ac.uk
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <curand_kernel.h>
namespace cg = cooperative_groups;

#include <assert.h>

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	/* comment to remove low-pass filter */
	// cov[0][0] += 0.3f;
	// cov[1][1] += 0.3f;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const float* nu_degree,
	const float* time,
	const float* duration,
	const float* velocity,
	const float* acceleration,
	const float* jerk,
	const float* snap,
	const float camera_timestamp,
	const float noise_std,
	float3* dyn_noise)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Load original 3D position
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	
	// Apply motion function: µx(t) = µx + v·(t−µt) + 0.5·a·(t−µt)² + (1/6)·j·(t−µt)³ + (1/24)·s·(t−µt)⁴
	// where t is camera_timestamp, µt is time[idx], v is velocity, a is acceleration, j is jerk, and s is snap
	float dt = camera_timestamp - time[idx];
	float dt2 = dt * dt;
	float dt3 = dt2 * dt;
	float dt4 = dt3 * dt;
	float3 p_transformed;

	// Add Brownian motion noise: N(0, ε²I)·Δt where noise variance scales with time
	// Standard deviation scales as sqrt(dt) for accumulated Brownian motion
	float noise_x = 0.0f, noise_y = 0.0f, noise_z = 0.0f;
	if (noise_std > 0.0f) {
		curandState state;
		curand_init(idx + (unsigned long long)(camera_timestamp * 1000000.0f), 0, 0, &state);
		float noise_scale = noise_std * sqrtf(fabsf(dt));
		noise_x = curand_normal(&state) * noise_scale;
		noise_y = curand_normal(&state) * noise_scale;
		noise_z = curand_normal(&state) * noise_scale;
	}

	dyn_noise[idx] = { noise_x, noise_y, noise_z };

	p_transformed.x = p_orig.x + (velocity[3 * idx + 0] * dt + 0.5f * acceleration[3 * idx + 0] * dt2 + (1.0f / 6.0f) * jerk[3 * idx + 0] * dt3 + (1.0f / 24.0f) * snap[3 * idx + 0] * dt4) + noise_x;
	p_transformed.y = p_orig.y + (velocity[3 * idx + 1] * dt + 0.5f * acceleration[3 * idx + 1] * dt2 + (1.0f / 6.0f) * jerk[3 * idx + 1] * dt3 + (1.0f / 24.0f) * snap[3 * idx + 1] * dt4) + noise_y;
	p_transformed.z = p_orig.z + (velocity[3 * idx + 2] * dt + 0.5f * acceleration[3 * idx + 2] * dt2 + (1.0f / 6.0f) * jerk[3 * idx + 2] * dt3 + (1.0f / 24.0f) * snap[3 * idx + 2] * dt4) + noise_z;

	// Perform near culling with TRANSFORMED position
	// Transform point to view space to check depth
	float3 p_view = transformPoint4x3(p_transformed, viewmatrix);
	if (p_view.z <= 0.2f)
	{
		return;
	}

	// Transform point by projecting (use transformed position)
	float4 p_hom = transformPoint4x4(p_transformed, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix using transformed position
	float3 cov = computeCov2D(p_transformed, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));

	/* 68–95–99.7 rule (empirical rule) for Gaussian*/
	// float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

	/* For Student's t, this part could be further improved */
	float empirical = 3.0f;
	int vv = floor(nu_degree[idx]);
	switch (vv) {
		case 1:
			empirical = (63.657 - 9.925) * (nu_degree[idx] - vv) + 9.925;
			break;
		case 2:
			empirical = (9.925 - 5.841) * (nu_degree[idx] - vv) + 5.841;
			break;
		case 3:
			empirical = (5.841 - 4.604) * (nu_degree[idx] - vv) + 4.604;
			break;
		case 4:
			empirical = (4.604 - 4.032) * (nu_degree[idx] - vv) + 4.032;
			break;
		case 5:
			empirical = (4.032 - 3.707) * (nu_degree[idx] - vv) + 3.707;
			break;
		case 6:
			empirical = (3.707 - 3.499) * (nu_degree[idx] - vv) + 3.499;
			break;
		case 7:
			empirical = (3.499 - 3.355) * (nu_degree[idx] - vv) + 3.355;
			break;
		case 8:
			empirical = (3.355 - 3.250) * (nu_degree[idx] - vv) + 3.250;
			break;
		case 9:
			empirical = (3.250 - 3.169) * (nu_degree[idx] - vv) + 3.169;
			break;
		case 10:
			empirical = (3.169 - 3.106) * (nu_degree[idx] - vv) + 3.106;
			break;
		case 11:
			empirical = (3.106 - 3.055) * (nu_degree[idx] - vv) + 3.055;
			break;
		case 12:
			empirical = (3.055 - 3.012) * (nu_degree[idx] - vv) + 3.012;
			break;
		default:
			empirical = 3.0f;
			break;
	}
	float my_radius = ceil(empirical * sqrt(max(lambda1, lambda2)));


	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color using the transformed position.
	// The view direction d(µx(t)) is computed from the moved position µx(t) to camera.
	if (colors_precomp == nullptr)
	{
		// Convert transformed position to glm::vec3 for SH computation
		glm::vec3 p_transformed_glm = glm::vec3(p_transformed.x, p_transformed.y, p_transformed.z);
		glm::vec3 dir = p_transformed_glm - *cam_pos;
		dir = dir / glm::length(dir);
		
		// Compute color from SH with the view direction based on transformed position
		glm::vec3* sh = ((glm::vec3*)shs) + idx * M;
		glm::vec3 result = SH_C0 * sh[0];

		if (D > 0)
		{
			float x = dir.x;
			float y = dir.y;
			float z = dir.z;
			result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

			if (D > 1)
			{
				float xx = x * x, yy = y * y, zz = z * z;
				float xy = x * y, yz = y * z, xz = x * z;
				result = result +
					SH_C2[0] * xy * sh[4] +
					SH_C2[1] * yz * sh[5] +
					SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
					SH_C2[3] * xz * sh[7] +
					SH_C2[4] * (xx - yy) * sh[8];

				if (D > 2)
				{
					result = result +
						SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
						SH_C3[1] * xy * z * sh[10] +
						SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
						SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
						SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
						SH_C3[5] * z * (xx - yy) * sh[14] +
						SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
				}
			}
		}
		result += 0.5f;

		// RGB colors are clamped to positive values
		clamped[3 * idx + 0] = (result.x < 0);
		clamped[3 * idx + 1] = (result.y < 0);
		clamped[3 * idx + 2] = (result.z < 0);
		result = glm::max(result, 0.0f);
		
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	
	// Compute temporal opacity: σ(t) = exp(-0.5 * ((t - µt) / s)^2)
	// where t is camera_timestamp, µt is time[idx], s is duration[idx]
	float temporal_opacity = 1.0f;
	if (duration[idx] > 0.0f)
	{
		float time_diff = camera_timestamp - time[idx];
		float normalized_time = time_diff / duration[idx];
		temporal_opacity = expf(-0.5f * normalized_time * normalized_time);
	}
	
	// Final opacity: σ(t) * σ
	// The spatial Gaussian term exp(-0.5 * (x - µx(t))^T Σ^-1 (x - µx(t))) is computed
	// during rendering per-pixel, not in preprocessing. At the Gaussian's center (x = µx(t)),
	// this term equals 1.0, so we only store the temporal and base opacity here.
	float final_opacity = opacities[idx] * temporal_opacity;
	
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, final_opacity };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, bool INFERENCE_ONLY>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	const float* __restrict__ prev_likelihoods,
	float* __restrict__ out_color,
	float* __restrict__ out_rendered_likelihoods,
	float* __restrict__ out_velocity,
	float* __restrict__ out_acceleration,
	float* __restrict__ out_jerk,
	float* __restrict__ out_snap,
	float* __restrict__ out_likelihoods,
	const float* __restrict__ nu_degree,
	const float* __restrict__ time,
	const float* __restrict__ duration,
	const float* __restrict__ velocity,
	const float* __restrict__ acceleration,
	const float* __restrict__ jerk,
	const float* __restrict__ snap)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_nu_degree[BLOCK_SIZE];
	__shared__ float collected_prev_likelihoods[BLOCK_SIZE];
	__shared__ float collected_vel_mag[BLOCK_SIZE];
	__shared__ float collected_acc_mag[BLOCK_SIZE];
	__shared__ float collected_jerk_mag[BLOCK_SIZE];
	__shared__ float collected_snap_mag[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float L = { 0 };
	float V = { 0 };  // For velocity magnitude rendering
	float A = { 0 };  // For acceleration magnitude rendering
	float J = { 0 };  // For jerk magnitude rendering
	float S = { 0 };  // For snap magnitude rendering

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_nu_degree[block.thread_rank()] = nu_degree[coll_id];
			if (!INFERENCE_ONLY) {
				collected_prev_likelihoods[block.thread_rank()] = prev_likelihoods[coll_id];
				
				// Precompute magnitudes to avoid repeated global memory access and sqrt
				float vx = velocity[coll_id * 3 + 0];
				float vy = velocity[coll_id * 3 + 1];
				float vz = velocity[coll_id * 3 + 2];
				collected_vel_mag[block.thread_rank()] = sqrtf(vx * vx + vy * vy + vz * vz);
				
				float ax = acceleration[coll_id * 3 + 0];
				float ay = acceleration[coll_id * 3 + 1];
				float az = acceleration[coll_id * 3 + 2];
				collected_acc_mag[block.thread_rank()] = sqrtf(ax * ax + ay * ay + az * az);
				
				float jx = jerk[coll_id * 3 + 0];
				float jy = jerk[coll_id * 3 + 1];
				float jz = jerk[coll_id * 3 + 2];
				collected_jerk_mag[block.thread_rank()] = sqrtf(jx * jx + jy * jy + jz * jz);
				
				float sx = snap[coll_id * 3 + 0];
				float sy = snap[coll_id * 3 + 1];
				float sz = snap[coll_id * 3 + 2];
				collected_snap_mag[block.thread_rank()] = sqrtf(sx * sx + sy * sy + sz * sz);
			}
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float nu_degree_j = collected_nu_degree[j];
			if (nu_degree_j < 1.f)
				continue;

			/* This is the base of 2D Student T */
			float base = (con_o.x * d.x * d.x + con_o.z * d.y * d.y) + 2 * con_o.y * d.x * d.y;
			if (base <= 0)
				continue;
			float power = -(nu_degree_j + 2.0f) / 2.0f;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 

			/* This is the probability density of Student T multiply with opacity */
			float probability_density = pow(1.0f + (1.0f / nu_degree_j) * base, power);
			
			float likelihood = probability_density;
			if (!INFERENCE_ONLY) {
				// Soft visibility P(R | C_theta): response weighted by the transmittance
				// reaching this component (occlusion-aware). T here is the transmittance
				// before component j is blended (updated to test_T further below).
				atomicAdd(&out_likelihoods[collected_id[j]], likelihood * T);
			}
			float alpha = min(0.99f, con_o.w * probability_density);

			if (alpha < 1.0f / 255.0f)
				continue;

			float test_T = T * (1 - alpha);
			
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			if (!INFERENCE_ONLY) {
				L += (collected_prev_likelihoods[j] + likelihood) * T;
				V += collected_vel_mag[j] * alpha * T;
				A += collected_acc_mag[j] * alpha * T;
				J += collected_jerk_mag[j] * alpha * T;
				S += collected_snap_mag[j] * alpha * T;
			}
			
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		if (!INFERENCE_ONLY) {
			out_rendered_likelihoods[pix_id] = L;
			out_velocity[pix_id] = V;
			out_acceleration[pix_id] = A;
			out_jerk[pix_id] = J;
			out_snap[pix_id] = S;
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	const float* prev_likelihoods,
	float* out_color,
	float* out_rendered_likelihoods,
	float* out_velocity,
	float* out_acceleration,
	float* out_jerk,
	float* out_snap,
	float* out_likelihoods,
	const float* nu_degree,
	const float* time,
	const float* duration,
	const float* velocity,
	const float* acceleration,
	const float* jerk,
	const float* snap,
	const bool inference_only)
{
	if (inference_only)
	{
		renderCUDA<NUM_CHANNELS, true> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means2D,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			prev_likelihoods,
			out_color,
			out_rendered_likelihoods,
			out_velocity,
			out_acceleration,
			out_jerk,
			out_snap,
			out_likelihoods,
			nu_degree,
			time,
			duration,
			velocity,
			acceleration,
			jerk,
			snap);
	}
	else
	{
		renderCUDA<NUM_CHANNELS, false> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			means2D,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			prev_likelihoods,
			out_color,
			out_rendered_likelihoods,
			out_velocity,
			out_acceleration,
			out_jerk,
			out_snap,
			out_likelihoods,
			nu_degree,
			time,
			duration,
			velocity,
			acceleration,
			jerk,
			snap);
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	const float* nu_degree,
	const float* time,
	const float* duration,
	const float* velocity,
	const float* acceleration,
	const float* jerk,
	const float* snap,
	const float camera_timestamp,
	const float noise_std,
	float3* dyn_noise)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		nu_degree,
		time,
		duration,
		velocity,
		acceleration,
		jerk,
		snap,
		camera_timestamp,
		noise_std,
		dyn_noise
		);
}