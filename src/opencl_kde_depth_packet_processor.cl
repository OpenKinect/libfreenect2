/*
 * This code implements a depth packet processor using the phase unwrapping
 * algorithm described in the paper "Efficient Phase Unwrapping using Kernel
 * Density Estimation", ECCV 2016, Felix Järemo Lawin, Per-Erik Forssen and
 * Hannes Ovren, see http://www.cvl.isy.liu.se/research/datasets/kinect2-dataset/.
 */


/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */


/*******************************************************************************
 * Process pixel stage 1
 ******************************************************************************/

#define PHASE (float3)(PHASE_IN_RAD0, PHASE_IN_RAD1, PHASE_IN_RAD2)
#define AB_MULTIPLIER_PER_FRQ (float3)(AB_MULTIPLIER_PER_FRQ0, AB_MULTIPLIER_PER_FRQ1, AB_MULTIPLIER_PER_FRQ2)
#define NUM_HYPOTHESES 30

float decodePixelMeasurement(global const ushort *data, global const short *lut11to16, const uint sub, const uint x, const uint y)
{
  uint row_idx = (424 * sub + y) * 352;
  uint idx = (((x >> 2) + ((x << 7) & BFI_BITMASK)) * 11) & (uint)0xffffffff;

  uint col_idx = idx >> 4;
  uint upper_bytes = idx & 15;
  uint lower_bytes = 16 - upper_bytes;

  uint data_idx0 = row_idx + col_idx;
  uint data_idx1 = row_idx + col_idx + 1;

  return (float)lut11to16[(x < 1 || 510 < x || col_idx > 352) ? 0 : ((data[data_idx0] >> upper_bytes) | (data[data_idx1] << lower_bytes)) & 2047];
}

void kernel processPixelStage1(global const short *lut11to16, global const float *z_table, global const float3 *p0_table, global const ushort *data,
                               global float3 *a_out, global float3 *b_out, global float3 *n_out, global float *ir_out)
{
  const uint i = get_global_id(0);

  const uint x = i % 512;
  const uint y = i / 512;

  const uint y_tmp = (423 - y);
  const uint y_in = (y_tmp < 212 ? y_tmp + 212 : 423 - y_tmp);

  const int3 invalid = (int)(0.0f >= z_table[i]);
  const float3 p0 = p0_table[i];
  float3 p0x_sin, p0y_sin, p0z_sin;
  float3 p0x_cos, p0y_cos, p0z_cos;

  p0x_sin = -sincos(PHASE + p0.x, &p0x_cos);
  p0y_sin = -sincos(PHASE + p0.y, &p0y_cos);
  p0z_sin = -sincos(PHASE + p0.z, &p0z_cos);

  int3 invalid_pixel = (int3)(invalid);

  const float3 v0 = (float3)(decodePixelMeasurement(data, lut11to16, 0, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 1, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 2, x, y_in));
  const float3 v1 = (float3)(decodePixelMeasurement(data, lut11to16, 3, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 4, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 5, x, y_in));
  const float3 v2 = (float3)(decodePixelMeasurement(data, lut11to16, 6, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 7, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 8, x, y_in));

  float3 a = (float3)(dot(v0, p0x_cos),
                      dot(v1, p0y_cos),
                      dot(v2, p0z_cos)) * AB_MULTIPLIER_PER_FRQ;
  float3 b = (float3)(dot(v0, p0x_sin),
                      dot(v1, p0y_sin),
                      dot(v2, p0z_sin)) * AB_MULTIPLIER_PER_FRQ;

  a = select(a, (float3)(0.0f), invalid_pixel);
  b = select(b, (float3)(0.0f), invalid_pixel);
  float3 n = sqrt(a * a + b * b);

  int3 saturated = (int3)(any(isequal(v0, (float3)(32767.0f))),
                          any(isequal(v1, (float3)(32767.0f))),
                          any(isequal(v2, (float3)(32767.0f))));

  a_out[i] = select(a, (float3)(0.0f), saturated);
  b_out[i] = select(b, (float3)(0.0f), saturated);
  n_out[i] = n;
  ir_out[i] = min(dot(select(n, (float3)(65535.0f), saturated), (float3)(0.333333333f  * AB_MULTIPLIER * AB_OUTPUT_MULTIPLIER)), 65535.0f);
}

/*******************************************************************************
 * Filter pixel stage 1
 ******************************************************************************/
void kernel filterPixelStage1(global const float3 *a, global const float3 *b, global const float3 *n,
                              global float3 *a_out, global float3 *b_out, global uchar *max_edge_test)
{
    const uint i = get_global_id(0);

    const uint x = i % 512;
    const uint y = i / 512;

    const float3 self_a = a[i];
    const float3 self_b = b[i];

    const float gaussian[9] = {GAUSSIAN_KERNEL_0, GAUSSIAN_KERNEL_1, GAUSSIAN_KERNEL_2, GAUSSIAN_KERNEL_3, GAUSSIAN_KERNEL_4, GAUSSIAN_KERNEL_5, GAUSSIAN_KERNEL_6, GAUSSIAN_KERNEL_7, GAUSSIAN_KERNEL_8};

    if(x < 1 || y < 1 || x > 510 || y > 422)
    {
        a_out[i] = self_a;
        b_out[i] = self_b;
        max_edge_test[i] = 1;
    }
    else
    {
        float3 threshold = (float3)(JOINT_BILATERAL_THRESHOLD);
        float3 joint_bilateral_exp = (float3)(JOINT_BILATERAL_EXP);

        const float3 self_norm = n[i];
        const float3 self_normalized_a = self_a / self_norm;
        const float3 self_normalized_b = self_b / self_norm;

        float3 weight_acc = (float3)(0.0f);
        float3 weighted_a_acc = (float3)(0.0f);
        float3 weighted_b_acc = (float3)(0.0f);
        float3 dist_acc = (float3)(0.0f);

        const int3 c0 = isless(self_norm * self_norm, threshold);

        threshold = select(threshold, (float3)(0.0f), c0);
        joint_bilateral_exp = select(joint_bilateral_exp, (float3)(0.0f), c0);

        for(int yi = -1, j = 0; yi < 2; ++yi)
        {
            uint i_other = (y + yi) * 512 + x - 1;

            for(int xi = -1; xi < 2; ++xi, ++j, ++i_other)
            {
                const float3 other_a = a[i_other];
                const float3 other_b = b[i_other];
                const float3 other_norm = n[i_other];
                const float3 other_normalized_a = other_a / other_norm;
                const float3 other_normalized_b = other_b / other_norm;

                const int3 c1 = isless(other_norm * other_norm, threshold);

                const float3 dist = 0.5f * (1.0f - (self_normalized_a * other_normalized_a + self_normalized_b * other_normalized_b));
                const float3 weight = select(gaussian[j] * exp(-1.442695f * joint_bilateral_exp * dist), (float3)(0.0f), c1);

                weighted_a_acc += weight * other_a;
                weighted_b_acc += weight * other_b;
                weight_acc += weight;
                dist_acc += select(dist, (float3)(0.0f), c1);
            }
        }

        const int3 c2 = isless((float3)(0.0f), weight_acc.xyz);
        a_out[i] = select((float3)(0.0f), weighted_a_acc / weight_acc, c2);
        b_out[i] = select((float3)(0.0f), weighted_b_acc / weight_acc, c2);

        max_edge_test[i] = all(isless(dist_acc, (float3)(JOINT_BILATERAL_MAX_EDGE)));
    }
}



/*******************************************************************************
 * KDE phase unwrapping
 ******************************************************************************/

//arrays for hypotheses
float constant k_list[NUM_HYPOTHESES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
float constant n_list[NUM_HYPOTHESES] = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 5.0f, 6.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 8.0f, 8.0f, 7.0f, 8.0f, 9.0f, 9.0f};
float constant m_list[NUM_HYPOTHESES] = {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 7.0f, 7.0f, 8.0f, 8.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 13.0f, 13.0f, 14.0f};

void calcErr(const float k, const float n, const float m, const float t0, const float t1, const float t2, float* err1, float* err2, float* err3)
{
    //phase unwrapping equation residuals
    *err1 = 3.0f * n - 15.0f * k - (t1 - t0);
    *err2 = 3.0f * n - 2.0f * m - (t2 - t0);
    *err3 = 15.0f * k - 2.0f * m - (t2 - t1);
}

/********************************************************************************
 * Rank all 30 phase hypothses and returns the two most likley
 ********************************************************************************/
void phaseUnWrapper(float t0, float t1,float t2, float* phase_first, float* phase_second, float* err_w1, float* err_w2)
{
  float err;
  float err1,err2,err3;

  //unwrapping weight for cost function
  float w1 = 1.0f;
  float w2 = 10.0f;
  float w3 = 1.0218f;

  float err_min = 100000.0f;
  float err_min_second = 200000.0f;
  unsigned int ind_min, ind_second;

  float k, n, m;

  for(int i = 0; i < NUM_HYPOTHESES; i++)
  {
    m = m_list[i];
    n = n_list[i];
    k = k_list[i];
    calcErr(k, n, m, t0, t1, t2, &err1, &err2, &err3);
    err = w1 * err1 * err1 + w2 * err2 * err2 + w3 * err3 * err3;
    if(err < err_min)
    {
      err_min_second = err_min;
      ind_second = ind_min;
      err_min = err;
      ind_min = i;

    }
    else if(err < err_min_second)
    {
      err_min_second = err;
      ind_second = i;
    }
  }

  //decode ind_min
  float mvals = m_list[ind_min];
  float nvals = n_list[ind_min];
  float kvals = k_list[ind_min];

  //Weighted phases for phase fusion weighted average
  float phi2_out = (t2 / 2.0f + mvals);
  float phi1_out = (t1 / 15.0f + kvals);
  float phi0_out = (t0 / 3.0f + nvals);

  *err_w1 = err_min;

  //phase fusion
  *phase_first = (phi2_out + phi1_out + phi0_out) / 3.0f;

  mvals = m_list[ind_second];
  nvals = n_list[ind_second];
  kvals = k_list[ind_second];

  //Weighted phases for phase fusion weighted average
  phi2_out = (t2 / 2.0f + mvals);
  phi1_out = (t1 / 15.0f + kvals);
  phi0_out = (t0 / 3.0f + nvals);

  *err_w2 = err_min_second;

  //phase fusion
  *phase_second = (phi2_out + phi1_out + phi0_out) / 3.0f;

}

/*******************************************************************************
 * Predict phase variance from amplitude direct quadratic model
 ******************************************************************************/
void calculatePhaseUnwrappingVarDirect(float3 ir, float3* var)
{
  //Model: sigma = 1/(gamma0*a+gamma1*a^2+gamma2). Possibly better than calculatePhaseUnwrappingVar
  //The gammas are optimized using lsqnonlin in matlab.
  //For more details see the paper "Efficient Phase Unwrapping using Kernel Density Estimation"
  //section 3.3 and 4.4.
  float sigma_max = 0.5f * M_PI_F;

  //Set sigma = pi/2 as a maximum standard deviation of the phase. Cut off function after root
  float q0 = ir.x > 5.244404f ? 0.7919451669f * ir.x - 0.002363097609f * ir.x * ir.x - 3.088285897f : 1.0f / sigma_max;
  float q1 = ir.y > 4.084835f ? 1.214266794f * ir.y - 0.00581082634f * ir.y * ir.y - 3.863119924f : 1.0f / sigma_max;
  float q2 = ir.z > 6.379475f ? 0.6101457464f * ir.z - 0.00113679233f * ir.z * ir.z - 2.84614442f : 1.0f / sigma_max;
  float3 q = (float3)(q0, q1, q2);
  float3 roots = (float3)(5.244404f, 4.084835f, 6.379475f);
  float3 sigma = (float3)(1.0f)/q;
  sigma = select(sigma, (float3)(sigma_max), isless((float3)(sigma_max), sigma));
  *var = sigma;
}


/*******************************************************************************
 * Predict phase variance from amplitude quadratic atan model
 ******************************************************************************/
//void calculatePhaseUnwrappingVar(float3 ir, float* var0, float* var1, float* var2)
void calculatePhaseUnwrappingVar(float3 ir, float3 *var)
{
  //Model: sigma = atan(sqrt(1/(gamma0*a+gamma1*a^2+gamma2)-1)). The gammas are optimized using lsqnonlin in matlab.
  //For more details see the paper "Efficient Phase Unwrapping using Kernel Density Estimation",
  //section 3.3 and 4.4.
  float q0 = 0.8211288451f * ir.x - 0.002601348899f * ir.x * ir.x - 3.549793908f;
  float q1 = 1.259642407f * ir.y - 0.005478390508f * ir.y * ir.y - 4.335841127f;
  float q2 = 0.6447928035f * ir.z - 0.0009627273649f * ir.z * ir.z - 3.368205575f;
  float3 q = (float3)(q0, q1, q2);
  q *= q;
  float3 roots = (float3)(5.64173671f, 4.31705182f, 6.84453530f);
  float3 sigma = select(select((float3)(0.5f * M_PI_F), roots * 0.5f * M_PI_F / ir, isless(roots,ir)), atan(sqrt((float3)(1.0f) / (q - (float3)(1.0f)))), isless((float3)(1.0f), q));
  sigma = select(sigma, (float3)(0.001f), isless(sigma, (float3)(0.001f)));
  *var = sigma*sigma;
}

void kernel processPixelStage2_phase(global const float3 *a_in, global const float3 *b_in, global float4 *phase_conf_vec)
{
  const uint i = get_global_id(0);

  //read complex number real (a) and imaginary part (b)
  float3 a = a_in[i];
  float3 b = b_in[i];

  //calculate complex argument
  float3 phase = atan2(b, a);
  phase = select(phase, (float3)(0.0f), isnan(phase));
  phase = select(phase, phase + 2.0f * M_PI_F, isless(phase, (float3)(0.0f)));

  //calculate amplitude or the absolute value
  float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;

  float ir_sum = ir.x + ir.y + ir.z;

  float phase_first = 0.0f;
  float phase_second = 0.0f;

  float J_1, J_2, unwrapping_likelihood1, unwrapping_likelihood2;

  //scale with least common multiples of modulation frequencies
  float3 t = phase / (2.0f * M_PI_F) * (float3)(3.0f, 15.0f, 2.0f);

  float t0 = t.x;
  float t1 = t.y;
  float t2 = t.z;

  //rank and extract two most likely phase hypothises
  phaseUnWrapper(t0, t1, t2, &phase_first, &phase_second, &J_1, &J_2);

  float phase_likelihood;

  //check if near saturation
  if(ir_sum < 0.4f * 65535.0f)
  {
    //calculate phase likelihood from amplitude
    //float var0, var1, var2;
    float3 var;
    calculatePhaseUnwrappingVar(ir, &var);
    phase_likelihood = exp(-(var.x + var.y + var.z) / (2.0f * PHASE_CONFIDENCE_SCALE));
    phase_likelihood = select(phase_likelihood, 0.0f, isnan(phase_likelihood));
  }
  else
  {
    phase_likelihood = 0.0f;
  }

  //merge phase likelihood with phase likelihood
  unwrapping_likelihood1 = phase_likelihood*exp(-J_1 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));
  unwrapping_likelihood2 = phase_likelihood*exp(-J_2 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));

  //suppress confidence if phase is beyond allowed range
  unwrapping_likelihood1 = phase_first > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood1;
  unwrapping_likelihood2 = phase_second > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood2;

  phase_conf_vec[i] = (float4)(phase_first,phase_second, unwrapping_likelihood1, unwrapping_likelihood2);

}

void kernel filter_kde(global const float4* phase_conf_vec, global const float* gauss_filt_array, global const float* z_table, global const float* x_table, global float* depth)
{
  const uint i = get_global_id(0);
  float kde_val_1, kde_val_2;

  const int loadX = i % 512;
  const int loadY = i / 512;

  int k, l;
  float sum_1, sum_2;

  //initialize neighborhood boundaries
  int from_x = (loadX > KDE_NEIGBORHOOD_SIZE ? -KDE_NEIGBORHOOD_SIZE : -loadX + 1);
  int from_y = (loadY > KDE_NEIGBORHOOD_SIZE ? -KDE_NEIGBORHOOD_SIZE : -loadY + 1);
  int to_x = (loadX < 511 - KDE_NEIGBORHOOD_SIZE - 1 ? KDE_NEIGBORHOOD_SIZE: 511 - loadX - 1);
  int to_y = (loadY < 423 - KDE_NEIGBORHOOD_SIZE ? KDE_NEIGBORHOOD_SIZE: 423 - loadY);

  kde_val_1 = 0.0f;
  kde_val_2 = 0.0f;
  float2 phase_local = phase_conf_vec[i].xy;

  if(loadX >= 1 && loadX < 511 && loadY >= 0 && loadY<424)
  {
    sum_1=0.0f;
    sum_2=0.0f;
    float gauss;
    float sum_gauss = 0.0f;

    float phase_1_local;
    float phase_2_local;
    float conf1_local;
    float conf2_local;
    float4 phase_conf_local;
    uint ind;

    float diff11, diff21, diff12, diff22;

    //calculate KDE for all hypothesis within the neigborhood
    for(k = from_y; k <= to_y; k++)
      for(l = from_x; l <= to_x; l++)
      {
        ind = (loadY + k) * 512 + (loadX + l);
        phase_conf_local = phase_conf_vec[ind];
        conf1_local = phase_conf_local.z;
        conf2_local = phase_conf_local.w;

        phase_1_local = phase_conf_local.x;
        phase_2_local = phase_conf_local.y;

        gauss = gauss_filt_array[k + KDE_NEIGBORHOOD_SIZE] * gauss_filt_array[l + KDE_NEIGBORHOOD_SIZE];
        sum_gauss += gauss * (conf1_local + conf2_local);
        diff11 = phase_1_local - phase_local.x;
        diff21 = phase_2_local - phase_local.x;
        diff12 = phase_1_local - phase_local.y;
        diff22 = phase_2_local - phase_local.y;
        sum_1 += gauss * (conf1_local * exp(-diff11 * diff11 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff21 * diff21 / (2 * KDE_SIGMA_SQR)));
        sum_2 += gauss * (conf1_local * exp(-diff12 * diff12 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff22 * diff22 / (2 * KDE_SIGMA_SQR)));
      }
    kde_val_1 = sum_gauss > 0.5f ? sum_1 / sum_gauss : sum_1 * 2.0f;
    kde_val_2 = sum_gauss > 0.5f ? sum_2 / sum_gauss : sum_2 * 2.0f;
  }

  //select hypothesis
  int val_ind = kde_val_2 <= kde_val_1 ? 1: 0;

  float phase_final = val_ind ? phase_local.x: phase_local.y;
  float max_val = val_ind ? kde_val_1: kde_val_2;

  float zmultiplier = z_table[i];
  float xmultiplier = x_table[i];

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * UNAMBIGIOUS_DIST * 2.0f;

  bool cond1 =  true && 0.0f < depth_linear && 0.0f < max_depth;

  xmultiplier = (xmultiplier * 90.0f) / (max_depth * max_depth * 8192.0f);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);
  depth_fit = depth_fit < 0.0f ? 0.0f : depth_fit;

  float d = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z

  max_val = d < MIN_DEPTH || d > MAX_DEPTH ? 0.0f: max_val;

  //set to zero if confidence is low
  depth[i] = max_val >= KDE_THRESHOLD ? d: 0.0f;
}


/*****************************************************************
 * THREE HYPOTHESIS
 *****************************************************************/

void phaseUnWrapper3(float t0, float t1,float t2, float* phase_first, float* phase_second, float* phase_third, float* err_w1, float* err_w2, float* err_w3)
{
  float err;
  float err1, err2, err3;

  //unwrapping weight for cost function
  float w1 = 1.0f;
  float w2 = 10.0f;
  float w3 = 1.0218f;

  float err_min = 100000.0f;
  float err_min_second = 200000.0f;
  float err_min_third = 300000.0f;
  unsigned int ind_min, ind_second, ind_third;

  float k, n, m;

  for(int i = 0; i < NUM_HYPOTHESES; i++)
  {
    m = m_list[i];
    n = n_list[i];
    k = k_list[i];
    calcErr(k, n, m, t0, t1, t2, &err1, &err2, &err3);
    err = w1 * err1 * err1 + w2 * err2 * err2 + w3 * err3 * err3;
    if(err < err_min)
    {
      err_min_third = err_min_second;
      ind_third = ind_second;
      err_min_second = err_min;
      ind_second = ind_min;
      err_min = err;
      ind_min = i;

    }
    else if(err<err_min_second)
    {
      err_min_third = err_min_second;
      ind_third = ind_second;
      err_min_second = err;
      ind_second = i;
    }
    else if(err<err_min_third)
    {
      err_min_third = err;
      ind_third = i;
    }
  }

  //decode ind_min
  float mvals = m_list[ind_min];
  float nvals = n_list[ind_min];
  float kvals = k_list[ind_min];

  //Weighted phases for phase fusion weighted average
  float phi2_out = (t2 / 2.0f + mvals);
  float phi1_out = (t1 / 15.0f + kvals);
  float phi0_out = (t0 / 3.0f + nvals);

  *err_w1 = err_min;

  *phase_first = (phi2_out + phi1_out + phi0_out) / 3.0f;

  mvals = m_list[ind_second];
  nvals = n_list[ind_second];
  kvals = k_list[ind_second];

  //Weighted phases for phase fusion weighted average
  phi2_out = (t2 / 2.0f + mvals);
  phi1_out = (t1 / 15.0f + kvals);
  phi0_out = (t0 / 3.0f + nvals);

  *err_w2 = err_min_second;
  *phase_second = (phi2_out + phi1_out + phi0_out) / 3.0f;

  mvals = m_list[ind_third];
  nvals = n_list[ind_third];
  kvals = k_list[ind_third];

  //Weighted phases for phase fusion weighted average
  phi2_out = (t2 / 2.0f + mvals);
  phi1_out = (t1 / 15.0f + kvals);
  phi0_out = (t0 / 3.0f + nvals);

  *err_w3 = err_min_third;
  *phase_third = (phi2_out + phi1_out + phi0_out) / 3.0f;
}


void kernel processPixelStage2_phase3(global const float3 *a_in, global const float3 *b_in, global float *phase_1, global float *phase_2, global float *phase_3, global float *conf1, global float *conf2, global float *conf3)
{
  const uint i = get_global_id(0);

  //read complex number real (a) and imaginary part (b)
  float3 a = a_in[i];
  float3 b = b_in[i];

  //calculate complex argument
  float3 phase = atan2(b, a);
  phase = select(phase, (float3)(0.0f), isnan(phase));
  phase = select(phase, phase + 2.0f * M_PI_F, isless(phase, (float3)(0.0f)));

  //calculate amplitude or the absolute value
  float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;

  float ir_sum = ir.x + ir.y + ir.z;

  float phase_first = 0.0f;
  float phase_second = 0.0f;
  float phase_third = 0.0f;
  float J_1, J_2, J_3, unwrapping_likelihood1, unwrapping_likelihood2, unwrapping_likelihood3;

  //scale with least common multiples of modulation frequencies
  float3 t = phase / (2.0f * M_PI_F) * (float3)(3.0f, 15.0f, 2.0f);

  float t0 = t.x;
  float t1 = t.y;
  float t2 = t.z;

  //rank and extract three most likely phase hypothises
  phaseUnWrapper3(t0, t1, t2, &phase_first, &phase_second, &phase_third, &J_1, &J_2, &J_3);

  phase_1[i] = phase_first;
  phase_2[i] = phase_second;
  phase_3[i] = phase_third;

  float phase_likelihood;
  //check if near saturation
  if(ir_sum < 0.4f * 65535.0f)
  {
    //calculate phase likelihood from amplitude
    float3 var;
    calculatePhaseUnwrappingVar(ir, &var);
    phase_likelihood = exp(-(var.x + var.y + var.z)/(2.0f * PHASE_CONFIDENCE_SCALE));
    phase_likelihood = select(phase_likelihood, 0.0f, isnan(phase_likelihood));
  }
  else
  {
    phase_likelihood = 0.0f;
  }

  //merge unwrapping likelihood with phase likelihood
  unwrapping_likelihood1 = phase_likelihood * exp(-J_1 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));
  unwrapping_likelihood2 = phase_likelihood * exp(-J_2 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));
  unwrapping_likelihood3 = phase_likelihood * exp(-J_3 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));

  //suppress confidence if phase is beyond allowed range
  unwrapping_likelihood1 = phase_first > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood1;
  unwrapping_likelihood2 = phase_second > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood2;
  unwrapping_likelihood3 = phase_third > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood3;

  conf1[i] = unwrapping_likelihood1;
  conf2[i] = unwrapping_likelihood2;
  conf3[i] = unwrapping_likelihood3;

}



void kernel filter_kde3(global const float *phase_1, global const float *phase_2, global const float *phase_3, global const float* conf1, global const float* conf2, global const float* conf3, global const float* gauss_filt_array, global const float* z_table, global const float* x_table, global float* depth)
{
  const uint i = get_global_id(0);
  float kde_val_1, kde_val_2, kde_val_3;

  const int loadX = i % 512;
  const int loadY = i / 512;
  int k, l;
  float sum_1, sum_2, sum_3;

  //initialize neighborhood boundaries
  int from_x = (loadX > KDE_NEIGBORHOOD_SIZE ? -KDE_NEIGBORHOOD_SIZE : -loadX + 1);
  int from_y = (loadY > KDE_NEIGBORHOOD_SIZE ? -KDE_NEIGBORHOOD_SIZE : -loadY + 1);
  int to_x = (loadX < 511 - KDE_NEIGBORHOOD_SIZE - 1 ? KDE_NEIGBORHOOD_SIZE: 511 - loadX - 1);
  int to_y = (loadY < 423 - KDE_NEIGBORHOOD_SIZE ? KDE_NEIGBORHOOD_SIZE: 423 - loadY);

  kde_val_1 = 0.0f;
  kde_val_2 = 0.0f;
  kde_val_3 = 0.0f;
  float phase_first = phase_1[i];
  float phase_second = phase_2[i];
  float phase_third = phase_3[i];
  if(loadX >= 1 && loadX < 511 && loadY >= 0 && loadY < 424)
  {
    //Filter kernel
    sum_1 = 0.0f;
    sum_2 = 0.0f;
    sum_3 = 0.0f;
    float gauss;
    float sum_gauss = 0.0f;

    float phase_1_local;
    float phase_2_local;
    float phase_3_local;
    float conf1_local;
    float conf2_local;
    float conf3_local;
    float diff11, diff12, diff13, diff21, diff22, diff23, diff31, diff32, diff33;
    uint ind;

    //calculate KDE for all hypothesis within the neigborhood
    for(k = from_y; k <= to_y; k++)
      for(l = from_x; l <= to_x; l++)
      {
        ind = (loadY + k) * 512 + (loadX + l);
        conf1_local = conf1[ind];
        conf2_local = conf2[ind];
        conf3_local = conf3[ind];
        phase_1_local = phase_1[ind];
        phase_2_local = phase_2[ind];
        phase_3_local = phase_3[ind];
        diff11 = phase_1_local - phase_first;
        diff12 = phase_1_local - phase_second;
        diff13 = phase_1_local - phase_third;
        diff21 = phase_2_local - phase_first;
        diff22 = phase_2_local - phase_second;
        diff23 = phase_2_local - phase_third;
        diff31 = phase_3_local - phase_first;
        diff32 = phase_3_local - phase_second;
        diff33 = phase_3_local - phase_third;
        gauss = gauss_filt_array[k + KDE_NEIGBORHOOD_SIZE] * gauss_filt_array[l + KDE_NEIGBORHOOD_SIZE];
        sum_gauss += gauss * (conf1_local + conf2_local + conf3_local);
        sum_1 += gauss*(conf1_local * exp(-diff11 * diff11 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff21 * diff21 / (2 * KDE_SIGMA_SQR)) + conf3_local * exp(-diff31 * diff31 /(2 * KDE_SIGMA_SQR)));
        sum_2 += gauss * (conf1_local * exp(-diff12 * diff12 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff22 * diff22/(2 * KDE_SIGMA_SQR))+conf3_local * exp(-diff32 * diff32 / (2 * KDE_SIGMA_SQR)));
        sum_3 += gauss * (conf1_local * exp(-diff13 * diff13 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff23 * diff23 / (2 * KDE_SIGMA_SQR)) + conf3_local * exp(-diff33 * diff33 / (2 * KDE_SIGMA_SQR)));
      }
    kde_val_1 = sum_gauss > 0.5f ? sum_1 / sum_gauss : sum_1 * 2.0f;
    kde_val_2 = sum_gauss > 0.5f ? sum_2 / sum_gauss : sum_2 * 2.0f;
    kde_val_3 = sum_gauss > 0.5f ? sum_3 / sum_gauss : sum_3 * 2.0f;
  }

  //select hypothesis
  float phase_final, max_val;
  if(kde_val_2 > kde_val_1 || kde_val_3 > kde_val_1)
  {
    if(kde_val_3 > kde_val_2)
    {
      phase_final = phase_third;
      max_val = kde_val_3;
    }
    else
    {
      phase_final = phase_second;
      max_val = kde_val_2;
    }
  }
  else
  {
    phase_final = phase_first;
    max_val = kde_val_1;
  }

  float zmultiplier = z_table[i];
  float xmultiplier = x_table[i];

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * UNAMBIGIOUS_DIST * 2.0;

  bool cond1 =  true && 0.0f < depth_linear && 0.0f < max_depth;

  xmultiplier = (xmultiplier * 90.0f) / (max_depth * max_depth * 8192.0f);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);
  depth_fit = depth_fit < 0.0f ? 0.0f : depth_fit;

  float d = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z

  max_val = depth_linear < MIN_DEPTH || depth_linear > MAX_DEPTH ? 0.0f: max_val;

  //set to zero if confidence is low
  depth[i] = max_val >= KDE_THRESHOLD ? d: 0.0f;
}


