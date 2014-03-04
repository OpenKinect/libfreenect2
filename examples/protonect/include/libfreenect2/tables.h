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

#ifndef _TABLES_H_
#define _TABLES_H_

// "P0" coefficient tables, input to the deconvolution code
struct __attribute__ ((__packed__)) p0tables {

  uint32_t headersize;
  uint32_t unknown1;
  uint32_t unknown2;
  uint32_t tablesize;
  uint32_t unknown3;
  uint32_t unknown4;
  uint32_t unknown5;
  uint32_t unknown6;

  uint16_t unknown7;
  uint16_t p0table0[512*424]; // row[0] == row[511] == 0x2c9a
  uint16_t unknown8;

  uint16_t unknown9;
  uint16_t p0table1[512*424]; // row[0] == row[511] == 0x08ec
  uint16_t unknownA;

  uint16_t unknownB;
  uint16_t p0table2[512*424]; // row[0] == row[511] == 0x42e8
  uint16_t unknownC;

  uint8_t  unknownD[];
};


// probably some combination of color camera intrinsics + depth coefficient tables
struct __attribute__ ((__packed__)) CameraParams {

  // unknown, always seen as 1 so far
  uint8_t table_id;

  // this block contains at least some color camera intrinsic params
  float intrinsics[25];

  // perhaps related to xtable/ztable in the deconvolution code.
  // data seems to be arranged into two tables of 28*23, which
  // matches the depth image aspect ratio of 512*424 very closely
  float table1[28*23*4];
  float table2[28*23];
};


// depth camera intrinsic & distortion parameters
struct __attribute__ ((__packed__)) DepthCameraParams {

  // intrinsics (this is pretty certain)
  float fx;
  float fy;
  float unknown1; // assumed to be always zero
  float cx;
  float cy;

  // radial distortion (educated guess based on calibration data from Kinect SDK)
  float k1;
  float k2;
  float p1; // always seen as zero so far, so purely a guess
  float p2; // always seen as zero so far, so purely a guess
  float k3;

  float unknown2[13]; // assumed to be always zero
};

#endif // _TABLES_H_
