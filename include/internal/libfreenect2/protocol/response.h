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

#ifndef RESPONSE_H_
#define RESPONSE_H_

#include <vector>
#include <sstream>
#include <iomanip>
#include <stdint.h>
#include <algorithm>
#include <libfreenect2/config.h>
#include <libfreenect2/libfreenect2.hpp>

namespace libfreenect2
{
namespace protocol
{

class SerialNumberResponse
{
private:
  std::string serial_;
public:
  SerialNumberResponse(const std::vector<unsigned char> &data)
  {
    int length = data.size();
    char *c = new char[length / 2 + 1]();

    for(int i = 0, j = 0; i < length; i += 2, ++j)
    {
      c[j] = (char)data[i];
      if(c[j] == 0) break;
    }

    serial_.assign(c);

    delete[] c;
  }

  std::string toString()
  {
    return serial_;
  }
};

class FirmwareVersionResponse
{
private:
  struct FWSubsystemVersion
  {
    uint32_t maj_min;
    uint32_t revision;
    uint32_t build;
    uint32_t reserved0;

    FWSubsystemVersion()
    {
      maj_min = 0;
      revision = 0;
      build = 0;
    }
  };

  std::vector<FWSubsystemVersion> versions_;
public:
  FirmwareVersionResponse(const std::vector<unsigned char> &data)
  {
    int length = data.size();
    int n = length / sizeof(FWSubsystemVersion);
    const FWSubsystemVersion *sv = reinterpret_cast<const FWSubsystemVersion *>(&data[0]);

    for(int i = 0; i < 7 && i < n; ++i)
    {
      versions_.push_back(sv[i]);
    }
  }

  std::string toString()
  {
    FWSubsystemVersion max;
    std::stringstream version_string;
    // the main blob's index
    size_t i = 3;
    if (i < versions_.size())
    {
      const FWSubsystemVersion &ver = versions_[i];
      version_string << (ver.maj_min >> 16) << "." << (ver.maj_min & 0xffff) << "." << ver.revision << "." << ver.build;
    }

    return version_string.str();
  }
};

class Status0x090000Response
{
private:
  uint32_t status_;
public:
  Status0x090000Response(const std::vector<unsigned char> &data)
  {
    status_ = *reinterpret_cast<const uint32_t *>(&data[0]);
  }

  uint32_t toNumber()
  {
    return status_;
  }
};

class GenericResponse
{
private:
  std::string dump_;
public:
  GenericResponse(const std::vector<unsigned char> &data)
  {
    int length = data.size();
    std::stringstream dump;
    dump << length << " bytes of raw data" << std::endl;

    int lines = length >> 4;
    if (length % 16 != 0) lines += 1;

    for (int i = 0; i < lines; i++)
    {
      dump << "0x" << std::hex << std::setfill('0') << std::setw(4) << (i*16) << ":  ";
      for (int j = 0; j < 16; j++)
      {
        if (j < length) dump << std::hex << std::setfill('0') << std::setw(2) << int(data[i*16+j]) << " ";
        else dump << "   ";
      }
      dump << "   ";
      for (int j = 0; (j < 16) && (j < length); j++)
      {
        unsigned char c = data[i*16+j];
        dump << (((c<32)||(c>128))?'.':c);
      }
      dump << std::endl;
      length -= 16;
    }

    dump_ = dump.str();
  }

  std::string toString()
  {
    return dump_;
  }
};

// probably some combination of color camera intrinsics + depth coefficient tables
LIBFREENECT2_PACK(struct RgbCameraParamsResponse
{
  // unknown, always seen as 1 so far
  uint8_t table_id;

  // color -> depth mapping parameters
  float color_f;
  float color_cx;
  float color_cy;

  float shift_d;
  float shift_m;

  float mx_x3y0; // xxx
  float mx_x0y3; // yyy
  float mx_x2y1; // xxy
  float mx_x1y2; // yyx
  float mx_x2y0; // xx
  float mx_x0y2; // yy
  float mx_x1y1; // xy
  float mx_x1y0; // x
  float mx_x0y1; // y
  float mx_x0y0; // 1

  float my_x3y0; // xxx
  float my_x0y3; // yyy
  float my_x2y1; // xxy
  float my_x1y2; // yyx
  float my_x2y0; // xx
  float my_x0y2; // yy
  float my_x1y1; // xy
  float my_x1y0; // x
  float my_x0y1; // y
  float my_x0y0; // 1

  // perhaps related to xtable/ztable in the deconvolution code.
  // data seems to be arranged into two tables of 28*23, which
  // matches the depth image aspect ratio of 512*424 very closely
  float table1[28 * 23 * 4];
  float table2[28 * 23];

  RgbCameraParamsResponse(const std::vector<unsigned char> &data)
  {
    *this = *reinterpret_cast<const RgbCameraParamsResponse *>(&data[0]);
  }

  Freenect2Device::ColorCameraParams toColorCameraParams()
  {
    Freenect2Device::ColorCameraParams params;
    params.fx = color_f;
    params.fy = color_f;
    params.cx = color_cx;
    params.cy = color_cy;

    params.shift_d = shift_d;
    params.shift_m = shift_m;

    params.mx_x3y0 = mx_x3y0; // xxx
    params.mx_x0y3 = mx_x0y3; // yyy
    params.mx_x2y1 = mx_x2y1; // xxy
    params.mx_x1y2 = mx_x1y2; // yyx
    params.mx_x2y0 = mx_x2y0; // xx
    params.mx_x0y2 = mx_x0y2; // yy
    params.mx_x1y1 = mx_x1y1; // xy
    params.mx_x1y0 = mx_x1y0; // x
    params.mx_x0y1 = mx_x0y1; // y
    params.mx_x0y0 = mx_x0y0; // 1

    params.my_x3y0 = my_x3y0; // xxx
    params.my_x0y3 = my_x0y3; // yyy
    params.my_x2y1 = my_x2y1; // xxy
    params.my_x1y2 = my_x1y2; // yyx
    params.my_x2y0 = my_x2y0; // xx
    params.my_x0y2 = my_x0y2; // yy
    params.my_x1y1 = my_x1y1; // xy
    params.my_x1y0 = my_x1y0; // x
    params.my_x0y1 = my_x0y1; // y
    params.my_x0y0 = my_x0y0; // 1
    return params;
  }
});


// depth camera intrinsic & distortion parameters
LIBFREENECT2_PACK(struct DepthCameraParamsResponse
{
  // intrinsics (this is pretty certain)
  float fx;
  float fy;
  float unknown0; // assumed to be always zero
  float cx;
  float cy;

  // radial distortion (educated guess based on calibration data from Kinect SDK)
  float k1;
  float k2;
  float p1; // always seen as zero so far, so purely a guess
  float p2; // always seen as zero so far, so purely a guess
  float k3;

  float unknown1[13]; // assumed to be always zero

  DepthCameraParamsResponse(const std::vector<unsigned char> &data)
  {
    *this = *reinterpret_cast<const DepthCameraParamsResponse *>(&data[0]);
  }

  Freenect2Device::IrCameraParams toIrCameraParams()
  {
    Freenect2Device::IrCameraParams params;
    params.fx = fx;
    params.fy = fy;
    params.cx = cx;
    params.cy = cy;
    params.k1 = k1;
    params.k2 = k2;
    params.k3 = k3;
    params.p1 = p1;
    params.p2 = p2;
    return params;
  }
});

// "P0" coefficient tables, input to the deconvolution code
LIBFREENECT2_PACK(struct P0TablesResponse
{
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
});

} /* namespace protocol */
} /* namespace libfreenect2 */
#endif /* RESPONSE_H_ */
