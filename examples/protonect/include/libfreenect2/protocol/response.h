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

namespace libfreenect2
{
namespace protocol
{

class SerialNumberResponse
{
private:
  std::string serial_;
public:
  SerialNumberResponse(const unsigned char *data, int length)
  {
    char *c = new char[length / 2];

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
    uint16_t minor;
    uint16_t major;
    uint16_t build;
    uint16_t revision;
    uint16_t reserved0[4];

    FWSubsystemVersion()
    {
      major = 0;
      minor = 0;
      build = 0;
      revision = 0;
    }
  };

  std::vector<FWSubsystemVersion> versions_;
public:
  FirmwareVersionResponse(const unsigned char *data, int length)
  {
    int n = length / sizeof(FWSubsystemVersion);
    const FWSubsystemVersion *sv = reinterpret_cast<const FWSubsystemVersion *>(data);

    for(int i = 0; i < n && sv->major > 0; ++i, ++sv)
    {
      versions_.push_back(*sv);
    }
  }

  std::string toString()
  {
    FWSubsystemVersion max;
    for(int i = 0; i < versions_.size(); ++i)
    {
      max.major = std::max<uint16_t>(max.major, versions_[i].major);
      max.minor = std::max<uint16_t>(max.minor, versions_[i].minor);
      max.build = std::max<uint16_t>(max.build, versions_[i].build);
      max.revision = std::max<uint16_t>(max.revision, versions_[i].revision);
    }
    std::stringstream version_string;
    version_string << max.major << "." << max.minor << "." << max.build << "." << max.revision << "." << versions_.size();

    return version_string.str();
  }
};

class GenericResponse
{
private:
  std::string dump_;
public:
  GenericResponse(const unsigned char *data, int length)
  {
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
        char c = data[i*16+j];
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

  // this block contains at least some color camera intrinsic params
  float intrinsics[25];

  // perhaps related to xtable/ztable in the deconvolution code.
  // data seems to be arranged into two tables of 28*23, which
  // matches the depth image aspect ratio of 512*424 very closely
  float table1[28 * 23 * 4];
  float table2[28 * 23];
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
