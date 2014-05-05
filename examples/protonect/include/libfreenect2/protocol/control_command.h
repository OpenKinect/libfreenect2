/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2011 individual OpenKinect contributors. See the CONTRIB file
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

#ifndef CONTROL_COMMAND_H_
#define CONTROL_COMMAND_H_

#include <stdint.h>

#define KCMD_READ_FIRMWARE_VERSIONS 0x02
#define KCMD_INIT_STREAMS 0x09
#define KCMD_READ_DATA_0x14 0x14
#define KCMD_READ_STATUS 0x16
#define KCMD_READ_DATA_PAGE 0x22
#define KCMD_READ_DATA_0x26 0x26

#define KCMD_SET_STREAMING 0x2B
#define KCMD_SET_MODE 0x4B

namespace libfreenect2
{

namespace protocol
{

template<uint32_t TCommand, uint32_t TMaxResponseLength, int TNParam>
struct ControlCommand
{
  static const uint32_t MagicNumber = 0x06022009;
  static const uint32_t Command = TCommand;
  static const uint32_t MaxResponseLength = TMaxResponseLength;
  static const uint32_t NParam = TNParam;

  static const int Size = (5 + NParam) * sizeof(uint32_t);

  uint32_t magic;
  uint32_t sequence;
  uint32_t max_response_length;
  uint32_t command;
  uint32_t reserved0;
  uint32_t parameters[NParam];

  ControlCommand(uint32_t seq) :
    magic(MagicNumber),
    sequence(seq),
    max_response_length(MaxResponseLength),
    command(Command),
    reserved0(0)
  {
    for(int i = 0; i < NParam; ++i)
      parameters[i] = 0;
  }

  char* data() const
  {
    return reinterpret_cast<char*>(this);
  }

  int size() const
  {
    return Size;
  }
};

template<uint32_t Command, uint32_t MaxResponseLength>
struct ControlCommandWith0Params : public ControlCommand<Command, MaxResponseLength, 0>
{
};

template<uint32_t Command, uint32_t MaxResponseLength, uint32_t Param1>
struct ControlCommandWith1Param : public ControlCommand<Command, MaxResponseLength, 1>
{
  ControlCommandWith1Param(uint32_t seq) : ControlCommand<Command, MaxResponseLength, 1>(seq)
  {
    this->parameters[0] = Param1;
  }
};

typedef ControlCommandWith0Params<0x02, 0x200> ReadFirmwareVersionsCommand;

typedef ControlCommandWith0Params<KCMD_READ_DATA_0x14, 0x5C> ReadData0x14Command;

typedef ControlCommandWith0Params<KCMD_INIT_STREAMS, 0x00> InitStreamsCommand;

typedef ControlCommandWith1Param<KCMD_READ_DATA_PAGE, 0x80, 0x01> ReadData0x22_0x01Command;
typedef ControlCommandWith1Param<KCMD_READ_DATA_PAGE, 0x1C0000, 0x02> ReadP0TablesCommand;
typedef ControlCommandWith1Param<KCMD_READ_DATA_PAGE, 0x1C0000, 0x03> ReadDepthCameraParametersCommand;
typedef ControlCommandWith1Param<KCMD_READ_DATA_PAGE, 0x1C0000, 0x04> ReadRgbCameraParametersCommand;

typedef ControlCommandWith1Param<KCMD_READ_STATUS, 0x04, 0x090000> ReadStatus0x090000Command;
typedef ControlCommandWith1Param<KCMD_READ_STATUS, 0x04, 0x100007> ReadStatus0x100007Command;

typedef ControlCommandWith1Param<KCMD_READ_DATA_0x26, 0x10, 0x00> ReadData0x26_0x00Command;

typedef ControlCommandWith1Param<KCMD_SET_STREAMING, 0x00, 0x00> SetStreamDisabledCommand;
typedef ControlCommandWith1Param<KCMD_SET_STREAMING, 0x00, 0x01> SetStreamEnabledCommand;

template<uint32_t Param1>
struct SetModeControlCommand : public ControlCommand<KCMD_SET_MODE, 0x00, 4>
{
  SetModeControlCommand(uint32_t seq) : ControlCommand<KCMD_SET_MODE, 0x00, 4>(seq)
  {
    this->parameters[0] = Param1;
  }
};

typedef SetModeControlCommand<0x00> SetModeDisabledCommand;
typedef SetModeControlCommand<0x01> SetModeEnabledCommand;
} /* namespace protocol */
} /* namespace libfreenect2 */
#endif /* CONTROL_COMMAND_H_ */
