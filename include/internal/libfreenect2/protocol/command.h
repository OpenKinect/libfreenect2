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

#ifndef COMMAND_H_
#define COMMAND_H_

#include <stdint.h>
#include "libfreenect2/protocol/response.h"

#define KCMD_READ_FIRMWARE_VERSIONS 0x02
#define KCMD_INIT_STREAMS 0x09
#define KCMD_READ_HARDWARE_INFO 0x14
#define KCMD_READ_STATUS 0x16
#define KCMD_READ_DATA_PAGE 0x22
#define KCMD_READ_DATA_0x26 0x26

#define KCMD_SET_STREAMING 0x2B
#define KCMD_SET_MODE 0x4B

#define KCMD_0x46 0x46
#define KCMD_0x47 0x47

// observed in sensor stop/shutdown sequence
#define KCMD_STOP 0x0A
#define KCMD_SHUTDOWN 0x00

namespace libfreenect2
{

namespace protocol
{

template<int NParam>
struct CommandData
{
  uint32_t magic;
  uint32_t sequence;
  uint32_t max_response_length;
  uint32_t command;
  uint32_t reserved0;
  uint32_t parameters[NParam];

  CommandData()
  {
    for(int i = 0; i < NParam; ++i)
      parameters[i] = 0;
  }
};

template<>
struct CommandData<0>
{
  uint32_t magic;
  uint32_t sequence;
  uint32_t max_response_length;
  uint32_t command;
  uint32_t reserved0;
};

struct CommandBase
{
  virtual ~CommandBase() {}

  virtual uint32_t sequence() const = 0;
  virtual uint32_t maxResponseLength() const = 0;
  virtual uint32_t minResponseLength() const = 0;

  virtual const uint8_t *data() const = 0;
  virtual uint32_t size() const = 0;
};

template<uint32_t CommandId, uint32_t MaxResponseLength, uint32_t MinResponseLength, uint32_t NParam>
class Command : public CommandBase
{
public:
  typedef CommandData<NParam> Data;

  static const uint32_t MagicNumber = 0x06022009;
  static const uint32_t Size = sizeof(Data);

  uint32_t min_response_length;

  Command(uint32_t seq)
  {
    data_.magic = MagicNumber;
    data_.sequence = seq;
    data_.max_response_length = MaxResponseLength;
    data_.command = CommandId;
    data_.reserved0 = 0;
    min_response_length = MinResponseLength;
  }

  virtual ~Command()
  {
  }

  virtual uint32_t sequence() const
  {
    return data_.sequence;
  }

  virtual uint32_t maxResponseLength() const
  {
    return data_.max_response_length;
  }

  virtual uint32_t minResponseLength() const
  {
    return min_response_length;
  }

  virtual const uint8_t *data() const
  {
    return reinterpret_cast<const uint8_t *>(&data_);
  }

  virtual uint32_t size() const
  {
    return Size;
  }

protected:
  Data data_;
};

template<uint32_t CommandId, uint32_t MaxResponseLength>
struct CommandWith0Params : public Command<CommandId, MaxResponseLength, MaxResponseLength, 0>
{
  CommandWith0Params(uint32_t seq) : Command<CommandId, MaxResponseLength, MaxResponseLength, 0>(seq)
  {
  }
};

template<uint32_t CommandId, uint32_t MaxResponseLength, uint32_t Param1>
struct CommandWith1Param : public Command<CommandId, MaxResponseLength, MaxResponseLength, 1>
{
  CommandWith1Param(uint32_t seq) : Command<CommandId, MaxResponseLength, MaxResponseLength, 1>(seq)
  {
    this->data_.parameters[0] = Param1;
  }
};

template<uint32_t CommandId, uint32_t MaxResponseLength, uint32_t MinResponseLength, uint32_t Param1>
struct CommandWith1ParamLarge : public Command<CommandId, MaxResponseLength, MinResponseLength, 1>
{
  CommandWith1ParamLarge(uint32_t seq) : Command<CommandId, MaxResponseLength, MinResponseLength, 1>(seq)
  {
    this->data_.parameters[0] = Param1;
  }
};

template<uint32_t CommandId, uint32_t MaxResponseLength, uint32_t Param1, uint32_t Param2 = 0, uint32_t Param3 = 0, uint32_t Param4 = 0>
struct CommandWith4Params : public Command<CommandId, MaxResponseLength, MaxResponseLength, 4>
{
  CommandWith4Params(uint32_t seq) : Command<CommandId, MaxResponseLength, MaxResponseLength, 4>(seq)
  {
    this->data_.parameters[0] = Param1;
    this->data_.parameters[1] = Param2;
    this->data_.parameters[2] = Param3;
    this->data_.parameters[3] = Param4;
  }
};

typedef CommandWith0Params<0x02, 0x200> ReadFirmwareVersionsCommand;

typedef CommandWith0Params<KCMD_READ_HARDWARE_INFO, 0x5C> ReadHardwareInfoCommand;

typedef CommandWith0Params<KCMD_INIT_STREAMS, 0x00> InitStreamsCommand;

typedef CommandWith1Param<KCMD_READ_DATA_PAGE, 0x80, 0x01> ReadSerialNumberCommand;
typedef CommandWith1ParamLarge<KCMD_READ_DATA_PAGE, 0x1C0000, sizeof(P0TablesResponse), 0x02> ReadP0TablesCommand;
typedef CommandWith1ParamLarge<KCMD_READ_DATA_PAGE, 0x1C0000, sizeof(DepthCameraParamsResponse), 0x03> ReadDepthCameraParametersCommand;
typedef CommandWith1ParamLarge<KCMD_READ_DATA_PAGE, 0x1C0000, sizeof(RgbCameraParamsResponse), 0x04> ReadRgbCameraParametersCommand;

typedef CommandWith1Param<KCMD_READ_STATUS, 0x04, 0x090000> ReadStatus0x090000Command;
typedef CommandWith1Param<KCMD_READ_STATUS, 0x04, 0x100007> ReadStatus0x100007Command;
// TODO: are they ever used?!
typedef CommandWith1Param<KCMD_READ_STATUS, 0x04, 0x02006F> ReadStatus0x02006FCommand;
typedef CommandWith1Param<KCMD_READ_STATUS, 0x04, 0x020070> ReadStatus0x020070Command;

// TODO: is the following actually correct?
typedef CommandWith0Params<KCMD_READ_DATA_0x26, 0x10> ReadData0x26Command;
//typedef CommandWith1Param<KCMD_READ_DATA_0x26, 0x10, 0x00> ReadData0x26_0x00Command;

typedef CommandWith1Param<KCMD_SET_STREAMING, 0x00, 0x00> SetStreamDisabledCommand;
typedef CommandWith1Param<KCMD_SET_STREAMING, 0x00, 0x01> SetStreamEnabledCommand;

// TODO: are they ever used?!
typedef CommandWith4Params<KCMD_0x46, 0x00, 0x00, 0x00003840, 0x00000037, 0x00> Unknown0x46Command;
typedef CommandWith0Params<KCMD_0x47, 0x10> Unknown0x47Command;

typedef CommandWith0Params<KCMD_STOP, 0x00> StopCommand;
typedef CommandWith0Params<KCMD_SHUTDOWN, 0x00> ShutdownCommand;

typedef CommandWith4Params<KCMD_SET_MODE, 0x00, 0x00> SetModeDisabledCommand;
typedef CommandWith4Params<KCMD_SET_MODE, 0x00, 0x01> SetModeEnabledCommand;
typedef CommandWith4Params<KCMD_SET_MODE, 0x00, 0x01, 0x00640064> SetModeEnabledWith0x00640064Command;
typedef CommandWith4Params<KCMD_SET_MODE, 0x00, 0x01, 0x00500050> SetModeEnabledWith0x00500050Command;
} /* namespace protocol */
} /* namespace libfreenect2 */
#endif /* COMMAND_H_ */
