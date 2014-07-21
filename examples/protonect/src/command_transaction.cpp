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

#include <libfreenect2/protocol/command_transaction.h>

#include <stdint.h>
#include <iostream>

namespace libfreenect2
{
namespace protocol
{
CommandTransaction::Result::Result() :
    code(Error), data(NULL), capacity(0), length(0)
{
}

CommandTransaction::Result::~Result()
{
  deallocate();
}

void CommandTransaction::Result::allocate(size_t size)
{
  if (capacity < size)
  {
    deallocate();
    data = new unsigned char[size];
    capacity = size;
  }
  length = 0;
}

void CommandTransaction::Result::deallocate()
{
  if (data != NULL)
  {
    delete[] data;
  }
  length = 0;
  capacity = 0;
}

bool CommandTransaction::Result::notSuccessfulThenDeallocate()
{
  bool not_successful = (code != Success);

  if (not_successful)
  {
    deallocate();
  }

  return not_successful;
}

CommandTransaction::CommandTransaction(libusb_device_handle *handle, int inbound_endpoint, int outbound_endpoint) :
  handle_(handle),
  inbound_endpoint_(inbound_endpoint),
  outbound_endpoint_(outbound_endpoint),
  timeout_(1000)
{
  response_complete_result_.allocate(ResponseCompleteLength);
}

CommandTransaction::~CommandTransaction() {}

void CommandTransaction::execute(const CommandBase& command, Result& result)
{
  result.allocate(command.maxResponseLength());

  // send command
  result.code = send(command);

  if(result.notSuccessfulThenDeallocate()) return;

  bool complete = false;

  // receive response data
  if(command.maxResponseLength() > 0)
  {
    receive(result);
    complete = isResponseCompleteResult(result, command.sequence());

    if(complete)
    {
      std::cerr << "[CommandTransaction::execute] received premature response complete!" << std::endl;
      result.code = Error;
    }

    if(result.notSuccessfulThenDeallocate()) return;
  }

  // receive response complete
  receive(response_complete_result_);
  complete = isResponseCompleteResult(response_complete_result_, command.sequence());

  if(!complete)
  {
    std::cerr << "[CommandTransaction::execute] missing response complete!" << std::endl;
    result.code = Error;
  }

  result.notSuccessfulThenDeallocate();
}

CommandTransaction::ResultCode CommandTransaction::send(const CommandBase& command)
{
  ResultCode code = Success;

  int transferred_bytes = 0;
  int r = libusb_bulk_transfer(handle_, outbound_endpoint_, const_cast<uint8_t *>(command.data()), command.size(), &transferred_bytes, timeout_);

  if(r != LIBUSB_SUCCESS)
  {
    std::cerr << "[CommandTransaction::send] bulk transfer failed! libusb error " << r << ": " << libusb_error_name(r) << std::endl;
    code = Error;
  }

  if(transferred_bytes != command.size())
  {
    std::cerr << "[CommandTransaction::send] sent number of bytes differs from expected number! expected: " << command.size() << " got: " << transferred_bytes << std::endl;
    code = Error;
  }

  return code;
}

void CommandTransaction::receive(CommandTransaction::Result& result)
{
  result.code = Success;
  result.length = 0;

  int r = libusb_bulk_transfer(handle_, inbound_endpoint_, result.data, result.capacity, &result.length, timeout_);

  if(r != LIBUSB_SUCCESS)
  {
    std::cerr << "[CommandTransaction::receive] bulk transfer failed! libusb error " << r << ": " << libusb_error_name(r) << std::endl;
    result.code = Error;
  }
}

bool CommandTransaction::isResponseCompleteResult(CommandTransaction::Result& result, uint32_t sequence)
{
  bool complete = false;

  if(result.code == Success && result.length == ResponseCompleteLength)
  {
    uint32_t *data = reinterpret_cast<uint32_t *>(result.data);

    if(data[0] == ResponseCompleteMagic)
    {
      complete = true;

      if(data[1] != sequence)
      {
        std::cerr << "[CommandTransaction::isResponseCompleteResult] response complete with wrong sequence number! expected: " << sequence << " got: " << data[1]<< std::endl;
      }
    }
  }

  return complete;
}


} /* namespace protocol */
} /* namespace libfreenect2 */
