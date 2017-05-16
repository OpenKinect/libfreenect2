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

/** @file command_transaction.cpp Protocol transactions for device. */

#include <libfreenect2/protocol/command_transaction.h>
#include <libfreenect2/logging.h>

#include <stdint.h>

#define WRITE_LIBUSB_ERROR(__RESULT) libusb_error_name(__RESULT) << " " << libusb_strerror((libusb_error)__RESULT)

namespace libfreenect2
{
namespace protocol
{
CommandTransaction::CommandTransaction(libusb_device_handle *handle, int inbound_endpoint, int outbound_endpoint) :
  handle_(handle),
  inbound_endpoint_(inbound_endpoint),
  outbound_endpoint_(outbound_endpoint),
  timeout_(1000)
{
}

CommandTransaction::~CommandTransaction() {}

bool CommandTransaction::execute(const CommandBase& command, Result& result)
{
  result.resize(command.maxResponseLength());
  response_complete_result_.resize(ResponseCompleteLength);

  // send command
  if (!send(command))
    return false;

  // receive response data
  if(command.maxResponseLength() > 0)
  {
    if (!receive(result, command.minResponseLength()))
      return false;
    if (isResponseCompleteResult(result, command.sequence()))
    {
      LOG_ERROR << "received premature response complete!";
      return false;
    }
  }

  // receive response complete
  if (!receive(response_complete_result_, ResponseCompleteLength))
    return false;
  if (!isResponseCompleteResult(response_complete_result_, command.sequence()))
  {
    LOG_ERROR << "missing response complete!";
    return false;
  }

  return true;
}

bool CommandTransaction::send(const CommandBase& command)
{
  int transferred_bytes = 0;
  int r = libusb_bulk_transfer(handle_, outbound_endpoint_, const_cast<uint8_t *>(command.data()), command.size(), &transferred_bytes, timeout_);

  if(r != LIBUSB_SUCCESS)
  {
    LOG_ERROR << "bulk transfer failed: " << WRITE_LIBUSB_ERROR(r);
    return false;
  }

  if((size_t)transferred_bytes != command.size())
  {
    LOG_ERROR << "sent number of bytes differs from expected number! expected: " << command.size() << " got: " << transferred_bytes;
    return false;
  }

  return true;
}

bool CommandTransaction::receive(CommandTransaction::Result& result, uint32_t min_length)
{
  int length = 0;

  int r = libusb_bulk_transfer(handle_, inbound_endpoint_, &result[0], result.size(), &length, timeout_);
  result.resize(length);

  if(r != LIBUSB_SUCCESS)
  {
    LOG_ERROR << "bulk transfer failed: " << WRITE_LIBUSB_ERROR(r);
    return false;
  }

  if ((uint32_t)length < min_length)
  {
    LOG_ERROR << "bulk transfer too short! expected at least: " << min_length << " got : " << length;
    return false;
  }

  return true;
}

bool CommandTransaction::isResponseCompleteResult(CommandTransaction::Result& result, uint32_t sequence)
{
  if(result.size() == ResponseCompleteLength)
  {
    uint32_t *data = reinterpret_cast<uint32_t *>(&result[0]);

    if(data[0] == ResponseCompleteMagic)
    {
      if(data[1] != sequence)
      {
        LOG_ERROR << "response complete with wrong sequence number! expected: " << sequence << " got: " << data[1];
      }
      return true;
    }
  }

  return false;
}


} /* namespace protocol */
} /* namespace libfreenect2 */
