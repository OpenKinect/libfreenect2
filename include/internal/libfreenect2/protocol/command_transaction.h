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

#ifndef COMMAND_TRANSACTION_H_
#define COMMAND_TRANSACTION_H_

#include <vector>
#include <libusb.h>
#include <libfreenect2/protocol/command.h>

namespace libfreenect2
{
namespace protocol
{

class CommandTransaction
{
public:
  static const int ResponseCompleteLength = 16;
  static const uint32_t ResponseCompleteMagic = 0x0A6FE000;

  typedef std::vector<unsigned char> Result;

  CommandTransaction(libusb_device_handle *handle, int inbound_endpoint, int outbound_endpoint);
  ~CommandTransaction();

  bool execute(const CommandBase& command, Result& result);
private:
  libusb_device_handle *handle_;
  int inbound_endpoint_, outbound_endpoint_, timeout_;
  Result response_complete_result_;

  bool send(const CommandBase& command);

  bool receive(Result& result, uint32_t min_length);

  bool isResponseCompleteResult(Result& result, uint32_t sequence);
};

} /* namespace protocol */
} /* namespace libfreenect2 */
#endif /* COMMAND_TRANSACTION_H_ */
