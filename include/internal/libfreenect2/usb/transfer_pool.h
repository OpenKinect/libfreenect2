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

#ifndef TRANSFER_POOL_H_
#define TRANSFER_POOL_H_

#include <vector>
#include <libusb.h>

#include <libfreenect2/data_callback.h>
#include <libfreenect2/threading.h>

namespace libfreenect2
{

namespace usb
{

class TransferPool
{
public:
  TransferPool(libusb_device_handle *device_handle, unsigned char device_endpoint);
  virtual ~TransferPool();

  void deallocate();

  void enableSubmission();

  void disableSubmission();

  bool enabled();

  bool submit();

  void cancel();

  void setCallback(DataCallback *callback);
protected:
  libfreenect2::mutex stopped_mutex;
  struct Transfer
  {
    libusb_transfer *transfer;
    TransferPool *pool;
    bool stopped;
    Transfer(libusb_transfer *transfer, TransferPool *pool):
      transfer(transfer), pool(pool), stopped(true) {}
    void setStopped(bool value)
    {
      libfreenect2::lock_guard guard(pool->stopped_mutex);
      stopped = value;
    }
    bool getStopped()
    {
      libfreenect2::lock_guard guard(pool->stopped_mutex);
      return stopped;
    }
  };

  void allocateTransfers(size_t num_transfers, size_t transfer_size);

  virtual libusb_transfer *allocateTransfer() = 0;
  virtual void fillTransfer(libusb_transfer *transfer) = 0;

  virtual void processTransfer(libusb_transfer *transfer) = 0;

  DataCallback *callback_;
private:
  typedef std::vector<Transfer> TransferQueue;

  libusb_device_handle *device_handle_;
  unsigned char device_endpoint_;

  TransferQueue transfers_;
  unsigned char *buffer_;
  size_t buffer_size_;

  bool enable_submit_;

  static void onTransferCompleteStatic(libusb_transfer *transfer);

  void onTransferComplete(Transfer *transfer);
};

class BulkTransferPool : public TransferPool
{
public:
  BulkTransferPool(libusb_device_handle *device_handle, unsigned char device_endpoint);
  virtual ~BulkTransferPool();

  void allocate(size_t num_transfers, size_t transfer_size);

protected:
  virtual libusb_transfer *allocateTransfer();
  virtual void fillTransfer(libusb_transfer *transfer);
  virtual void processTransfer(libusb_transfer *transfer);
};

class IsoTransferPool : public TransferPool
{
public:
  IsoTransferPool(libusb_device_handle *device_handle, unsigned char device_endpoint);
  virtual ~IsoTransferPool();

  void allocate(size_t num_transfers, size_t num_packets, size_t packet_size);

protected:
  virtual libusb_transfer *allocateTransfer();
  virtual void fillTransfer(libusb_transfer *transfer);
  virtual void processTransfer(libusb_transfer *transfer);

private:
  size_t num_packets_;
  size_t packet_size_;
};

} /* namespace usb */
} /* namespace libfreenect2 */
#endif /* TRANSFER_POOL_H_ */
