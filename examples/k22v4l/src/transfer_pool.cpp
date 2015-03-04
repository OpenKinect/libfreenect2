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

#include <libfreenect2/usb/transfer_pool.h>
#include <iostream>
#include <algorithm>

namespace libfreenect2
{
namespace usb
{

TransferPool::TransferPool(libusb_device_handle* device_handle, unsigned char device_endpoint) :
    callback_(0),
    device_handle_(device_handle),
    device_endpoint_(device_endpoint),
    buffer_(0),
    buffer_size_(0),
    enable_submit_(false)
{
}

TransferPool::~TransferPool()
{
  deallocate();
}

void TransferPool::enableSubmission()
{
  enable_submit_ = true;
}

void TransferPool::disableSubmission()
{
  enable_submit_ = false;
}

void TransferPool::deallocate()
{
  for(TransferQueue::iterator it = idle_transfers_.begin(); it != idle_transfers_.end(); ++it)
  {
    libusb_free_transfer(*it);
  }
  idle_transfers_.clear();

  if(buffer_ != 0)
  {
    delete[] buffer_;
    buffer_ = 0;
    buffer_size_ = 0;
  }
}

void TransferPool::submit(size_t num_parallel_transfers)
{
  if(!enable_submit_)
  {
    std::cerr << "[TransferPool::submit] transfer submission disabled!" << std::endl;
    return;
  }

  if(idle_transfers_.size() < num_parallel_transfers)
  {
    std::cerr << "[TransferPool::submit] too few idle transfers!" << std::endl;
  }

  for(size_t i = 0; i < num_parallel_transfers; ++i)
  {
    libusb_transfer *transfer = idle_transfers_.front();
    idle_transfers_.pop_front();

    int r = libusb_submit_transfer(transfer);

    // put transfer in pending queue on success otherwise put it back in the idle queue
    if(r == LIBUSB_SUCCESS)
    {
      pending_transfers_.push_back(transfer);
    }
    else
    {
      idle_transfers_.push_back(transfer);
      std::cerr << "[TransferPool::submit] failed to submit transfer" << std::endl;
    }
  }
}

void TransferPool::cancel()
{
  for(TransferQueue::iterator it = pending_transfers_.begin(); it != pending_transfers_.end(); ++it)
  {
    int r = libusb_cancel_transfer(*it);

    if(r != LIBUSB_SUCCESS)
    {
      // TODO: error reporting
    }
  }

  //idle_transfers_.insert(idle_transfers_.end(), pending_transfers_.begin(), pending_transfers_.end());
}

void TransferPool::setCallback(DataCallback *callback)
{
  callback_ = callback;
}

void TransferPool::allocateTransfers(size_t num_transfers, size_t transfer_size)
{
  buffer_size_ = num_transfers * transfer_size;
  buffer_ = new unsigned char[buffer_size_];

  unsigned char *ptr = buffer_;

  for(size_t i = 0; i < num_transfers; ++i)
  {
    libusb_transfer *transfer = allocateTransfer();
    fillTransfer(transfer);

    transfer->dev_handle = device_handle_;
    transfer->endpoint = device_endpoint_;
    transfer->buffer = ptr;
    transfer->length = transfer_size;
    transfer->timeout = 1000;
    transfer->callback = (libusb_transfer_cb_fn) &TransferPool::onTransferCompleteStatic;
    transfer->user_data = this;

    idle_transfers_.push_back(transfer);

    ptr += transfer_size;
  }
}

void TransferPool::onTransferCompleteStatic(libusb_transfer* transfer)
{
  reinterpret_cast<TransferPool*>(transfer->user_data)->onTransferComplete(transfer);
}

void TransferPool::onTransferComplete(libusb_transfer* transfer)
{
  // remove transfer from pending queue - should be fast as it is somewhere at the front
  TransferQueue::iterator it = std::find(pending_transfers_.begin(), pending_transfers_.end(), transfer);

  if(it == pending_transfers_.end())
  {
    // TODO: error reporting
  }

  pending_transfers_.erase(it);

  // process data
  processTransfer(transfer);

  // put transfer back in idle queue
  idle_transfers_.push_back(transfer);

  // submit new transfer
  submit(1);
}

BulkTransferPool::BulkTransferPool(libusb_device_handle* device_handle, unsigned char device_endpoint) :
    TransferPool(device_handle, device_endpoint)
{
}

BulkTransferPool::~BulkTransferPool()
{
}

void BulkTransferPool::allocate(size_t num_transfers, size_t transfer_size)
{
  allocateTransfers(num_transfers, transfer_size);
}

libusb_transfer* BulkTransferPool::allocateTransfer()
{
  return libusb_alloc_transfer(0);
}

void BulkTransferPool::fillTransfer(libusb_transfer* transfer)
{
  transfer->type = LIBUSB_TRANSFER_TYPE_BULK;
}

void BulkTransferPool::processTransfer(libusb_transfer* transfer)
{
  if(transfer->status != LIBUSB_TRANSFER_COMPLETED) return;

  if(callback_)
    callback_->onDataReceived(transfer->buffer, transfer->actual_length);
}

IsoTransferPool::IsoTransferPool(libusb_device_handle* device_handle, unsigned char device_endpoint) :
    TransferPool(device_handle, device_endpoint),
    num_packets_(0),
    packet_size_(0)
{
}

IsoTransferPool::~IsoTransferPool()
{
}

void IsoTransferPool::allocate(size_t num_transfers, size_t num_packets, size_t packet_size)
{
  num_packets_ = num_packets;
  packet_size_ = packet_size;

  allocateTransfers(num_transfers, num_packets_ * packet_size_);
}

libusb_transfer* IsoTransferPool::allocateTransfer()
{
  return libusb_alloc_transfer(num_packets_);
}

void IsoTransferPool::fillTransfer(libusb_transfer* transfer)
{
  transfer->type = LIBUSB_TRANSFER_TYPE_ISOCHRONOUS;
  transfer->num_iso_packets = num_packets_;

  libusb_set_iso_packet_lengths(transfer, packet_size_);
}

void IsoTransferPool::processTransfer(libusb_transfer* transfer)
{
  unsigned char *ptr = transfer->buffer;

  for(size_t i = 0; i < num_packets_; ++i)
  {
    if(transfer->iso_packet_desc[i].status != LIBUSB_TRANSFER_COMPLETED) continue;

    if(callback_)
      callback_->onDataReceived(ptr, transfer->iso_packet_desc[i].actual_length);

    ptr += transfer->iso_packet_desc[i].length;
  }
}

} /* namespace usb */
} /* namespace libfreenect2 */

