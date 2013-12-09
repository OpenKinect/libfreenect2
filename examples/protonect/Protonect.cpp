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

#include "Protonect.h"

bool should_resubmit = true;
uint32_t num_iso_requests_outstanding = 0;

int KSetSensorStatus(libusb_device_handle *handle, KSensorStatus KSensorStatus)
{
	uint16_t wFeature = 0;
	uint16_t wIndex = KSensorStatus;
	unsigned char* data = NULL;
	uint16_t wLength = 0;
	unsigned int timeout = 1000;

	printf("Setting sensor status: %s\n", (KSensorStatus == KSENSOR_ENABLE ? "Enable" : "Disable"));

	int r = libusb_control_transfer(handle,
		LIBUSB_RECIPIENT_INTERFACE,
		LIBUSB_REQUEST_SET_FEATURE,
		wFeature,
		wIndex,
		data,
		wLength,
		timeout);

	if (r < 0)
	{
		perr("Set sensor status error: %d\n", r);
		return K_ERROR;
	}

	return K_SUCCESS;
}

int KSetStreamingInterfaceStatus(libusb_device_handle *handle, KStreamStatus KStreamStatus)
{
	int altSetting = KStreamStatus;
	int interfaceNum = 1;

	printf("Setting stream status: %s\n", (KStreamStatus == KSTREAM_ENABLE ? "Enable" : "Disable"));

	int r = libusb_set_interface_alt_setting(handle, interfaceNum, altSetting);

	if (r < 0)
	{
		perr("Set stream status error: %d\n", r);
		return K_ERROR;
	}

	return K_SUCCESS;
}

cmd_header KInitCommand()
{
	cmd_header cmd;
	memset(&cmd, 0, sizeof(cmd_header));

	cmd.magic = KCMD_MAGIC;
	cmd.sequence = cmd_seq;
	cmd_seq++;

	return cmd;
}

int KSendCommand(libusb_device_handle *handle, cmd_header& cmd, int length)
{
	uint8_t endpoint = 0x2;
	int transferred = 0;
	int timeout = 1000;

	uint8_t* p_data = (uint8_t*)(&cmd);

	printf("Cmd seq %u func %#04x (%#x)\n", cmd.sequence, cmd.command, cmd.parameter);
	int r = libusb_bulk_transfer(handle,
		endpoint,
		p_data,
		length,
		&transferred,
		timeout);

	if (r != LIBUSB_SUCCESS)
	{
		perr("Cmd error: %d\n", r);
		return K_ERROR;
	}

	printf("Cmd sent, %u bytes sent\n", transferred);

	return K_SUCCESS;
}

int KReadCommandResponse(libusb_device_handle *handle, cmd_header& cmd, uint32_t dataLen, uint8_t* data, int* transferred)
{
	if (NULL == transferred)
	{
		perr("Cannot read command response with transferred == NULL");
		return K_ERROR;
	}

	uint8_t endpoint = 0x81;
	int timeout = 1000;

	printf("Cmd response seq %u func %#04x (%#x)\n", cmd.sequence, cmd.command, cmd.parameter);
	int r = libusb_bulk_transfer(handle,
		endpoint,
		data,
		dataLen,
		transferred,
		timeout);

	uint32_t completedBytes = *transferred;

	if (r != LIBUSB_SUCCESS)
	{
		perr("Cmd response error: %d\n", r);
		return K_ERROR;
	}

	printf("Cmd response success, %u bytes received\n", completedBytes);

	if (completedBytes > dataLen)
	{
		perr("Warning, buffer length (%u) smaller than transferred bytes (%u).", dataLen, completedBytes);
	}

	if (completedBytes == KCMD_RESPONSE_COMPLETE_LEN)
	{
		uint32_t* i_data = (uint32_t*)data;

		uint32_t tag = i_data[0];
		uint32_t seq = i_data[1];

		if (tag == KCMD_RESPONSE_COMPLETE_MAGIC)
		{
			printf("Cmd response completed\n");
			if (seq != cmd.sequence)
			{
				perr("Cmd response completed with wrong sequence number. Expected %u, Got %u\n", cmd.sequence, seq);
			}
			return K_RESPONSE_COMPLETE;
		}
	}
	return K_RESPONSE_DATA;
}

int KSendCommandReadResponse(libusb_device_handle *handle, cmd_header& cmd, int length, uint8_t* data, int* transferred)
{
	if (NULL == transferred)
	{
		perr("Cannot read command response with transferred == NULL");
		return K_ERROR;
	}

	printf("\n===\n\n");
	int r;

	r = KSendCommand(handle, cmd, length);

	if (r == K_ERROR)
	{
		return K_ERROR;
	}

	if (data != NULL && cmd.responseDataLen > 0)
	{
		//Read data response
		r = KReadCommandResponse(handle, cmd, cmd.responseDataLen, data, transferred);

		uint32_t completedBytes = *transferred;

		if (r == K_RESPONSE_DATA)
		{
			printf("Received cmd data %#04x (%#x), length: %u\n", cmd.command, cmd.parameter, completedBytes);
		}
		else if (r == K_RESPONSE_COMPLETE)
		{
			perr("Premature response complete for %#04x cmd (%#x)\n", cmd.command, cmd.parameter);
			return K_ERROR;
		}
		else
		{
			perr("Error in response for %#04x cmd (%#x)\n", cmd.command, cmd.parameter);
			return K_ERROR;
		}
	}

	//Read expected response complete
	uint8_t responseData[512];
	int response_xf = 0;
	r = KReadCommandResponse(handle, cmd, 512, responseData, &response_xf);

	if (r == K_RESPONSE_COMPLETE)
	{
		printf("Response complete for cmd %#04x (%#x)\n", cmd.command, cmd.parameter);
	}
	else
	{
		perr("Missing expected response complete for cmd %#04x (%#x)\n", cmd.command, cmd.parameter);
		return K_ERROR;
	}

	return K_SUCCESS;
}

int KReadData02(libusb_device_handle *handle)
{
	int length = 20;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x200;
	cmd.command = KCMD_READ_DATA1;

	uint8_t* data = new uint8_t[cmd.responseDataLen];
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO parse data
	}

	delete data;
	return ret;
}

int KReadData14(libusb_device_handle *handle)
{
	int length = 20;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x5C;
	cmd.command = KCMD_READ_VERSIONS;

	uint8_t* data = new uint8_t[cmd.responseDataLen];
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO parse data
	}

	delete data;
	return ret;
}

int KReadData22_1(libusb_device_handle *handle)
{
	int length = 24;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x80;
	cmd.command = KCMD_READ_DATA_PAGE;
	cmd.parameter = 0x1;

	uint8_t* data = new uint8_t[cmd.responseDataLen];
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO parse data
	}

	delete data;
	return ret;
}

int KReadData22_2(libusb_device_handle *handle)
{
	int length = 24;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x1C0000;
	cmd.command = KCMD_READ_DATA_PAGE;
	cmd.parameter = 0x2;

	uint8_t* data = new uint8_t[cmd.responseDataLen];
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO parse data
	}

	delete data;
	return ret;
}

int KReadData22_3(libusb_device_handle *handle)
{
	int length = 24;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x1C0000;
	cmd.command = KCMD_READ_DATA_PAGE;
	cmd.parameter = 0x3;

	uint8_t* data = new uint8_t[cmd.responseDataLen];
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO parse data
	}

	delete data;
	return ret;
}

int KReadData22_4(libusb_device_handle *handle)
{
	int length = 24;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x1C0000;
	cmd.command = KCMD_READ_DATA_PAGE;
	cmd.parameter = 0x4;

	uint8_t* data = new uint8_t[cmd.responseDataLen];
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO parse data
	}

	delete data;
	return ret;
}

int KReadData16_90000(libusb_device_handle *handle)
{
	int length = 24;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x4;
	cmd.command = KCMD_READ_STATUS;
	cmd.parameter = 0x90000;

	uint8_t* data = new uint8_t[cmd.responseDataLen];
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO parse data
		uint32_t* data32 = (uint32_t*)data;
		uint32_t value = *data32;
		printf("Received status %#04x (%#x): %d\n", cmd.command, cmd.parameter, value);
	}

	delete data;
	return ret;
}

int KReadData16_100007(libusb_device_handle *handle)
{
	int length = 24;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x4;
	cmd.command = KCMD_READ_STATUS;
	cmd.parameter = 0x100007;

	uint8_t* data = new uint8_t[cmd.responseDataLen];
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO parse data
		uint32_t* data32 = (uint32_t*)data;
		uint32_t value = *data32;
		printf("Received status %#04x (%#x): %d\n", cmd.command, cmd.parameter, value);
	}

	delete data;
	return ret;
}

int KInitStreams09(libusb_device_handle *handle)
{
	int length = 20;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x0;
	cmd.command = KCMD_INIT_STREAMS;

	uint8_t* data = NULL;
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO success
	}

	return ret;
}

int KSetStreamCommand2B(libusb_device_handle *handle, KStreamStatus KStreamStatus)
{
	int length = 24;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x0;
	cmd.command = KCMD_SET_STREAMING;
	cmd.parameter = KStreamStatus;

	uint8_t* data = NULL;
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO success
		printf("Set stream status success: %s\n", (KStreamStatus == KSTREAM_ENABLE ? "Enable" : "Disable"));

	}

	return ret;
}

int KSetModeCommand4B(libusb_device_handle *handle, KModeStatus KModeStatus)
{
	int length = 36;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x0;
	cmd.command = KCMD_SET_MODE;
	cmd.parameter = KModeStatus;

	uint8_t* data = NULL;
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO success
		printf("Set mode status success: %s\n", (KModeStatus == KSTREAM_ENABLE ? "Enable" : "Disable"));

	}

	return ret;
}

int KReadData26(libusb_device_handle *handle)
{
	int length = 24;

	cmd_header cmd = KInitCommand();
	cmd.responseDataLen = 0x10;
	cmd.command = KCMD_READ_COUNT;

	uint8_t* data = new uint8_t[cmd.responseDataLen];
	int transferred = 0;

	int r = KSendCommandReadResponse(handle, cmd, length, data, &transferred);

	int ret = K_SUCCESS;

	if (r == K_ERROR)
	{
		printf("Error in cmd protocol %#04x (%#x)\n", cmd.command, cmd.parameter);
		ret = K_ERROR;
	}
	else
	{
		//TODO parse data
		uint16_t* data16 = (uint16_t*)data;
		int numValues = transferred / 2;
		for (int i = 0; i < numValues; i++)
		{
			uint16_t value = *data16;
			data16++;
			printf("Received status %#04x (%#x) %d: %d\n", cmd.command, cmd.parameter, i, value);
		}
	}

	delete data;
	return ret;
}

void InitKinect(libusb_device_handle *handle)
{
	if (NULL == handle)
		return;

	printf("running kinect...\n");
	int r;
	uint8_t bmRequestType;
	uint8_t bRequest;
	uint16_t wValue;
	uint16_t wIndex;
	unsigned char* data = NULL;
	uint16_t wLength = 0;
	unsigned int timeout = 1000;

	bmRequestType = LIBUSB_RECIPIENT_DEVICE;
	bRequest = LIBUSB_SET_ISOCH_DELAY;
	wValue = 40; //ms?
	wIndex = 0;
	wLength = 0;
	data = NULL;

	printf("Control transfer 1 - set isoch delay\n");
	r = libusb_control_transfer(handle,
		bmRequestType,
		bRequest,
		wValue,
		wIndex,
		data,
		wLength,
		timeout);
	if (r < 0)
	{
		perr("Control transfer error: %d\n", r);
	}


	bmRequestType = LIBUSB_RECIPIENT_DEVICE;
	bRequest = LIBUSB_REQUEST_SET_SEL;
	wValue = 0;
	wIndex = 0;
	wLength = 6;
	unsigned char seldata[] = { 0x55, 0x00, 0x55, 0x00, 0x00, 0x00 };

	printf("Control transfer 2 - set sel u1/u2\n");
	r = libusb_control_transfer(handle,
		bmRequestType,
		bRequest,
		wValue,
		wIndex,
		seldata,
		wLength,
		timeout);
	if (r < 0)
	{
		perr("Control transfer error: %d\n", r);
	}

	int configId = 1;
	printf("Setting config: %d\n", configId);
	r = libusb_set_configuration(handle, configId);
	if (r != LIBUSB_SUCCESS)
	{
		perr("  Can't set configuration\n");
	}

	printf("\nSetting interface alt setting...\n");
	r = KSetStreamingInterfaceStatus(handle, KSTREAM_DISABLE);
	if (r != LIBUSB_SUCCESS) {
		perr("   Failed: %d.\n", r);
	}

	bmRequestType = LIBUSB_RECIPIENT_DEVICE;
	bRequest = LIBUSB_REQUEST_SET_FEATURE;
	wValue = U1_ENABLE;
	wIndex = 0;
	wLength = 0;

	printf("Control transfer 3 - enable u1\n");
	r = libusb_control_transfer(handle,
		bmRequestType,
		bRequest,
		wValue,
		wIndex,
		NULL,
		wLength,
		timeout);
	if (r < 0)
	{
		perr("Control transfer error: %d\n", r);
	}


	bmRequestType = LIBUSB_RECIPIENT_DEVICE;
	bRequest = LIBUSB_REQUEST_SET_FEATURE;
	wValue = U2_ENABLE;
	wIndex = 0;
	wLength = 0;

	printf("Control transfer 4 - enable u2\n");
	r = libusb_control_transfer(handle,
		bmRequestType,
		bRequest,
		wValue,
		wIndex,
		NULL,
		wLength,
		timeout);
	if (r < 0)
	{
		perr("Control transfer error: %d\n", r);
	}

	printf("Control transfer 5 - set feature 768\n");
	KSetSensorStatus(handle, KSENSOR_DISABLE);

	printf("Kinect init done\n\n");
}

void IsoCallback(libusb_transfer* transfer)
{
	num_iso_requests_outstanding--;

	if (NULL == transfer)
	{
		perr("IsoCallback received NULL transfer\n");
		return;
	}

	if (LIBUSB_TRANSFER_COMPLETED == transfer->status)
	{
		int numPackets = transfer->num_iso_packets;
		printf("Iso transfer completed. Num packets: %d  Len: %d/%d\n", numPackets, transfer->actual_length, transfer->length);

		for (int i = 0; i < numPackets; i++)
		{
			libusb_iso_packet_descriptor packet = transfer->iso_packet_desc[i];

			if (packet.status == LIBUSB_TRANSFER_COMPLETED)
			{
				uint32_t len = packet.actual_length;
				uint8_t* packetData = libusb_get_iso_packet_buffer_simple(transfer, i);
			}
			else
			{
				printf("  %d) Error: %d\n", i, packet.status);
			}
		}
	}
	else
	{
		printf("Iso transfer error status: %d\n", transfer->status);
	}
	
	if (should_resubmit)
	{
		int r = libusb_submit_transfer(transfer);

		if (r != LIBUSB_SUCCESS)
		{
			//TODO: free buffer?
			perr("Resubmit iso transfer error: %d\n", r);
		}
		else
		{
			num_iso_requests_outstanding++;
			//printf("Resubmit iso transfer done\n");
		}
	}
	else
	{
		free(transfer->buffer);
		transfer->buffer = NULL;
		libusb_free_transfer(transfer);
	}
}

void StartIsochronousTransfers(libusb_device_handle* handle, libusb_device* dev)
{
	uint8_t endpoint = 0x84;

	int numPackets = 8;
	libusb_transfer* transfer = libusb_alloc_transfer(numPackets);

	int maxPacketSize = libusb_get_max_iso_packet_size(dev, endpoint);
	
	//nominally 270336 = 33792 * 8
	int bufferLen = maxPacketSize * numPackets;
	uint8_t* buffer = (uint8_t*)malloc(bufferLen);
	memset(buffer, 0, bufferLen);
	int timeout = 1000;

	libusb_fill_iso_transfer(transfer,
		handle,
		endpoint,
		buffer,
		bufferLen,
		numPackets,
		(libusb_transfer_cb_fn)&IsoCallback,
		NULL,
		0);

	libusb_set_iso_packet_lengths(transfer, maxPacketSize);
	
	int r = libusb_submit_transfer(transfer);

	if (r != LIBUSB_SUCCESS)
	{
		perr("Submit iso transfer error: %d\n", r);

		//TODO: free buffer?
	}
	else
	{
		num_iso_requests_outstanding++;
		printf("Submit iso transfer done\n");
	}
}

void RunKinect(libusb_device_handle *handle)
{
	if (NULL == handle)
		return;

	printf("running kinect...\n");
	int r;

	r = KSetSensorStatus(handle, KSENSOR_ENABLE);

	r = KReadData02(handle);

	r = KReadData14(handle);

	r = KReadData22_1(handle);

	r = KReadData22_3(handle);

	r = KReadData22_2(handle);

	r = KReadData22_4(handle);

	r = KReadData16_90000(handle);

	r = KInitStreams09(handle);

	r = KSetStreamingInterfaceStatus(handle, KSTREAM_ENABLE);

	r = KReadData16_90000(handle);

	r = KSetStreamCommand2B(handle, KSTREAM_ENABLE);

}


void CloseKinect(libusb_device_handle *handle)
{
	printf("closing kinect...\n");
	int r;
	r = KSetSensorStatus(handle, KSENSOR_DISABLE);
}


int main(void) {

	uint16_t vid = 0x045E;
	uint16_t pid = 0x02C4;
	uint16_t mi = 0x00;

	bool debug_mode = false;

	libusb_device_handle *handle;
	libusb_device *dev;
	uint8_t bus;
	const char* speed_name[5] = { "Unknown", "1.5 Mbit/s (USB LowSpeed)", "12 Mbit/s (USB FullSpeed)",
		"480 Mbit/s (USB HighSpeed)", "5000 Mbit/s (USB SuperSpeed)" };

	int r;

	const struct libusb_version* version;
	version = libusb_get_version();
	printf("Using libusbx v%d.%d.%d.%d\n\n", version->major, version->minor, version->micro, version->nano);

	r = libusb_init(NULL);
	if (r < 0)
		return r;

	libusb_set_debug(NULL, debug_mode ? LIBUSB_LOG_LEVEL_DEBUG : LIBUSB_LOG_LEVEL_INFO);

	printf("Opening device %04X:%04X...\n", vid, pid);
	handle = libusb_open_device_with_vid_pid(NULL, vid, pid);

	if (handle == NULL) {
		perr("  Failed.\n");
		system("PAUSE");
		return -1;
	}

	dev = libusb_get_device(handle);
	bus = libusb_get_bus_number(dev);
	/*
	struct libusb_device_descriptor dev_desc;

	printf("\nReading device descriptor:\n");
	CALL_CHECK(libusb_get_device_descriptor(dev, &dev_desc));
	printf("            length: %d\n", dev_desc.bLength);
	printf("      device class: %d\n", dev_desc.bDeviceClass);
	printf("               S/N: %d\n", dev_desc.iSerialNumber);
	printf("           VID:PID: %04X:%04X\n", dev_desc.idVendor, dev_desc.idProduct);
	printf("         bcdDevice: %04X\n", dev_desc.bcdDevice);
	printf("   iMan:iProd:iSer: %d:%d:%d\n", dev_desc.iManufacturer, dev_desc.iProduct, dev_desc.iSerialNumber);
	printf("          nb confs: %d\n", dev_desc.bNumConfigurations);
	*/

	r = libusb_get_device_speed(dev);
	if ((r<0) || (r>4)) r = 0;
	printf("             speed: %s\n", speed_name[r]);


	int iface = 0;
	printf("\nClaiming interface %d...\n", iface);
	r = libusb_claim_interface(handle, iface);
	if (r != LIBUSB_SUCCESS) {
		perr("   Failed: %d.\n", r);
	}

	iface = 1;
	printf("\nClaiming interface %d...\n", iface);
	r = libusb_claim_interface(handle, iface);
	if (r != LIBUSB_SUCCESS) {
		perr("   Failed: %d.\n", r);
	}

	InitKinect(handle);

	r = libusb_get_device_speed(dev);
	if ((r<0) || (r>4)) r = 0;
	printf("             speed: %s\n", speed_name[r]);

	RunKinect(handle);

	r = libusb_get_device_speed(dev);
	if ((r<0) || (r>4)) r = 0;
	printf("             speed: %s\n", speed_name[r]);

	should_resubmit = true;
	printf("should_resubmit: true\n");
	StartIsochronousTransfers(handle, dev);
	
	r = libusb_get_device_speed(dev);
	if ((r<0) || (r>4)) r = 0;
	printf("             speed: %s\n", speed_name[r]);

	int numIsoEventsToProcess = 2000;
	printf("Waiting for iso transfer...\n");
	for (int i = 0; i <= numIsoEventsToProcess; i++)
	{
		r = libusb_handle_events(NULL);
		if (i % 1000 == 0)
			printf("handle events: %d\n", i);
	}

	should_resubmit = false;
	printf("\nshould_resubmit: false\n\n");
	
	//continue processing events while there are outstanding transfers
	//TODO cancel transfers
	while (num_iso_requests_outstanding > 0)
	{
		r = libusb_handle_events(NULL);
	}

	printf("\n");
	
	r = libusb_get_device_speed(dev);
	if ((r<0) || (r>4)) r = 0;
	printf("             speed: %s\n", speed_name[r]);

	CloseKinect(handle);

	r = libusb_get_device_speed(dev);
	if ((r<0) || (r>4)) r = 0;
	printf("             speed: %s\n", speed_name[r]);

	iface = 0;
	printf("Releasing interface %d...\n", iface);
	libusb_release_interface(handle, iface);

	iface = 1;
	printf("Releasing interface %d...\n", iface);
	libusb_release_interface(handle, iface);

	printf("Closing device...\n");
	libusb_close(handle);

	libusb_exit(NULL);

	system("PAUSE");
	return 0;
}
