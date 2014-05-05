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

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <stdarg.h>
#include <string.h>

#include "libusb.h"

static int perr(char const *format, ...)
{
	va_list args;
	int r;

	va_start(args, format);
	r = vfprintf(stderr, format, args);
	va_end(args);
	
	return r;
}

#define ERR_EXIT(errcode) do { perr("   %s\n", libusb_strerror((enum libusb_error)errcode)); return -1; } while (0)
#define CALL_CHECK(fcall) do { r=fcall; if (r < 0) ERR_EXIT(r); } while (0);

#define U1_ENABLE 0x30
#define U2_ENABLE 0x31

#define K_SUCCESS 0
#define K_ERROR -1
#define K_RESPONSE_COMPLETE 1
#define K_RESPONSE_DATA 2

enum KSensorStatus 
{
	KSENSOR_ENABLE = 0x0, 
	KSENSOR_DISABLE = 0x300 
};

enum KStreamStatus
{
	KSTREAM_DISABLE = 0x0,
	KSTREAM_ENABLE = 0x1
};

enum KModeStatus
{
	KMODE_DISABLE = 0x0,
	KMODE_ENABLE = 0x1
};

//Same magic from Kinect v1 audio protocol, representing June 2, 2009, the day after Project Natal announcement at E3 2009
#define KCMD_MAGIC 0x06022009

//Magic header for response complete packet
#define KCMD_RESPONSE_COMPLETE_MAGIC 0x0A6FE000
#define KCMD_RESPONSE_COMPLETE_LEN 16

#define KCMD_READ_FIRMWARE_VERSIONS 0x02
#define KCMD_INIT_STREAMS 0x09
#define KCMD_READ_VERSIONS 0x14
#define KCMD_READ_STATUS 0x16
#define KCMD_READ_DATA_PAGE 0x22
#define KCMD_READ_COUNT 0x26
#define KCMD_SET_STREAMING 0x2B
#define KCMD_SET_MODE 0x4B

typedef struct
{
	uint32_t magic;		//0x06022009
	uint32_t sequence;	//incremented sequence number, echoed in response complete packet
	uint32_t responseDataLen;	//requested max length of response (sometimes response is smaller)
	uint32_t command;	//function this command represents
	uint32_t reserved0;	//always zero (so far)
	uint32_t parameter;	//optional parameter, varies by command
	uint32_t reserved1; //always zero (so far)
	uint32_t reserved2; //always zero (so far)
	uint32_t reserved3; //always zero (so far)
} cmd_header;

uint32_t cmd_seq = 0;

