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

#ifndef _P0TABLES_H_
#define _P0TABLES_H_

struct __attribute__ ((__packed__)) p0tables {

	uint32_t headersize;
	uint32_t unknown1;
	uint32_t unknown2;
	uint32_t planesize;
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

};

#endif // _P0TABLES_H_
