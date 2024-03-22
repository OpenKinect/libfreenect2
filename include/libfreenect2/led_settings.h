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

#ifndef LED_SETTINGS_H_
#define LED_SETTINGS_H_

namespace libfreenect2
{

// The following information was found by using the library released by Microsoft under MIT license,
// https://github.com/Microsoft/MixedRealityCompanionKit/tree/master/KinectIPD/NuiSensor
// Debugging the library assembly shows the original struct name was _PETRA_LED_STATE.
struct LedSettings
{
  uint16_t LedId;         // LED index  [0, 1]
  uint16_t Mode;          // 0 = constant, 1 = blink between StartLevel, StopLevel every IntervalInMs ms
  uint16_t StartLevel;    // LED intensity  [0, 1000]
  uint16_t StopLevel;     // LED intensity  [0, 1000]
  uint32_t IntervalInMs;  // Blink interval for Mode=1 in milliseconds
  uint32_t Reserved;      // 0
};

} /* namespace libfreenect2 */

#endif /* LED_SETTINGS_H_ */
