/*
 * ROS package of publishing depth image of Kinect V2 using libfreenect 2.
 *
 * Copyright (c) 2014 spiralray
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


#include <iostream>
#include <signal.h>

#include <opencv2/opencv.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

void thread_main( libfreenect2::Freenect2Device *dev );

bool sflag = false;

int main(int argc, char** argv){
	bool s;

	ros::init(argc, argv, "kinectv2");

	libfreenect2::Freenect2 freenect2;
	libfreenect2::Freenect2Device *dev = freenect2.openDefaultDevice();

	if(dev == 0)
	{
		std::cout << "no device connected or failure opening the default one!" << std::endl;
		return -1;
	}

	pthread_t thread;
	pthread_create( &thread, NULL, (void* (*)(void*))thread_main, dev );

	ros::spin();

	sflag = true;
	pthread_join( thread, NULL );

	dev->stop();
	dev->close();

	return 0;
}

void thread_main( libfreenect2::Freenect2Device *dev ){
  ROS_INFO("New thread Created.");

  libfreenect2::SyncMultiFrameListener listener(/*libfreenect2::Frame::Color | libfreenect2::Frame::Ir | */libfreenect2::Frame::Depth);
  libfreenect2::FrameMap frames;

  //dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);
  dev->start();

  std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
  std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

  ros::NodeHandle n;
  image_transport::ImageTransport it(n);
  image_transport::Publisher image_pub = it.advertise("kinect/depth", 1);

  while(!sflag)
  {
    listener.waitForNewFrame(frames);
    //libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    //libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    //cv::imshow("rgb", cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data));
    //cv::imshow("ir", cv::Mat(ir->height, ir->width, CV_32FC1, ir->data) / 20000.0f);
    //cv::imshow("depth", cv::Mat(depth->height, depth->width, CV_32FC1, depth->data) / 4500.0f);

    cv::Mat depthMat = cv::Mat(depth->height, depth->width, CV_32FC1, depth->data) / 4500.0f;
    depthMat.convertTo(depthMat, CV_16UC1, 65535.0f);
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", depthMat ).toImageMsg();
    image_pub.publish(img_msg);

    listener.release(frames);
    //libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100));
  }
}
