#include <fstream>
#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

#include <stdarg.h>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char **argv) {
  std::vector<cv::Mat> imgs;

  for(int i = 0; i < 10; i++)
  {
    std::string filename;
    std::cin >> filename;
    imgs.push_back(cv::imread(filename, -1));
  }

  cv::Mat sum = imgs[0] * 0.1 + imgs[1] * 0.1  + imgs[2] * 0.1 + imgs[3] * 0.1 + imgs[4] * 0.1 + imgs[5] * 0.1 + imgs[6] * 0.1 + imgs[7] * 0.1 + imgs[8] * 0.1 - imgs[9] * 0.1;


  cv::Mat upper(sum, cv::Rect(0, 0, 512, 212));
  cv::Mat lower(sum, cv::Rect(0, 211, 512, 212)), lower_flipped;

  cv::flip(lower, lower_flipped, 0);

  cv::imshow("up", upper);
  cv::imshow("lo", lower_flipped);

  lower_flipped.copyTo(lower);
  cv::imshow("sum", sum);
  cv::Mat out = imgs[0].clone();
  out.setTo(0.0);

  for(int y = 0; y < 424; y++)
  {
    //for(int x = 0; x < 16; x++)
    {
      for(int i = 0; i < 10; i++)
      {
        std::cout << (int)imgs[i].at<uint16_t>(y, 1) << " ";
        //out.at<char>(y,i*4 + 0+10) = (char)(imgs[i].at<uint16_t>(y,0+4*9) / 2048.0f * 255.0f);
        //out.at<char>(y,i*4 + 1+10) = (char)(imgs[i].at<uint16_t>(y,1+4*9) / 2048.0f * 255.0f);
        //out.at<char>(y,i*4 + 2+10) = (char)(imgs[i].at<uint16_t>(y,2+4*9) / 2048.0f * 255.0f);
        //out.at<char>(y,i*4 + 3+10) = (char)(imgs[i].at<uint16_t>(y,3+4*9) / 2048.0f * 255.0f);
      }
      std::cout << std::endl;
    }
  }
  cv::imshow("interleaved", out);

  cv::waitKey(0);
}
