#include <fstream>
#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <stdarg.h>
#include <string.h>

#include <opencv2/opencv.hpp>

#define ISO_BUFFER_SIZE 600000

static inline void convert_packed_to_16bit(uint8_t *src, uint16_t *dest, int vw, int n)
{
  unsigned int mask = (1 << vw) - 1;
  uint32_t buffer = 0;
  int bitsIn = 0;
  int k = 0;
  while (n--)
  {
    while (bitsIn < vw)
    {
      buffer = (buffer << 8) | *(src++);
      bitsIn += 8;
    }
    bitsIn -= vw;
    *(dest++) = (buffer >> bitsIn) & mask;
  }
}

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cerr << "usage: decode <file>.bin" << std::endl;
    return -1;
  }

  uint8_t* raw_buffer = new uint8_t[ISO_BUFFER_SIZE];
  uint16_t* decoded_buffer = new uint16_t[ISO_BUFFER_SIZE];

  std::string filename(argv[1]);

  std::ifstream file(filename.c_str(), std::ifstream::binary);
  file.read((char*) raw_buffer, ISO_BUFFER_SIZE);

  size_t n = file.gcount();

  file.close();

  for (int shift = 0; shift <= 0; shift += 1)
  {
    convert_packed_to_16bit(raw_buffer + shift, decoded_buffer, 11, n - shift);
    std::cerr << shift << std::endl;
    //std::cerr << "start marker: " <<  *((uint8_t*)raw_buffer) << std::endl;
    //std::cerr << "start marker: " <<  *((uint8_t*)decoded_buffer) << std::endl;
    std::cerr << "decoded values " << ((n - shift) * 8) / 11 << std::endl;
    std::cerr << "decoded values " << ((n - shift) * 8) % 11 << std::endl;
    std::cerr << "decoded values " << (((n - shift) * 8 / 11) % 525) * 11 / 8 << std::endl;
    int cols = 512;
    int rows = n * 8 / 11 / cols;
    cv::Mat raw(n / 704, 704, CV_8UC1, raw_buffer);
    cv::imshow("raw", raw);

    if (rows < 424)
      return 0;

    cv::Mat img(rows + 1, cols, CV_16UC1, decoded_buffer);
    cv::Mat out, out2, out3;

    for (int i = 0; i < 10; ++i)
    {
      std::cerr << i << ": " << (int) *(uint16_t*) (raw_buffer + i) << " ";
    }

    std::cerr << std::endl << std::endl;

    img.convertTo(out, CV_8UC1, 1.0 / 2048.0 * 255.0);

    out2 = out.clone();
    out3.create(rows, out2.cols, CV_8UC1);
    out3.setTo(0);

    for (size_t y = 0; y < rows; ++y)
    {
      for (size_t i = 0; i < 128; ++i)
      {
        if (i % 1 == 0)
        {
          out2.at<char>(y, i * 4 + 0) = out.at<char>(y, 0 * 128 + i + 0);
          out2.at<char>(y, i * 4 + 1) = out.at<char>(y, 1 * 128 + i + 0);
          out2.at<char>(y, i * 4 + 2) = out.at<char>(y, 2 * 128 + i + 0);
          out2.at<char>(y, i * 4 + 3) = out.at<char>(y, 3 * 128 + i + 0);
        }
        else
        {
          out2.at<char>(y, i * 4 + 0) = 0;
          out2.at<char>(y, i * 4 + 1) = 0;
          out2.at<char>(y, i * 4 + 2) = 0;
          out2.at<char>(y, i * 4 + 3) = 0;
        }
      }
    }

    std::cerr << out2.rows << " " << out2.cols << std::endl;

    cv::imshow("decoded0 " + filename, out);
    cv::imshow("decoded1 " + filename, out2);
    cv::waitKey(0);
  }
  delete[] raw_buffer;
  delete[] decoded_buffer;

  return 0;
}
