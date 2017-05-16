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

/** @file generate_resources.cpp Generator of the resource file, to load tables from in-program data. */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

/**
 * Add the content of the given filename to the resources.
 * @param filename File to add to the resources.
 */
void dumpFile(const std::string& filename)
{
  using namespace std;

  ifstream f(filename.c_str(), ios::binary);

  unsigned char buffer[1024];

  while(!f.eof())
  {
    f.read(reinterpret_cast<char *>(buffer), sizeof(buffer));
    size_t n = f.gcount();

    if(n == 0) break;

    cout << hex << setw(2) << setfill('0') << "  ";
    for(size_t i = 0; i < n; ++i)
    {
      cout << "0x" << int(buffer[i]) << ", ";
    }
    cout << endl;
  }
  cout << dec;
}

/**
 * Main application entry point.
 * Arguments: List of files to add as resource.
 */
int main(int argc, char **argv)
{
  if(argc < 2) return -1;

  using namespace std;

  string basefolder(argv[1]);

  for(int i = 2; i < argc; ++i)
  {
    cout << "static unsigned char resource" << (i - 2) << "[] = {" << endl;
    dumpFile(basefolder + "/" + argv[i]);
    cout << "};" << endl;
  }

  cout << "static ResourceDescriptor resource_descriptors[] = {" << endl;

  for(int i = 2; i < argc; ++i)
  {
    string path(argv[i]);
    size_t last_slash = path.find_last_of("\\/");
    if (last_slash != std::string::npos)
      path.erase(0, last_slash + 1);
    cout << "  { \"" << path << "\", resource" << (i - 2) << ", " << "sizeof(resource" << (i - 2) << ") }," << endl;
  }

  cout << "  {NULL, NULL, 0}," << endl;
  cout << "};" << endl;
  cout << "static int resource_descriptors_length = " << (argc - 2) << ";" << endl;
  
  return 0;
}
