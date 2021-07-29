#include <iostream>
#include <fstream>
#include <vector>
#include "../wrapperjpeg.hpp"


int main(void) {
  const char *filename = "input.jpg";
  const char *output_filename = "output.jpg";

  std::ifstream oInputStream(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if(!(oInputStream.is_open()))
  {
      std::cerr << "Cannot open image: " << filename << std::endl;
      return 1;
  }

  // Get the size.
  std::streamsize nSize = oInputStream.tellg();
  oInputStream.seekg(0, std::ios::beg);
  std::vector<char> vBuffer(nSize);

  if (!oInputStream.read(vBuffer.data(), nSize))
  {
    std::cerr << "Cannot allocate memory"<< std::endl;
    return 1;
  }

  struct compressedImage *cm = compressImage((unsigned char *)vBuffer.data(), nSize);
  if (cm == NULL) {
    std::cerr << "Cannot compress image"<< std::endl;
    return 1;
  }

  std::ofstream outputFile(output_filename, std::ios::out | std::ios::binary);
  outputFile.write(
    reinterpret_cast<const char *>(cm->bufferEncodedImage),
    static_cast<int>(cm->sizeBufferEncodedImage)
  );

  return 0;
}