#include <stdio.h>
#include <stdlib.h>
#include "../wrapperjpeg.hpp"


int main(void) {
  FILE *f = fopen("input.jpg", "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  /* same as rewind(f); */

  unsigned char *string = malloc(fsize + 1);
  fread(string, 1, fsize, f);
  fclose(f);

  string[fsize] = 0;

  struct compressedImage *cm =compressImage(string, fsize);
  if (cm == NULL) {
    fprintf(stderr, "Cannot compress image_n");
    return 1;
  }

  FILE *output_file = fopen("output.jpg", "w+");
  fwrite(cm->bufferEncodedImage, sizeof(unsigned char), cm->sizeBufferEncodedImage, output_file);
  fclose(output_file);

  return 0;
}