#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "aoimage.h"

int main(void) {
    setvbuf(stderr, NULL, _IOLBF, BUFSIZ);
    setvbuf(stdout, NULL, _IOLBF, BUFSIZ);

    aoimage_t img;

    if(!aoimage_read_jpg("../testfiles/test_tile.jpg", &img)) {
        printf("Error in loading the image\n");
        exit(1);
    }

    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", img.width, img.height, img.channels);

    aoimage_t img_2;
    aoimage_reduce_2(&img, &img_2);
    aoimage_dump("reduced", &img_2);
    aoimage_write_jpg("test_tile_2.jpg", &img_2, 90);

    aoimage_paste(&img, &img_2, 1024, 0);
    aoimage_write_jpg("test_tile_p.jpg", &img, 90);

    aoimage_t bg_img;
    if(!aoimage_read_jpg("1072_693_11_BI.jpg", &bg_img)) {
        printf("Error in loading the image\n");
        exit(1);
    }

    aoimage_t e16_img;
    if(!aoimage_enlarge_2(&bg_img, &e16_img, 4, 0)) {
        printf("Error enlarging image\n");
        exit(1);
    }

    aoimage_write_jpg("en_16.jpg", &e16_img, 50);

    aoimage_t e16_r0_img;
    if(!aoimage_enlarge_2(&bg_img, &e16_r0_img, 4, 128)) {
        printf("Error enlarging image\n");
        exit(1);
    }

    aoimage_write_jpg("en_16_r0.jpg", &e16_r0_img, 50);

    aoimage_desaturate(&img, 0.6);
    aoimage_write_jpg("desaturated.jpg", &img, 50);
}
