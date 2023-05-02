#!/usr/bin/env python

import os
import sys
from ctypes import *
import platform


import logging
log = logging.getLogger(__name__)

class AoImage(Structure):
    _fields_ = [
        ('_data', c_uint64),    # ctypes pointers are tricky when changed under the hud so we treat it as number
        ('_width', c_uint32),
        ('_height', c_uint32),
        ('_stride', c_uint32),
        ('_channels', c_uint32),
        ('_errmsg', c_char*80)  #possible error message to be filled by the C routines
    ]

    has_voids = False           # has void spots, to be propagated to the dds

    def __init__(self, has_voids = False):
        self._data = 0
        self._width = 0
        self._height = 0
        self._stride = 0
        self._channels = 0
        self._errmsg = b'';
        # beyond the struct
        self.has_voids = has_voids

    def __del__(self):
        _aoi.aoimage_delete(self)

    def __repr__(self):
        return f"ptr:  width: {self._width} height: {self._height} stride: {self._stride} channels: {self._channels}"

    def close(self):
        log.warning("AoImage.close() is obsolete and does nothing")

    def convert(self, mode):
        """
        Not really needed as AoImage always loads as RGBA
        """
        assert mode == "RGBA", "Sorry, only conversion to RGBA supported"
        new = AoImage(self.has_voids)
        if not _aoi.aoimage_2_rgba(self, new):
            log.error(f"AoImage.reduce_2 error: {new._errmsg.decode()}")
            return None

        return new

    def reduce_2(self, steps = 1):
        """
        Reduce image by factor 2.
        """
        assert steps >= 1, "useless reduce_2" # otherwise we must do a useless copy

        half = self
        while steps >= 1:
            orig = half
            half = AoImage(self.has_voids)
            if not _aoi.aoimage_reduce_2(orig, half):
                log.error(f"AoImage.reduce_2 error: {new._errmsg.decode()}")
                return None

            steps -= 1

        return half

    def enlarge_2(self, steps, height_only = None):
        """
        Enlarge by factor 2^steps
        """
        assert 1 <= steps and steps <= 4    # assert a reasonable range
        new = AoImage(self.has_voids)
        height = self._height;
        if height_only != None:
            assert height_only <= self._height
            height = height_only

        if not _aoi.aoimage_enlarge_2(self, new, steps, height):
            log.error(f"AoImage.enlarge_2 error: {new._errmsg.decode()}")
            return None

        return new

    def write_jpg(self, filename, quality = 90, height_only = 0):
        """
        Convenience function to write jpeg.
        """
        h = self._height
        if height_only > 0:
            self._height = height_only
        if not _aoi.aoimage_write_jpg(filename.encode(), self, quality):
            log.error(f"AoImage.write_jpg error: {self._errmsg.decode()}")
        self._height = h

    def tobytes(self):
        """
        Not really needed, high overhead. Use data_ptr instead.
        """
        buf = create_string_buffer(self._width * self._height * self._channels)
        _aoi.aoimage_tobytes(self, buf)
        return buf.raw

    def data_ptr(self):
        """
        Return ptr to image data. Valid only as long as the object lives.
        """
        return self._data

    def paste(self, p_img, pos):
        _aoi.aoimage_paste(self, p_img, pos[0], pos[1])
        return None

    def copy(self, height_only = 0):
        new = AoImage(self.has_voids)
        if not _aoi.aoimage_copy(self, new, height_only):
            log.error(f"AoImage.copy error: {self._errmsg.decode()}")
            return None

        return new

    @property
    def size(self):
        return self._width, self._height

## factories
def new(mode, wh, color):
    #print(f"{mode}, {wh}, {color}")
    assert(mode == "RGBA")
    new = AoImage()
    if not _aoi.aoimage_create(new, wh[0], wh[1], color[0], color[1], color[2]):
        log.error(f"AoImage.new error: {new._errmsg.decode()}")
        return None

    return new


def load_from_memory(mem, log_error = True):
    new = AoImage()
    if not _aoi.aoimage_from_memory(new, mem, len(mem)):
        if log_error:
            log.error(f"AoImage.load_from_memory error: {new._errmsg.decode()}")
        return None

    return new

def open(filename, log_error = True):
    new = AoImage()
    if not _aoi.aoimage_read_jpg(filename.encode(), new):
        if log_error:
            log.error(f"AoImage.open error for {filename}: {new._errmsg.decode()}")
        return None

    return new

# init code
if platform.system().lower() == 'linux':
    _aoi_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'aoimage.so')
elif platform.system().lower() == 'windows':
    _aoi_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'aoimage.dll')
else:
    log.error("System is not supported")
    exit()

_aoi = CDLL(_aoi_path)
_aoi.aoimage_read_jpg.argtypes = (c_char_p, POINTER(AoImage))
_aoi.aoimage_write_jpg.argtypes = (c_char_p, POINTER(AoImage), c_int32)
_aoi.aoimage_2_rgba.argtypes = (POINTER(AoImage), POINTER(AoImage))
_aoi.aoimage_reduce_2.argtypes = (POINTER(AoImage), POINTER(AoImage))
_aoi.aoimage_enlarge_2.argtypes = (POINTER(AoImage), POINTER(AoImage), c_uint32, c_uint32)
_aoi.aoimage_delete.argtypes = (POINTER(AoImage),)
_aoi.aoimage_create.argtypes = (POINTER(AoImage), c_uint32, c_uint32, c_uint32, c_uint32, c_uint32)
_aoi.aoimage_tobytes.argtypes = (POINTER(AoImage), c_char_p)
_aoi.aoimage_from_memory.argtypes = (POINTER(AoImage), c_char_p, c_uint32)
_aoi.aoimage_copy.argtypes = (POINTER(AoImage), POINTER(AoImage), c_uint32)
_aoi.aoimage_paste.argtypes = (POINTER(AoImage), POINTER(AoImage), c_uint32, c_uint32)

def main():
    logging.basicConfig(level = logging.DEBUG)
    width = 16
    height = 16
    black = new('RGBA', (256*width,256*height), (0,0,0))
    log.info(f"{black}")
    log.info(f"black._data: {black._data}")
    log.info(f"black.data_ptr(): {black.data_ptr()}")
    black.write_jpg("black.jpg")
    w, h = black.size
    black = None
    log.info(f"black done, {w} {h}")

    green = new('RGBA', (256*width,256*height), (0,230,0))
    log.info(f"green {green}")
    green.write_jpg("green.jpg")

    log.info("Trying nonexistent jpg")
    img = open("../testfiles/non_exitent.jpg")

    log.info("Trying non jpg")
    img = open("main.c")

    img = open("../testfiles/test_tile.jpg")
    log.info(f"AoImage.open {img}")

    img_hdr = img.copy(256)
    img_hdr.write_jpg("img_hdr.jpg")

    img2 = img.reduce_2()
    log.info(f"img2: {img2}")

    img2.write_jpg("test_tile_2.jpg")

    green.paste(img2, (1024, 1024))
    green.write_jpg("test_tile_p.jpg")

    img4 = img.reduce_2(2)
    log.info(f"img4 {img4}")

    img.paste(img4, (0, 2048))
    img.write_jpg("test_tile_p2.jpg")

    bg_img = open("1072_693_11_BI.jpg")
    en_16 = bg_img.enlarge_2(4)
    log.info(f"en_16 {en_16}")
    en_16.write_jpg("en_16.jpg")


if __name__ == "__main__":
    main()
