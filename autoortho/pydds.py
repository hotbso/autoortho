#!/usr/bin/env python

import os
import sys
from io import BytesIO
from binascii import hexlify
from ctypes import *
from PIL import Image
import platform
import threading

from functools import lru_cache, cache

#from memory_profiler import profile
from aoconfig import CFG

import logging
log = logging.getLogger(__name__)


#_stb = CDLL("/usr/lib/x86_64-linux-gnu/libstb.so")
if platform.system().lower() == 'linux':
    print("Linux detected")
    _stb_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'lib','linux','lib_stb_dxt.so')
    _ispc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'lib','linux','libispc_texcomp.so')
elif platform.system().lower() == 'windows':
    print("Windows detected")
    _stb_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'lib','windows','stb_dxt.dll')
    _ispc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'lib','windows','ispc_texcomp.dll')
else:
    print("System is not supported")
    exit()

_stb = CDLL(_stb_path)
_ispc = CDLL(_ispc_path)

DDSD_CAPS = 0x00000001          # dwCaps/dwCaps2 is enabled.
DDSD_HEIGHT = 0x00000002                # dwHeight is enabled.
DDSD_WIDTH = 0x00000004                 # dwWidth is enabled. Required for all textures.
DDSD_PITCH = 0x00000008                 # dwPitchOrLinearSize represents pitch.
DDSD_PIXELFORMAT = 0x00001000   # dwPfSize/dwPfFlags/dwRGB/dwFourCC and such are enabled.
DDSD_MIPMAPCOUNT = 0x00020000   # dwMipMapCount is enabled. Required for storing mipmaps.
DDSD_LINEARSIZE = 0x00080000    # dwPitchOrLinearSize represents LinearSize.
DDSD_DEPTH = 0x00800000                 # dwDepth is enabled. Used for 3D (Volume) Texture.


STB_DXT_NORMAL = 0
STB_DXT_DITHER = 1
STB_DXT_HIGHQUAL = 2


# def do_compress(img):
#
#     width, height = img.size
#
#     if (width < 4 or width % 4 != 0 or height < 4 or height % 4 != 0):
#         log.debug("Compressed images must have dimensions that are multiples of 4.")
#         return None
#
#     if img.mode == "RGB":
#         img = img.convert("RGBA")
#
#     data = img.tobytes()
#
#     is_rgba = True
#     blocksize = 16
#
#     dxt_size = ((width+3) >> 2) * ((height+3) >> 2) * 16
#     outdata = create_string_buffer(dxt_size)
#
#     _stb.compress_pixels.argtypes = (
#             c_char_p,
#             c_char_p,
#             c_uint64,
#             c_uint64,
#             c_bool)
#
#     result = _stb.compress_pixels(
#             outdata,
#             c_char_p(data),
#             c_uint64(width),
#             c_uint64(height),
#             c_bool(is_rgba))
#
#     if not result:
#         log.debug("Failed to compress")
#
#     return (dxt_size, outdata)
#
#def get_size(width, height):
#    return ((width+3) >> 2) * ((height+3) >> 2) * 16

class MipMap(object):
    def __init__(self, idx=0, startpos=0, endpos=0, length=0, retrieved=False):
        self.idx = idx
        self.startpos = startpos
        self.endpos = endpos
        self.length = length
        self.retrieved = retrieved

    def __repr__(self):
        return f"MipMap({self.idx}, {self.startpos}, {self.endpos}, {self.length}, {self.retrieved})"


class rgba_surface(Structure):
    _fields_ = [
        ('data', c_char_p),
        ('width', c_uint32),
        ('height', c_uint32),
        ('stride', c_uint32)
    ]


class DDS(Structure):
    _fields_ = [
        ('magic', c_char * 4),
        ('size', c_uint32),
        ('flags', c_uint32),
        ('height', c_uint32),
        ('width', c_uint32),
        ('pitchOrLinearSize', c_uint32),
        ('depth', c_uint32),
        ('mipMapCount', c_uint32),
        ('reserved1', c_char * 44),
        ('pfSize', c_uint32),
        ('pfFlags', c_uint32),
        ('fourCC', c_char * 4),
        ('rgbBitCount', c_uint32),
        ('rBitMask', c_uint32),
        ('gBitMask', c_uint32),
        ('bBitMask', c_uint32),
        ('aBitMask', c_uint32),
        ('caps', c_uint32),
        ('caps2', c_uint32),
        ('reservedCaps', c_uint32 * 2),
        ('reserved2', c_uint32)
    ]


    def __init__(self, width, height, ispc=True, dxt_format="BC3"):
        self.magic = b"DDS "
        self.size = 124
        self.flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT | DDSD_LINEARSIZE
        self.width = width
        self.height = height

        # https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
        # pitchOrLinearSize is the total number of bytes in the top level texture for a compressed texture

        #self.reserved1 = b"pydds"
        self.pfSize = 32
        self.pfFlags = 0x4

        if dxt_format == 'BC3':
            self.fourCC = b'DXT5'
            self.pitchOrLinearSize = ((width+3) >> 2) * ((height+3) >> 2) * 16
        else:
            self.fourCC = b'DXT1'
            self.pitchOrLinearSize = ((width+3) >> 2) * ((height+3) >> 2) * 8

        self.caps = 0x1000 | 0x400000
        self.mipMapCount = 0

        #self.mipmaps = []


        self.ispc = ispc
        self.dxt_format = dxt_format
        self.mipmap_map = {}

        #[pow(2,x)*pow(2,x) for x in range(int(math.log(width,2)),1,-1) ]

        # List of tuples [(byte_position, retrieved_bool)]
        self.mipmap_list = []

        curbytes = 128
        while (width >= 4) and (height >= 4):
            mipmap = MipMap()
            mipmap.idx = self.mipMapCount
            mipmap.startpos = curbytes
            if dxt_format == 'BC3':
                curbytes += width*height
            else:
                curbytes += int((width*height)/2)
            mipmap.length = curbytes - mipmap.startpos
            mipmap.endpos = mipmap.startpos + mipmap.length
            width = int(width/2)
            height = int(height/2)
            self.mipMapCount+=1
            self.mipmap_list.append(mipmap)
        # Size of all mipmaps: sum([pow(2,x)*pow(2,x) for x in range(12,1,-1) ])
        self.total_size = curbytes

        self.data = BytesIO(b'\0' * self.total_size)
        self.dump_header()

        for m in self.mipmap_list:
            log.debug(m)
        #log.debug(self.mipmap_list)
        log.debug(self.total_size)
        #print(self.total_size)
        log.debug(self.mipMapCount)

        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.ready.clear()

        self.compress_count = 0

    def write(self, filename):
        #self.dump_header()
        with open(filename, 'wb') as h:
            h.write(self.data.getbuffer()[0:self.total_size])

    def tell(self):
        return self.data.tell()

    def seek(self, offset):
        log.debug(f"PYDDS: SEEK: {offset}")
        self.data.seek(offset)

    def read(self, length):
        log.debug(f"PYDDS: READ: {length} bytes")
        return self.data.read(length)

    def dump_header(self):
        self.data.seek(0)
        self.data.write(self)

    #@profile
    def compress(self, width, height, data):
        if (width < 4 or width % 4 != 0 or height < 4 or height % 4 != 0):
            log.debug(f"Compressed images must have dimensions that are multiples of 4. We got {width}x{height}")
            return None

        if self.ispc and self.dxt_format == "BC3":
            blocksize = 16
            dxt_size = ((width+3) >> 2) * ((height+3) >> 2) * blocksize
            outdata = create_string_buffer(dxt_size)
            #print(f"LEN: {len(outdata)}")
            s = rgba_surface()
            s.data = c_char_p(data)
            s.width = c_uint32(width)
            s.height = c_uint32(height)
            s.stride = c_uint32(width * 4)

            #print("Will do ispc")
            _ispc.CompressBlocksBC3.argtypes = (
                POINTER(rgba_surface),
                c_char_p
            )

            _ispc.CompressBlocksBC3(
                s, outdata
            )
            result = True
        elif self.ispc and self.dxt_format == "BC1":
            #print("BC1")
            blocksize = 8
            dxt_size = ((width+3) >> 2) * ((height+3) >> 2) * blocksize
            outdata = create_string_buffer(dxt_size)
            #print(f"LEN: {len(outdata)}")

            s = rgba_surface()
            s.data = c_char_p(data)
            s.width = c_uint32(width)
            s.height = c_uint32(height)
            s.stride = c_uint32(width * 4)

            #print("Will do ispc")
            _ispc.CompressBlocksBC1.argtypes = (
                POINTER(rgba_surface),
                c_char_p
            )

            _ispc.CompressBlocksBC1(
                s, outdata
            )
            result = True
        else:
            is_rgba = True
            #print("Will use stb")
            blocksize = 16
            dxt_size = ((width+3) >> 2) * ((height+3) >> 2) * blocksize
            outdata = create_string_buffer(dxt_size)

            #print(f"LEN: {len(outdata)}")
            _stb.compress_pixels.argtypes = (
                    c_char_p,
                    c_char_p,
                    c_uint64,
                    c_uint64,
                    c_bool)

            result = _stb.compress_pixels(
                    outdata,
                    c_char_p(data),
                    c_uint64(width),
                    c_uint64(height),
                    c_bool(is_rgba))


        if not result:
            log.debug("Failed to compress")

        self.compress_count += 1
        return outdata

    #@profile
    def gen_mipmaps(self, img, startmipmap=0, maxmipmaps=0):

        #if maxmipmaps <= len(self.mipmap_list):
        #    maxmipmaps = len(self.mipmap_list)

        with self.lock:

            # Size of all mipmaps: sum([pow(2,x)*pow(2,x) for x in range(12,1,-1) ])

            width, height = img.size
            img_width, img_height = img.size
            mipmap = startmipmap

            log.debug(self.mipmap_list)

            while (width > 4) and (height > 4):

                ratio = pow(2,mipmap)
                desired_width = self.width / ratio
                desired_height = self.height / ratio
                m = self.mipmap_list[mipmap]
                #if True:
                if not m.retrieved:

                    # Only squares for now
                    reduction_ratio = int(img_width // desired_width)
                    if reduction_ratio < 1:
                        #log.debug("0 ratio. skip")
                        mipmap += 1
                        if maxmipmaps and mipmap >= maxmipmaps:
                            break
                        continue

                    timg = img.reduce(reduction_ratio)

                    imgdata = timg.tobytes()
                    width, height = timg.size
                    log.debug(f"MIPMAP: {mipmap} SIZE: {timg.size}")

                    try:
                        dxtdata = self.compress(width, height, imgdata)
                    finally:
                        pass
                        timg.close()
                        del(imgdata)
                        imgdata = None
                        timg = None

                    if dxtdata is not None:
                        self.data.seek(m.startpos)
                        l = self.data.write(dxtdata)
                        if l != m.length:
                            print(f"Oh no, actual: {l}, should be: {m.length}")
                        m.retrieved = True
                    dxtdata = None

                    #print(f"REF: {sys.getrefcount(dxtdata)}")

                mipmap += 1
                if maxmipmaps and mipmap >= maxmipmaps:
                    break

                if mipmap >= len(self.mipmap_list):
                    break

            self.dump_header()



def to_dds(img, outpath):
    if img.mode == "RGB":
        img = img.convert("RGBA")
    width, height = img.size

    dds = DDS(width, height)
    dds.gen_mipmaps(img)
    dds.write(outpath)


def main():
    inimg = sys.argv[1]
    outimg = sys.argv[2]
    img = Image.open(inimg)

    to_dds(img, outimg)


if __name__ == "__main__":
    main()
