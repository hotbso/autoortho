#!/usr/bin/env python

#from __future__ import with_statement

import os
import re
import sys
import time
import math
import types
import errno
import random
import psutil
import ctypes
import platform
import threading
import itertools

import flighttrack

from functools import wraps, lru_cache

from aoconfig import CFG
import logging
log = logging.getLogger(__name__)

#from fuse import FUSE, FuseOSError, Operations, fuse_get_context
from refuse.high import FUSE, FuseOSError, Operations, fuse_get_context, fuse_exit, _libfuse

import getortho

from xp_udp import DecodePacket, RequestDataRefs
import socket

#from memory_profiler import profile
import tracemalloc


def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
  return (xtile, ytile)


def tilemeters(lat_deg, zoom):
    y = 64120000 * math.cos(math.radians(lat_deg)) / (pow(2, zoom))
    x = 64120000 / (pow(2, zoom))
    return (x, y)

MEMTRACE=False


def locked(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            result = fn(self, *args, **kwargs)
        return result
    return wrapped


class AutoOrtho(Operations):

    path_dict = {}

    tile_base_fh = 1000

    default_uid = -1
    default_gid = -1

    startup = True


    def __init__(self, root, cache_dir='.cache', tile_cache = None):
        log.info(f"ROOT: {root}")
        self.dds_re = re.compile(".*/(\d+)[-_](\d+)[-_]((?!ZL)\S*)(\d{2}).dds")
        self.ktx2_re = re.compile(".*/(\d+)[-_](\d+)[-_]((?!ZL)\D*)(\d+).ktx2")
        self.dsf_re = re.compile(".*/[-+]\d+[-+]\d+.dsf")
        self.ter_re = re.compile(".*/\d+[-_]\d+[-_](\D*)(\d+).ter")
        self.root = root
        self.cache_dir = cache_dir

        if tile_cache is not None:
            self.tc = tile_cache
        else:
            self.tc = getortho.TileCacher(cache_dir)

        #self.path_condition = threading.Condition()
        #self.read_lock = threading.Lock()
        self._lock = threading.RLock()
        self.tile_fh = []


    # Helpers
    # =======

    @locked
    def _allocate_tile_fh(self, tile):
        nfh = len(self.tile_fh)
        for i in range(0, nfh):
            if self.tile_fh[i] is None:
                self.tile_fh[i] = tile
                return self.tile_base_fh + i

        self.tile_fh.append(tile)
        return self.tile_base_fh + nfh

    @locked
    def _release_tile_fh(self, fh):
        self.tile_fh[fh - self.tile_base_fh] = None

    @locked
    def _map_tile(self, fh):
        return self.tile_fh[fh - self.tile_base_fh]

    def _full_path(self, partial):
        if partial.startswith("/"):
            partial = partial[1:]
        path = os.path.abspath(os.path.join(self.root, partial))
        return path


    # Filesystem methods
    # ==================

    def _access(self, path, mode):
        log.info(f"ACCESS: {path}")
        #m = re.match(".*/(\d+)[-_](\d+)[-_](\D*)(\d+).dds", path)
        #if m:
        #    log.info(f"ACCESS: Found DDS file {path}: %s " % str(m.groups()))
        full_path = self._full_path(path)
        if not os.access(full_path, mode):
            raise FuseOSError(errno.EACCES)

    def chmod(self, path, mode):
        full_path = self._full_path(path)
        return os.chmod(full_path, mode)

    def chown(self, path, uid, gid):
        full_path = self._full_path(path)
        return os.chown(full_path, uid, gid)

    @lru_cache(maxsize=1024)
    def getattr(self, path, fh=None):
        log.debug(f"GETATTR {path}")

        m = self.dds_re.match(path)
        #if m and not exists:
        if m:
            log.debug(f"GETATTR: {path}: MATCH!")
            if not flighttrack.ft.running:
                # First matched file
                log.info(f"First matched DDS file {path} detected.  Start flight tracker.")
                flighttrack.ft.start()
                self.startup = False

            #row, col, maptype, zoom = m.groups()
            #log.debug(f"GETATTR: Fetch for {path}: %s" % str(m.groups()))

            if CFG.pydds.format == "BC1":
                dds_size = 11184952
            else:
                #dds_size = 22369744
                dds_size = 22369776

            attrs = {
                'st_atime': 1649857250.382081,
                'st_ctime': 1649857251.726115,
                'st_gid': self.default_gid,
                'st_uid': self.default_uid,
                #'st_mode': 33204,
                'st_mode': 33206,
                'st_mtime': 1649857251.726115,
                'st_nlink': 1,
                'st_size': dds_size,
                'st_blksize': 32768
                #'st_blksize': 16384
                #'st_blksize': 8192
                #'st_blksize': 4096
                # Windows specific stuff
                #'st_ino': 844424931910150,
                #'st_dev': 1421433975
            }
        elif path.endswith(".poison"):
            log.info("Poison pill.  Exiting!")
            fuse_ptr = ctypes.c_void_p(_libfuse.fuse_get_context().contents.fuse)
            #threading.Thread(target=do_fuse_exit, args=(fuse_ptr,)).start()
            do_fuse_exit(fuse_ptr)

            attrs = {
                'st_atime': 1649857250.382081,
                'st_ctime': 1649857251.726115,
                'st_gid': self.default_gid,
                'st_uid': self.default_uid,
                'st_mode': 33206,
                'st_mtime': 1649857251.726115,
                'st_nlink': 1,
                'st_size': 0,
                'st_blksize': 32768
            }
            return attrs
        else:
            full_path = self._full_path(path)
            exists = os.path.exists(full_path)
            log.debug(f"GETATTR FULLPATH {full_path}  Exists? {exists}")
            full_path = self._full_path(path)
            st = os.lstat(full_path)
            log.debug(f"GETATTR: Orig stat: {st}")
            attrs = dict((key, getattr(st, key)) for key in ('st_atime', 'st_ctime',
                        'st_gid', 'st_mode', 'st_mtime', 'st_nlink', 'st_size', 'st_uid', 'st_ino', 'st_dev'))

            #if os.path.isdir(full_path):
            #    attrs['st_nlink'] = 2

        #attrs['st_uid'] = self.default_uid
        #attrs['st_gid'] = self.default_gid

        #attrs['st_mode'] = 33204

        #log.info(f"GETATTR: FH: {fh}")
        log.debug(f"GETATTR: {path}: {attrs}")

        return attrs

    @lru_cache
    def readdir(self, path, fh):
        return ['.','..']
        log.info(f"READDIR: {path} {fh}")
        return ['.', '..']

        if path not in self.path_dict:
            full_path = self._full_path(path)
            dirents = ['.', '..']
            if os.path.isdir(full_path):
                dirents.extend(os.listdir(full_path))
            #self.path_dict[path] = dirents
        else:
            dirents = self.path_dict.get(path)

        return dirents
        #for r in dirents:
        #    yield r

    def readlink(self, path):
        pathname = os.readlink(self._full_path(path))
        if pathname.startswith("/"):
            # Path name is absolute, sanitize it.
            return os.path.relpath(pathname, self.root)
        else:
            return pathname

    def mknod(self, path, mode, dev):
        return os.mknod(self._full_path(path), mode, dev)

    def rmdir(self, path):
        full_path = self._full_path(path)
        return os.rmdir(full_path)

    def mkdir(self, path, mode):
        return os.mkdir(self._full_path(path), mode)

    @lru_cache
    def statfs(self, path):
        log.info(f"STATFS: {path}")
        full_path = self._full_path(path)
        if platform.system() == 'Windows':
            stats = {
                    'f_bavail':47602498,
                    'f_bfree':47602498,
                    'f_blocks':124699647,
                    'f_favail':1000000,
                    'f_ffree':1000000,
                    'f_files':999,
                    'f_frsize':4096,
                    'f_flag':1024,
                    'f_bsize':4096
            }
            return stats
            # st = os.stat(full_path)
            # return dict((key, getattr(st, key)) for key in ('f_bavail', 'f_bfree',
            #     'f_blocks', 'f_bsize', 'f_favail', 'f_ffree', 'f_files', 'f_flag',
            #     'f_frsize', 'f_namemax'))
        elif platform.system() == 'Linux':
            stv = os.statvfs(full_path)
            #log.info(stv)
            return dict((key, getattr(stv, key)) for key in ('f_bavail', 'f_bfree',
                'f_blocks', 'f_bsize', 'f_favail', 'f_ffree', 'f_files', 'f_flag',
                'f_frsize', 'f_namemax'))

    def unlink(self, path):
        return os.unlink(self._full_path(path))

    def symlink(self, name, target):
        return os.symlink(target, self._full_path(name))

    def rename(self, old, new):
        return os.rename(self._full_path(old), self._full_path(new))

    def link(self, target, name):
        return os.link(self._full_path(name), self._full_path(target))

    def utimens(self, path, times=None):
        return os.utime(self._full_path(path), times)

    # File methods
    # ============


    #@locked
    def open(self, path, flags):
        h = 0

        #log.info(f"READ CACHE {self.read.cache_info()}")
        #log.info(f"ATTR CACHE {self.getattr.cache_info()}")
        #log.info(f"DIR CACHE {self.readdir.cache_info()}")
        log.debug(f"OPEN: {path}, {flags}")
        full_path = self._full_path(path)
        log.debug(f"OPEN: FULL PATH: {full_path}")

        dsf_m = self.dsf_re.match(path)
        if dsf_m:
            log.debug(f"OPEN: Detected DSF open: {path}")
        #dsf_m = self.dsf_re.match(path)
        #ter_m = self.ter_re.match(path)
        dds_m = self.dds_re.match(path)
        if dds_m:
            row, col, maptype, zoom = dds_m.groups()
            row = int(row)
            col = int(col)
            zoom = int(zoom)
            t = self.tc._open_tile(row, col, maptype, zoom)
            h = self._allocate_tile_fh(t)
        elif platform.system() == 'Windows':
            h = os.open(full_path, flags|os.O_BINARY)
        else:
            h = os.open(full_path, flags)

        log.debug(f"OPEN: FH= {h}")
        return h

    def _create(self, path, mode, fi=None):
        uid, gid, pid = fuse_get_context()
        full_path = self._full_path(path)
        fd = os.open(full_path, os.O_WRONLY | os.O_CREAT, mode)
        os.chown(full_path,uid,gid) #chown to context uid & gid
        return fd

    #@profile
    def read(self, path, length, offset, fh):
        log.debug(f"READ: {path} {offset} {length} {fh}")
        data = None

        if fh >= self.tile_base_fh:
            log.debug(f"READ: DDS file {path}, offset {offset}, length {length}")
            ot = self._map_tile(fh)
            data = ot.read(offset, length)
        else:
            os.lseek(fh, offset, os.SEEK_SET)
            data = os.read(fh, length)
            log.debug(f"READ: Read {len(data)} bytes.")

        return data

    def _write(self, path, buf, offset, fh):
        os.lseek(fh, offset, os.SEEK_SET)
        return os.write(fh, buf)

    def truncate(self, path, length, fh=None):
        log.info(f"TRUNCATE")
        full_path = self._full_path(path)
        with open(full_path, 'r+') as f:
            f.truncate(length)

    def _flush(self, path, fh):
        log.info(f"FLUSH")
        m = self.dds_re.match(path)
        if m:
            log.info(f"RELEASE: {path}")
            return 0
        else:
            return os.fsync(fh)

    def _releasedir(self, path, fh):
        log.debug(f"RELEASEDIR: {path}")
        return 0

    #@locked
    def release(self, path, fh):
        log.debug(f"RELEASE: {path}")
        #ter_m = self.ter_re.match(path)
        # dsf_m = self.dsf_re.match(path)
        # if dsf_m:
            # log.info(f"RELEASE: Detected DSF close: {path}")

        if fh >= self.tile_base_fh:
            ot = self._map_tile(fh)
            self.tc._release_tile(ot)
            self._release_tile_fh(fh)
        else:
            return os.close(fh)

    def fsync(self, path, fdatasync, fh):
        log.info(f"FSYNC: {path}")
        return self.flush(path, fh)


def do_fuse_exit(fuse_ptr=None):
    print("fuse_exit called")
    #time.sleep(1)
    if not fuse_ptr:
        fuse_ptr = ctypes.c_void_p(_libfuse.fuse_get_context().contents.fuse)
    print(fuse_ptr)
    _libfuse.fuse_exit(fuse_ptr)


def run(ao, mountpoint, nothreads=False):
    log.info(f"MOUNT: {mountpoint}")

    FUSE(
        ao,
        mountpoint,
        nothreads=nothreads,
        foreground=True,
        allow_other=True,
        #auto_cache=True,
        #max_read=32768,
        #max_read=16384,
        #kernel_cache=True,
        uid=-1,
        gid=-1,
        #debug=True,
        #mode="0777",
        #umask="777",
        #FileSecurity="O:BAG:BAD:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;WD)",
        #FileSecurity="D:P(A;;FA;;;OW)",
        #FileSecurity="D:P(A;;0x1200A9;;;WD)",
        #FileSecurity="D:P(A;OICI;FA;;;WD)(A;OICI;FA;;;BU)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)",
        #FileSecurity="O:BAG:BAD:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;WD)",
        #umask=0,
        #create_dir_umask=0,
        #max_background=4,
        #kernel_cache=False,
        #max_readahead=0,
        #sync_read=True,
        #async_read=False,
        #max_readahead=8192,
        #max_readahead=0,
        #max_read=262144,
        #default_permissions=True,
        #direct_io=True
    )
