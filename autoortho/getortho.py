#!/usr/bin/env python3

import os
import sys
import time
import math
import platform
import threading
import gc

from urllib.request import urlopen, Request
from queue import Queue, PriorityQueue, Empty
from functools import wraps, lru_cache

import pydds

import psutil
from aoimage import AoImage

from aoconfig import CFG
from aostats import STATS, StatTracker, STATS_inc

MEMTRACE = False

import logging
log = logging.getLogger(__name__)

#from memory_profiler import profile

MAPID = "s2cloudless-2020_3857"
MATRIXSET = "g"

# Track average fetch times
tile_stats = StatTracker(20, 12)
mm_stats = StatTracker(0, 5)
partial_stats = StatTracker()

def _gtile_to_quadkey(til_x, til_y, zoomlevel):
    """
    Translates Google coding of tiles to Bing Quadkey coding.
    """
    quadkey=""
    temp_x=til_x
    temp_y=til_y
    for step in range(1,zoomlevel+1):
        size=2**(zoomlevel-step)
        a=temp_x//size
        b=temp_y//size
        temp_x=temp_x-a*size
        temp_y=temp_y-b*size
        quadkey=quadkey+str(a+2*b)
    return quadkey


def locked(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        #result = fn(self, *args, **kwargs)
        with self._lock:
            result = fn(self, *args, **kwargs)
        return result
    return wrapped


class Getter(object):
    queue = None
    workers = None
    WORKING = False

    def __init__(self, num_workers):

        self.count = 0
        self.queue = PriorityQueue()
        self.workers = []
        self.WORKING = True
        self.localdata = threading.local()

        for i in range(num_workers):
            t = threading.Thread(target=self.worker, args=(i,), daemon=True)
            t.start()
            self.workers.append(t)

        #self.stat_t = threading.Thread(target=self.show_stats, daemon=True)
        #self.stat_t.start()


    def stop(self):
        self.WORKING = False
        for t in self.workers:
            t.join()
        self.stat_t.join()

    def worker(self, idx):
        global STATS
        self.localdata.idx = idx
        while self.WORKING:
            try:
                obj, args, kwargs = self.queue.get(timeout=5)
                #log.debug(f"Got: {obj} {args} {kwargs}")
            except Empty:
                #log.debug(f"timeout, continue")
                #log.info(f"Got {self.counter}")
                continue

            #STATS.setdefault('count', 0) + 1
            STATS['count'] = STATS.get('count', 0) + 1


            try:
                if not self.get(obj, *args, **kwargs):
                    log.warning(f"Failed getting: {obj} {args} {kwargs}, re-submit.")
                    self.submit(obj, *args, **kwargs)
            except Exception as err:
                log.error(f"ERROR {err} getting: {obj} {args} {kwargs}, re-submit.")
                self.submit(obj, *args, **kwargs)

    def get(obj, *args, **kwargs):
        raise NotImplementedError

    def submit(self, obj, *args, **kwargs):
        self.queue.put((obj, args, kwargs))

    def show_stats(self):
        while self.WORKING:
            log.info(f"{self.__class__.__name__} got: {self.count}")
            print(f"{self.__class__.__name__}, qsize: {self.queue.qsize()}")
            time.sleep(10)
        log.info(f"Exiting {self.__class__.__name__} stat thread.  Got: {self.count} total")


class ChunkGetter(Getter):
    def get(self, obj, *args, **kwargs):
        if obj.ready.is_set():
            log.info(f"{obj} already retrieved.  Exit")
            return True

        kwargs['idx'] = self.localdata.idx
        #log.debug(f"{obj}, {args}, {kwargs}")
        return obj.get(*args, **kwargs)

chunk_getter = ChunkGetter(32)

log.info(f"chunk_getter: {chunk_getter}")

class EventRate:
    _nevent = 0
    _nevent_prev = 0
    rate = 0
    qsize = 150
    min_len = 10
    max_len = 250

    def __init__(self, name, tick_delta):
        self.tick_delta = tick_delta
        self.qsize = int(0.75 * self.max_len)
        self.name = name
        self.t = threading.Thread(target=self._ticker, daemon=True)
        self.t.start()

    def report_event(self):
        self._nevent += 1

    def _ticker(self):
        while True:
            self.rate = (self._nevent - self._nevent_prev) / self.tick_delta
            self._nevent_prev = self._nevent
            if self.rate > 0.0:
                self.qsize -= 2 * self.rate * self.tick_delta
            elif self.rate < 1.0:
                self.qsize += 5

            self.qsize = int(max(self.min_len, min(self.max_len, self.qsize)))
            print(f"{self.name} rate: {self.rate:1.2f} {self.qsize}")
            time.sleep(self.tick_delta)

timeout_rate = EventRate("timeout", 3.0)

class Chunk(object):
    col = -1
    row = -1
    source = None
    chunk_id = ""
    priority = 0
    width = 256
    height = 256
    cache_dir = 'cache'

    attempt = 0

    starttime = 0
    fetchtime = 0

    ready = None
    img = None
    deadline = 0

    serverlist=['a','b','c','d']

    def __init__(self, col, row, maptype, zoom, priority=0, cache_dir='.cache'):
        self.col = col
        self.row = row
        self.zoom = zoom
        self.maptype = maptype
        self.cache_dir = cache_dir

        # Hack override maptype
        #self.maptype = "BI"

        if not priority:
            self.priority = zoom
        self.chunk_id = f"{col}_{row}_{zoom}_{maptype}"
        self.ready = threading.Event()
        self.ready.clear()
        if maptype == "Null":
            self.maptype = "EOX"

        self.cache_path = os.path.join(self.cache_dir, f"{self.chunk_id}.jpg")

    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        #return f"Chunk({self.col},{self.row},{self.maptype},{self.zoom},{self.priority})"
        return f"Chunk({self.col},{self.row},{self.maptype},{self.zoom},{self.priority},{self.deadline - time.time():0.1f})"

    def get_cache(self):
        self.img = AoImage.open(self.cache_path, log_error = False)
        if self.img:
            STATS_inc('chunk_hit')
            return True
        else:
            STATS_inc('chunk_miss')
            return False

    def save_cache(self, data):
        with open(self.cache_path, 'wb') as h:
            h.write(data)

    def get(self, idx=0):
        #log.debug(f"Getting {self}")

        if self.get_cache():
            self.ready.set()
            return True

        remaining_time = self.deadline - time.time()

        # expired before being retrieved
        if remaining_time <= 0.3:
            log.info(f"deadline not met for {self}")
            self.ready.set()    # results in a black hole
            timeout_rate.report_event()
            return True

        if not self.starttime:
            self.startime = time.time()

        server_num = idx%(len(self.serverlist))
        server = self.serverlist[server_num]
        quadkey = _gtile_to_quadkey(self.col, self.row, self.zoom)

        # Hack override maptype
        #maptype = "ARC"

        MAPTYPES = {
            "EOX": f"https://{server}.s2maps-tiles.eu/wmts/?layer={MAPID}&style=default&tilematrixset={MATRIXSET}&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image%2Fjpeg&TileMatrix={self.zoom}&TileCol={self.col}&TileRow={self.row}",
            "BI": f"http://r{server_num}.ortho.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=136",
            #"GO2": f"http://khms{server_num}.google.com/kh/v=934?x={self.col}&y={self.row}&z={self.zoom}",
            "ARC": f"http://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{self.zoom}/{self.row}/{self.col}",
            "NAIP": f"http://naip.maptiles.arcgis.com/arcgis/rest/services/NAIP/MapServer/tile/{self.zoom}/{self.row}/{self.col}",
            "USGS": f"https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{self.zoom}/{self.row}/{self.col}",
            "FIREFLY": f"https://fly.maptiles.arcgis.com/arcgis/rest/services/World_Imagery_Firefly/MapServer/tile/{self.zoom}/{self.row}/{self.col}"
        }
        url = MAPTYPES[self.maptype.upper()]
        #log.debug(f"{self} getting {url}")
        header = {
                "user-agent": "curl/7.68.0"
        }

        #time.sleep((self.attempt/10))
        self.attempt += 1

        req = Request(url, headers=header)
        resp = 0
        data = None
        try:
            resp = urlopen(req, timeout=max(1, remaining_time))
            if resp.status != 200:
                log.warning(f"Failed with status {resp.status} to get chunk {self} on server {server}.")
                return False
            data = resp.read()
            if data[:3] != b'\xFF\xD8\xFF':
                # FFD8FF identifies image as a JPEG
                log.debug(f"Chunk {self} is not a JPEG! {data[:3]} URL: {url}")
                data = None
                # FALLTHROUGH
        except Exception as err:
            log.warning(f"Failed to get chunk {self} on server {server}. Err: {err}")
            timeout_rate.report_event()
            # FALLTHROUGH
        finally:
            if resp:
                resp.close()

        self.fetchtime = time.time() - self.starttime

        if data:
            STATS['bytes_dl'] = STATS.get('bytes_dl', 0) + len(data)
            self.save_cache(data)
            self.img = AoImage.load_from_memory(data, log_error = False)

        self.ready.set()
        return True

    def close(self):
        self.img = None

class BgWorkUnit:

    def __init__(self, img, pathname, zoom, chunk_names):
        self.img = img
        self.pathname = pathname
        self.zoom = zoom
        self.chunk_names = chunk_names

    def __repr__(self):
        return f"BgWorkUnit {self.pathname}"

    def execute(self):
        #print(f"Saving {self.pathname}")
        self.img.write_jpg(self.pathname, 50)
        for cn in self.chunk_names:
            try:
                os.remove(cn)
                #print(f"deleted {cn}")
            except FileNotFoundError:
                pass
                print(f"can't delete {cn}!")

class Tile(object):
    row = -1
    col = -1
    maptype = None
    zoom = -1
    min_zoom = 12
    width = 16
    height = 16

    max_mipmap = 4

    priority = -1
    #tile_condition = None
    _lock = None
    ready = None

    chunks = None
    hdr_im = None       # header image
    cache_file = None
    dds = None

    refs = None
    last_access = 0

    first_open = True

    # a global zoom out of everything
    global_zoom_out = 1

    def __init__(self, col, row, maptype, zoom, min_zoom=0, priority=0, cache_dir=None):
        self.row = int(row)
        self.col = int(col)
        self.maptype = maptype
        self.zoom = int(zoom)
        self.chunks = {}
        self.cache_file = (-1, None)
        self.ready = threading.Event()
        self._lock = threading.RLock()
        self.refs = 0

        self.bytes_read = 0
        self.lowest_offset = 99999999

        #self.tile_condition = threading.Condition()
        if min_zoom:
            self.min_zoom = int(min_zoom)

        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = CFG.paths.cache_dir

        # Hack override maptype
        #self.maptype = "BI"

        #self._find_cached_tiles()
        self.ready.clear()

        #self._find_cache_file()

        if not priority:
            self.priority = zoom

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        if CFG.pydds.compressor.upper() == "ISPC":
            use_ispc=True
        else:
            use_ispc=False

        self.dds = pydds.DDS(self.width*256, self.height*256, ispc=use_ispc,
                dxt_format=CFG.pydds.format)
        self.id = f"{row}_{col}_{maptype}_{zoom}"
        self.bg_work = []

    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        return f"Tile({self.col}, {self.row}, {self.maptype}, {self.zoom}, {self.min_zoom}, {self.cache_dir})"

    @locked
    def _create_chunks(self, quick_zoom=0):
        col, row, width, height, zoom, zoom_diff = self._get_quick_zoom(quick_zoom)

        if not self.chunks.get(zoom):
            self.chunks[zoom] = []

            for r in range(row, row+height):
                for c in range(col, col+width):
                    #chunk = Chunk(c, r, self.maptype, zoom, priority=self.priority)
                    chunk = Chunk(c, r, self.maptype, zoom, cache_dir=self.cache_dir)
                    self.chunks[zoom].append(chunk)

    def _get_quick_zoom(self, quick_zoom=0):
        if quick_zoom:
            # Max difference in steps this tile can support
            max_diff = min((self.zoom - int(quick_zoom)), self.max_mipmap)
            # Minimum zoom level allowed
            min_zoom = max((self.zoom - max_diff), self.min_zoom)

            # Effective zoom level we will use
            quick_zoom = max(int(quick_zoom), min_zoom)

            # Effective difference in steps we will use
            zoom_diff = min((self.zoom - int(quick_zoom)), self.max_mipmap)

            col = int(self.col/pow(2,zoom_diff))
            row = int(self.row/pow(2,zoom_diff))
            width = int(self.width/pow(2,zoom_diff))
            height = int(self.height/pow(2,zoom_diff))
            zoom = int(quick_zoom)
        else:
            col = self.col
            row = self.row
            width = self.width
            height = self.height
            zoom = self.zoom
            zoom_diff = 0

        return (col, row, width, height, zoom, zoom_diff)

    def find_mipmap_pos(self, offset):
        for m in self.dds.mipmap_list:
            if offset < m.endpos:
                return m.idx
        return self.dds.mipmap_list[-1].idx

    def get_bytes(self, offset, length, ot_ctx):

        mipmap = self.find_mipmap_pos(offset)
        log.debug(f"Get_bytes for mipmap {mipmap} ...")
        if mipmap > 4:
            # Just get the entire mipmap
            self.get_mipmap(ot_ctx, 4)
            return True

        # Exit if already retrieved
        if self.dds.mipmap_list[mipmap].retrieved:
            log.debug(f"We already have mipmap {mipmap} for {self}")
            return True

        mm = self.dds.mipmap_list[mipmap]
        if length >= mm.length:
            self.get_mipmap(ot_ctx, mipmap)
            return True

        log.debug(f"Retrieving {length} bytes from mipmap {mipmap} offset {offset}")

        if CFG.pydds.format == "BC1":
            bytes_per_row = 524288 >> mipmap
        else:
            bytes_per_row = 1048576 >> mipmap

        rows_per_mipmap = 16 >> mipmap

        # how deep are we in a mipmap
        mm_offset = max(0, offset - self.dds.mipmap_list[mipmap].startpos)
        log.debug(f"MM_offset: {mm_offset}  Offset {offset}.  Startpos {self.dds.mipmap_list[mipmap]}")

        if CFG.pydds.format == "BC1":
            # Calculate which row 'offset' is in
            startrow = mm_offset >> (19 - mipmap)
            # Calculate which row 'offset+length' is in
            endrow = (mm_offset + length) >> (19 - mipmap)
        else:
            # Calculate which row 'offset' is in
            startrow = mm_offset >> (20 - mipmap)
            # Calculate which row 'offset+length' is in
            endrow = (mm_offset + length) >> (20 - mipmap)

        log.debug(f"Startrow: {startrow} Endrow: {endrow}")

        new_im = self.get_img(mipmap, ot_ctx, startrow, endrow)
        if not new_im:
            log.debug("No updates, so no image generated")
            return True

        self.ready.clear()
        #log.info(new_im.size)

        start_time = time.time()

        # Only attempt partial compression from mipmap start
        if offset == 0:
            compress_len = length
            #compress_len = length - 128
        else:
            compress_len = 0

        try:
            self.dds.gen_mipmaps(new_im, mipmap, mipmap, compress_len)
        except:
            pass

        # We haven't fully retrieved so unset flag
        log.debug(f"UNSETTING RETRIEVED! {self}")
        self.dds.mipmap_list[mipmap].retrieved = False
        end_time = time.time()
        self.ready.set()

        if compress_len:
            #STATS['partial_mm'] = STATS.get('partial_mm', 0) + 1
            tile_time = end_time - start_time
            partial_stats.set(mipmap, tile_time)
            STATS['partial_mm_averages'] = partial_stats.averages
            STATS['partial_mm_counts'] = partial_stats.counts

        return True

    def read_dds_bytes(self, offset, length, ot_ctx):
        log.debug(f"READ DDS BYTES: {offset} {length}")

        if offset > 0 and offset < self.lowest_offset:
            self.lowest_offset = offset

        mm_idx = self.find_mipmap_pos(offset)
        mipmap = self.dds.mipmap_list[mm_idx]

        part1 = None
        if offset == 0:
            # If offset = 0, read the header
            log.debug("READ_DDS_BYTES: Read header")
            self.get_bytes(0, length, ot_ctx)
        #elif offset < 32768:
        #elif offset < 65536:
        elif offset < 131072:
        #elif offset < 262144:
        #elif offset < 1048576:
            # How far into mipmap 0 do we go before just getting the whole thing
            log.debug("READ_DDS_BYTES: Middle of mipmap 0")
            self.get_bytes(0, length + offset, ot_ctx)
        elif (offset + length) < mipmap.endpos:
            # Total length is within this mipmap.  Make sure we have it.
            log.debug(f"READ_DDS_BYTES: Detected middle read for mipmap {mipmap.idx}")
            if not mipmap.retrieved:
                log.debug(f"READ_DDS_BYTES: Retrieve {mipmap.idx}")
                self.get_mipmap(ot_ctx, mipmap.idx)
        else:
            log.debug(f"READ_DDS_BYTES: Start before this mipmap {mipmap.idx}")
            # We already know we start before the end of this mipmap
            # We must extend beyond the length.

            # If we seek here (i.e. we are not in a sequential read cycle) we conclude that
            # this is just a collateral effect of input blocking and data does not matter.
            # So we do not do a costly construction of the end row of this mm but return zeroes.
            if mm_idx < 4 and offset != ot_ctx.last_read_pos:
                #print(f"Seek into middle of mm {mm_idx} {offset} {length}")
                delta = mipmap.endpos - offset
                part1 = b'\x00' * delta
                offset += delta
                length -= delta
                assert length >= 0
            else:
                # Get bytes prior to this mipmap
                self.get_bytes(offset, length, ot_ctx)

            # Get the entire next mipmap
            self.get_mipmap(ot_ctx, mm_idx + 1)

        self.bytes_read += length

        # Seek and return data
        self.dds.seek(offset)
        if part1 is None:
            return self.dds.read(length)
        else:
            return part1 + self.dds.read(length)

    @locked
    def execute_bg_work(self):
        for wu in self.bg_work:
            print(f"Executing {wu}")
            wu.execute()
            self.chunks[wu.zoom] = []

        self.bg_work = []

    @locked
    def get_img(self, mipmap, ot_ctx, startrow=0, endrow=None):
        #
        # Get an image for a particular mipmap
        #

        #print(f"{self}")
        #print(f"1: zoom: {self.zoom}, Mipmap: {mipmap}, startrow: {startrow} endrow: {endrow}")

        req_mipmap = mipmap     # requested mm before gzo

        # Get effective zoom
        gzo_effective = min(self.global_zoom_out, max(0, 4 - mipmap))
        zoom = self.zoom - (mipmap + gzo_effective)

        log.debug(f"GET_IMG: Default zoom: {self.zoom}, Requested Mipmap: {mipmap}, Requested mipmap zoom: {zoom}")
        col, row, width, height, zoom, mipmap = self._get_quick_zoom(zoom)
        log.debug(f"Will use:  Zoom: {zoom},  Mipmap: {mipmap}")


        log.debug(f"GET_IMG: MM List before { {x.idx:x.retrieved for x in self.dds.mipmap_list} }")
        if self.dds.mipmap_list[req_mipmap].retrieved:
            log.debug(f"GET_IMG: We already have mipmap {req_mipmap} for {self}")
            return

        req_header = req_mipmap == 0 and startrow == 0 and endrow == 0  # header only requested
        req_full_img = startrow == 0 and endrow is None                 # full image requested

        new_im = None

        mm_jpg_path = os.path.join(self.cache_dir, f"mm_{self.col}_{self.row}_{self.maptype}_{zoom}_{mipmap}.jpg")
        hdr_jpg_path = os.path.join(self.cache_dir, f"hdr_{self.col}_{self.row}_{self.maptype}_{zoom}.jpg")

        if req_header:   # header only
            if self.hdr_im is None:
                self.hdr_im = AoImage.open(hdr_jpg_path, log_error = False)
                if self.hdr_im:
                    STATS_inc('jpg_hdr_dsk_hit')
                    #print(f"opened {hdr_jpg_path}")
            else:
                STATS_inc('jpg_hdr_mem_hit')

            if self.hdr_im:
                if gzo_effective > 0:
                    return self.hdr_im.enlarge_2(gzo_effective)
                return self.hdr_im

        if mipmap < 4:
            new_im = AoImage.open(mm_jpg_path, log_error = False)
            if new_im:      # whole image found?
                STATS['jpg_mm_hit'] = STATS.get('jpg_mm_hit', 0) + 1
                #print(f"opened {mm_jpg_path}")
                if gzo_effective > 0:
                    new_im = new_im.enlarge_2(gzo_effective)
                return new_im
            elif startrow == 0 and req_mipmap == 0 and self.hdr_im is None: # else try r0 for later
                self.hdr_im = AoImage.open(hdr_jpg_path, log_error = False)
                if self.hdr_im:
                    STATS['jpg_hdr_hit'] = STATS.get('jpg_hdr_hit', 0) + 1

        startchunk = 0
        endchunk = None
        # Determine start and end chunk
        chunks_per_row = 16  >> mipmap
        if startrow:
            startrow >>= gzo_effective
            startchunk = startrow * chunks_per_row
            if req_mipmap == 0 and self.hdr_im:
                startchunk += chunks_per_row
        if endrow is not None:
            endrow >>= gzo_effective
            endchunk = (endrow * chunks_per_row) + chunks_per_row

        self._create_chunks(zoom)
        chunks = self.chunks[zoom][startchunk:endchunk]
        log.debug(f"Start chunk: {startchunk}  End chunk: {endchunk}  Chunklen {len(self.chunks[zoom])}")

        log.debug(f"GET_IMG: {self} : Retrieve mipmap for ZOOM: {zoom} MIPMAP: {mipmap}")

        log.debug(f"GET_IMG: {self} retrieving/submitting chunks.")
        # load cached chunks
        have_all_chunks = True
        for chunk in chunks:
            if chunk.img is None:
                chunk.get_cache()

            if chunk.img is not None:
                chunk.ready.set()
            else:
                have_all_chunks = False

        bg_chunk = None

        # only submit missing chunks to the workers
        if not have_all_chunks:
            if mipmap <= 3:         # more than 4 chunks, its worth a bg image
                steps = 4 - mipmap  # steps to enlarge
                bg_col = col >> steps
                bg_row = row >> steps
                bg_width = width >> steps
                bg_height = height >> steps
                bg_zoom = zoom - steps

                #print(f"tile_zoom {self.zoom}, mipmap {mipmap}, width: {bg_width}, crz: {bg_col} {bg_row} {bg_zoom}")

                assert bg_width == 1
                # submit in front of the other chunks
                bg_chunk = Chunk(bg_col, bg_row, self.maptype, bg_zoom, -1, cache_dir=self.cache_dir)
                bg_chunk.deadline = ot_ctx.deadline - 0.5
                chunk_getter.submit(bg_chunk)

            # submitting more chunks than 'credits' will end in a timeout anyway
            # so just skip them
            ql = chunk_getter.queue.qsize()
            credits = max(0, timeout_rate.qsize - ql)

            # a header read on first open is just a probe so we provide
            # a background image only (=mm4 and that's needed anyway as next
            # operation probes the length by reading 8-( )
            if self.first_open and req_header:
                credits = 0
                STATS['fake_hdr'] = STATS.get('fake_hdr', 0) + 1

            for chunk in chunks:
                if chunk.img is None:
                    if credits > 0:
                        #log.info(f"SUBMIT: {chunk}")
                        chunk.ready.clear()
                        chunk.priority = ot_ctx.deadline
                        chunk.deadline = ot_ctx.deadline
                        chunk_getter.submit(chunk)
                    else:
                        STATS['submit_skip'] = STATS.get('submit_skip', 0) + 1
                        chunk.ready.set()
                credits -= 1

        log.debug(f"GET_IMG: Create new image: Zoom: {self.zoom} | {(256*width, 256*height)}")

        if bg_chunk:
            if bg_chunk.ready.wait() and bg_chunk.img:
                new_im = bg_chunk.img.enlarge_2(steps)
            else:
                log.warning(f"failed to retrieve bg_chunk {bg_chunk}")

        if new_im == None:
            if have_all_chunks:
                bg_color = (0, 0, 0)       # black is much cheaper
            else:
                bg_color = (85,74,41)      # dark shade of a soil like color
            new_im = AoImage.new('RGBA', (256*width,256*height), bg_color)

        if req_mipmap == 0 and self.hdr_im:
            new_im.paste(self.hdr_im, (0, 0))


        #log.info(f"NUM CHUNKS: {len(chunks)}")
        for chunk in chunks:
            chunk.ready.wait()
            if chunk.img is None:   # deadline not met
                new_im.has_voids = True
                continue

            start_x = int((chunk.width) * (chunk.col - col))
            start_y = int((chunk.height) * (chunk.row - row))

            new_im.paste(
                chunk.img,
                (
                    start_x,
                    start_y
                )
            )

        if (req_full_img or req_header) and mipmap < 4 and not new_im.has_voids:
            chunk_names = []
            for chunk in chunks:
                chunk_names.append(chunk.cache_path)
                chunk.img = None

            if req_full_img:
                wu = BgWorkUnit(new_im, mm_jpg_path, zoom, chunk_names)
            elif req_header:
                self.hdr_im = new_im.copy(256)
                wu = BgWorkUnit(self.hdr_im, hdr_jpg_path, zoom, chunk_names)

            self.bg_work.append(wu)

        log.debug(f"GET_IMG: DONE!  IMG created {new_im}")
        if gzo_effective > 0:
            height_only = None
            if endrow is not None:
                height_only = min(height, (endrow + 1)) * 256    # endrow may reach beyond the current img

            new_im = new_im.enlarge_2(gzo_effective, height_only)

        return new_im

    @locked
    def get_mipmap(self, ot_ctx, mipmap=0):
        #
        # Protect this method to avoid simultaneous threads attempting mm builds at the same time.
        # Otherwise we risk contention such as waiting get_img call attempting to build an image as
        # another thread closes chunks.
        #

        log.debug(f"GET_MIPMAP: {self}")

        if mipmap > self.max_mipmap:
            mipmap = self.max_mipmap

        # We can have multiple threads wait on get_img ...
        log.debug(f"GET_MIPMAP: Next call is get_img which may block!.............")
        new_im = self.get_img(mipmap, ot_ctx)
        if not new_im:
            log.debug("GET_MIPMAP: No updates, so no image generated")
            return True

        self.ready.clear()
        start_time = time.time()
        try:
            #self.dds.gen_mipmaps(new_im, mipmap)
            if mipmap == 0:
                self.dds.gen_mipmaps(new_im, mipmap, 1)
            else:
                self.dds.gen_mipmaps(new_im, mipmap)
        except:
            pass

        end_time = time.time()
        self.ready.set()

        zoom = self.zoom - mipmap
        tile_time = end_time - start_time
        mm_stats.set(mipmap, tile_time)

        #log.info(f"Compress MM {mipmap} for ZL {zoom} in {tile_time} seconds")
        #log.info(f"Average compress times: {mm_averages}")
        #log.info(f"MM counts: {mm_counts}")
        STATS['mm_counts'] = mm_stats.counts
        STATS['mm_averages'] = mm_stats.averages

        log.debug("Results:")
        log.debug(self.dds.mipmap_list)
        return True

    def should_close(self):
        if self.dds.mipmap_list[0].retrieved:
            if self.bytes_read < self.dds.mipmap_list[0].length:
                log.warning(f"TILE: {self} retrieved mipmap 0, but only read {self.bytes_read}. Lowest offset: {self.lowest_offset}")
                return False
            else:
                #log.info(f"TILE: {self} retrieved mipmap 0, full read of mipmap! {self.bytes_read}.")
                return True
        else:
            return True

    def close(self):
        log.debug(f"Closing {self}")

        #print(f"\n\nClosing {self} {len(self.bg_work)}")

        self.execute_bg_work()
        if self.dds.mipmap_list[0].retrieved:
            if self.bytes_read < self.dds.mipmap_list[0].length:
                log.warning(f"TILE: {self} retrieved mipmap 0, but only read {self.bytes_read}. Lowest offset: {self.lowest_offset}")
            else:
                log.debug(f"TILE: {self} retrieved mipmap 0, full read of mipmap! {self.bytes_read}.")

        if self.refs > 0:
            log.warning(f"TILE: Trying to close, but has refs: {self.refs}")
            return

        for chunks in self.chunks.values():
            for chunk in chunks:
                chunk.close()
        self.chunks = {}

class OpenTile_Ctx:
    """
    Context of an open tile.
    """

    _default_timeout = 5.0

    # public
    deadline = 0            # deadline for next read cycle
    last_read_pos = -1      # position after last read

    def __init__(self, tile):
        self.tile = tile
        
    def read(self, offset, length):
        self.tile.last_access = time.time()

        # new read cycle or continued read?
        if offset != self.last_read_pos:
            #print(f"new cycle: {self.last_read_pos} {offset}")
            self.deadline = time.time() + self._default_timeout

        self.last_read_pos = offset + length
        return self.tile.read_dds_bytes(offset, length, self)

class TileCacher(object):
    tiles = {}

    hits = 0
    misses = 0

    cache_mem_lim = pow(2,30) * 2
    cache_tile_lim = 100

    open_count = {}

    def __init__(self, cache_dir='.cache'):
        if MEMTRACE:
            tracemalloc.start()

        self.maptype_override = CFG.autoortho.maptype_override
        if self.maptype_override:
            log.info(f"Maptype override set to {self.maptype_override}")
        else:
            log.info(f"Maptype override not set, will use default.")
        log.info(f"Will use Compressor: {CFG.pydds.compressor}")
        self.tc_lock = threading.RLock()

        self.cache_dir = CFG.paths.cache_dir
        log.info(f"Cache dir: {self.cache_dir}")
        self.min_zoom = CFG.autoortho.min_zoom

        self.bg_processing_t = threading.Thread(target=self.bg_processing, daemon=True)
        self.bg_processing_t.start()

    def _to_tile_id(self, row, col, map_type, zoom):
        if self.maptype_override:
            map_type = self.maptype_override
        tile_id = f"{row}_{col}_{map_type}_{zoom}"
        return tile_id

    def bg_processing(self):
        log.info(f"Started tile bg_processing thread.  Mem limit {self.cache_mem_lim}")
        process = psutil.Process(os.getpid())

        while True:
            gc.collect(1)   # run garbage collector before judging bss
            time.sleep(1)   # collect is asynchronous, or?
            cur_mem = process.memory_info().rss
            rate = (self.hits * 100 ) // (1 + self.misses + self.hits)
            log.info(f"TILE CACHE:  MISS: {self.misses}  HIT: {self.hits} RATE: {rate}%")
            log.info(f"NUM OPEN TILES: {len(self.tiles)}.  TOTAL MEM: {cur_mem//1048576} MB")

            # sweep over tiles and run bg work

            # as that writes back jpegs it can be lengthy and we don't do that under
            # tc_lock. We extract them from the cache to a local queue.
            bg_q = []
            with self.tc_lock:
                for t in self.tiles.values():
                    if t.refs <= 0:
                        bg_q.append(t)

            for t in bg_q:
                t.execute_bg_work()

            del(bg_q)

            while len(self.tiles) >= self.cache_tile_lim and cur_mem > self.cache_mem_lim:
                log.debug("Hit cache limit.  Remove oldest 20")
                close_q = []
                with self.tc_lock:
                    by_age = sorted(self.tiles, key=lambda id: self.tiles.get(id).last_access)
                    for i in by_age[:20]:
                        t = self.tiles[i]
                        #print(f"age: {t.last_access}")
                        if t.refs <= 0:
                            del(self.tiles[i])
                            close_q.append(t)

                for t in close_q:
                    t.close()

                # remove all extra references to tiles so the are gc'ed
                del(by_age)
                del(close_q)
                del(t)

                cur_mem = process.memory_info().rss


            if MEMTRACE:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                log.info("[ Top 10 ]")
                for stat in top_stats[:10]:
                        log.info(stat)

            time.sleep(15)


    def _open_tile(self, row, col, map_type, zoom):
        if self.maptype_override:
            map_type = self.maptype_override
        idx = self._to_tile_id(row, col, map_type, zoom)

        log.debug(f"Get_tile: {idx}")
        with self.tc_lock:
            tile = self.tiles.get(idx)
            if not tile:
                self.misses += 1
                tile = Tile(col, row, map_type, zoom,
                    cache_dir = self.cache_dir,
                    min_zoom = self.min_zoom)
                self.tiles[idx] = tile
                self.open_count[idx] = self.open_count.get(idx, 0) + 1
                if self.open_count[idx] > 1:
                    tile.first_open = False
                    log.debug(f"Tile: {idx} opened for the {self.open_count[idx]} time.")
            elif tile.refs <= 0:
                # Only in this case would this cache have made a difference
                self.hits += 1

            tile.refs += 1
            #STATS_inc(f"open_{tile.refs}") # up to 8!
        return OpenTile_Ctx(tile)

    def _release_tile(self, open_tile):
        t = open_tile.tile
        with self.tc_lock:
            t.refs -= 1
            t.first_open = False

            # mark mms with voids as not retrieved but keep other cached data
            for m in t.dds.mipmap_list:
                if m.has_voids:
                    m.retrieved = False
                    m.has_voids = False

            self.last_read_pos = -1 # so a revive from the cache starts a new cycle
            return True
