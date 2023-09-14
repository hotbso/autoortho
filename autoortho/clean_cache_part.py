#clean_cache_part.py -rect 48.762564,8.040208,48.797761,8.117613

import sys, os, math, re
from aoconfig import CFG

def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
  return (xtile, ytile)


def usage():
    print( \
        """clean_cache -rect lower_left,upper_right [-dry_run]
            -dry_run    only list matching files

            Examples:
                clean_cache -rect +36+019,+40+025 
        """)
    sys.exit(2)

cache_dir = CFG.paths.cache_dir + '-ht'

dry_run = False
lat1 = None

i = 1
while i < len(sys.argv):
    if sys.argv[i] == "-rect":
        i = i + 1
        if i >= len(sys.argv):
            usage()


        lat1, lon1, lat2, lon2 = sys.argv[i].split(',')
        print(f"restricting to rect ({lat1},{lon1}) -> ({lat2},{lon2})")

    elif sys.argv[i] == "-dry_run":
        dry_run = True

    else:
        usage()

    i = i + 1

if lat1 is None:
    usage()

base_zoom = 16
max_zoom = 18
# (0, 0) is in the upper left corner, i.e. y is down
x1, y2 = deg2num(float(lat1), float(lon1), base_zoom)
# a dds has 16x16 google tiles
x1 = (x1 >> 4) << 4
y2 = ((y2 >> 4) << 4) + 16

x2, y1 = deg2num(float(lat2), float(lon2), base_zoom)
x2 = ((x2 >> 4) << 4) + 16
y1 = (y1 >> 4) << 4

print(f"x: [{x1},{x2}), y: [{y1},{y2})")

files = []

def process_hdr_mm(x1, x2, y1, y2, z1, z2):
    # pre assembled jpgs for dds tiles
    z = z1
    while z <= z2:
        for x in range(x1, x2, 16):
            for y in range(y1, y2, 16):
                for gzo in range(0, 2):
                    hdr_jpg_path = os.path.join(cache_dir, f"hdr_{x}_{y}_BI_{z}_{gzo}.jpg")
                    if os.path.isfile(hdr_jpg_path):
                        #print(hdr_jpg_path)
                        files.append(hdr_jpg_path)
                        
                for mm in range(0, 4):
                    mm_jpg_path = os.path.join(cache_dir, f"mm_{x}_{y}_BI_{z}_{mm}.jpg")
                    if os.path.isfile(mm_jpg_path):
                        #print(mm_jpg_path)
                        files.append(mm_jpg_path)

        x1 = x1 << 1
        x2 = x2 << 1
        y1 = y1 << 1
        y2 = y2 << 1
        z = z + 1

def process_jpg(x1, x2, y1, y2, z1, z2):
    z = z1
    while z <= z2:
        for x in range(x1, x2):
            for y in range(y1, y2):
                jpg_path = os.path.join(cache_dir, f"{x}_{y}_{z}_BI.jpg")
                if os.path.isfile(jpg_path):
                    #print(jpg_path)
                    files.append(jpg_path)

        x1 = x1 << 1
        x2 = x2 << 1
        y1 = y1 << 1
        y2 = y2 << 1
        z = z + 1


process_hdr_mm(x1, x2, y1, y2, base_zoom, max_zoom)
process_jpg(x1 >> 4, x2 >> 4, y1 >> 4, y2 >> 4, base_zoom - 4, max_zoom)

for f in files:
    print(f)
    if not dry_run:
        os.remove(f)