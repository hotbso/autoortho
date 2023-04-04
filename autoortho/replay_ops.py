import sys
import aoconfig
import getortho

import cProfile, pstats
#profiler = cProfile.Profile(builtins = False)
profiler = cProfile.Profile()

tc = getortho.TileCacher()

opened_tiles = {}

def replay():
    with open("ops_stream.txt", "r") as fh:
        for line in fh:
            fields = line.split()
            row = int(fields[0])
            col = int(fields[1])
            maptype = fields[2]
            zoom = int(fields[3])
            op = fields[4]
            key = (row, col, maptype, zoom)

            if op == "O":
                #print(f"open {key}")
                t = tc._open_tile(row, col, maptype, zoom)
                del(t)
                opened_tiles[key] = 1
            elif op == "C":
                #print(f"close {key}")
                tc._close_tile(*key)
                del(opened_tiles[key])
                #print(f"{len(opened_tiles)}")
 
            elif op == "R":
                offs = int(fields[5])
                length = int(fields[6])
                #
                #print(f"read {key} {offs} {length}")
                t = tc._get_tile(*key)
                buf = t.read_dds_bytes(offs, length)
                # lb = len(buf)
                # if lb != length:
                    # print(f"size mismatch {lb} {length}!")
                del(t)

def main(filename = "ao.stats"):
    profiler.enable()
    replay()
    print(f"{len(opened_tiles)}")
    profiler.dump_stats(filename)

if __name__ == "__main__":
    main(sys.argv[1])
