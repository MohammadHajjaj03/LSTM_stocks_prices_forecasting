import os
import cv2
import rasterio
import geopandas as gpd
import numpy as np
import random

from rasterio.windows import Window
from shapely.geometry import box, Polygon, MultiPolygon

# ======================
# CONFIG
# ======================

TIFF_PATH = r"C:\Users\mohma\Downloads\archive\New folder\JubailSep2025.tiff"
LABEL_GEOJSON = r"C:\Users\mohma\Downloads\archive\New folder\merged_roads.geojson"

OUTPUT_DIR = r"C:\Users\mohma\Downloads\archive\New folder\Dataset"

TILE_SIZE = 2048
TRAIN_RATIO = 0.8

CLASS_ID = 0
MIN_POLY_AREA = 5

random.seed(42)

# ======================
# CREATE OUTPUT
# ======================

for split in ["train","val"]:
    os.makedirs(os.path.join(OUTPUT_DIR,"images",split),exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR,"labels",split),exist_ok=True)

# ======================
# LOAD DATA
# ======================

src = rasterio.open(TIFF_PATH)
labels = gpd.read_file(LABEL_GEOJSON).to_crs(src.crs)

# ======================
# TILE LOOP
# ======================

tile_id = 0

for y in range(0, src.height, TILE_SIZE):
    for x in range(0, src.width, TILE_SIZE):

        window = Window(x,y,TILE_SIZE,TILE_SIZE)

        img = src.read(window=window)

        if img.size == 0:
            continue

        img = np.transpose(img,(1,2,0))
        img = img[:,:,:3]

        # تأكد أن التايل كاملة
        if img.shape[0] != TILE_SIZE or img.shape[1] != TILE_SIZE:
            continue

        # منع tiles السوداء
        if np.mean(img) < 5:
            continue

        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        bounds = rasterio.windows.bounds(window,src.transform)
        tile_poly = box(*bounds)

        tile_transform = rasterio.windows.transform(window,src.transform)

        objs = labels[labels.intersects(tile_poly)]

        split = "train" if random.random() < TRAIN_RATIO else "val"

        name = f"tile_{tile_id:06d}"

        img_path = os.path.join(OUTPUT_DIR,"images",split,name+".jpg")
        lbl_path = os.path.join(OUTPUT_DIR,"labels",split,name+".txt")

        cv2.imwrite(img_path,img)

        h,w = img.shape[:2]

        with open(lbl_path,"w") as f:

            for _,row in objs.iterrows():

                geom = row.geometry.intersection(tile_poly).buffer(0)

                if geom.is_empty:
                    continue

                if isinstance(geom,Polygon):
                    polygons=[geom]

                elif isinstance(geom,MultiPolygon):
                    polygons=list(geom.geoms)

                else:
                    continue

                for poly in polygons:

                    if poly.area < MIN_POLY_AREA:
                        continue

                    poly = poly.simplify(0.3,preserve_topology=True)

                    coords = np.array(poly.exterior.coords)

                    pixels=[]

                    for px,py in coords:

                        tx,ty = ~tile_transform * (px,py)

                        tx/=w
                        ty/=h

                        pixels.append(f"{tx:.6f} {ty:.6f}")

                    if len(pixels) < 3:
                        continue

                    line = f"{CLASS_ID} " + " ".join(pixels)

                    f.write(line+"\n")

        tile_id += 1

print("Done")
print("Tiles generated:",tile_id)