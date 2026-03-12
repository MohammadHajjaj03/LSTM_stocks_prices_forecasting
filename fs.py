# =====================================================
# GroundingDINO → SAM2 → VEHICLES (GUIDED SEGMENTATION)
# - SAM uses POINT PROMPT inside the DINO box (fixes random segmentation)
# - Filters are SOFT (area only) to avoid dropping true vehicles
# =====================================================

import os, json, cv2, torch, warnings
import numpy as np
from tqdm import tqdm
from PIL import Image

import rasterio
from rasterio.features import shapes
from rasterio.crs import CRS

import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid
from pyproj import Transformer

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from groundingdino.util.inference import load_model, predict
from groundingdino.datasets import transforms as T

warnings.filterwarnings("ignore")

# =====================================================
# PATHS
# =====================================================
TILES_DIR   = r"C:\Users\mohma\OneDrive\Desktop\Saqer\New folder\Cons"
INDEX_JSON  = r"C:\Users\mohma\OneDrive\Desktop\Saqer\New folder\Cons\tiles_index_clean.json"

DINO_CONFIG = r"C:\Users\mohma\PycharmProjects\pythonProject19\models\GroundingDINO_SwinT_OGC.py"
DINO_CKPT   = r"C:\Users\mohma\PycharmProjects\pythonProject19\models\groundingdino_swint_ogc.pth"

SAM_CONFIG  = r"C:\Users\mohma\PycharmProjects\pythonProject19\.venv\Lib\site-packages\sam2\configs\sam2\sam2_hiera_s.yaml"
SAM_CKPT    = r"C:\Users\mohma\PycharmProjects\pythonProject19\sam2_hiera_s.pt"

OUT_GEOJSON = r"C:\Users\mohma\OneDrive\Desktop\Saqer\CARS2_2ONL2Y_G23U33I2244DED.geojson"

# =====================================================
# DINO PROMPT
# =====================================================
TEXT_PROMPT = "car. vehicle."

BOX_THRESHOLD  = 0.38
TEXT_THRESHOLD = 0.85

MIN_AREA_M2 = 5.0
MAX_AREA_M2 = 150.0

# =====================================================
# HELPERS
# =====================================================
def affine_from_list(a):
    return rasterio.Affine(*a[:6])

def safe_make_valid(g):
    try:
        return make_valid(g)
    except:
        return g.buffer(0)

to_m = Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True).transform

def geom_area_m2(g):
    return shp_transform(to_m, g).area

# =====================================================
# MASK SMOOTHING (LIGHT)
# =====================================================
def smooth_mask(mask):
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    blur = cv2.GaussianBlur(mask.astype(np.float32),(3,3),0)
    return (blur > 0.45).astype(np.uint8)

# =====================================================
# SELECT BEST MASK (middle area is usually safest)
# (smallest can be noise, largest can be ground)
# =====================================================
def select_best_mask(masks):
    areas = np.array([m.sum() for m in masks], dtype=np.float32)
    order = np.argsort(areas)
    idx = order[len(order)//2]
    return masks[idx].astype(np.uint8)

# =====================================================
# DINO DETECTION
# =====================================================
def dino_boxes(img_rgb, model, device):
    H,W,_ = img_rgb.shape
    tfm = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    timg,_ = tfm(Image.fromarray(img_rgb),None)
    timg = timg.to(device)

    boxes, scores, _ = predict(
        model=model,
        image=timg,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    out=[]
    for (cx,cy,bw,bh), sc in zip(boxes, scores):
        x1=int((cx-bw/2)*W); y1=int((cy-bh/2)*H)
        x2=int((cx+bw/2)*W); y2=int((cy+bh/2)*H)
        if x2>x1 and y2>y1:
            out.append([x1,y1,x2,y2])
    return out

# =====================================================
# SAM2 GUIDED PREDICT (POINT PROMPT INSIDE BOX)
# =====================================================
def sam_guided_mask(sam, box, H, W):
    x1,y1,x2,y2 = box

    # center point (and 2 small offsets) لتقليل احتمالية انه النقطة تقع على ظل/فراغ
    cx = int((x1+x2)/2); cy = int((y1+y2)/2)
    p1 = [cx, cy]
    p2=[int(cx+0.08 * (x2-x1)),cy]
    p3=[cx,int(cy+0.08 * (y2-y1))]

    point_coords = np.array([p1, p2, p3], dtype=np.float32)
    point_labels = np.array([1, 1, 1], dtype=np.int32)

    masks, _, _ = sam.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=np.array(box, dtype=np.float32),
        multimask_output=True
    )

    if masks is None or len(masks) == 0:
        return None

    mask = smooth_mask(select_best_mask(masks))

    # قص الماسك داخل البوكس فقط (يقلل التوسع الغلط)
    clipped = np.zeros_like(mask, dtype=np.uint8)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped

# =====================================================
# LOAD MODELS
# =====================================================
with open(INDEX_JSON) as f:
    tiles = json.load(f)

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print("Tiles:",len(tiles),"| Device:",DEVICE)

dino = load_model(DINO_CONFIG, DINO_CKPT).to(DEVICE)
sam  = SAM2ImagePredictor(build_sam2(SAM_CONFIG, SAM_CKPT, device=DEVICE))

# =====================================================
# PIPELINE
# =====================================================
cars=[]

for meta in tqdm(tiles, desc="🚗 Cars (Guided)"):
    img_path = os.path.join(TILES_DIR, meta["filename"])
    img = cv2.imread(img_path)
    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape

    boxes = dino_boxes(img_rgb, dino, DEVICE)
    if not boxes:
        continue

    affine = affine_from_list(meta["transform"])
    src_crs = CRS.from_string(meta["crs"])
    to_wgs84 = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)

    sam.set_image(img_rgb)

    for box in boxes:
        mask = sam_guided_mask(sam, box, H, W)
        if mask is None:
            continue
        if mask.sum() < 10:  # فلتر خفيف جدًا للضجيج
            continue

        for geom, val in shapes(mask, mask=mask, transform=affine):
            if val != 1:
                continue

            g = safe_make_valid(shape(geom))

            # تنعيم هندسي خفيف (بدون قتل التفاصيل)
            g = g.simplify(0.25, preserve_topology=True)

            # تحويل إحداثيات
            g = shp_transform(lambda x,y: to_wgs84.transform(x,y), g)

            if not g.is_valid or g.is_empty:
                continue

            area = geom_area_m2(g)
            if MIN_AREA_M2 <= area <= MAX_AREA_M2:
                cars.append(g)

# =====================================================
# SAVE
# =====================================================
if not cars:
    raise RuntimeError("❌ No cars detected")

gdf = gpd.GeoDataFrame(geometry=cars, crs="EPSG:4326")
gdf.to_file(OUT_GEOJSON, driver="GeoJSON")

print("====================================")
print("✅ DONE – GUIDED SAM (LESS RANDOM)")
print("Cars detected:", len(gdf))
print("Saved to:", OUT_GEOJSON)
print("====================================")
