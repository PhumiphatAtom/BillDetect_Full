# uvicorn --host 192.168.1.45 --port 80  main:app --reload
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File
import uvicorn
from fastapi.responses import FileResponse
import numpy as np
import cv2

chunk = 1024
FLANN_INDEX_KDTREE = 5

app = FastAPI()

def init_feature():
    detector = cv2.xfeatures2d.SIFT_create()
    norm = cv2.NORM_L1
    matcher = cv2.BFMatcher(norm)
    return detector, matcher

def filter_matches(kp1, kp2, matches, ratio=0.75):  # ratio = 0.75
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs

def match_and_draw(found, count, MIN_POINT, kp1, desc1, kp2, desc2):
    if len(kp2) > 0:
        # matching feature
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        if len(p1) >= MIN_POINT:
            found = True
            count = count + 1
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, confidence=0.9)
            return True, 0
        if count > 3:
            return True, 0
    return found, count


detector, matcher = init_feature()

# preLoad resource
img_source1 = cv2.imread("img_source/20f.bmp", 0)
temp_kp1, temp_desc1 = detector.detectAndCompute(img_source1, None)

img_source2 = cv2.imread("img_source/20b.bmp", 0)
temp_kp2, temp_desc2 = detector.detectAndCompute(img_source2, None)

img_source3 = cv2.imread("img_source/50f.bmp", 0)
temp_kp3, temp_desc3 = detector.detectAndCompute(img_source3, None)

img_source4 = cv2.imread("img_source/50b.bmp", 0)
temp_kp4, temp_desc4 = detector.detectAndCompute(img_source4, None)

img_source5 = cv2.imread("img_source/100f.bmp", 0)
temp_kp5, temp_desc5 = detector.detectAndCompute(img_source5, None)

img_source6 = cv2.imread("img_source/100b.bmp", 0)
temp_kp6, temp_desc6 = detector.detectAndCompute(img_source6, None)

img_source7 = cv2.imread("img_source/500f.bmp", 0)
temp_kp7, temp_desc7 = detector.detectAndCompute(img_source7, None)

img_source8 = cv2.imread("img_source/500b.bmp", 0)
temp_kp8, temp_desc8 = detector.detectAndCompute(img_source8, None)

img_source9 = cv2.imread("img_source/1000f.bmp", 0)
temp_kp9, temp_desc9 = detector.detectAndCompute(img_source9, None)

img_source10 = cv2.imread("img_source/1000b.bmp", 0)
temp_kp10, temp_desc10 = detector.detectAndCompute(img_source10, None)

# new bank

img_source11 = cv2.resize(cv2.imread("img_source/20f2.bmp", 0), (300, 156))
temp_kp11, temp_desc11 = detector.detectAndCompute(img_source11, None)

img_source12 = cv2.resize(cv2.imread("img_source/20b2.bmp", 0), (300, 156))
temp_kp12, temp_desc12 = detector.detectAndCompute(img_source12, None)

img_source13 = cv2.resize(cv2.imread("img_source/50f2.bmp", 0), (300, 156))
temp_kp13, temp_desc13 = detector.detectAndCompute(img_source13, None)

img_source14 = cv2.resize(cv2.imread("img_source/50b2.bmp", 0), (300, 156))
temp_kp14, temp_desc14 = detector.detectAndCompute(img_source14, None)

img_source15 = cv2.resize(cv2.imread("img_source/100f2.bmp", 0), (300, 156))
temp_kp15, temp_desc15 = detector.detectAndCompute(img_source15, None)

img_source16 = cv2.resize(cv2.imread("img_source/100b2.bmp", 0), (300, 156))
temp_kp16, temp_desc16 = detector.detectAndCompute(img_source16, None)

img_source17 = cv2.resize(cv2.imread("img_source/500f2.bmp", 0), (300, 156))
temp_kp17, temp_desc17 = detector.detectAndCompute(img_source17, None)

img_source18 = cv2.resize(cv2.imread("img_source/500b2.bmp", 0), (300, 156))
temp_kp18, temp_desc18 = detector.detectAndCompute(img_source18, None)

img_source19 = cv2.resize(cv2.imread("img_source/1000f2.bmp", 0), (300, 156))
temp_kp19, temp_desc19 = detector.detectAndCompute(img_source19, None)

img_source20 = cv2.resize(cv2.imread("img_source/1000b2.bmp", 0), (300, 156))
temp_kp20, temp_desc20 = detector.detectAndCompute(img_source20, None)


def detect_bill(frame):
    MIN_BACK = 30
    MIN_FRONT = 80
    MIN_POINT = MIN_FRONT
    found = False
    searchIndex = 1
    count = 0
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    while True:
        if not found:
            if searchIndex <= 20:
                if searchIndex == 1:
                    kp1 = temp_kp1
                    desc1 = temp_desc1
                    showText = "20"
                elif searchIndex == 2:
                    kp1 = temp_kp2
                    desc1 = temp_desc2
                    showText = "20"
                elif searchIndex == 3:
                    kp1 = temp_kp3
                    desc1 = temp_desc3
                    showText = "50"
                elif searchIndex == 4:
                    kp1 = temp_kp4
                    desc1 = temp_desc4
                    showText = "50"
                elif searchIndex == 5:
                    kp1 = temp_kp5
                    desc1 = temp_desc5
                    showText = "100"
                elif searchIndex == 6:
                    kp1 = temp_kp6
                    desc1 = temp_desc6
                    showText = "100"
                elif searchIndex == 7:
                    kp1 = temp_kp7
                    desc1 = temp_desc7
                    showText = "500"
                elif searchIndex == 8:
                    kp1 = temp_kp8
                    desc1 = temp_desc8
                    showText = "500"
                elif searchIndex == 9:
                    kp1 = temp_kp9
                    desc1 = temp_desc9
                    showText = "1000"
                elif searchIndex == 10:
                    kp1 = temp_kp10
                    desc1 = temp_desc10
                    showText = "1000"
                elif searchIndex == 11:
                    kp1 = temp_kp11
                    desc1 = temp_desc11
                    showText = "20"
                    MIN_POINT = MIN_FRONT
                elif searchIndex == 12:
                    kp1 = temp_kp12
                    desc1 = temp_desc12
                    showText = "20"
                    MIN_POINT = MIN_BACK
                elif searchIndex == 13:
                    kp1 = temp_kp13
                    desc1 = temp_desc13
                    showText = "50"
                    MIN_POINT = MIN_FRONT
                elif searchIndex == 14:
                    kp1 = temp_kp14
                    desc1 = temp_desc14
                    showText = "50"
                    MIN_POINT = MIN_BACK
                elif searchIndex == 15:
                    kp1 = temp_kp15
                    desc1 = temp_desc15
                    showText = "100"
                    MIN_POINT = MIN_FRONT
                elif searchIndex == 16:
                    kp1 = temp_kp16
                    desc1 = temp_desc16
                    showText = "100"
                    MIN_POINT = MIN_BACK
                elif searchIndex == 17:
                    kp1 = temp_kp17
                    desc1 = temp_desc17
                    showText = "500"
                    MIN_POINT = MIN_FRONT
                elif searchIndex == 18:
                    kp1 = temp_kp18
                    desc1 = temp_desc18
                    showText = "500"
                    MIN_POINT = MIN_BACK
                elif searchIndex == 19:
                    kp1 = temp_kp19
                    desc1 = temp_desc19
                    showText = "1000"
                    MIN_POINT = MIN_FRONT
                elif searchIndex == 20:
                    kp1 = temp_kp20
                    desc1 = temp_desc20
                    showText = "1000"
                    MIN_POINT = MIN_BACK
                searchIndex = searchIndex + 1
            else:
                return "not found"
        else:
            searchIndex = 1
        kp2, desc2 = detector.detectAndCompute(img2, None)
        found, count = match_and_draw(
            found, count, MIN_POINT, kp1, desc1, kp2, desc2)
        if(found == True):
            return showText

@app.post("/upload-file/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    img = cv2.imdecode(np.fromstring(uploaded_file.file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    detected = detect_bill(img)
    return {"Detect": detected}

# @app.post("/upload-file/")
# async def create_upload_file(uploaded_file: UploadFile = File(...)):
#     file_location = f"image/{uploaded_file.filename}"
#     with open(file_location, "wb+") as file_object:
#         shutil.copyfileobj(uploaded_file.file, file_object)
#     detected = detect_bill(cv2.imread("image/" + uploaded_file.filename))
#     return {"detect": detected}

# @app.get("/")
# async def connected():
#     return {"Hello": "hello"}

# @app.post("/upload/")
# async def upload_image(files: List[UploadFile] = File(...)):
#     for img in files:
#         with open(f"image/{img.filename}", "wb") as buffer:
#             shutil.copyfileobj(img.file, buffer)
#     return {"file": img.filename}

# @app.get("/{filename}")
# async def show_image(filename):
#     detected = detect_bill(cv2.imread("image/"+filename))
#     return {"detect": detected}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
