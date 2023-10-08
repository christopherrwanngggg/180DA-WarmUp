import numpy as np
import cv2


cap = cv2.VideoCapture(0)
n_clusters = 1

while True:
    status, image = cap.read()
    if not status:
        break
    crp_image = image[300:500, 550:750]
    # to reduce complexity resize the image
    data = cv2.resize(crp_image, (100, 100)).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)

    cluster_sizes = np.bincount(labels.flatten())

    palette = []
    for cluster_idx in np.argsort(-cluster_sizes):
        palette.append(np.full((image.shape[0], image.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
    palette = np.hstack(palette)

    sf = image.shape[1] / palette.shape[1]
    out = np.vstack([image, cv2.resize(palette, (0, 0), fx=sf, fy=sf)])
    start_point = (550 , 300) 
    end_point = (750, 500) 

    color = (0, 255, 0) 
    cv2.rectangle(out, start_point, end_point, color, 2)
    cv2.imshow("dominant_colors", out)
    cv2.waitKey(1)