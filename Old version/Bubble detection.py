import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def non_max_suppression(boxes, overlapThresh):
    # Si aucune boîte n'est fournie, retournez une liste vide
    if len(boxes) == 0:
        return []

    # Initialisation de la liste des index sélectionnés
    pick = []

    # Coordonnées des boîtes et calcul de l'aire
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    area = (x2 - x1) * (y2 - y1)

    # Triez les boîtes selon l'axe des y
    idxs = np.argsort(y2)

    # Tant qu'il reste des index dans la liste
    while len(idxs) > 0:
        # Prenez le dernier index de la liste et ajoutez l'index à la liste des sélectionnés
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Trouvez la plus grande (x, y) coordonnées pour le début de la boîte
        # et la plus petite (x, y) coordonnées pour la fin de la boîte
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Calculez la largeur et la hauteur de la boîte
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Calculez l'aire de chevauchement et l'aire de chevauchement sur l'aire totale
        overlap = (w * h) / area[idxs[:last]]

        # Supprimez les index de la liste qui se chevauchent au-delà du seuil
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Retournez seulement les boîtes sélectionnées
    return boxes[pick].astype("int")

def merge_close_boxes(boxes, distance_thresh):
    """
    Fusionne des boîtes englobantes proches en une seule boîte.

    Args:
        boxes (list of lists/tuples): Liste des boîtes englobantes, où chaque boîte est représentée par [x, y, w, h].
        distance_thresh (float): Seuil de distance pour considérer les boîtes comme proches.

    Returns:
        list of lists: Liste des boîtes après la fusion.
    """
    # Si aucune boîte ou une seule boîte est fournie, pas besoin de fusionner
    if len(boxes) <= 1:
        return boxes

    # Convertir en array pour faciliter les opérations
    boxes_array = np.array(boxes)
    merged_boxes = []

    while len(boxes_array):
        # Prendre la première boîte
        base = boxes_array[0]
        # Calculer le coin bas droit de la première boîte
        base_br = base[:2] + base[2:]

        # Initialiser la boîte englobante minimale à la boîte de base
        min_x, min_y, max_x, max_y = base[0], base[1], base_br[0], base_br[1]

        # Pour garder une trace des boîtes à fusionner
        merge_with_base = [0]

        # Parcourir les autres boîtes et trouver celles à fusionner
        for i in range(1, len(boxes_array)):
            other = boxes_array[i]
            other_br = other[:2] + other[2:]

            # Calculer la distance entre les coins les plus proches
            distance = np.min(np.sqrt((base[:2] - other[:2])**2 + (base_br - other_br)**2))

            # Si la distance est inférieure au seuil, planifier la fusion
            if distance <= distance_thresh:
                merge_with_base.append(i)
                # Mettre à jour la boîte englobante minimale
                min_x = min(min_x, other[0])
                min_y = min(min_y, other[1])
                max_x = max(max_x, other_br[0])
                max_y = max(max_y, other_br[1])

        # Ajouter la nouvelle boîte englobante minimale fusionnée
        merged_boxes.append([min_x, min_y, max_x - min_x, max_y - min_y])

        # Supprimer les boîtes qui ont été fusionnées
        boxes_array = np.delete(boxes_array, merge_with_base, axis=0)

    return merged_boxes
 
def process_frame(frame):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
 
    # Subtract the background
    fg_mask = backSub.apply(blurred)
 
    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
 
    # Dilate the thresholded image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=2)
 
    # Find contours on the thresholded image
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    return contours
 
# Initialize the video capture
video_path = 'video_cassure.avi'
cap = cv2.VideoCapture(video_path)
 
# Check if video opened successfully
if not cap.isOpened():
    print(f"Error opening video file {video_path}")
    exit()
 
# Create a background subtractor object
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    contours = process_frame(frame)

    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, w, h])

    # Fusionnez les boîtes proches avant d'appliquer la NMS
    distance_thresh = 10  # Ajustez le seuil selon les besoins
    boxes = merge_close_boxes(boxes, distance_thresh)

    # Maintenant, appliquez la NMS sur les boîtes fusionnées
    pick = non_max_suppression(np.array(boxes), 0) # Ajustez le seuil selon les besoins

    for (x, y, w, h) in pick:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Moving Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()