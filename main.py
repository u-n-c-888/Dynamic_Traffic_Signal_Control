import cv2
from ultralytics import YOLO
from pathlib import Path

# ----------------------------------------------------------
# Dynamic Traffic Signal System
# YOLO is used to count vehicles.
# Ambulance roads are currently given as manual input.
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"

# YOLO automatically downloads yolov8n.pt the first time.
model = YOLO("yolov8n.pt")

ROAD_IMAGES = {
    "Road 1": IMAGE_DIR / "road1.png",
    "Road 2": IMAGE_DIR / "road2.png",
    "Road 3": IMAGE_DIR / "road3.png",
    "Road 4": IMAGE_DIR / "road4.png",
}

# Change this according to the image used for the demo.
# Examples:
# AMBULANCE_ROADS = ["Road 4"]
# AMBULANCE_ROADS = ["Road 2", "Road 4"]
# No ambulance: AMBULANCE_ROADS = []
AMBULANCE_ROADS = ["Road 4"]

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

MIN_TIME = 5
MAX_TIME = 30


def count_vehicles(image_path):
    """Detect and count cars, motorcycles, buses and trucks."""
    result = model.predict(
        source=str(image_path),
        conf=0.35,
        verbose=False
    )[0]

    count = 0

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in VEHICLE_CLASSES:
            count += 1

    return count, result


def main():
    counts = {}

    print("\n==============================================")
    print("       DYNAMIC TRAFFIC SIGNAL SYSTEM")
    print("==============================================\n")
    print("========== ANALYZING ROADS ==========\n")

    for road, image_path in ROAD_IMAGES.items():
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"{road}: IMAGE NOT FOUND -> {image_path}")
            continue

        counts[road], result = count_vehicles(image_path)

        if road in AMBULANCE_ROADS:
            print(
                f"AMBULANCE DETECTED IN {road} "
                f"-> FIRST PRIORITY"
            )
        else:
            print(f"{road}: {counts[road]} vehicles")

        cv2.imshow(road, result.plot())

    if not counts:
        print("\nNo road images found.")
        print("Put road1.png, road2.png, road3.png and road4.png")
        print("inside the images folder.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    maximum = max(counts.values())
    green_time = {}

    for road, count in counts.items():

        # Emergency road gets maximum green time.
        if road in AMBULANCE_ROADS:
            green_time[road] = MAX_TIME

        # No vehicles = minimum time.
        elif count == 0 or maximum == 0:
            green_time[road] = MIN_TIME

        # Dynamic timing based on traffic density.
        else:
            green_time[road] = round(
                MIN_TIME
                + (count / maximum) * (MAX_TIME - MIN_TIME)
            )

    # Ambulance roads first; if there are multiple,
    # higher vehicle count gets priority among them.
    priority = sorted(
        counts,
        key=lambda road: (
            road in AMBULANCE_ROADS,
            counts[road]
        ),
        reverse=True
    )

    print("\n==============================================")
    print("          TRAFFIC SIGNAL RESULT")
    print("==============================================\n")

    print(
        f"{'Priority':<10}"
        f"{'Road':<10}"
        f"{'Vehicles':<12}"
        f"{'Ambulance':<12}"
        f"Green Time"
    )
    print("-" * 60)

    for number, road in enumerate(priority, 1):
        emergency = "YES" if road in AMBULANCE_ROADS else "NO"

        print(
            f"{number:<10}"
            f"{road:<10}"
            f"{counts[road]:<12}"
            f"{emergency:<12}"
            f"{green_time[road]} seconds"
        )

    print("\n========== SIGNAL SEQUENCE ==========\n")

    for road in priority:
        if road in AMBULANCE_ROADS:
            print(
                f"AMBULANCE -> {road} -> FIRST PRIORITY -> "
                f"GREEN FOR {green_time[road]} SECONDS"
            )
        else:
            print(
                f"{road} -> GREEN FOR "
                f"{green_time[road]} SECONDS"
            )

    print("\n==============================================")
    print("       PROJECT COMPLETED SUCCESSFULLY")
    print("==============================================")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
