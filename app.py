import streamlit as st
import cv2
from ultralytics import YOLO
from pathlib import Path

# ==========================================================
# DYNAMIC TRAFFIC SIGNAL SYSTEM - WEB VERSION
# ==========================================================

st.set_page_config(
    page_title="Dynamic Traffic Signal System",
    page_icon="🚦",
    layout="wide"
)

# ----------------------------------------------------------
# Project paths
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"

ROAD_IMAGES = {
    "Road 1": IMAGE_DIR / "road1.png",
    "Road 2": IMAGE_DIR / "road2.png",
    "Road 3": IMAGE_DIR / "road3.png",
    "Road 4": IMAGE_DIR / "road4.png"
}

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}

MIN_TIME = 5
MAX_TIME = 30


# ----------------------------------------------------------
# Load YOLO model
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# ----------------------------------------------------------
# Vehicle detection
# ----------------------------------------------------------

def count_vehicles(image_path):

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


# ----------------------------------------------------------
# Calculate green signal time
# ----------------------------------------------------------

def calculate_green_time(counts, ambulance_roads):

    maximum = max(counts.values())

    green_time = {}

    for road, count in counts.items():

        # Ambulance gets maximum priority
        if road in ambulance_roads:

            green_time[road] = MAX_TIME

        # No vehicles
        elif count == 0 or maximum == 0:

            green_time[road] = MIN_TIME

        # Dynamic timing based on traffic
        else:

            green_time[road] = round(
                MIN_TIME
                + (count / maximum)
                * (MAX_TIME - MIN_TIME)
            )

    return green_time


# ----------------------------------------------------------
# Main interface
# ----------------------------------------------------------

st.title("🚦 Dynamic Traffic Signal System")

st.write(
    "An AI-based traffic signal system that dynamically "
    "allocates green signal time based on vehicle density."
)

st.divider()

# ----------------------------------------------------------
# Ambulance selection
# ----------------------------------------------------------

st.subheader("🚑 Emergency Vehicle Priority")

ambulance_roads = st.multiselect(
    "Select road(s) containing an ambulance:",
    list(ROAD_IMAGES.keys()),
    default=["Road 4"]
)

st.info(
    "Ambulance roads receive first priority and maximum "
    "green signal time."
)

st.divider()

# ----------------------------------------------------------
# Run analysis button
# ----------------------------------------------------------

if st.button("🔍 Analyze Traffic", type="primary"):

    counts = {}
    results = {}

    st.subheader("📷 Road Analysis")

    # Display roads in columns
    columns = st.columns(2)

    for index, (road, image_path) in enumerate(ROAD_IMAGES.items()):

        with columns[index % 2]:

            st.markdown(f"### {road}")

            image = cv2.imread(str(image_path))

            if image is None:

                st.error(
                    f"{road} image not found: {image_path}"
                )

                continue

            count, result = count_vehicles(image_path)

            counts[road] = count
            results[road] = result

            # YOLO result image
            detected_image = result.plot()

            # Convert BGR → RGB
            detected_image = cv2.cvtColor(
                detected_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                detected_image,
                use_container_width=True
            )

            if road in ambulance_roads:

                st.warning(
                    f"🚑 AMBULANCE | {count} vehicles detected"
                )

            else:

                st.write(
                    f"🚗 Vehicles detected: **{count}**"
                )

    # ------------------------------------------------------
    # Calculate result
    # ------------------------------------------------------

    if counts:

        green_time = calculate_green_time(
            counts,
            ambulance_roads
        )

        # Priority order
        priority = sorted(
            counts,
            key=lambda road: (
                road in ambulance_roads,
                counts[road]
            ),
            reverse=True
        )

        st.divider()

        st.subheader("🚦 Traffic Signal Result")

        # --------------------------------------------------
        # Result table
        # --------------------------------------------------

        table_data = []

        for number, road in enumerate(priority, 1):

            ambulance = (
                "YES"
                if road in ambulance_roads
                else "NO"
            )

            table_data.append({
                "Priority": number,
                "Road": road,
                "Vehicles": counts[road],
                "Ambulance": ambulance,
                "Green Time": f"{green_time[road]} seconds"
            })

        st.table(table_data)

        # --------------------------------------------------
        # Signal sequence
        # --------------------------------------------------

        st.subheader("🔄 Signal Sequence")

        for road in priority:

            if road in ambulance_roads:

                st.success(
                    f"🚑 {road} → FIRST PRIORITY → "
                    f"GREEN FOR {green_time[road]} SECONDS"
                )

            else:

                st.write(
                    f"🚦 {road} → GREEN FOR "
                    f"{green_time[road]} SECONDS"
                )

        st.divider()

        # --------------------------------------------------
        # Highest priority
        # --------------------------------------------------

        first_road = priority[0]

        st.header(
            f"🚦 Next Green Signal: {first_road}"
        )

        st.write(
            f"Green time: **{green_time[first_road]} seconds**"
        )

        if first_road in ambulance_roads:

            st.success(
                "🚑 Ambulance detected — emergency "
                "priority activated!"
            )