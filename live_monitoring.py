import cv2
import csv
import os
import re
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

LOG_FILE = "data_log.csv"
MODEL_PATH = "best.pt"
YOUTUBE_URL = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
FALLBACK_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4"
ENABLE_FALLBACK_VIDEO = False
USE_BROWSER_COOKIES = True
COOKIE_BROWSER = "chrome"
COOKIE_PROFILES = ("Default", "Profile 2", "Profile 6", "Profile 19", "Profile 13")
COOKIES_FILE = "cookies.txt"
FORMAT_SELECTORS = (
    "best[ext=mp4][protocol^=http]/best[ext=mp4]/best",
    "best",
)
VEHICLE_CLASS_NAMES = {"vehicle", "car", "truck", "bus", "motorcycle", "motorbike", "van"}

# Start/end line koordinatalari shu o'lchamdagi kamera kadriga moslab berilgan.
# Rasmga qarab yo'lning yuqori-o'ng qismidan pastki-chap qismiga ketadigan oqim uchun sozlandi.
ROI_BASE_WIDTH = 1280
ROI_BASE_HEIGHT = 720
START_LINE = ((240, 372), (740, 290))
END_LINE = ((0, 648), (1140, 648))
SHOW_ROI_OVERLAY = True
SHOW_COUNTING_LINES = True
LINE_TOUCH_MARGIN = 16
END_TOUCH_MARGIN_MULTIPLIER = 6
END_PROGRESS_THRESHOLD = 0.50
END_PROGRESS_CROSS_THRESHOLD = 0.72
DETECTION_CONFIDENCE = 0.20
DETECTION_IMAGE_SIZE = 640
PROCESS_FRAME_SIZE = (1280, 720)
FRAME_PROCESS_INTERVAL = 1
MIN_VEHICLE_WIDTH_RATIO = 0.020
MIN_VEHICLE_HEIGHT_RATIO = 0.020
MIN_VEHICLE_AREA_RATIO = 0.00035
MIN_VEHICLE_ASPECT_RATIO = 0.55
DUPLICATE_EVENT_SECONDS = 1.0
DUPLICATE_EVENT_DISTANCE_RATIO = 0.03


def get_base_roi_polygon():
    # ROI start va end line'larning uchlaridan avtomatik quriladi.
    return [START_LINE[0], START_LINE[1], END_LINE[1], END_LINE[0]]


def get_scaled_roi(frame_shape):
    height, width = frame_shape[:2]
    scale_x = width / ROI_BASE_WIDTH
    scale_y = height / ROI_BASE_HEIGHT
    return np.array(
        [(int(x * scale_x), int(y * scale_y)) for x, y in get_base_roi_polygon()],
        dtype=np.int32,
    )


def get_scaled_line(line, frame_shape):
    height, width = frame_shape[:2]
    scale_x = width / ROI_BASE_WIDTH
    scale_y = height / ROI_BASE_HEIGHT
    return tuple((int(x * scale_x), int(y * scale_y)) for x, y in line)


def line_midpoint(line):
    return ((line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2)


def moving_from_start_to_end(prev_point, curr_point, start_line, end_line):
    if prev_point is None or curr_point is None:
        return False

    start_mid = line_midpoint(start_line)
    end_mid = line_midpoint(end_line)
    direction_x = end_mid[0] - start_mid[0]
    direction_y = end_mid[1] - start_mid[1]
    movement_x = curr_point[0] - prev_point[0]
    movement_y = curr_point[1] - prev_point[1]
    return (movement_x * direction_x + movement_y * direction_y) > 0


def progress_from_start_to_end(point, start_line, end_line):
    start_mid = line_midpoint(start_line)
    end_mid = line_midpoint(end_line)
    direction_x = end_mid[0] - start_mid[0]
    direction_y = end_mid[1] - start_mid[1]
    length_sq = direction_x * direction_x + direction_y * direction_y
    if length_sq == 0:
        return 0

    point_x = point[0] - start_mid[0]
    point_y = point[1] - start_mid[1]
    return (point_x * direction_x + point_y * direction_y) / length_sq


def is_point_inside_roi(point, roi_polygon):
    return cv2.pointPolygonTest(roi_polygon, point, False) >= 0


def looks_like_car_box(x1, y1, x2, y2, frame_shape):
    height, width = frame_shape[:2]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    box_area = box_width * box_height
    frame_area = width * height
    aspect_ratio = box_width / box_height

    return (
        box_width >= width * MIN_VEHICLE_WIDTH_RATIO
        and box_height >= height * MIN_VEHICLE_HEIGHT_RATIO
        and box_area >= frame_area * MIN_VEHICLE_AREA_RATIO
        and aspect_ratio >= MIN_VEHICLE_ASPECT_RATIO
    )


def is_duplicate_event(point, recent_events, now, frame_shape):
    height, width = frame_shape[:2]
    max_distance = width * DUPLICATE_EVENT_DISTANCE_RATIO
    fresh_events = []
    duplicate = False

    for event_time, event_point in recent_events:
        if now - event_time > DUPLICATE_EVENT_SECONDS:
            continue

        fresh_events.append((event_time, event_point))
        distance = ((point[0] - event_point[0]) ** 2 + (point[1] - event_point[1]) ** 2) ** 0.5
        if distance <= max_distance:
            duplicate = True

    return duplicate, fresh_events


def _orientation(a, b, c):
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def _on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def line_side(point, line_start, line_end):
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (
        line_end[1] - line_start[1]
    ) * (point[0] - line_start[0])


def point_to_segment_distance(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return ((px - nearest_x) ** 2 + (py - nearest_y) ** 2) ** 0.5


def crossed_segment(prev_point, curr_point, line_start, line_end):
    if prev_point is None or curr_point is None or prev_point == curr_point:
        return False

    o1 = _orientation(prev_point, curr_point, line_start)
    o2 = _orientation(prev_point, curr_point, line_end)
    o3 = _orientation(line_start, line_end, prev_point)
    o4 = _orientation(line_start, line_end, curr_point)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(prev_point, line_start, curr_point):
        return True
    if o2 == 0 and _on_segment(prev_point, line_end, curr_point):
        return True
    if o3 == 0 and _on_segment(line_start, prev_point, line_end):
        return True
    if o4 == 0 and _on_segment(line_start, curr_point, line_end):
        return True
    return False


def crossed_counting_line(prev_point, curr_point, line_start, line_end, margin=LINE_TOUCH_MARGIN):
    if prev_point is None or curr_point is None:
        return False

    if crossed_segment(prev_point, curr_point, line_start, line_end):
        return True

    prev_side = line_side(prev_point, line_start, line_end)
    curr_side = line_side(curr_point, line_start, line_end)
    side_changed = prev_side * curr_side < 0
    close_to_line = (
        point_to_segment_distance(prev_point, line_start, line_end) <= margin
        or point_to_segment_distance(curr_point, line_start, line_end) <= margin
    )
    return side_changed and close_to_line


def draw_roi_overlay(frame, roi_polygon):
    cv2.polylines(frame, [roi_polygon], True, (0, 0, 255), 3)
    cv2.putText(
        frame,
        "ROI",
        tuple(roi_polygon[0]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )


def draw_counting_lines(frame, start_line, end_line):
    cv2.line(frame, start_line[0], start_line[1], (0, 255, 255), 3)
    cv2.line(frame, end_line[0], end_line[1], (0, 255, 0), 3)
    cv2.putText(frame, "START", start_line[1], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "END", end_line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def get_vehicle_class_ids(model):
    names = getattr(model, "names", {}) or {}
    vehicle_ids = [
        int(cls_id)
        for cls_id, name in names.items()
        if str(name).strip().lower() in VEHICLE_CLASS_NAMES
    ]
    return vehicle_ids or None


def class_label(model, cls_id):
    names = getattr(model, "names", {}) or {}
    return str(names.get(cls_id, f"class_{cls_id}"))


def normalize_youtube_url(youtube_url):
    parsed = urlparse(youtube_url)
    query = parse_qs(parsed.query)

    if "v" in query and query["v"]:
        return f"https://www.youtube.com/watch?v={query['v'][0]}"

    match = re.search(r"(?:youtube\.com/(?:live|shorts)/|youtu\.be/)([^?&/]+)", youtube_url)
    if match:
        return f"https://www.youtube.com/watch?v={match.group(1)}"

    return youtube_url


def select_stream_url(info):
    if not isinstance(info, dict):
        return None

    if info.get("url"):
        return info["url"]

    formats = info.get("formats") or []
    stream_formats = [fmt for fmt in formats if fmt.get("url")]
    if not stream_formats:
        return None

    def format_score(fmt):
        protocol = str(fmt.get("protocol") or "")
        height = fmt.get("height") or 0
        fps = fmt.get("fps") or 0
        has_video = fmt.get("vcodec") not in (None, "none")
        has_audio = fmt.get("acodec") not in (None, "none")
        return (
            1 if "m3u8" in protocol else 0,
            1 if has_video else 0,
            1 if has_audio else 0,
            height,
            fps,
        )

    return max(stream_formats, key=format_score)["url"]


def resolve_youtube_stream(youtube_url):
    if YoutubeDL is None:
        print("⚠️ yt_dlp kutubxonasi o'rnatilmagan. `pip install yt-dlp` bilan o'rnating.")
        return None

    youtube_url = normalize_youtube_url(youtube_url)
    print(f"🔗 Normalized YouTube URL: {youtube_url}")

    ydl_attempts = []

    if USE_BROWSER_COOKIES:
        for profile in COOKIE_PROFILES:
            ydl_attempts.append(
                {
                    "name": f"{COOKIE_BROWSER} cookie / {profile}",
                    "options": {"cookiesfrombrowser": (COOKIE_BROWSER, profile)},
                }
            )

    if os.path.exists(COOKIES_FILE):
        ydl_attempts.append(
            {
                "name": f"{COOKIES_FILE} fayli",
                "options": {"cookiefile": COOKIES_FILE},
            }
        )

    ydl_attempts.append({"name": "cookiesiz", "options": {}})

    base_ydl_options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    for attempt in ydl_attempts:
        for format_selector in FORMAT_SELECTORS:
            ydl_options = {
                **base_ydl_options,
                **attempt["options"],
                "format": format_selector,
            }
            try:
                with YoutubeDL(ydl_options) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                stream_url = select_stream_url(info)
                if stream_url:
                    print(f"✅ YouTube direct stream URL topildi ({attempt['name']}).")
                    return stream_url
            except Exception as e:
                print(f"⚠️ yt-dlp orqali URL olish bo'lmadi ({attempt['name']}): {e}")

    return None


def init_csv():
    # Agar fayl yo'q bo'lsa yoki eski formatda bo'lsa, 2 ustunli standart formatda yaratamiz
    needs_init = True
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, mode='r', encoding='utf-8') as f:
                header = f.readline().strip().split(',')
                if len(header) == 2 and header[1] == "Vehicles":
                    needs_init = False
        except Exception:
            pass
            
    if needs_init:
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Vehicles"])

def start_monitoring():
    init_csv()
    print(f"⏳ Model yuklanmoqda... ({MODEL_PATH} - transport vositalari rejimida)")
    model = YOLO(MODEL_PATH)
    vehicle_class_ids = get_vehicle_class_ids(model)
    print(f"🚗 Model classlari: {model.names}")
    if vehicle_class_ids is None:
        print("⚠️ Vehicle class nomi topilmadi. Modeldagi barcha classlar kuzatiladi.")
    else:
        print(f"✅ Kuzatiladigan class IDlar: {vehicle_class_ids}")
    
    youtube_url = normalize_youtube_url(YOUTUBE_URL)
    fallback_url = FALLBACK_URL
    
    print(f"📡 Video oqim ulanmoqda: {youtube_url}")
    
    is_youtube = False
    is_live_capture = False
    stream = None
    cap = None
    active_capture_source = None
    
    active_source = youtube_url
    active_stream_mode = True
    successful_options = {}

    youtube_stream_url = resolve_youtube_stream(youtube_url)
    
    # 1-usul: To'g'ridan-to'g'ri YouTube oqimining direct URLsi orqali ulanish (stream_mode=False)
    if youtube_stream_url:
        print("🔎 YouTube real oqim URLsi topildi va shu URL orqali ulanilmoqda...")
        try:
            print("🔄 Direct URL orqali OpenCV ulanmoqda...")
            cap = cv2.VideoCapture(youtube_stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("⏳ Oqim buferlanmoqda (5 soniya)...")
            time.sleep(5)
            if not cap.isOpened():
                raise RuntimeError("OpenCV direct streamni ocholmadi")
            is_live_capture = True
            active_source = youtube_stream_url
            active_capture_source = youtube_stream_url
        except Exception as e:
            print(f"⚠️ Direct URL orqali ulanib bo'lmadi: {e}")
            if cap is not None:
                cap.release()
                cap = None

    if not is_youtube and not is_live_capture:
        if not ENABLE_FALLBACK_VIDEO:
            print("❌ YouTube live oqimiga ulanib bo'lmadi. Fallback video o'chirilgan, noto'g'ri video ochilmaydi.")
            print(f"   Chrome profilelar tekshirildi: {', '.join(COOKIE_PROFILES)}")
            print(f"   Yoki shu papkaga {COOKIES_FILE} faylini qo'ying.")
            return

        print(f"⚠️ Barcha YouTube ulanish usullari cheklandi. Namunaviy videoga ulanmoqda: {fallback_url}")
        cap = cv2.VideoCapture(fallback_url)
        if not cap.isOpened():
            print("❌ Namunaviy videoga ham ulanib bo'lmadi. Internet aloqasini tekshiring.")
            return

    print("✅ Kuzatuv boshlandi. 'q' tugmasini bosing to'xtatish uchun.")

    interval_seconds = 10  # Har 10 soniyada bazaga yozib boramiz
    last_log_time = time.time()
    
    tracked_vehicles = {}

    interval_vehicles = 0
    start_vehicles = 0
    end_vehicles = 0
    total_vehicles = 0
    recent_start_events = []
    recent_end_events = []
    recent_total_events = []
    frame_count = 0

    consecutive_none_frames = 0
    while True:
        if is_youtube:
            frame = stream.read()
            if frame is None:
                consecutive_none_frames += 1
                if consecutive_none_frames > 100:  # Qayta ulanish
                    print("⚠️ Video oqim uzildi. Qayta ulanishga harakat qilinmoqda...")
                    try:
                        stream.stop()
                    except Exception:
                        pass
                    time.sleep(3)
                    stream = CamGear(source=active_source, stream_mode=active_stream_mode, logging=True, **successful_options).start()
                    time.sleep(5)  # Buffer new stream
                    consecutive_none_frames = 0
                time.sleep(0.1)
                continue
            consecutive_none_frames = 0
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_live_capture:
                    consecutive_none_frames += 1
                    if consecutive_none_frames > 100:
                        print("⚠️ Direct live oqim uzildi. Qayta ulanishga harakat qilinmoqda...")
                        cap.release()
                        time.sleep(3)
                        cap = cv2.VideoCapture(active_capture_source)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        consecutive_none_frames = 0
                    time.sleep(0.1)
                    continue

                # Namunaviy video tugasa, cheksiz aylanishi uchun boshiga qaytaramiz
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            consecutive_none_frames = 0

        frame_count += 1
        if frame_count % FRAME_PROCESS_INTERVAL != 0:
            continue

        process_frame = cv2.resize(frame, PROCESS_FRAME_SIZE)
        roi_polygon = get_scaled_roi(process_frame.shape)
        start_line = get_scaled_line(START_LINE, process_frame.shape)
        end_line = get_scaled_line(END_LINE, process_frame.shape)

        # TRACKING: best.pt ichidagi transport class(lar)i kuzatiladi.
        results = model.track(
            process_frame,
            classes=vehicle_class_ids,
            conf=DETECTION_CONFIDENCE,
            imgsz=DETECTION_IMAGE_SIZE,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        
        # Model ko'rgan transportlarni chizamiz, counting esa filterlardan keyin ishlaydi.
        annotated_frame = process_frame.copy()
        if SHOW_ROI_OVERLAY:
            draw_roi_overlay(annotated_frame, roi_polygon)
        if SHOW_COUNTING_LINES:
            draw_counting_lines(annotated_frame, start_line, end_line)

        current_zone_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                if boxes.id[i] is None:
                    continue
                obj_id = int(boxes.id[i].item())
                cls_id = int(boxes.cls[i].item())
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
                foot_point = ((x1 + x2) // 2, y2)
                center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                is_in_roi = (
                    is_point_inside_roi(foot_point, roi_polygon)
                    or is_point_inside_roi(center_point, roi_polygon)
                )
                
                if vehicle_class_ids is not None and cls_id not in vehicle_class_ids:
                    continue

                countable_vehicle = looks_like_car_box(x1, y1, x2, y2, process_frame.shape)
                box_color = (255, 0, 0) if countable_vehicle else (120, 120, 120)
                if is_in_roi:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(annotated_frame, foot_point, 4, (0, 255, 255), -1)
                    cv2.putText(
                        annotated_frame,
                        f"id:{obj_id} {class_label(model, cls_id)} {conf:.2f}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                if not countable_vehicle:
                    continue

                if obj_id not in tracked_vehicles:
                    tracked_vehicles[obj_id] = {
                        "prev_foot": None,
                        "started": False,
                        "start_counted": False,
                        "end_counted": False,
                        "wrong_direction": False,
                        "counted": False,
                        "last_progress": None,
                        "last_seen": time.time(),
                    }

                vehicle_state = tracked_vehicles[obj_id]
                prev_foot = vehicle_state["prev_foot"]
                now = time.time()

                if is_in_roi:
                    current_zone_ids.add(obj_id)

                crossed_start = crossed_counting_line(
                    prev_foot,
                    foot_point,
                    start_line[0],
                    start_line[1],
                )
                crossed_end = crossed_counting_line(
                    prev_foot,
                    foot_point,
                    end_line[0],
                    end_line[1],
                )
                valid_direction = moving_from_start_to_end(prev_foot, foot_point, start_line, end_line)
                end_progress = progress_from_start_to_end(foot_point, start_line, end_line)
                prev_progress = vehicle_state["last_progress"]
                progressing_towards_end = (
                    prev_progress is None
                    or end_progress >= prev_progress - 0.15
                    or valid_direction
                )
                close_to_end_line = (
                    point_to_segment_distance(foot_point, end_line[0], end_line[1])
                    <= LINE_TOUCH_MARGIN * END_TOUCH_MARGIN_MULTIPLIER
                )
                crossed_end_progress = (
                    prev_progress is not None
                    and prev_progress < END_PROGRESS_CROSS_THRESHOLD <= end_progress
                )
                reached_end_area = (
                    crossed_end
                    or close_to_end_line
                    or end_progress >= END_PROGRESS_THRESHOLD
                    or crossed_end_progress
                )
                near_end_after_start = (
                    vehicle_state["started"]
                    and progressing_towards_end
                    and reached_end_area
                )
                end_after_possible_id_switch = (
                    not vehicle_state["started"]
                    and progressing_towards_end
                    and end_progress >= END_PROGRESS_THRESHOLD
                    and reached_end_area
                )

                if crossed_end and not vehicle_state["started"]:
                    vehicle_state["wrong_direction"] = True

                if (
                    crossed_start
                    and valid_direction
                    and not vehicle_state["wrong_direction"]
                    and not vehicle_state["start_counted"]
                ):
                    is_duplicate, recent_start_events = is_duplicate_event(
                        foot_point,
                        recent_start_events,
                        now,
                        process_frame.shape,
                    )
                    vehicle_state["start_counted"] = True
                    vehicle_state["started"] = True
                    if not is_duplicate:
                        recent_start_events.append((now, foot_point))
                        start_vehicles += 1
                        print(f"➡️ Start line kesildi: ID {obj_id}. Start jami: {start_vehicles}")

                if (
                    (crossed_end or near_end_after_start or end_after_possible_id_switch)
                    and (vehicle_state["started"] or end_after_possible_id_switch)
                    and not vehicle_state["end_counted"]
                ):
                    vehicle_state["end_counted"] = True
                    is_duplicate, recent_end_events = is_duplicate_event(
                        foot_point,
                        recent_end_events,
                        now,
                        process_frame.shape,
                    )
                    if not is_duplicate:
                        recent_end_events.append((now, foot_point))
                        end_vehicles += 1
                        print(f"⬅️ End line kesildi: ID {obj_id}. End jami: {end_vehicles}")

                if (
                    (vehicle_state["started"] or end_after_possible_id_switch)
                    and not vehicle_state["counted"]
                    and (crossed_end or near_end_after_start or end_after_possible_id_switch)
                ):
                    vehicle_state["counted"] = True
                    is_duplicate, recent_total_events = is_duplicate_event(
                        foot_point,
                        recent_total_events,
                        now,
                        process_frame.shape,
                    )
                    if not is_duplicate:
                        recent_total_events.append((now, foot_point))
                        interval_vehicles += 1
                        total_vehicles += 1
                        print(f"✅ Startdan kirib enddan chiqdi: ID {obj_id}. Jami: {total_vehicles}")

                vehicle_state["prev_foot"] = foot_point
                vehicle_state["last_progress"] = end_progress
                vehicle_state["last_seen"] = time.time()

        now = time.time()
        tracked_vehicles = {
            vehicle_id: state
            for vehicle_id, state in tracked_vehicles.items()
            if now - state["last_seen"] <= 30
        }

        # Har 10 soniyada CSV faylga yozamiz
        current_time = time.time()
        if current_time - last_log_time >= interval_seconds:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp_str, interval_vehicles])
            
            print(f"[{timestamp_str}] Bazaga yozildi: +{interval_vehicles} transport")
            
            # Keyingi oraliq uchun nollab qo'yamiz (jami hisob qoladi)
            interval_vehicles = 0
            last_log_time = current_time

        # Ekranga ma'lumot chiqarish (HUD)
        
        # Statik ma'lumotlarni chiroyli formatda chiqarish (transport vositalari)
        cv2.rectangle(annotated_frame, (10, 10), (410, 150), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Start: {start_vehicles}", (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"End: {end_vehicles}", (25, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Zonada: {len(current_zone_ids)}", (25, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Start->End: {total_vehicles}", (25, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)
        
        # Headless-safe visual display
        try:
            cv2.imshow("ASSBI - Live Vehicle Counter (Tracking & Logging)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            # Headless rejimda bo'lsak, CPU 100% bo'lib ketmasligi va oqim to'lib ketmasligi uchun ozroq kutamiz (~30 FPS)
            time.sleep(0.03)

    if is_youtube and stream is not None:
        stream.stop()
    elif cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_monitoring()
