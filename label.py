"""Streamlit-based bounding-box labeling tool used in the paper.

Three experimental treatments are exposed via the `treatment` URL parameter:

* sigma_0 – Baseline: standard bounding boxes, no uncertainty cue (control).
* sigma_1 – Uncertainty visualization: per-detection aleatoric localization
            uncertainty is mapped onto the box border colour (blue → red).
* sigma_2 – Same as sigma_1 but boxes whose normalized uncertainty is below
            a threshold are dimmed; used internally for development.

In addition, passing `labeler=relabel` in the URL switches the tool into
the authors' relabeling pass: instead of the per-participant per-bin
sample, the full set of images on disk is shown so the authors can
re-annotate the experimental pool to produce the gold-standard ground
truth (Table S4 of the paper).

Authentication, image preloading, timing instrumentation, and the
write-back to a private results repo are unchanged from the version used
to run the experiment.
"""

import ast
import base64
import datetime
import os
import random
import time
from typing import Dict, List

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import streamlit_authenticator as stauth  # noqa: F401  (kept for hashing utility)
import yaml
from yaml.loader import SafeLoader

st.set_page_config(
    page_title="Bounding Box Labeling Tool",
    page_icon="📦",
    layout="wide",
)

# ---- paths -----------------------------------------------------------------
IMAGES_DIR = "data/ground_truth/images"
DETECTIONS_FILE = "data/detector_predictions.txt"
LOW_BIN = "data/low_uncertainty.txt"
MID_BIN = "data/mid_uncertainty.txt"
HIGH_BIN = "data/high_uncertainty.txt"

# ---- relabeling mode -------------------------------------------------------
# In addition to the three experiment treatments (sigma_0/1/2) the tool
# supports a single relabeling pass used by the authors to produce the
# gold-standard ground truth: when `?labeler=relabel` is passed in the
# URL, the tool shows ALL 97 images instead of the per-participant
# uncertainty-bin sample.
RELABEL_FLAG = "relabel"

DEV_MODE = False  # set True locally to expose developer controls


def uncertainty_to_color(uncertainty, min_uncertainty=0.0, max_uncertainty=5.0):
    """Map an uncertainty value to a colorblind-friendly blue→red colour."""
    if max_uncertainty > min_uncertainty:
        normalized = (uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)
    else:
        normalized = 0.0
    normalized = max(0.0, min(1.0, normalized))

    # Two-step gradient: (16,101,171) → (249,249,249) → (179,21,41)
    if normalized <= 0.5:
        t = normalized * 2
        r = int(16 + t * (249 - 16))
        g = int(101 + t * (249 - 101))
        b = int(171 + t * (249 - 171))
    else:
        t = (normalized - 0.5) * 2
        r = int(249 + t * (179 - 249))
        g = int(249 + t * (21 - 249))
        b = int(249 + t * (41 - 249))

    return f"rgb({r},{g},{b})", f"rgba({r},{g},{b},0.2)"


def push_file_to_github(content: str, path_in_repo: str, commit_msg: str, branch: str = "main"):
    """Commit `content` to a private results repo via the GitHub Contents API."""
    token = st.secrets["GITHUB_TOKEN"]
    repo_owner = "mos-ks"
    repo_name = "labeling_results"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path_in_repo}?ref={branch}"
    r = requests.get(url, headers=headers)
    sha = r.json()["sha"] if r.status_code == 200 else None

    payload = {
        "message": commit_msg,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    put_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path_in_repo}"
    resp = requests.put(put_url, headers=headers, json=payload)
    return resp.status_code, resp.json()


class BoundingBoxLabeler:
    # ------------------------------------------------------------------ config
    def sidebar_config(self):
        if DEV_MODE:
            st.sidebar.header("🔧 Configuration")
            images_dir = st.sidebar.text_input("Images Directory", value=IMAGES_DIR)
            detections_file = st.sidebar.text_input("Detections File", value=DETECTIONS_FILE)
        else:
            images_dir = IMAGES_DIR
            detections_file = DETECTIONS_FILE

        user = st.session_state.current_user
        treatment = st.session_state.treatment.lower()
        output_file_detections = f"updated_detections_{user}_{treatment}.txt"
        output_file_time = f"timing_data_{user}_{treatment}.txt"
        return images_dir, detections_file, output_file_detections, output_file_time

    # -------------------------------------------------------------- detections
    def load_detections(self, path: str, valid_image_names: List[str] = None) -> Dict[str, List[Dict]]:
        """Load per-image detections, optionally restricting to `valid_image_names`."""
        detections: Dict[str, List[Dict]] = {}
        if not os.path.exists(path):
            st.warning(f"Detections file not found: {path}")
            return detections

        # Normalize the filter set so any of "id", "id.png", "id.jpg" matches.
        valid_set = set()
        if valid_image_names:
            for name in valid_image_names:
                valid_set.add(name)
                base = os.path.splitext(name)[0]
                valid_set.add(base)
                valid_set.add(base + ".jpg")

        try:
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            for line_num, line in enumerate(lines, 1):
                try:
                    data = ast.literal_eval(line)
                except Exception as e:
                    print(f"Line {line_num} skipped: {e}")
                    continue
                if not (isinstance(data, dict) and "image_name" in data and "bbox" in data):
                    continue
                if valid_image_names is not None and data["image_name"] not in valid_set:
                    continue
                detections.setdefault(data["image_name"], []).append(data)
        except Exception as e:
            st.error(f"Error loading detections: {e}")
        return detections

    # ------------------------------------------------------------- uncertainty
    def render_uncertainty_controls(self, current_detections):
        """Toggle/configure the uncertainty visualisation for the current treatment."""
        if "show_uncertainty" not in st.session_state:
            st.session_state.show_uncertainty = False

        # In sigma_1/sigma_2 the uncertainty layer is on by default; flip it
        # exactly once on first render to match the experimental conditions.
        if not st.session_state.show_uncertainty and st.session_state.treatment in ("sigma_1", "sigma_2"):
            st.session_state.show_uncertainty = True
            st.session_state.refresh_counter = st.session_state.get("refresh_counter", 0) + 1
            st.rerun()

        if st.session_state.show_uncertainty:
            if "uncertainty_type" not in st.session_state:
                st.session_state.uncertainty_type = "albox"
            options = ["confidence", "entropy", "mcbox", "albox"]
            current_index = options.index(st.session_state.uncertainty_type)

            if DEV_MODE:
                uncertainty_type = st.radio(
                    "Uncertainty Type:",
                    options,
                    index=current_index,
                    format_func=lambda x: {
                        "confidence": "🎯 Confidence Score",
                        "entropy": "📊 Entropy (Classification)",
                        "mcbox": "📦 MC Box (Monte Carlo)",
                        "albox": "🎲 AL Box (Aleatoric)",
                    }[x],
                    horizontal=True,
                )
            else:
                uncertainty_type = "albox"

            if st.session_state.uncertainty_type != uncertainty_type:
                st.session_state.uncertainty_type = uncertainty_type
                st.rerun()
        else:
            uncertainty_type = "confidence"
            st.session_state.uncertainty_type = uncertainty_type

        if st.session_state.show_uncertainty:
            min_unc, max_unc = self.get_uncertainty_stats(current_detections, uncertainty_type)[:2]
        else:
            min_unc, max_unc = None, None
        return st.session_state.show_uncertainty, uncertainty_type, min_unc, max_unc

    # --------------------------------------------------------------- zoom UI
    def render_zoom_controls(self, scale, img_width, img_height):
        """Render the three image-button zoom controls and return the new scale."""
        # Each button gets a CSS rule that targets the n-th element-container,
        # restyles it as a 30×30 image button, and pulls it up to stack the
        # three controls vertically.
        for key, image_path, container_idx, top_offset, scale_change in [
            ("zoomin_btn",    "./zoomin.png",    3, -17, lambda s: min(s * 1.5, 5.0)),
            ("zoomout_btn",   "./zoomout.png",   5, -34, lambda s: max(s / 1.5, 0.1)),
            ("zoomreset_btn", "./zoomreset.png", 7, -51, lambda s: 1.0),
        ]:
            with open(image_path, "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode()
            st.markdown(
                f"""
                <style>
                .element-container:nth-of-type({container_idx}) button {{
                    background-image: url('data:image/png;base64,{image_b64}') !important;
                    background-size: 30px 30px !important;
                    background-repeat: no-repeat !important;
                    background-position: center !important;
                    position: relative;
                    top: {top_offset}px;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            if st.button("", key=key):
                new_scale = scale_change(scale)
                if new_scale != scale:
                    st.session_state.zoom_scale = new_scale
                    st.rerun()
        return scale

    # ----------------------------------------------------------- timer state
    def initialize_timer(self):
        for key in (
            "image_start_times", "editing_start_times", "editing_last_action_times",
            "image_times", "editing_times", "has_started_editing", "initial_box_counts",
        ):
            if key not in st.session_state:
                st.session_state[key] = {}

    def start_image_timer(self, img_name):
        if img_name not in st.session_state.image_start_times:
            st.session_state.image_start_times[img_name] = time.time()
            st.session_state.has_started_editing[img_name] = False

    def start_editing_timer(self, img_name):
        if not st.session_state.has_started_editing.get(img_name, False):
            st.session_state.editing_start_times[img_name] = time.time()
            st.session_state.has_started_editing[img_name] = True
        st.session_state.editing_last_action_times[img_name] = time.time()

    def check_canvas_interaction(self, img_name, canvas_result, initial_detection_count):
        """Start the edit timer once the box count diverges from the initial state."""
        if canvas_result.json_data is None:
            return False
        current_count = len(canvas_result.json_data.get("objects", []))
        st.session_state.initial_box_counts.setdefault(img_name, initial_detection_count)
        if current_count != st.session_state.initial_box_counts[img_name]:
            self.start_editing_timer(img_name)
            return True
        return False

    def stop_image_timer(self, img_name):
        if img_name in st.session_state.image_start_times:
            start = st.session_state.image_start_times[img_name]
            st.session_state.image_times[img_name] = time.time() - start

    def stop_editing_timer(self, img_name):
        if (img_name in st.session_state.editing_start_times
                and img_name in st.session_state.editing_last_action_times):
            start = st.session_state.editing_start_times[img_name]
            end = st.session_state.editing_last_action_times[img_name]
            st.session_state.editing_times[img_name] = end - start

    def render_timer_display(self, current_img):
        st.markdown("### ⏱️ Current Session Times")
        col1, col2 = st.columns(2)
        with col1:
            if current_img in st.session_state.image_start_times:
                elapsed = time.time() - st.session_state.image_start_times[current_img]
                st.metric("Time on Image", f"{elapsed:.1f}s")
            else:
                st.metric("Time on Image", "0.0s")
        with col2:
            if (current_img in st.session_state.editing_start_times
                    and current_img in st.session_state.editing_last_action_times):
                start = st.session_state.editing_start_times[current_img]
                end = st.session_state.editing_last_action_times[current_img]
                st.metric("Editing Time", f"{end - start:.1f}s")
            else:
                st.metric("Editing Time", "0.0s")

    # --------------------------------------------------------------- saving
    def save_detections(self, detections: Dict[str, List[Dict]], path_detections: str, path_time: str) -> bool:
        """Push timing and detection results to the private results repo."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            timing_lines = ["# Timing Data", "# Format: image_name: time_per_image(s), editing_time(s)", ""]
            all_images = set(st.session_state.image_times) | set(st.session_state.editing_times)
            for img_name in sorted(all_images):
                t_img = st.session_state.image_times.get(img_name, 0.0)
                t_edit = st.session_state.editing_times.get(img_name, 0.0)
                timing_lines.append(f"{img_name}: {t_img:.2f}, {t_edit:.2f}")
            timing_content = "\n".join(timing_lines)

            timing_repo_path = os.path.join(st.session_state.treatment, path_time.split("/")[-1])
            timing_status, timing_response = push_file_to_github(
                timing_content, timing_repo_path,
                f"Update timing data from Streamlit app - {timestamp}",
            )

            det_lines = []
            for det_list in detections.values():
                for det in det_list:
                    if isinstance(det, dict):
                        det_lines.append(str(det))
            detections_content = "\n".join(det_lines)

            detections_repo_path = os.path.join(st.session_state.treatment, path_detections.split("/")[-1])
            detections_status, detections_response = push_file_to_github(
                detections_content, detections_repo_path,
                f"Update detections from Streamlit app - {timestamp}",
            )

            timing_ok = timing_status in (200, 201)
            detections_ok = detections_status in (200, 201)
            if timing_ok and detections_ok:
                st.success("Both timing data and detections pushed to GitHub successfully!")
                return True
            if timing_ok:
                st.warning(f"Timing data pushed successfully, but detections failed: {detections_response}")
            elif detections_ok:
                st.warning(f"Detections pushed successfully, but timing data failed: {timing_response}")
            else:
                st.error(f"Both pushes failed - Timing: {timing_response}, Detections: {detections_response}")
            return False
        except Exception as e:
            st.error(f"Error pushing data to GitHub: {e}")
            return False

    # ----------------------------------------------------------- image utils
    def get_image_list(self, images_dir: str) -> List[str]:
        if not os.path.exists(images_dir):
            return []
        supported = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
        try:
            return sorted(f for f in os.listdir(images_dir) if f.lower().endswith(supported))
        except Exception as e:
            st.error(f"Error reading images directory: {e}")
            return []

    def normalize_image_key(self, filename: str) -> str:
        # Detection records are keyed on `<id>.jpg` regardless of the on-disk
        # extension, because that is how the prediction file was written.
        return os.path.splitext(filename)[0] + ".jpg"

    def get_uncertainty_stats(self, detections, uncertainty_type):
        if not detections:
            return 0.0, 1.0, 0.5
        values = []
        for det in detections:
            if uncertainty_type == "confidence":
                values.append(1.0 - det.get("det_score", 1.0))  # invert: high confidence = low uncertainty
            elif uncertainty_type == "entropy":
                values.append(det.get("entropy", 0.0))
            elif uncertainty_type == "mcbox":
                v = det.get("uncalib_mcbox", [0, 0, 0, 0])
                values.append(sum(v) / len(v) if v else 0.0)
            elif uncertainty_type == "albox":
                v = det.get("rel_iso_perclscoo_albox", [0, 0, 0, 0])
                values.append(sum(v) / len(v) if v else 0.0)
            else:
                values.append(0.0)
        return min(values), max(values), sorted(values)[len(values) // 2]

    # ------------------------------------------------------ canvas conversions
    def bbox_to_canvas(self, detections, scale=1.0, show_uncertainty=False,
                       uncertainty_type="entropy", min_uncertainty=None,
                       max_uncertainty=None, score_threshold=0.0):
        """Render detections as fabric.js objects, colour-coded by uncertainty."""
        objects = []
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            y1, x1, y2, x2 = bbox[:4]
            x1_s, y1_s = float(x1 * scale), float(y1 * scale)
            w_s, h_s = float((x2 - x1) * scale), float((y2 - y1) * scale)
            if w_s <= 0 or h_s <= 0:
                continue

            if uncertainty_type == "confidence":
                uncertainty = 1.0 - det.get("det_score", 1.0)
            elif uncertainty_type == "entropy":
                uncertainty = det.get("entropy", 0.0)
            elif uncertainty_type == "mcbox":
                v = det.get("uncalib_mcbox", [0, 0, 0, 0])
                uncertainty = sum(v) / len(v) if v else 0.0
            elif uncertainty_type == "albox":
                v = det.get("rel_iso_perclscoo_albox", [0, 0, 0, 0])
                uncertainty = sum(v) / len(v) if v else 0.0
            else:
                uncertainty = 0.0

            is_filtered = show_uncertainty and uncertainty < score_threshold * (max_uncertainty or 0)
            if is_filtered and st.session_state.treatment == "sigma_2":
                stroke_color, fill_color, stroke_width = "#1065AB", "rgba(16,101,171,0.1)", 1
            elif show_uncertainty:
                stroke_color, fill_color = uncertainty_to_color(uncertainty, min_uncertainty, max_uncertainty)
                stroke_width = 2
            else:
                stroke_color, fill_color, stroke_width = "#1065AB", "rgba(16,101,171,0.1)", 1

            objects.append({
                "type": "rect",
                "left": x1_s, "top": y1_s, "width": w_s, "height": h_s,
                "stroke": stroke_color, "strokeWidth": stroke_width, "fill": fill_color,
                "selectable": True, "evented": True,
            })
        return {"objects": objects}

    def canvas_to_bbox(self, canvas_data, scale=1.0, img_w=None, img_h=None,
                       image_key=None, original_detections=None):
        """Convert canvas rectangles back to detection dicts, preserving any
        original metadata (logits, uncertainty, etc.) by matching on bbox
        coordinates."""
        detections = []
        if not canvas_data or "objects" not in canvas_data:
            return detections

        # Index original detections by rounded bbox so we can re-attach
        # metadata after the user moves/resizes a box.
        bbox_index = {}
        if original_detections:
            for det in original_detections:
                bb = det.get("bbox", [0, 0, 0, 0])
                if len(bb) >= 4:
                    y1, x1, y2, x2 = bb[:4]
                    bbox_index[(round(x1), round(y1), round(x2), round(y2))] = det

        MIN_BOX_SIZE = 10  # pixels; smaller boxes are treated as misclicks

        for obj in canvas_data["objects"]:
            if obj.get("type") != "rect":
                continue
            left, top = float(obj.get("left", 0)), float(obj.get("top", 0))
            ow, oh = float(obj.get("width", 0)), float(obj.get("height", 0))
            sx, sy = float(obj.get("scaleX", 1)), float(obj.get("scaleY", 1))
            rendered_w, rendered_h = ow * sx, oh * sy
            if rendered_w < MIN_BOX_SIZE or rendered_h < MIN_BOX_SIZE:
                continue

            x1, y1 = left / scale, top / scale
            x2, y2 = (left + rendered_w) / scale, (top + rendered_h) / scale
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            if img_w is not None and img_h is not None:
                x1, x2 = max(0, min(x1, img_w)), max(0, min(x2, img_w))
                y1, y2 = max(0, min(y1, img_h)), max(0, min(y2, img_h))
            if x2 <= x1 or y2 <= y1:
                continue

            key = (round(x1), round(y1), round(x2), round(y2))
            original_det = bbox_index.get(key)

            # If no exact match, fall back to nearest-centre within 10px so a
            # small drag still re-attaches the right metadata.
            if original_det is None and original_detections:
                best, best_d = None, float("inf")
                for det in original_detections:
                    bb = det.get("bbox", [0, 0, 0, 0])
                    if len(bb) < 4:
                        continue
                    oy1, ox1, oy2, ox2 = bb[:4]
                    dx = ((ox1 + ox2) / 2) - ((x1 + x2) / 2)
                    dy = ((oy1 + oy2) / 2) - ((y1 + y2) / 2)
                    d = (dx * dx + dy * dy) ** 0.5
                    if d < best_d and d <= 10:
                        best, best_d = det, d
                original_det = best

            if original_det is not None:
                new_det = original_det.copy()
                new_det["bbox"] = [y1, x1, y2, x2]
                new_det["image_name"] = image_key
            else:
                new_det = {
                    "image_name": image_key,
                    "score_thresh": 0.1,
                    "top_5scores": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "det_score": 1.0,
                    "class": 1,
                    "logits": [0.0] * 7,
                    "entropy": 0.0,
                    "probab": [1.0] + [0.0] * 6,
                    "bbox": [y1, x1, y2, x2],
                }
            detections.append(new_det)
        return detections

    # ----------------------------------------------------------- score filter
    def render_score_filter(self, detections, show_uncertainty=False):
        if DEV_MODE:
            st.markdown("### 🎯 Score Filtering")
            if not detections:
                st.info("No detections to filter")
                return detections, 0.0

        unc_type_key = {
            "confidence": "det_score",
            "entropy": "entropy",
            "mcbox": "uncalib_mcbox",
            "albox": "rel_iso_perclscoo_albox",
        }[st.session_state.uncertainty_type]

        if st.session_state.uncertainty_type in ("confidence", "entropy"):
            scores = [det.get(unc_type_key, 0) for det in detections]
        else:
            scores = [sum(det.get(unc_type_key, [0, 0, 0, 0])) / 4 for det in detections]

        if not scores:
            return detections, 0.0
        min_score, max_score = min(scores), max(scores)
        if min_score == max_score:
            if DEV_MODE:
                st.info(f"All detections have same score: {min_score:.3f}")
            return detections, float(min_score)

        if "score_threshold" not in st.session_state:
            st.session_state.score_threshold = 0.0

        if DEV_MODE:
            new_threshold = st.slider(
                "Minimum Confidence Score", 0.0, 1.0,
                value=st.session_state.score_threshold, step=0.01, format="%.3f",
                key="score_slider",
            )
            if new_threshold != st.session_state.score_threshold:
                st.session_state.score_threshold = new_threshold
                st.rerun()
        else:
            st.session_state.score_threshold = 0.8 if st.session_state.treatment == "sigma_2" else st.session_state.score_threshold

        threshold = st.session_state.score_threshold
        if st.session_state.treatment.lower() == "sigma_2":
            threshold *= max_score

        if show_uncertainty:
            return detections, threshold
        return [d for d in detections if d.get(unc_type_key, 0) >= st.session_state.score_threshold], st.session_state.score_threshold

    # ----------------------------------------------------------- delete-mode
    def delete_box_at_position(self, click_x, click_y, detections, scale):
        for i, det in enumerate(detections):
            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            y1, x1, y2, x2 = bbox[:4]
            if (x1 * scale) <= click_x <= (x2 * scale) and (y1 * scale) <= click_y <= (y2 * scale):
                detections.pop(i)
                return True, i
        return False, -1

    # ------------------------------------------------------------ colorbar UI
    def show_uncertainty_colorbar(self, uncertainty_type, current_detections):
        min_unc, max_unc, _ = self.get_uncertainty_stats(current_detections, uncertainty_type)
        if uncertainty_type == "confidence":
            gradient = "linear-gradient(to bottom, rgb(16,101,171) 0%, rgb(249,249,249) 50%, rgb(179,21,41) 100%)"
            top_label, bottom_label = "Low", "High"
        else:
            gradient = "linear-gradient(to bottom, rgb(179,21,41) 0%, rgb(249,249,249) 50%, rgb(16,101,171) 100%)"
            top_label, bottom_label = "High", "Low"
        st.markdown(
            f"""
            <div style="position: relative; display: flex; flex-direction: column;
                        align-items: center; justify-content: flex-start;
                        height: 100%; padding-top: 10px; width: 100%;">
                <div style="margin-bottom: 8px; font-size: 11px; font-weight: bold;">{top_label}</div>
                <div style="width: 27px; height: 250px; background: {gradient};
                            border: 1px solid #ccc; border-radius: 5px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></div>
                <div style="margin-top: 8px; font-size: 11px; font-weight: bold;">{bottom_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return min_unc, max_unc

    # ------------------------------------------------------------ canvas save
    def save_canvas_state(self, current_img, img_key, session_key, scale, img_width, img_height):
        if st.session_state.get("previous_canvas_data") is None:
            return False
        self.start_editing_timer(current_img)
        new_dets = self.canvas_to_bbox(
            st.session_state.previous_canvas_data, scale, img_width, img_height,
            img_key, st.session_state[session_key],
        )
        old_bb = [tuple(d.get("bbox", [])) for d in st.session_state[session_key]]
        new_bb = [tuple(d.get("bbox", [])) for d in new_dets]
        if old_bb != new_bb:
            st.session_state[session_key] = new_dets
            return True
        return False

    # ------------------------------------------------- image sampling per run
    def select_images_for_run(self, all_image_files: List[str]) -> List[str]:
        """Decide which images this session should show.

        Default (experiment treatments sigma_0/1/2): draw 5 random images
        from the mid-uncertainty bin and 10 from the high-uncertainty bin
        (15 per participant).

        `labeler_suffix == RELABEL_FLAG`: show all images on disk so the
        author can re-annotate the full 97-image experimental pool used to
        produce the gold standard reported in Table S4 of the paper.
        """
        def _load(path: str) -> List[str]:
            try:
                with open(path) as f:
                    return [ln.strip() for ln in f if ln.strip()]
            except Exception:
                return []

        if st.session_state.get("labeler_suffix") == RELABEL_FLAG:
            return sorted(os.path.splitext(f)[0] for f in all_image_files)

        mid, high = _load(MID_BIN), _load(HIGH_BIN)
        selected = random.sample(mid, min(5, len(mid))) + random.sample(high, min(10, len(high)))
        random.shuffle(selected)
        return selected

    # ----------------------------------------------------------------- run
    def run(self):
        left_spacer, main_content, right_spacer = st.columns([1, 80, 1])
        with main_content:
            images_dir, detections_file, output_file_detections, output_file_time = self.sidebar_config()

            if "image_names" not in st.session_state:
                image_names = self.select_images_for_run(self.get_image_list(images_dir))
                if not image_names:
                    st.error("No images selected for this session!")
                    st.stop()
                st.session_state.image_names = image_names

            if "all_detections" not in st.session_state:
                st.session_state.all_detections = (
                    self.load_detections(detections_file, st.session_state.image_names)
                    if os.path.exists(detections_file) else {}
                )

            # Preload all images for the run so navigation between them is
            # instant (the canvas can stutter otherwise on slow connections).
            if "preloaded_images" not in st.session_state:
                st.session_state.preloaded_images = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, img_name in enumerate(st.session_state.image_names):
                    img_path = os.path.join(images_dir, img_name + ".png")
                    try:
                        if os.path.exists(img_path):
                            status_text.text(f"Loading image {i + 1}/{len(st.session_state.image_names)}: {img_name}")
                            img = Image.open(img_path)
                            img.load()
                            st.session_state.preloaded_images[img_name] = img
                        else:
                            st.warning(f"Image not found: {img_path}")
                    except Exception as e:
                        st.error(f"Failed to load {img_name}: {e}")
                        continue
                    progress_bar.progress((i + 1) / len(st.session_state.image_names))
                progress_bar.empty()
                status_text.empty()
                if DEV_MODE:
                    st.success(f"✅ Loaded {len(st.session_state.preloaded_images)} images!")
                st.rerun()

            if "img_index" not in st.session_state:
                st.session_state.img_index = 0
            current_idx = min(st.session_state.img_index, len(st.session_state.image_names) - 1)
            current_img = st.session_state.image_names[current_idx]
            img_key = self.normalize_image_key(current_img)

            if current_img not in st.session_state.preloaded_images:
                st.error(f"Image {current_img} not found in preloaded images!")
                st.stop()
            img = st.session_state.preloaded_images[current_img].copy()

            if "zoom_scale" not in st.session_state:
                st.session_state.zoom_scale = 1.0

            session_key = f"detections_{img_key}"
            if session_key not in st.session_state:
                st.session_state[session_key] = st.session_state.all_detections.get(img_key, []).copy()

            self.initialize_timer()
            self.start_image_timer(current_img)

            scale = st.session_state.zoom_scale
            display_w, display_h = int(img.width * scale), int(img.height * scale)
            img_display = img.resize((display_w, display_h), Image.Resampling.LANCZOS)
            img_display.load()

            st.markdown(
                f"""
                <style>
                .main .block-container {{
                    max-width: {int(1600 * st.session_state.zoom_scale)}px;
                    margin: 0 auto;
                    padding: 2rem;
                }}
                .stButton {{ display: flex; justify-content: center; }}
                .row-widget.stColumns {{
                    display: flex; justify-content: center; align-items: flex-start;
                    gap: 10px; margin: 0 auto;
                }}
                .metric-container {{ display: flex; justify-content: center; gap: 20px; margin: 20px 0; }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            # --- mode selection (Draw / Edit / Delete) -----------------------
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(
                    """
                    <style>
                    .stRadio > div { padding: 0; margin: -3rem 0; }
                    .stRadio > div > label { margin-bottom: 0; padding: 0.25rem; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                drawing_mode = st.radio(
                    " ",
                    ["rect", "transform", "point"],
                    format_func=lambda x: {"rect": "🖊️ Draw", "transform": "✏️ Edit", "point": "🗑️ Delete"}[x],
                    horizontal=True,
                )
            with col2:
                blurb = {
                    "rect": "🖊️ <strong>Draw Mode</strong>: Click and drag on image to create new boxes",
                    "transform": "✏️ <strong>Edit Mode</strong>: Click boxes to select, drag to move, drag corners to resize",
                    "point": "🗑️ <strong>Delete Mode</strong>: Click on any box to delete it",
                }[drawing_mode]
                st.markdown(f'<div style="width: fit-content;">{blurb}</div>', unsafe_allow_html=True)

            current_detections = st.session_state[session_key]
            show_uncertainty = st.session_state.get("show_uncertainty", False)
            uncertainty_type = st.session_state.get("uncertainty_type", "confidence")
            if st.session_state.treatment.lower() == "sigma_2":
                st.session_state.score_threshold = 0.7

            if show_uncertainty and current_detections:
                min_uncertainty, max_uncertainty, _ = self.get_uncertainty_stats(current_detections, uncertainty_type)
            else:
                min_uncertainty, max_uncertainty = None, None

            canvas_data = self.bbox_to_canvas(
                current_detections, scale, show_uncertainty, uncertainty_type,
                min_uncertainty, max_uncertainty,
                st.session_state.get("score_threshold", 0.0),
            )

            # Detect mode change so we can persist any in-progress edits before
            # rebuilding the canvas under a new fabric.js drawing mode.
            if "last_drawing_mode" not in st.session_state:
                st.session_state.last_drawing_mode = drawing_mode
            if "previous_canvas_data" not in st.session_state:
                st.session_state.previous_canvas_data = None
            if st.session_state.last_drawing_mode != drawing_mode and st.session_state.previous_canvas_data is not None:
                self.save_canvas_state(current_img, img_key, session_key, scale, img.width, img.height)
            st.session_state.last_drawing_mode = drawing_mode

            canvas_key = f"canvas_{current_idx}_{drawing_mode}_{int(scale * 100)}"

            show_uncertainty, uncertainty_type, min_uncertainty, max_uncertainty = self.render_uncertainty_controls(current_detections)
            current_detections, _ = self.render_score_filter(current_detections, show_uncertainty)

            col_left1, col_left2, col_canvas = st.columns([1, 2, 40])

            with col_left1:
                st.markdown("<hr style='margin: 2.5px 0;'>", unsafe_allow_html=True)
                st.session_state.zoom_scale = self.render_zoom_controls(
                    st.session_state.zoom_scale, img.width, img.height,
                )
                st.markdown("<hr style='margin: -55px 0;'>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <style>
                    .element-container:nth-of-type(10) button { position: relative; top: -73px; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("🔄"):
                    self.start_editing_timer(current_img)
                    st.session_state[session_key] = st.session_state.all_detections.get(img_key, []).copy()
                    st.session_state.previous_canvas_data = None
                    st.rerun()
                st.markdown("<hr style='margin: -75px 0;'>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <style>
                    .element-container:nth-of-type(13) button { position: relative; top: -91px; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("⚠️"):
                    st.rerun()

            with col_left2:
                if show_uncertainty:
                    self.show_uncertainty_colorbar(uncertainty_type, current_detections)

            with col_canvas:
                canvas_result = st_canvas(
                    stroke_width=2,
                    stroke_color="#1065AB",
                    fill_color="rgba(16,101,171,0.1)",
                    background_image=img_display,
                    update_streamlit=True,
                    height=display_h,
                    width=display_w,
                    drawing_mode=drawing_mode,
                    key=canvas_key,
                    initial_drawing=canvas_data,
                    display_toolbar=False,
                )

                # Throttle expensive canvas → bbox conversions: store the
                # raw fabric.js state on every tick but only normalise it
                # every ~2.5s.
                if "last_canvas_process_time" not in st.session_state:
                    st.session_state.last_canvas_process_time = 0
                now = time.time()
                if canvas_result.json_data is not None:
                    st.session_state.previous_canvas_data = canvas_result.json_data
                    if (now - st.session_state.last_canvas_process_time) > 2.5:
                        st.session_state.last_canvas_process_time = now

                # Delete-mode adds an invisible point on every click; if it
                # lands inside a box, remove that detection and re-render.
                if drawing_mode == "point" and canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    if objects and objects[-1].get("type") == "circle":
                        click_obj = objects[-1]
                        click_x, click_y = click_obj.get("left", 0), click_obj.get("top", 0)
                        current_detections = st.session_state[session_key].copy()
                        deleted, _ = self.delete_box_at_position(click_x, click_y, current_detections, scale)
                        if deleted:
                            self.start_editing_timer(current_img)
                            st.session_state[session_key] = current_detections
                            st.session_state.previous_canvas_data = None
                            st.session_state.pop("delete_miss_count", None)
                            st.rerun()
                        else:
                            st.session_state.delete_miss_count = st.session_state.get("delete_miss_count", 0) + 1
                            if DEV_MODE and st.session_state.delete_miss_count >= 2:
                                st.info("💡 Tip: Click directly on a box to delete it")

            if canvas_result.json_data is not None:
                st.session_state.previous_canvas_data = canvas_result.json_data

            self.check_canvas_interaction(current_img, canvas_result, len(st.session_state[session_key]))

            # --- navigation -----------------------------------------------------
            _, col_next = st.columns(2)
            with col_next:
                is_last_image = (current_idx == len(st.session_state.image_names) - 1)
                if not is_last_image:
                    if st.button("Next →", disabled=is_last_image):
                        self.stop_image_timer(current_img)
                        self.stop_editing_timer(current_img)
                        self.save_canvas_state(current_img, img_key, session_key, scale, img.width, img.height)
                        save_data = {}
                        for img_name in st.session_state.image_names:
                            img_k = self.normalize_image_key(img_name)
                            sk = f"detections_{img_k}"
                            if sk in st.session_state:
                                save_data[img_k] = st.session_state[sk]
                        self.save_detections(save_data, output_file_detections, output_file_time)
                        st.session_state.img_index = current_idx + 1
                        time.sleep(2)
                        st.rerun()
                else:
                    if st.button("Proceed to Final Survey", key="proceed_final_survey"):
                        self.stop_image_timer(current_img)
                        self.stop_editing_timer(current_img)
                        self.save_canvas_state(current_img, img_key, session_key, scale, img.width, img.height)
                        save_data = {}
                        for img_name in st.session_state.image_names:
                            img_k = self.normalize_image_key(img_name)
                            sk = f"detections_{img_k}"
                            if sk in st.session_state:
                                save_data[img_k] = st.session_state[sk]
                        self.save_detections(save_data, output_file_detections, output_file_time)
                        survey_url = (
                            "https://soscisurvey.scc.kit.edu/ula/?q=posttask"
                            f"&PROLIFIC_PID={st.session_state.get('current_user', '')}"
                            f"&treatment={st.session_state.get('treatment', '')}"
                        )
                        st.success("Thank you! Redirecting to survey...")
                        st.markdown(
                            f"""
                            <meta http-equiv="refresh" content="1; url={survey_url}">
                            <script>window.location.href = "{survey_url}";</script>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.stop()

            if DEV_MODE:
                st.sidebar.markdown("---")
                st.sidebar.header("📊 Statistics")
                total_boxes, images_with_boxes = 0, 0
                for img_name in st.session_state.image_names:
                    img_k = self.normalize_image_key(img_name)
                    boxes = st.session_state.get(f"detections_{img_k}", st.session_state.all_detections.get(img_k, []))
                    if boxes:
                        total_boxes += len(boxes)
                        images_with_boxes += 1
                st.sidebar.metric("Current Image", f"{current_idx + 1} / {len(st.session_state.image_names)}")
                st.sidebar.metric("Boxes on Current", len(st.session_state[session_key]))
                st.sidebar.metric("Total Images", len(st.session_state.image_names))
                st.sidebar.metric("Images with Boxes", images_with_boxes)
                st.sidebar.metric("Total Boxes", total_boxes)
                self.render_timer_display(current_img)


# ---- URL-driven authentication ----------------------------------------------
def clean_url_with_history_api():
    """Strip query parameters from the URL bar so credentials don't linger."""
    components.html(
        """
        <script>
            if (window.location.search) {
                const cleanUrl = window.location.origin + window.location.pathname;
                window.history.replaceState({}, document.title, cleanUrl);
            }
        </script>
        """,
        height=0,
    )


def main():
    for key, default in (
        ("authenticated", False),
        ("current_user", None),
        ("treatment", None),
        ("auth_done", False),
        ("labeler_suffix", None),
    ):
        st.session_state.setdefault(key, default)

    query_params = st.experimental_get_query_params()

    if query_params and not st.session_state.auth_done:
        username = query_params.get("username", [None])[0]
        password = query_params.get("password", [None])[0]
        prolific_pid = query_params.get("PROLIFIC_PID", [None])[0]
        treatment = query_params.get("treatment", [None])[0]
        labeler = query_params.get("labeler", [None])[0]

        if username and password:
            with open("./config.yaml") as f:
                config = yaml.load(f, Loader=SafeLoader)
            credentials = config["credentials"]["usernames"]["labeler"]
            if username == credentials["name"] and password == credentials["password"]:
                st.session_state.authenticated = True
                st.session_state.current_user = prolific_pid
                t = (treatment or "").lower()
                st.session_state.treatment = t if t in ("sigma_1", "sigma_2") else "sigma_0"
                # `labeler=relabel` switches into the authors' relabeling
                # pass (all images shown, no per-bin sampling).
                if labeler == RELABEL_FLAG:
                    st.session_state.labeler_suffix = labeler
                st.session_state.auth_done = True
                clean_url_with_history_api()
                st.experimental_set_query_params()
            else:
                st.error("❌ Authentication failed")
                st.stop()

    if st.session_state.authenticated and query_params:
        clean_url_with_history_api()

    if not st.session_state.authenticated:
        st.error("❌ Not authenticated")
        st.info(
            "Access this app using URL parameters:\n\n"
            "`?username=USER&password=PASS&PROLIFIC_PID=ID&treatment=sigma_0|sigma_1|sigma_2`\n\n"
            "Add `&labeler=relabel` to enter the authors' relabeling pass."
        )
        st.stop()

    components.html(
        """
        <script>
            setTimeout(function() {
                if (window.location.search) {
                    window.history.replaceState({}, '', window.location.pathname);
                }
            }, 100);
        </script>
        """,
        height=0,
    )
    time.sleep(1)
    BoundingBoxLabeler().run()


if __name__ == "__main__":
    main()
