# rotate_cams.py — Tool to visually adjust camera orientation, FOV, and position

import cv2
import json
import map_view
import config


def save_configs(configs, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cameras": configs}, f, indent=2)
    print(f"Saved changes to {path}")


def main():
    map_json = "map.json"
    map_img_path = "MAP.png"

    try:
        camera_configs = config.load_camera_configs(map_json)
    except Exception as e:
        print(f"Error loading map.json: {e}")
        return

    map_base = map_view.load_map_base(map_img_path)
    canvas_h, canvas_w = map_base.shape[:2]

    # ✅ Create windows
    cv2.namedWindow("Map View", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)

    # ✅ FORCE window creation (fix for NULL window error)
    dummy = map_base.copy()
    cv2.imshow("Map View", dummy)
    cv2.imshow("Controls", dummy)
    cv2.waitKey(1)

    # ✅ Resize + Position windows
    cv2.resizeWindow("Map View", canvas_w, canvas_h)
    cv2.resizeWindow("Controls", 400, canvas_h)

    cv2.moveWindow("Map View", 0, 0)
    cv2.moveWindow("Controls", canvas_w + 20, 0)

    # ✅ Create trackbars
    for cam in camera_configs:
        name = cam["name"]

        cv2.createTrackbar(f"{name} Rot", "Controls",
                           int(cam.get("orientation", 0)), 360, lambda x: None)

        cv2.createTrackbar(f"{name} FOV", "Controls",
                           int(cam.get("fov", 70)), 180, lambda x: None)

        cv2.createTrackbar(f"{name} X", "Controls",
                           int(cam["position"][0]), canvas_w, lambda x: None)

        cv2.createTrackbar(f"{name} Y", "Controls",
                           int(cam["position"][1]), canvas_h, lambda x: None)

        cv2.createTrackbar(f"{name} Range", "Controls",
                           int(cam.get("range_radius", 120)), 500, lambda x: None)

        cv2.createTrackbar(f"{name} Tilt", "Controls",
                           int(cam.get("tilt", 15)), 40, lambda x: None)

    print("Adjust properties using trackbars.")
    print("Press 'S' to save, 'Q' to quit.")

    while True:
        # ✅ Update camera configs from trackbars
        for cam in camera_configs:
            name = cam["name"]

            cam["orientation"] = cv2.getTrackbarPos(f"{name} Rot", "Controls")
            cam["fov"] = max(1, cv2.getTrackbarPos(f"{name} FOV", "Controls"))
            cam["position"][0] = cv2.getTrackbarPos(f"{name} X", "Controls")
            cam["position"][1] = cv2.getTrackbarPos(f"{name} Y", "Controls")
            cam["range_radius"] = max(10, cv2.getTrackbarPos(f"{name} Range", "Controls"))
            cam["tilt"] = max(1, cv2.getTrackbarPos(f"{name} Tilt", "Controls"))

        # ✅ Render map
        display = map_view.render_map(map_base, camera_configs, {}, show_cameras=True)

        # Help text
        cv2.putText(display, "S: Save | Q: Quit",
                    (15, canvas_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ✅ Show map on left
        cv2.imshow("Map View", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            save_configs(camera_configs, map_json)

        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()