import cv2
import mediapipe as mp
import threading


class GestureController:
    def __init__(self):
        # ── Existing gesture states ───────────────────────────────────────────
        self.left_fist   = False   # Gesture 1: black hole form
        self.right_open  = False   # Gesture 1 modifier: jet mode

        # ── New gesture states ────────────────────────────────────────────────
        self.right_fist  = False   # Gesture 2: vortex
        self.left_open   = False   # Gesture 3: shockwave (edge-triggered in main)
        self.both_open   = False   # Gesture 4: expansion field

        self.debug_frame = None

        threading.Thread(target=self._run, daemon=True).start()

    # ── Detectors ─────────────────────────────────────────────────────────────

    def _is_fist(self, lm):
        """4 fingers folded (thumb ignored)."""
        tips = [8, 12, 16, 20]
        folded = 0
        for tip in tips:
            if lm.landmark[tip].y > lm.landmark[tip - 2].y:
                folded += 1
        return folded >= 3

    def _is_open_palm(self, lm):
        """4 fingers extended and separated — all three joints ascending."""
        fingers_up = 0
        for tip in [8, 12, 16, 20]:
            if (
                lm.landmark[tip].y <
                lm.landmark[tip - 1].y <
                lm.landmark[tip - 2].y
            ):
                fingers_up += 1
        return fingers_up >= 3

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self):
        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils

        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame     = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(frame_rgb)

            left_fist  = False
            right_open = False
            right_fist = False
            left_open  = False

            if results.multi_hand_landmarks and results.multi_handedness:
                for lm, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label

                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                    if label == "Left":
                        if self._is_fist(lm):
                            left_fist = True
                        if self._is_open_palm(lm):
                            left_open = True

                    if label == "Right":
                        if self._is_open_palm(lm):
                            right_open = True
                        if self._is_fist(lm):
                            right_fist = True

            self.left_fist  = left_fist
            self.right_open = right_open
            self.right_fist = right_fist
            self.left_open  = left_open
            self.both_open  = left_open and right_open

            # ── Debug overlays ────────────────────────────────────────────────
            labels = [
                (f"Left Fist:    {self.left_fist}",  (0, 255,   0)),
                (f"Right Open:   {self.right_open}", (0, 255,   0)),
                (f"Jet Active:   {self.left_fist and self.right_open}",
                                                      (0, 100, 255)),
                (f"Right Fist:   {self.right_fist}", (0, 255, 200)),
                (f"Left Open:    {self.left_open}",  (0, 255, 200)),
                (f"Both Open:    {self.both_open}",  (0, 200, 255)),
            ]

            for i, (text, color) in enumerate(labels):
                cv2.putText(
                    frame, text,
                    (10, 30 + i * 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, 2
                )

            self.debug_frame = frame
