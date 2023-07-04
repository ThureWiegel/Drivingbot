from time import time
import pyautogui
from pynput.keyboard import Key,Controller
import cv2
import numpy as np

keyboard = Controller()

# istrue = True

# while istrue:
#     keyboard.press('w')
#     print('press s')
#     time.sleep(10)
#     keyboard.release('w')
#     print('release s')
#     time.sleep(10)

loop_time = time()
while True:
    screenshot = pyautogui.screenshot(region=(0,0,1920,1080))
    image = np.array(screenshot)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # image = image[:,:,::-1].copy()
    cv2.imshow('a', image)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

print('done')
