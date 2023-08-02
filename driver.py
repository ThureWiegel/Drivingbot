import cv2
import time
import numpy as np
from grabscreen import grab_screen
from keras.models import load_model
from directkeys import PressKey, ReleaseKey, W, A, D


def straight():
	PressKey(W)
	ReleaseKey(A)
	ReleaseKey(D)


def left():
	PressKey(W)
	PressKey(A)
	ReleaseKey(D)


def right():
	PressKey(W)
	PressKey(D)
	ReleaseKey(A)


def main():
	for i in list(range(4))[::-1]:
		print(i + 1)
		time.sleep(1)

	model = load_model('model.h5')

	while (True):
		# 800x600 windowed mode
		screen = grab_screen(region=(0, 40, 800, 640))
		screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
		screen = cv2.resize(screen, (160, 120))
		cv2.imshow('im', screen)

		model_input = screen.reshape(-1, 160, 120, 1)
		model_input = np.asarray(model_input)

		model_output = model(model_input, training=False)

		if model_output[0][0] > model_output[0][1] and model_output[0][0] > model_output[0][2]:
			left()
		elif model_output[0][1] > model_output[0][0] and model_output[0][1] > model_output[0][2]:
			straight()
		elif model_output[0][2] > model_output[0][1] and model_output[0][2] > model_output[0][0]:
			right()

		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break


main()
