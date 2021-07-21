# import pafy
# import cv2

# url = "https://www.youtube.com/watch?v=GbAZX-NDPLg"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")

# capture = cv2.VideoCapture(best.url)
# print(best.url)
# while True:
#     grabbed, frame = capture.read()
#     cv2.imshow('',frame)


import cv2, pafy

# TODO import API key from .env

url = "https://www.youtube.com/watch?v=GbAZX-NDPLg"
print(url)
video = pafy.new(url)
print(video)
# best  = video.getbest(preftype="webm")
# #documentation: https://pypi.org/project/pafy/

# capture = cv2.VideoCapture(best.url)
# check, frame = capture.read()
# print (check, frame)

# cv2.imshow('frame',frame)
# cv2.waitKey(10)

# capture.release()
# cv2.destroyAllWindows()