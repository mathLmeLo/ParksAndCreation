import cv2 as open_cv
import logging


class TakePicture:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, image_path, cam_port = 0):
        self.image_path = image_path
        self.cam_port = cam_port


    def take_picture(self):
        cam = open_cv.VideoCapture(self.cam_port)
        open_cv.namedWindow(self.image_path)


        while True:
            result, frame = cam.read()

            if frame is None:
                break

            if not result:
                logging.error("failed to grab frame")
                raise CaptureReadError("Error reading frame %s" % str(frame))
            
            open_cv.imshow(self.image_path, frame)

            k = open_cv.waitKey(1)

            if k == ord('q'): 
                # writes pic to file
                open_cv.imwrite(self.image_path, frame)

                logging.debug("initial picture saved: %s", self.image_path)
                break               
                           

        cam.release()
        open_cv.destroyAllWindows()

        return True


class CaptureReadError(Exception):
    pass
