import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_BLUE
import torch


class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, coordinates, start_frame):
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

    # returns a pandas dataframe like
    #              xmin        ymin        xmax        ymax  confidence  class name
    # 0  290.822479  274.026245  394.356293  319.500305    0.891700      2  car
    # 1  573.885315  326.437714  741.645447  393.699036    0.868255      2  car
    def inference(self, img):
        # Inference
        results = self.model(img)

        # Results
        #results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
        inference = results.pandas().xyxy[0]
        print(inference)
        return inference
    
    # gets the coordinates of the detected vehicles in the following format: [["xmin", "ymin"], ["xmax", "ymin"], ["xmax", "ymax"], ["xmin", "ymax"]]
    def detected_vehicle_coordinates(self, inference):
        vehicles = inference[inference["name"].isin(["car", "bus", "truck"])]            
        return self.convert_format(vehicles.loc[:, ["xmin", "ymin", "xmax", 'ymax']].to_numpy())


    def detect_live_motion(self):
        capture = open_cv.VideoCapture(0)
        capture.set(open_cv.CAP_PROP_POS_FRAMES, self.start_frame)

        coordinates_data = self.coordinates_data
        logging.debug("coordinates data: %s", coordinates_data)

        for p in coordinates_data:
            coordinates = self._coordinates(p)
            
            logging.debug("coordinates: %s", coordinates)

            rect = open_cv.boundingRect(coordinates)
            logging.debug("rect: %s", rect)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]
            logging.debug("new_coordinates: %s", new_coordinates)

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = open_cv.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=open_cv.LINE_8)

            mask = mask == 255
            self.mask.append(mask)
            logging.debug("mask: %s", self.mask)

        statuses = [False] * len(coordinates_data)
        times = [None] * len(coordinates_data)

        while capture.isOpened():
            result, frame = capture.read()
            if frame is None:
                break

            if not result:
                raise CaptureReadError("Error reading video capture on frame %s" % str(frame))
            
            # For each frame gets the inference result and compares it with the previous 
            # calls check_collision with the point coordinate and each of the vehicled detected in inference
            # fot that uses the detected_vehicle_coordinates

            blurred = open_cv.GaussianBlur(frame.copy(), (5, 5), 3)
            grayed = open_cv.cvtColor(blurred, open_cv.COLOR_BGR2GRAY)
            new_frame = frame.copy()
            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(open_cv.CAP_PROP_POS_MSEC) / 1000.0

            for index, c in enumerate(coordinates_data):
                status = self.__apply(grayed, index, c)

                if times[index] is not None and self.same_status(statuses, index, status):
                    times[index] = None
                    continue

                if times[index] is not None and self.status_changed(statuses, index, status):
                    if position_in_seconds - times[index] >= MotionDetector.DETECT_DELAY:
                        statuses[index] = status
                        times[index] = None
                    continue

                if times[index] is None and self.status_changed(statuses, index, status):
                    times[index] = position_in_seconds

            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)

                color = COLOR_GREEN if statuses[index] else COLOR_BLUE
                draw_contours(new_frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)

            open_cv.imshow('Result', new_frame)
            k = open_cv.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        open_cv.destroyAllWindows()    

    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)

        rect = self.bounds[index]
        logging.debug("rect: %s", rect)

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = open_cv.Laplacian(roi_gray, open_cv.CV_64F)
        logging.debug("laplacian: %s", laplacian)

        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        status = np.mean(np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN
        logging.debug("status: %s", status)

        return status
    
        # Função para verificar a colisão entre dois retângulos
    @staticmethod
    def check_collision(rect1, rect2):
        # Extrair os pontos dos retângulos
        rect1_pts = np.array(rect1)
        rect2_pts = np.array(rect2)

        # Calcular os limites dos retângulos
        rect1_x_min, rect1_y_min = np.min(rect1_pts, axis=0)
        rect1_x_max, rect1_y_max = np.max(rect1_pts, axis=0)
        
        rect2_x_min, rect2_y_min = np.min(rect2_pts, axis=0)
        rect2_x_max, rect2_y_max = np.max(rect2_pts, axis=0)

        # Verificar a colisão
        if (rect1_x_max < rect2_x_min or rect1_x_min > rect2_x_max or
            rect1_y_max < rect2_y_min or rect1_y_min > rect2_y_max):
            return False
        else:
            return True
        
    

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])
    
    # converts ["xmin", "ymin", "xmax", "ymax"]] to [["xmin", "ymin"], ["xmax", "ymin"], ["xmax", "ymax"], ["xmin", "ymax"]]
    @staticmethod
    def convert_format(input_array):    
        # Extrair os valores de xmin, ymin, xmax e ymax
        output_array = []

        for element in input_array:        
            xmin, ymin, xmax, ymax = element
            # Criar o numpy array de saída no formato desejado
            output_array.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

        return np.array(output_array)    

    @staticmethod
    def same_status(coordinates_status, index, status):
        return status == coordinates_status[index]

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]


class CaptureReadError(Exception):
    pass
