import cv2


def capture(save_path):
    cam = cv2.VideoCapture(0)

    for i in range(200):
        _, img = cam.read()
        cv2.imshow('main', img)

        if i % 20 == 0:
            print(i)
            if not cv2.imwrite(f'save_path/{i+1:05}.png', img):
                raise Exception("Could not write image")
        
        cv2.waitKey(1)

    cam.release()
    cv2.destroyAllWindows()

    print('done...')
