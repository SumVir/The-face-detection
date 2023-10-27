import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))
camera = cv2.VideoCapture(0)

# Define face information (names and ages) for each detected face
face_info = [
    {"name": "Nursaid", "age": 18},

    # Add more entries for each detected face as needed
]

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
     gray,
     scaleFactor =1.1,
     minNeighbors =5,
     minSize = (30,30),
     flags = cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0),  2)
       
        # Display name and age above the rectangle
    if len(faces) > 0:
     for i, info in enumerate(face_info):
        face_info_text = f"Name:  {info['name']} | Age: {info['age']}"
        cv2.putText(frame,  face_info_text,  (x,y - 10 - i * 20),  cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 0), 2)
 
            

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()