from matplotlib import pyplot
import cv2 
from ultralytics import YOLO


def toarabic(en_chr):
    arabic_chars = {
        "9.0": "أ",
        "10.0": "ع",
        "11.0": "ب",
        "12.0": "د",
        "13.0": "ف",
        "14.0": "ج",
        "15.0": "ه",
        "16.0": "ك",
        "17.0": "ل",
        "18.0": "م",
        "19.0": "ن",
        "20.0": "ر",
        "21.0": "س",
        "22.0": "ص",
        "23.0": "ط",
        "24.0": "و",
        "25.0": "ى",
        "0.0": "١",
        "1.0": "٢",
        "2.0": "٣",
        "3.0": "٤",
        "4.0": "٥",
        "5.0": "٦",
        "6.0": "٧",
        "7.0": "٨",
        "8.0": "٩"
    }

    for i in range(26):
        if en_chr == f"{i}.0":
            return arabic_chars[f"{i}.0"]

    return None  # Return None if the tensor value is not found


#models
plate_model=YOLO("./models/polo.pt")
ocr_model=YOLO("./models/best.pt")    #theres another one called wolo.pt in the same folder


#path
frame=cv2.imread("./photos/1737.jpg")
frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
pyplot.imshow(frame)


#find plate
plate_model_tranied=plate_model(source=frame,save_txt=False,conf=.1)


#crop plate
for plate in plate_model_tranied:
    carBox=plate.boxes
    if len(carBox)!=0:
        x1,y1,x2,y2=carBox.xyxy[0]
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        cren_chrping_plate=frame[y1:y2,x1:x2]
        # pyplot.imshow(cren_chrping_plate)

        # cv2.imshow("The Plate", cren_chrping_plate)
        # if cv2.waitKey(0) & 0xFF == ord("q"):
        #     break



#find chars
result=ocr_model.predict(cren_chrping_plate,conf=.3,iou=.1,max_det=7,device="cuda")

lis=[]
for en_chr in result:
    # print(en_chr.tojson())
    for i in en_chr :
        w=i.boxes

        x1,y1,x2,y2=w.xyxy[0]
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        # print(str(w.cls.item()))
        lis.append({"class":toarabic(str(w.cls.item())),"x":x1,"y":y1})
lis=sorted(lis,key=lambda x:x["x"])


#combine all chars in one string
plat_number=""
for i in lis:
  plat_number+=i["class"]
  plat_number+=" "

print(reversed(plat_number))


