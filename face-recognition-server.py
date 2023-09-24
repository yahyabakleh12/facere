import face_recognition
import asyncio
import websockets
import json, io
import requests
import pickle
import numpy as np
import codecs
# from deepface import DeepFace
#######################################my code###########################################################
userList = []
encodListknown = []
userPath=[]
userid = []
def sync():
    x = requests.get('https://e9ed-94-200-29-94.ngrok-free.app/get_faces')
    x.raise_for_status()
    # access JSOn content
    jsonResponse = x.json()
    # y = json.loads(x)
    emptyjson= {}
    json_object = json.dumps(emptyjson)
    # Writing to XXXXX.json
    with open("DB/employee.json", "w") as outfile:
        outfile.write(json_object)
    json_object = json.dumps(jsonResponse)
    # Writing to XXXXX.json
    with open("DB/employee.json", "w") as outfile:
        outfile.write(json_object)
    with open('DB/employee.json', 'r') as openfile:
        # Reading from json file
        jsonResponse = json.load(openfile)
    for user in jsonResponse:
        if user[3] == None:
            pass
        else:
            face_data = ''
            face_data = pickle.loads(codecs.decode(
                user[3].encode(), "base64"))
            face_data = pickle.loads(face_data)
            encodListknown.append(face_data)
            userList.append(user[1])
            userPath.append(user[4])
            userid.append(user[0])
sync()
#######################################my code###########################################################

# picture_of_me = face_recognition.load_image_file("me.jpeg")
# my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

async def websocket_handler(websocket, path):
    try:
        sync()
        async for message in websocket:
            response = recognize_face(message)
            await websocket.send(json.dumps(response))
            
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")

def recognize_face(message):
   
    try:
        unknown_picture = face_recognition.load_image_file(io.BytesIO(message))
        unknown_face_encodings = face_recognition.face_encodings(unknown_picture)
        if len(unknown_face_encodings) > 0:
            unknown_face_encoding = unknown_face_encodings[0]
        else:
            return {"status": True, "message": "No Face Detected", "data": 0}
        matches = face_recognition.compare_faces(encodListknown, unknown_face_encoding,tolerance=0.45)
        facedis = face_recognition.face_distance(encodListknown, unknown_face_encoding)
        
        matchIndex = np.argmin(facedis)
        if matches[matchIndex]:
            name = userList[matchIndex].upper()
            # basepic = "http://127.0.0.1:5000/face_pic/"+str(userid[matchIndex])
            # result = DeepFace.verify("attendes/User." +str(count) + ".jpg",basepic,model_name="Facenet")
            return {"status": True, "message": "Recognition successful welcome "+str(name), "data": 2}
        else:
            return {"status": True, "message": "Recognition unsuccessful", "data": 1}

    except Exception as e:
        return {"status": False, "message": str(e)}

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        websockets.serve(websocket_handler, "3.254.51.228", 8765)
    )
    loop.run_forever()
