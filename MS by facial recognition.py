#music

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
col1, col2, col3 = st.columns([1,6,1])
with col1:
    st.write("")

#with col2:
    #st.image("./Images/logo.png" , width=530, use_column_width=True)
with col2:
    st.write("")
st.title("ZENE")
st.write('ZENE is facial emotion detection based music reccommendation system. To get reccommended songs, start by allowing mic and camera for this web app.')
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils
if "run" not in st.session_state:
    st.session_state["run"] = "true"
try:
    detected_emotion = np.load("detected_emotion.npy")[0]
except:
    detected_emotion = ""
if not(detected_emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"
class EmotionDetector:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1) #Flipping the frame from left to right
        res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []
        #Storing Landmark data
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        lst = np.array(lst).reshape(1,-1)
        pred = label[np.argmax(model.predict(lst))]
        print(pred)
        cv2.putText(frm, pred, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
        np.save("detected_emotion.npy", np.array([pred]))
        drawing.draw_landmarks(frm, res.face_landmarks,holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(frm, res.left_hand_landmarks,hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks,hands.HAND_CONNECTIONS)
        return av.VideoFrame.from_ndarray(frm, format="bgr24")
        lang = st.text_input("Enter your preferred language")
        artist = st.text_input("Enter your preferred artist")
        if lang and artist and st.session_state["run"] != "false":
            webrtc_streamer(key="key", desired_playing_state=True,video_processor_factory=EmotionDetector)
        btn = st.button("Recommend music")
        if btn:
              if not(detected_emotion):
                  st.warning("Please let me capture your emotion first!")
                  st.session_state["run"] = "true"
              else:
                  webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{detected_emotion}+songs+{artist}")
                  np.save("detected_emotion.npy", np.array([""]))
                  st.session_state["run"] = "false"
        st.write('Made with ❤ by KRCE')
        #Streamlit Customisation
        st.markdown(""" <style>header {visibility: hidden;}footer {visibility: hidden;}</style> """, unsafe_allow_html=True)

#DATA COLLECTION 
import mediapipe as mp
import numpy as np
import cv2
cap = cv2.VideoCapture(0)
name = input("Enter the name of the data : ")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
X = []
data_size = 0
while True:
    lst = []
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        X.append(lst)
        data_size = data_size+1
    drawing.draw_landmarks(frm, res.face_landmarks,holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks,hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks,hands.HAND_CONNECTIONS)
    cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.imshow("window", frm)
    if cv2.waitKey(1) == 27 or data_size>99:
        cv2.destroyAllWindows()
        cap.release()
        break
np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)

#DATA TRANING
import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
is_init = False
size = -1
label = []
dictionary = {}
c = 0
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
        if not(is_init):
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))
        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c = c+1
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")
### hello = 0 nope = 1 ---> [1,0] ... [0,1]
y = to_categorical(y)
X_new = X.copy()
y_new = y.copy()
counter = 0 
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)
for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter = counter + 1
ip = Input(shape=(X.shape[1]))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 
model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
model.fit(X, y, epochs=50)
model.save("model.h5")
np.save("labels.npy", np.array(label))

#INTERENCE
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
while True:
    lst = []
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)
        lst = np.array(lst).reshape(1, -1)
        pred = label[np.argmax(model.predict(lst))]
        print(pred)
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
    drawing.draw_landmarks(frm, res.face_landmarks,holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks,hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks,hands.HAND_CONNECTIONS)
    cv2.imshow("window", frm)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break





