import tkinter as tk, numpy as np, pickle as pk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

model = load_model('add the weights path')
with open('add ResultMap path', 'rb') as handle:
    ResultMap = pk.load(handle)

def preprocess_image(image_path, target_size=(64, 64)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, 0)
    image /= 255.0
    return image

def classify_image():
    global image_path
    if image_path:
        image = preprocess_image(image_path)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0] 
        result_label.config(text=f'Classified as: {ResultMap[predicted_class]}\nImage Path: {image_path}')
        load_and_display_image(image_path)

def load_and_display_image(path):
    global panel_image
    img = Image.open(path)
    panel_image = ImageTk.PhotoImage(img)
    panel.config(image=panel_image)


def select_image():
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        classify_image()

def clear_image():
    panel.config(image='')
    result_label.config(text='')

root = tk.Tk()
root.title("Eye Classification")

btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="")
result_label.pack()

btn_clear = tk.Button(root, text="Clear Image", command=clear_image)
btn_clear.pack()

image_path = ''
panel_image = None

root.mainloop()
