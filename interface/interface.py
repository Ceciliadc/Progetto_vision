import os
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk
import PIL.Image
import ctypes
import shutil
from retrieval.retrieval import data_of_aspecific_image
from pygame import mixer
import tkinter as tk
from tkinter import messagebox as mb

playing = False


class ResizingCanvas(Canvas):
    def _init_(self, parent, **kwargs):
        Canvas._init_(self, parent, **kwargs)
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all", 0, 0, wscale, hscale)


finestra = Tk()

# assegna dimensione alla finestra
finestra.title('Inside the Gallery')
photo = PhotoImage(file="icon2.png")
finestra.iconphoto(True, photo)

user32 = ctypes.windll.user32
myframe = Frame(finestra)
myframe.pack(fill=BOTH, expand=YES)
mycanvas = ResizingCanvas(myframe, width=user32.GetSystemMetrics(0), height=user32.GetSystemMetrics(1), bg="red",
                          highlightthickness=0)
mycanvas.pack(fill=BOTH, expand=YES)


def display_images(folder, root, filenames, people=0):
    gray_im = PIL.Image.open('gray.png')
    row, col = 0, 0
    image_count = 0
    window = Toplevel(root)
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    canvas = Canvas(window, width=width, height=height, scrollregion=(0, 0, width, height))
    canvas.grid(row=0, column=0, sticky="ns")

    frame_image = Frame(canvas)
    frame_image.pack(expand=True, fill="both")
    canvas.create_window((100, 0), window=frame_image, anchor="nw")

    next_clicked = BooleanVar()

    next_button = Frame(canvas, bg='#ffffff', bd=5)
    next_button.place(relx=0.0, rely=0.4, relwidth=0.12, relheight=0.1, anchor='n')
    next = Button(next_button, text='Succ.', command=lambda: next_clicked.set(True), bg='gray', font=11, fg='red')
    next.place(relx=0.55, rely=0.3, relwidth=0.4, relheight=0.5)

    for name in filenames:
        image_count += 1
        im = PIL.Image.open(os.path.join(folder, name))
        resized = im.resize((int((width - 100) / 3), (int((height - 100) / 2))), PIL.Image.ANTIALIAS)
        tkimage = ImageTk.PhotoImage(resized)
        myvar = Label(frame_image, image=tkimage)
        myvar.image = tkimage
        myvar.grid(row=row, column=col)

        im_buttons = Frame(myvar, bg='gray', bd=1)
        im_buttons.place(relx=0.5, rely=0, relwidth=0.3, relheight=0.1, anchor='n')

        def buttonClick(file_name):
            title, author, room, image, audio = data_of_aspecific_image(file_name)
            window.destroy()
            global playing
            playing = False
            ultima_pagina("image_ultima_pagina/", finestra, title, author, image, room, audio, filenames, people)

        Button(im_buttons, text='Info', command=lambda m=name: buttonClick(m), bg='gray',
               font=11, fg='red').place(relx=0.1,
                                        rely=0.2,
                                        relwidth=0.8,
                                        relheight=0.6)
        if row == 1 and col == 2:
            next_button.wait_variable(next_clicked)

            if next_clicked.get():
                next_clicked.set(False)

                row = 0
                col = 0

                while row < 3 and col < 3:
                    col += 1

                    if col == 3:
                        col = 0
                        row += 1

                    # we assume to put 3 images for each row
                    myvar.grid(row=row, column=col)

                row = 0
                col = -1

        col += 1

        if col == 3:
            col = 0
            row += 1

    resized = gray_im.resize((int(width / 3), (int(width / 3))), PIL.Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(resized)
    myvar = Label(frame_image, image=tkimage)
    myvar.image = tkimage
    window.mainloop()


def delete_old_data():
    to_be_deleted = ['./inference/output', '../rectification/painting_rect1', '../rectification/painting',
                     '../people_detection/images', '../people_detection/output_test',
                     '../people_detection/reshaped_images', '../people_detection/output',
                     '../pose_estimation/headpose/denoised_test', '../pose_estimation/headpose/images',
                     '../people_detection/labels']

    try:
        os.remove('images_found.txt')
    except:
        print('file images_found not existing')

    try:
        os.remove('people_found.txt')
    except:
        print('file people not existing')

    for folder in to_be_deleted:
        try:
            shutil.rmtree(folder)
        except:
            print('folder %s does not exist' % folder)


def ultima_pagina(folder, root, title, author, image, room, audio, filenames, people):
    window = Toplevel(root)
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    canvas = Canvas(window, width=width, height=height)
    canvas.grid(row=2, column=2, sticky="news")
    frame_image = Frame(canvas)
    frame_image.pack(expand=True, fill="both")
    canvas.create_window((0, 0), window=frame_image, anchor="nw")

    def callback():
        if playing:
            mixer.music.stop()
        window.destroy()
        display_images("../sift/images/", root, filenames, people)

    window.protocol("WM_DELETE_WINDOW", callback)
    painting = PIL.Image.open('../sift/images/' + image)
    description = open('../descriptions/' + image[:-3] + 'txt', 'r', encoding="utf8").read().split()

    resized = painting.resize((int((width - 50) / 2), (int((height + 200) / 2))), PIL.Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(resized)
    img = Label(frame_image, image=tkimage)
    img.image = tkimage
    img.grid(row=0, column=0)

    description = 'TITOLO:\t' + title + '\n\n' + 'AUTORE:\t' + author + '\n\nDESCRIZIONE:\t' + ' '.join(description)
    text = tk.Text(frame_image, height=30, width=90)
    text.insert(tk.END, '\n')
    text.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
    text.insert(tk.END, description)
    text.grid(row=0, column=1)

    if str(audio) != 'nan':
        mixer.init()
        mixer.music.load('../audio_description/' + audio)
        play_button = Frame(canvas, bg='#ffffff', bd=5)
        play_button.place(relx=0.2, rely=0.7, relwidth=0.12, relheight=0.1, anchor='n')

        def play():
            global playing

            if not playing:
                button['text'] = 'Stop'
                playing = True
                mixer.music.play()
            else:
                button['text'] = 'Play'
                playing = False
                mixer.music.pause()

        button = Button(play_button, text='Play', bg='#ff7f7f', font=11, command=play,
                        fg='#ffffff')
        button.place(relx=0.25, rely=0.25, relwidth=0.4, relheight=0.6)

    if len(room) <= 2:
        room = Label(frame_image,
                     text='\n\nIl dipinto è nella stanza ' + str(room) + '\n e sono presenti %s persone' % people)
    else:
        room = Label(frame_image,
                     text='\n\n' + str(room) + '\n Sono presenti %s persone' % people)
    room.config(font=("Courier", 18))  # width=100)
    room.grid(row=1, column=1)


def main():
    bg_image = PhotoImage(file=r"background2.png")

    print(ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))
    Label(finestra, image=bg_image).place(x=0, y=0, width=1920,
                                          height=1080)

    def open_file():
        frame_path.filename = filedialog.askopenfilename(initialdir="/", title="Seleziona un video", filetypes=(
            ("mp4 files", ".mp4"), ("MOV files", ".MOV"), ("avi files", ".avi"), ("All files", ".*")))
        e.delete(0, END)
        e.insert(0, frame_path.filename)

    frame_path = Frame(finestra, bg='#ffffff', bd=5)
    frame_path.place(relx=0.5, rely=0.5, relwidth=0.4, relheight=0.15, anchor='n')

    Label(frame_path, width=34, fg="#696969", bg='#ffffff', font=5, text='Inserisci o seleziona il percorso del video:',
          anchor='w').place(relx=0.2, rely=0.15, relwidth=0.96, relheight=0.3)
    e = Entry(frame_path, fg="#696969", bg='#ffffff', font=9)
    e.place(relx=0.02, rely=0.55, relwidth=0.84, relheight=0.30)
    e.insert(END, "Inserisci il percorso")
    Button(frame_path, text=" ... ", command=open_file, bg='#ffffff').place(relx=0.88, rely=0.55, relwidth=0.1,
                                                                            relheight=0.30)

    frame_buttons = Frame(finestra, bg='#ffffff', bd=5)
    frame_buttons.place(relx=0.5, rely=0.65, relwidth=0.4, relheight=0.1, anchor='n')

    def buttonClick():
        if hasattr(frame_path, 'filename'):

            filenames = []
            delete_old_data()

            os.system("python ../yolov5/detect.py --weights ../yolov5/runs/exp75/weights/last.pt --img 416 --conf "
                      "0.6 --source %s" % frame_path.filename)
            os.system("python ../rectification/video_frame.py")
            os.system("python ../rectification/rectification.py")
            os.system("python ../people_detection/ssd.py %s" % frame_path.filename)
            os.system("python ../sift/sift.py")
            os.system("python ../pose_estimation/head_pose.py")

            with open('./people_found.txt') as people_file:
                people = people_file.readline()

            with open('./images_found.txt', 'r') as file:
                for name in file:
                    filenames.append(name[:-1])

            filenames = list(dict.fromkeys(filenames))
            display_images("../sift/images/", finestra, filenames, people)
        else:
            mb.showerror("Attenzione", "Selezionare un percorso")

    Button(frame_buttons, text='Conferma', command=buttonClick, bg='beige', font=11,
           fg='brown').place(relx=0.3,
                             rely=0.2,
                             relwidth=0.4,
                             relheight=0.6)
    finestra.mainloop()


if __name__ == "__main__":
    main()