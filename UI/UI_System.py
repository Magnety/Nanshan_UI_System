import tkinter as tk
import SimpleITK as sitk
from tkinter import  *
from tkinter import filedialog
import os
import numpy as np
from PIL import Image,ImageTk
import threading
import random
import tkinter.messagebox
import cv2
from Segmentation.inference.output_predict import Output
from multiprocessing import Lock,Process
import shutil
from Segmentation.mcd import get_tumor
class UI(Frame):
    def __init__(self,master):
        Frame.__init__(self,master=None)
        master.title('Nanshan Hospital ABVS Diagnosis System')
        self.window_width = 800+400
        self.window_height = 400+100
        master.geometry('%dx%d'%(self.window_width,self.window_height))
        #master.geometry('' )
        self.master = master
        #open file
        self.btn_openfile = Button(master, text='Open Image', width=10, height=1, command=self.open_image)
        self.btn_openfile.place(x=5, y=5)

        self.btn_analyze = Button(master, text='Analyze', width=10, height=1, command=self.analyze)
        self.btn_analyze.place(x=810, y=450)

        # patient name
        self.patient_name = tk.StringVar()
        self.patient_name.set("No Image Opened")
        self.patient_from = Label(master, text="From: ")
        self.patient_from.place(x=100, y=10)
        self.patient_display = Label(master, textvariable=self.patient_name)
        self.patient_display.place(x=140, y=10)

        # 初始空白图像
        self.empty_img_np = np.zeros((800, 800))
        self.img_np = np.zeros((800, 800))
        self.empty_img = Image.fromarray(self.empty_img_np)
        self.empty_render = ImageTk.PhotoImage(self.empty_img)
        # 图片标签
        self.nii_label = Label(master, text='ABVS Image')
        self.nii_label.place(x=5, y=35)
        self.tumor_label = Label(self.master, text=' Tumor_No.                  Posibility')
        self.tumor_label.place(x=self.window_width - 280, y=30)
        # 原始图像框
        self.img_canvas = Canvas(master, width=800, height=400)
        # self.nii_img.create_image(0, 0, anchor=NW, image=self.empty_render)
        self.img_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.img_canvas.bind("<B1-Motion>", self.on_move_press)
        self.img_canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.img_canvas.place(x=0, y=60)
        self.img_canvas.image = self.empty_render
        self.img_canvas.create_image(0, 0, anchor=NW, image=self.empty_render)
        """# 分割标签
        self.mask_label = Label(master, text='Seg Img')
        self.mask_label.place(x=905, y=35)
        # seg框
        self.nii_mask = Canvas(master, width=800, height=800)
        # self.nii_mask.create_image(0, 0, anchor=NW, image=self.empty_render)
        self.nii_mask.place(x=905, y=60)"""


        #slice num
        self.scale = Scale(master, orient=tk.VERTICAL, length=360, from_=0, to=317, command=self.show_nii)
        self.scale.place(x=820, y=60)

        # 绑定鼠标滚轮
        self.scale.bind("<MouseWheel>", self.wheel)
        # Button(root,text = '获取位置',command = show).pack()#用command回调函数获取位置
        self.timer = threading.Timer(1, self.show)
        self.timer.start()

        self.tumor_buttons = []
        self.object_size = None
        self.object_x_max = None
        self.object_x_min = None
        self.object_y_max = None
        self.object_y_min = None
        self.object_z_max = None
        self.object_z_min = None
        #self.create_tumor_list_button(tumor_num=3)
    # 绘制矩形框
    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # create rectangle if not yet exist
        # if not self.rect:
        self.rect1 = self.nii_img.create_rectangle(self.x, self.y, 1, 1, fill="", outline='red')
        self.rect2 = self.nii_mask.create_rectangle(self.x, self.y, 1, 1, fill="", outline='red')

        print("x:%d" % self.start_x)
        print("y:%d" % self.start_y)

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)
        self.end_x = curX
        self.end_y = curY
        # expand rectangle as you drag the mouse
        self.nii_img.coords(self.rect1, self.start_x, self.start_y, curX, curY)
        self.nii_mask.coords(self.rect2, self.start_x, self.start_y, curX, curY)

        print("curx:%d" % curX)
        print("cury:%d" % curY)

    def on_button_release(self, event):
        pass

    #鼠标滚轮
    def wheel(self,e):
        if e.delta > 0:
            self.scale.set(self.scale.get() + 1)
        else:
            self.scale.set(self.scale.get() - 1)

    def show(self):
        self.timer = threading.Timer(1, self.show)
        self.timer.start()

    def analyze(self):
        """print("analyzing")
        if os.path.exists('/home/ubuntu/liuyiyao/Nanshan_UI_System/Output_segmentation/case_00000.nii.gz'):
            os.remove('/home/ubuntu/liuyiyao/Nanshan_UI_System/Output_segmentation/case_00000.nii.gz')"""
        tk.messagebox.showinfo("Analyzing","The ABVS Image is analyzing! Please wait!")
        #Output()
        self.mask_sitk = sitk.ReadImage('/home/ubuntu/liuyiyao/Nanshan_UI_System/Output_segmentation/case_00000.nii.gz')
        self.mask_np = sitk.GetArrayFromImage(self.mask_sitk)
        self.mask_np[self.mask_np == 1] = 255
        print("get_tumor")
        self.object_sizes,self.object_x_max,self.object_x_min,self.object_y_max,self.object_y_min,self.object_z_max,self.object_z_min = get_tumor()
        self.create_tumor_list_button(len(self.object_sizes))
        self.scale.set((self.object_x_max[0]+self.object_x_min[0])//2)
        self.show_nii(self.scale.get())


    def create_tumor_list_button(self,tumor_num):
        self.tumor_label.destroy()
        self.tumor_label = Label(self.master, text=' Tumor_No.                  Posibility')
        self.tumor_label.place(x=self.window_width - 280, y=30)
        for tumor_button in self.tumor_buttons:
            tumor_button.destroy()
        self.tumor_buttons=[]

        for i in range(tumor_num):
            posibility = random.random()
            if posibility>0.5:
                self.tumor_buttons.append(Button(self.master, text='Tumor_%02d:                    %.4f'%((i+1),posibility), foreground='Red',width=40,activebackground ='LightCyan',height=1, command= lambda arg = i:self.display_tumor(arg)))
            else:
                self.tumor_buttons.append(Button(self.master, text='Tumor_%02d:                    %.4f'%((i+1),posibility), foreground='Blue',width=40,activebackground ='LightCyan',height=1, command= lambda arg = i:self.display_tumor(arg)))

        index = 0
        for tumor_button in self.tumor_buttons:
            tumor_button.place(x=self.window_width-300, y=50+index*30)
            index += 1

    def display_tumor(self,index=0):
        print(index)
        self.scale.set((self.object_x_max[index]+self.object_x_min[index])//2)

        self.show_nii(val=(self.object_x_max[index]+self.object_x_min[index])//2)

    def open_image(self):
        default_dir = '/home/ubuntu/liuyiyao/Nanshan_UI_System/Input_Test_image'
        temp_dir = '/home/ubuntu/liuyiyao/Nanshan_UI_System/Temp_image'
        path = filedialog.askopenfilename(title=u"choose file", initialdir=(os.path.expanduser(default_dir)))
        print("patient_name:",path.split('/')[-1].split('.')[0])
        self.patient_name.set(path.split('/')[-1].split('.')[0])
        self.img_path=path
        print("img_path:",self.img_path)
        shutil.copy(self.img_path,temp_dir+'/case_00000_0000.nii.gz')
        img_sitk = sitk.ReadImage(path)
        self.x_spacing =img_sitk.GetSpacing()[0]
        self.y_spacing =img_sitk.GetSpacing()[1]


        if self.x_spacing>self.y_spacing:
            self.x_scale=1
            self.y_scale=self.x_spacing/self.y_spacing
        if self.x_spacing < self.y_spacing:
            self.y_scale =1
            self.x_scale =  self.y_spacing / self.x_spacing
        if self.x_spacing == self.y_spacing:
            self.x_scale = 1
            self.y_scale = 1


        print("x_spacing:",self.x_spacing)
        print("y_spacing:",self.y_spacing)
        print("x_scale:",self.x_scale)
        print("y_scale:",self.y_scale)



        self.img_np = sitk.GetArrayFromImage(img_sitk)

        print("image_shape:",self.img_np.shape)
        print(self.empty_img_np.shape)
        self.mask_np = np.zeros_like(self.img_np)
        self.show_nii(self.scale.get())
        #self.create_tumor_list_button(5)

    def show_nii(self,val):
        slice = int(val)
        self.img_canvas.delete(tk.ALL)
        #self.nii_mask.delete(tk.ALL)
        img = Image.fromarray(self.img_np[slice, :, :])
        img = img.convert("RGBA")
        mask = np.zeros((self.img_np.shape[1],self.img_np.shape[2],3), 'uint8')
        mask[:,:,0] = self.mask_np[slice,:,:]

        mask = Image.fromarray(mask)
        mask = mask.convert("RGBA")
        img_mask = Image.blend(img, mask, 0.3)
        img_mask = img_mask.resize((int((self.img_np.shape[2] / self.x_scale)*2), int((self.img_np.shape[1] / self.y_scale)*2)),Image.BICUBIC)
        #print("width:",int((self.img_np.shape[2] / self.x_scale)*2)," height:",int((self.img_np.shape[1] / self.y_scale)*2))
        self.window_width = int((self.img_np.shape[2] / self.x_scale)*2)+400
        self.window_height = int((self.img_np.shape[1] / self.y_scale)*2)+100
        self.btn_analyze.place(x=int((self.img_np.shape[2] / self.x_scale)*2)+10,y=int((self.img_np.shape[1] / self.y_scale)*2)+70)
        self.scale.config(length=int((self.img_np.shape[1] / self.y_scale)*2))
        self.master.geometry('%dx%d' % (self.window_width, self.window_height))
        img_mask_render = ImageTk.PhotoImage(img_mask)
        self.img_canvas.config(width=int((self.img_np.shape[2] / self.x_scale)*2), height=int((self.img_np.shape[1] / self.y_scale)*2))

        self.img_canvas.image = img_mask_render
        self.img_canvas.create_image(0, 0, anchor=NW, image=img_mask_render)
        self.scale.place(x=int((self.img_np.shape[2] / self.x_scale)*2)+20, y=60)



if __name__ == "__main__":
    root = Tk()
    app = UI(root)
    root.mainloop()