import torch
import torchvision.transforms as transforms
from torch.utils import data
import os
import scipy.io as scio
from PIL import Image
import numpy as np
import re
# import imageio
# import imgaug as ia
# import imgaug.augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapOnImage
# ia.seed(1)
import cv2

class MHPdataset(data.Dataset):
    def __init__(self, root='./',scale=True,keep_wh=True, file_list=None,mode='train',rand_sacle=(0.7,1.5,0.8,1.3,0.2)):
        super(MHPdataset, self).__init__()
        self.root = root
        self.img_path=os.path.join(self.root,'images')
        self.parse_path=os.path.join(self.root,'parsing_annos')
        self.pose_path=os.path.join(self.root,'pose_annos')
        if file_list is not None:
            self.file_list=[]
            with open(file_list,'r') as f:
                while True:
                    line=f.readline().strip()
                    if not line:
                        break
                    self.file_list.append(line)
        else:
            self.file_list = os.listdir(self.img_path)
            for idx,name in enumerate(self.file_list):
                self.file_list[idx]=name[:-4]
        self.mode=mode
        self.img_list=[]

        if self.mode=='train' or self.mode=='val':
            parse_file_list=os.listdir(self.parse_path)
            self.repattern = re.compile(r'^\d+_\d+_')
            self.name_dict={}
            maxcnt=0
            for file in parse_file_list:
                tempstr=self.repattern.match(file)[0]
                cnt=int(tempstr[-3:-1])
                f_name=tempstr[:-4]
                self.name_dict[f_name]=cnt
                if cnt>maxcnt:
                    maxcnt=cnt
            print(maxcnt)


            # self.repattern = re.compile(r'^\d*_')
            for name in self.file_list:
                img_file=os.path.join(self.img_path,"%s.jpg"%name)
                person_pose,person_cnt=self.load_mat(os.path.join(self.pose_path,"%s.mat"%name)   )
                # seg_cnt=0
                # for file in parse_file_list:
                #     if self.repattern.match(file)[0]==name+'_':
                #         seg_cnt+=1
                # parse_file=[ os.path.join(self.parse_path,"%s_%02d_%02d.png"%(name,seg_cnt,x)) for x in range(1,seg_cnt+1)]

                parse_file=[ os.path.join(self.parse_path,"%s_%02d_%02d.png"%(name,self.name_dict[name],x)) for x in range(1,self.name_dict[name]+1)]
                self.img_list.append({
                    "img": img_file,
                    "pose_count":person_cnt,
                    "pose": person_pose,
                    "parse": parse_file,
                    "parse_count": self.name_dict[name]
                })
        else:
            for name in self.file_list:
                img_file=os.path.join(self.img_path,"%s.jpg"%name)
                self.img_list.append({
                    "img": img_file,
                })

        self.size_h, self.size_w = (512,512)
        self.is_scale = scale
        self.keep_wh=keep_wh
        if self.mode=='train':
            # self.seq = iaa.Sequential([
            #     iaa.Dropout([0.05, 0.2]),
            #     iaa.Sharpen((0.0, 1.0)),
            #     iaa.Affine(rotate=(-45, 45)),
            #     iaa.ElasticTransformation(alpha=50, sigma=5)
            # ], random_order=True)
            self.rand_sacle=rand_sacle

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        datafiles = self.img_list[index]

        '''load the datas'''
        if self.mode=='test':
            name = datafiles["img"]
            image = Image.open(name).convert('RGB')
            img_w, img_h = image.size
            image,out_size=self.img_resize(image)
            image = np.asarray(image, np.float32)
            image = image.transpose((2, 0, 1))/255.
            return image,name,(img_w, img_h),out_size
        if self.mode=='val':
            name = datafiles["img"]
            image = Image.open(name).convert('RGB')
            parse_count = datafiles["parse_count"]
            pose_count = datafiles["pose_count"]
            img_w,img_h=image.size
            seg=np.zeros((img_h,img_w),np.uint8)
            instance=np.zeros((img_h,img_w),np.uint8)
            parse=np.zeros((img_h,img_w),np.uint8)
            for p_idx,parse_name in enumerate(datafiles["parse"]):
                seg_img = np.array(Image.open(parse_name).convert('RGB'),np.uint8)[:,:,0]
                seg+=np.array(np.greater(seg_img,0),np.uint8)
                parse+=seg_img
                instance+=np.array(np.greater(seg_img,0),np.uint8)*(p_idx+1)
            seg=Image.fromarray(seg,'L')
            parse=Image.fromarray(parse,'L')
            instance=Image.fromarray(instance,'L')
            image,out_size=self.img_resize(image)
            seg,_=self.seg_resize(seg)
            parse,_=self.seg_resize(parse)
            instance,_=self.seg_resize(instance)
            image = np.asarray(image, np.float32)
            image = image.transpose((2, 0, 1))/255.
            seg = np.asarray(seg, np.float32)[:, : ,np.newaxis]
            seg = seg.transpose((2, 0, 1))
            instance = np.asarray(instance, np.float32)[:, : ,np.newaxis]
            instance = instance.transpose((2, 0, 1))
            parse = np.asarray(parse, np.float32)[:, : ,np.newaxis]
            parse = parse.transpose((2, 0, 1))
            return image,seg,parse,instance,parse_count, pose_count, name, (img_w, img_h), out_size
        if self.mode == 'train':
            name = datafiles["img"]
            image = Image.open(name).convert('RGB')
            parse_count = datafiles["parse_count"]
            pose_count = datafiles["pose_count"]
            img_w, img_h = image.size
            seg = np.zeros((img_h, img_w), np.uint8)
            instance = np.zeros((img_h, img_w), np.uint8)
            parse = np.zeros((img_h, img_w), np.uint8)
            for p_idx, parse_name in enumerate(datafiles["parse"]):
                seg_img = np.array(Image.open(parse_name).convert('RGB'), np.uint8)[:,:,0]
                seg += np.array(np.greater(seg_img, 0), np.uint8)
                parse += seg_img
                instance += np.array(np.greater(seg_img, 0), np.uint8) * (p_idx + 1)
            seg = Image.fromarray(seg, 'L')
            parse = Image.fromarray(parse, 'L')
            instance = Image.fromarray(instance, 'L')
            max_wh = True if img_w / img_h > self.size_w / self.size_h else False
            random_scale1=np.random.rand()
            random_scale2=np.random.rand()
            if max_wh:
                resize_w = int(self.size_w*(self.rand_sacle[0]+(self.rand_sacle[1]-self.rand_sacle[0])*random_scale1))
                resize_h = int(img_h / img_w * self.size_w*(self.rand_sacle[0]+ (self.rand_sacle[1]-self.rand_sacle[0])*random_scale1)*(self.rand_sacle[2]+ (self.rand_sacle[3]-self.rand_sacle[2])*random_scale2))
            else:
                resize_h = int(self.size_h*(self.rand_sacle[0]+(self.rand_sacle[1]-self.rand_sacle[0])*random_scale1))
                resize_w = int(img_w / img_h * self.size_h*(self.rand_sacle[0]+ (self.rand_sacle[1]-self.rand_sacle[0])*random_scale1)*(self.rand_sacle[2]+ (self.rand_sacle[3]-self.rand_sacle[2])*random_scale2))
            img=image.resize((resize_w,resize_h),Image.LANCZOS)
            seg=seg.resize((resize_w,resize_h),Image.NEAREST)
            parse=parse.resize((resize_w,resize_h),Image.NEAREST)
            instance=instance.resize((resize_w,resize_h),Image.NEAREST)
            img=transforms.ColorJitter(0.3,0.3,0.3,0.15)(img)
            if resize_w>self.size_w:
                offset_w=int((resize_w-self.size_w)*np.random.rand())
                crop_w=self.size_w
                pad_w=0
            else:
                offset_w=0
                crop_w=resize_w
                pad_w=self.size_w-resize_w
            if resize_h>self.size_h:
                offset_h=int((resize_h-self.size_h)*np.random.rand())
                crop_h=self.size_h
                pad_h=0
            else:
                offset_h=0
                crop_h=resize_h
                pad_h=self.size_h-resize_h
            img=transforms.functional.crop(img,offset_h, offset_w, crop_h, crop_w)
            seg=transforms.functional.crop(seg,offset_h, offset_w, crop_h, crop_w)
            parse=transforms.functional.crop(parse,offset_h, offset_w, crop_h, crop_w)
            instance=transforms.functional.crop(instance,offset_h, offset_w, crop_h, crop_w)
            is_flip=np.random.randint(0,2)
            if is_flip:
                img=transforms.functional.hflip(img)
                seg=transforms.functional.hflip(seg)
                parse=transforms.functional.hflip(parse)
                instance=transforms.functional.hflip(instance)
            img = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(img)
            seg=transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(seg)
            parse=transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(parse)
            instance=transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(instance)

            image = np.asarray(img, np.float32)
            image = image.transpose((2, 0, 1))/255.
            seg = np.asarray(seg, np.float32)[:, : ,np.newaxis]
            seg = seg.transpose((2, 0, 1))
            instance = np.asarray(instance, np.float32)[:, : ,np.newaxis]
            instance = instance.transpose((2, 0, 1))
            parse = np.asarray(parse, np.float32)[:, : ,np.newaxis]
            parse = parse.transpose((2, 0, 1))
            return image,seg,parse,instance,parse_count, pose_count, name, (img_w, img_h), (crop_w,crop_h)


    def load_mat(self,path):
        #"left-elbow", "left-wrist", "right-hip", "right-knee", "right-ankle", "left-hip", "left-knee", "left-ankle", "head", "neck", "spine" and "pelvis"
        #Each key point has a flag indicating whether it is visible-0/occluded-1/out-of-image-2) and head & instance bounding boxes are also provided to facilitate Multi-Human Pose Estimation research.
        person_dict=scio.loadmat(path)
        persons=[]
        person_count=0
        for key in person_dict.keys():
            if key[:6]=='person':
                person_count+=1
                persons.append(person_dict[key])
        return persons,person_count

    def img_resize(self,img):
        img_w, img_h = img.size
        resize_w=img_w
        resize_h=img_h
        if self.is_scale:
            if not self.keep_wh:
                resize_w = self.size_w
                resize_h = self.size_h
                img = transforms.Resize((self.size_h, self.size_w), Image.LANCZOS)(img)
            else:
                max_wh=True if img_w/img_h >self.size_w/self.size_h else False
                if max_wh:
                    resize_w=self.size_w
                    resize_h=int(img_h/img_w*self.size_w)
                else:
                    resize_h=self.size_h
                    resize_w=int(img_w/img_h*self.size_h)
                pad_w = max(self.size_w - resize_w, 0)
                pad_h = max(self.size_h - resize_h, 0)
                img = transforms.Resize((resize_h, resize_w), Image.LANCZOS)(img)
                img = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(img)
        return img,(resize_w,resize_h)

    def seg_resize(self,img):
        img_w, img_h = img.size
        resize_w=img_w
        resize_h=img_h
        if self.is_scale:
            if not self.keep_wh:
                resize_w = self.size_w
                resize_h = self.size_h
                img = transforms.Resize((self.size_h, self.size_w), Image.NEAREST)(img)
            else:
                max_wh=True if img_w/self.size_w >img_h/self.size_h else False
                if max_wh:
                    resize_w=self.size_w
                    resize_h=int(img_h/img_w*self.size_w)
                else:
                    resize_h=self.size_h
                    resize_w=int(img_w/img_h*self.size_h)
                pad_w = max(self.size_w - resize_w, 0)
                pad_h = max(self.size_h - resize_h, 0)
                img = transforms.Resize((resize_h, resize_w), Image.NEAREST)(img)
                img = transforms.Pad(padding=(0, 0, pad_w, pad_h), fill=0, padding_mode='constant')(img)
        return img,(resize_w,resize_h)



if __name__=='__main__':
    a=MHPdataset(root='G:/LV-MHP-v2/train/',mode='train',scale=True)
    trainloader = data.DataLoader(a, batch_size=1,num_workers=0,shuffle=True)
    # for i, data in enumerate(trainloader):
    #     image,seg,parse,instance,count1,c2, name, (img_w, img_h), out_size = data
    #     temp=image.detach().numpy()
    #     temp=temp[0].transpose((1, 2,0))
    #     temp=np.array(temp,np.uint8)
    #     temp=cv2.cvtColor(temp,cv2.COLOR_RGB2BGR)
    #     # print(temp.shape)
    #     temp2=instance.detach().numpy()
    #     print(np.max(temp2))
    #     # cv2.imshow('temp',temp)
    #     # cv2.waitKey(1000)
    #     # a=np.array(instance)
    #     # print(np.max(a))
    #     # a=np.array(parse)
    #     # print(np.max(a))
    #     # a=np.array(seg)
    #     # print(np.max(a))
    #     #
    #
