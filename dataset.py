import os
import struct
import torch
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    def __init__(self, mod='train'):
        path = r'./dataset/'
        files = [
            'train-images.idx3-ubyte',
            'train-labels.idx1-ubyte',
            't10k-images.idx3-ubyte',
            't10k-labels.idx1-ubyte'
        ]
        self.mod = mod
        
        if mod == 'train':
            self.imgs = self.decode_idx3(file=path + files[0])
            self.labels = self.decode_idx1(file=path + files[1])
        elif mod == 'evl':
            ...
        elif mod == 'test':
            self.imgs = self.decode_idx3(file=path + files[2])
            self.labels = self.decode_idx1(file=path + files[3])
        elif mod == 'recognize':
            self.imgs, self.img_names = self.loadUnlabeledImgFiles()
    
    def __getitem__(self, index):
        if self.mod == 'recognize':
            return self.imgs[index], self.img_names[index]
        return self.imgs[index], self.labels[index]
    
    def __len__(self):
        return len(self.imgs)
    
    def decode_idx3(self, file):
        with open(file, 'rb') as fp:
            bin_data = fp.read()
            
            # 解析文件中的头信息
            # 从文件头部依次读取四个32位，分别为：
            # magic，numImgs, numRows, numCols
            # 偏置
            offset = 0
            # 读取格式: 大端
            # fmt_header = '>iiii'
            fmt_header = '>4i'
            magic, numImgs, numRows, numCols = struct.unpack_from(fmt_header, bin_data, offset)
            
            print('image ubyte: ')
            print(magic, numImgs, numRows, numCols)
            
            # 解析图片数据
            # 偏置掉头文件信息
            offset = struct.calcsize(fmt_header)
            # 读取格式
            fmt_image = '>' + str(numImgs * numRows * numCols) + 'B'
            data = torch.tensor(struct.unpack_from(fmt_image, bin_data, offset), dtype=torch.float) \
                .reshape(numImgs, numRows, numCols)
            
            print('image shape: ' + str(data.shape))
            
            return data
    
    def decode_idx1(self, file):
        with open(file, 'rb') as fp:
            bin_data = fp.read()
            
            offset = 0
            fmt_header = '>2i'
            
            magic, numLabel = struct.unpack_from(fmt_header, bin_data, offset)
            
            print('label ubyte: ')
            print(magic, numLabel)
            
            offset = struct.calcsize(fmt_header)
            
            fmt_label = '>' + str(numLabel) + 'B'
            
            data = torch.tensor(struct.unpack_from(fmt_label, bin_data, offset)).reshape(numLabel)
            
            print('label shape: ' + str(data.shape))
            
            return data
    
    def loadUnlabeledImgFiles(self):
        imgs_names_arr = os.listdir(os.getcwd() + '/dataset/unlabeled_imgs/')
        imgs = torch.empty(0, 28, 28)
        for img_name in imgs_names_arr:
            with open(r'dataset/unlabeled_imgs/'+img_name, 'rb') as fp:
                f_type = str(fp.read(2))  # 文件类型 2个字节
                file_size_byte = fp.read(4)  # 文件的大小 4个字节
                fp.seek(fp.tell() + 4)  # 无用的四个字节
                file_ofset_byte = fp.read(4)  # 读取位图数据的偏移量
                fp.seek(fp.tell() + 4)  # 无用的两个字节
                file_wide_byte = fp.read(4)  # 读取宽度字节
                file_height_byte = fp.read(4)  # 读取高度字节
                fp.seek(fp.tell() + 2)  # 无用的两个字节
                file_bitcount_byte = fp.read(4)  # 得到每个像素占位大小
        
                # 将读取的字节转换成指定的类型
                f_size, = struct.unpack('l', file_size_byte)
                f_offset, = struct.unpack('l', file_ofset_byte)
                f_wide, = struct.unpack('l', file_wide_byte)
                f_height, = struct.unpack('l', file_height_byte)
                f_bitcount, = struct.unpack('i', file_bitcount_byte)
                print("Name: "+img_name+", 类型:", f_type, "大小:", f_size, "位图数据偏移量:", f_offset,
                      "宽度:", f_wide, "高度:", f_height, "位图:", f_bitcount)
        
                img = torch.tensor(
                    struct.unpack_from(str(28 * 28) + 'B', open('dataset/unlabeled_imgs/'+img_name, 'rb').read(), f_offset),
                    dtype=torch.float).reshape(28, 28)
                
                # plt.plot(img)
                # plt.show()
                
                img = torch.unsqueeze(img, 0)
                imgs = torch.cat((imgs, img), 0)
                
                fp.close()
        return imgs, imgs_names_arr


def prep_loader(mod, batch_size, n_jobs=0):
    ds = MNISTDataset(mod=mod)

    if mod == 'recognize':
        return DataLoader(
            ds, batch_size, shuffle=False,
            num_workers=n_jobs, pin_memory=True
        )
        
    loader = DataLoader(
        ds, batch_size=batch_size,
        shuffle=(mod == 'train'),
        drop_last=False,
        num_workers=n_jobs, pin_memory=True
    )
    return loader
