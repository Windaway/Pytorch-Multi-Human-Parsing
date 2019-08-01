import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlocks(nn.Module):
    def __init__(self,inc,outc,stride=2):
        super(ConvBlocks, self).__init__()
        self.in0=nn.InstanceNorm2d(inc)
        self.c1=nn.Conv2d(inc,outc,3,padding=(1,1), bias=False)
        self.in1=nn.InstanceNorm2d(outc)
        self.c2=nn.Conv2d(outc,outc,3,stride,padding=(1,1),bias=False)
        self.c3=nn.Conv2d(inc,outc,1,stride,padding=0,bias=False)
    def forward(self, n):
        n=self.in0(n)
        n=F.relu(n)
        short=self.c3(n)
        n=self.c1(n)
        n=self.in1(n)
        n=F.relu(n)
        n=self.c2(n)
        n=n+short
        return n

class IdentityBlock(nn.Module):
    def __init__(self,channels):
        super(IdentityBlock,self).__init__()
        self.in0=nn.InstanceNorm2d(channels)
        self.c1=nn.Conv2d(channels,channels,3,1,padding=(1,1),bias=False)
        self.in1=nn.InstanceNorm2d(channels)
        self.c2=nn.Conv2d(channels,channels,3,1,padding=(1,1),bias=False)

    def forward(self, x):
        short=x
        n=self.in0(x)
        n=self.c1(n)
        n=self.in1(n)
        n=F.relu(n)
        n=self.c2(n)
        n=n+short
        return n

class Block5(nn.Module):
    def __init__(self,channels):
        super(Block5, self).__init__()
        self.in0=nn.InstanceNorm2d(channels)
        self.c1=nn.Conv2d(channels,512,3,1,(1,1),bias=False)
        self.in1=nn.InstanceNorm2d(channels)
        self.c2=nn.Conv2d(512,1024,3,1,(2,2),2,bias=False)
        self.c3=nn.Conv2d(channels,1024,1,bias=False)
    def forward(self, x):
        n=self.in0(x)
        n=F.relu(n)
        short=self.c3(n)
        n=self.c1(n)
        n=self.in1(n)
        n=F.relu(n)
        n=self.c2(n)
        n=short+n
        return n

class ID5Block(nn.Module):
    def __init__(self):
        super(ID5Block, self).__init__()
        self.in0=nn.InstanceNorm2d(1024)
        self.c1=nn.Conv2d(1024,512,3,1,(2,2),2,bias=False)
        self.in1=nn.InstanceNorm2d(512)
        self.c2=nn.Conv2d(512,1024,3,1,(2,2),2,bias=False)
    def forward(self, x):
        short=x
        n=self.in0(x)
        n=F.relu(n)
        n=self.c1(n)
        n=self.in1(n)
        n=F.relu(n)
        n=self.c2(n)
        n=n+short
        return n

class LastBlock(nn.Module):
    def __init__(self,inc,fmaps,dialation_rate):
        super(LastBlock, self).__init__()
        self.dialation=dialation_rate
        self.in0=nn.InstanceNorm2d(inc)
        self.c1=nn.Conv2d(inc,fmaps[0],1,bias=False)
        self.in1=nn.InstanceNorm2d(fmaps[0])
        self.c2=nn.Conv2d(fmaps[0],fmaps[1],3,1,padding=(self.dialation,self.dialation),dilation=self.dialation,bias=False)
        self.in2=nn.InstanceNorm2d(fmaps[1])
        self.c3=nn.Conv2d(fmaps[1],fmaps[2],1,bias=False)
        self.c4=nn.Conv2d(inc,fmaps[2],1,bias=False)
    def forward(self, x):
        n=self.in0(x)
        n=F.relu(n)
        short=self.c4(n)
        n=self.c1(n)
        n=self.in1(n)
        n=F.relu(n)
        n=self.c2(n)
        n=self.in2(n)
        n=F.relu(n)
        n=self.c3(n)
        n=n+short
        return n

class MainBody(nn.Module):
    def __init__(self):
        super(MainBody, self).__init__()
        #512
        self.c0=nn.Conv2d(3,64,3,1,1,bias=False)
        #512
        self.r1_0 = ConvBlocks(64,64)
        self.r1_1 = IdentityBlock(64)
        self.r1_2 = IdentityBlock(64)
        #256
        self.r2_0 = ConvBlocks(64,128)
        self.r2_1 = IdentityBlock(128)
        self.r2_2 = IdentityBlock(128)
        #128
        self.r3_0 = ConvBlocks(128,256)
        self.r3_1 = IdentityBlock(256)
        self.r3_2 = IdentityBlock(256)
        self.r3_3 = IdentityBlock(256)
        self.r3_4 = IdentityBlock(256)
        self.r3_5 = IdentityBlock(256)
        #64
        self.r4_0 = Block5(256)
        self.r4_1 = ID5Block()
        self.r4_2 = ID5Block()
        self.r4_3 = LastBlock(1024,[512, 1024, 2048], 4)
        self.r4_4 = LastBlock(2048,[1024, 2048, 4096], 4)
        self.bn4 = nn.InstanceNorm2d(4096)
        self.c4 = nn.Conv2d(4096, 512, 3, dilation=12,padding=12, bias=False)

    def forward(self, x):
        c0=x=self.c0(x)
        x = self.r1_0(x)
        x = self.r1_1(x)
        c1 = x = self.r1_2(x)
        x = self.r2_0(x)
        x = self.r2_1(x)
        c2 = x = self.r2_2(x)
        x = self.r3_0(x)
        x = self.r3_1(x)
        x = self.r3_2(x)
        x = self.r3_3(x)
        x = self.r3_4(x)
        c3 = x = self.r3_5(x)
        x = self.r4_0(x)
        x = self.r4_1(x)
        x = self.r4_2(x)
        x = self.r4_3(x)
        x = self.r4_4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.c4(x)
        c4 = x =F.relu(x)
        return c0,c1,c2,c3,c4

BaseMainBody=MainBody()

class gSeg(nn.Module):
    '''FrontGround and BackGround Seg'''
    def __init__(self,class_num=2,dilation_rate=1):
        super(gSeg, self).__init__()
        #512
        self.body=BaseMainBody
        #64
        self.seg_layer=nn.Conv2d(512,class_num,3,1,dilation_rate,dilation_rate)
        self.upsample=nn.UpsamplingBilinear2d(scale_factor=4)
        #256
    def forward(self, x):
        n=self.body(x)
        n=self.seg_layer(n[-1])
        n=self.upsample(n)
        n=torch.sigmoid(n)
        return n

class gParseSeg(nn.Module):
    def __init__(self,class_num=59, dilation_rate=1):
        super(gParseSeg, self).__init__()
        self.merging=nn.Conv2d(4,3,1,bias=False)
        self.body=BaseMainBody
        self.seg_layer=nn.Conv2d(512,class_num,3,1,dilation_rate,dilation_rate)
        self.upsample=nn.UpsamplingBilinear2d(scale_factor=4)
    def forward(self, x):
        n=self.merging(x)
        n=self.body(n)
        n=self.seg_layer(n[-1])
        n=self.upsample(n)
        n=torch.softmax(n,1)
        return n

class gInSegModule(nn.Module):
    def __init__(self,channels):
        super(gInSegModule, self).__init__()
        self.c1 = nn.Conv2d(channels, 128,3,1,1)
        self.c2 = nn.Conv2d(128, 128,3,1,1)
        self.c3 = nn.Conv2d(channels, 6,3,1,1)
    def forward(self, x):
        n=self.c1(x)
        n=F.relu(n)
        n=self.c2(n)
        n=F.relu(n)
        n=self.c3(n)
        return n

class gInSeg(nn.Module):
    def __init__(self,channels):
        super(gInSeg, self).__init__()
        self.merging=nn.Conv2d(channels,3,1,bias=False)
        self.body=BaseMainBody
        self.streams=nn.ModuleList([gInSegModule(x) for x in [128,256,256,512]])
        self.streans_fusion=nn.Conv2d(24,6,1,bias=False)

    def forward(self, x,y,z):
        n=torch.cat([x,y,z],1)
        n=self.merging(n)
        features=self.body(n)
        features=features[1:]
        streams_outputs=[]
        sample_scale=[2,4,8,8]
        for i,f in enumerate(features):
            stream=self.streams[i](f)
            stream=F.upsample_bilinear(stream,scale_factor=sample_scale[i])
            streams_outputs.append(stream)
        stream_fusion=self.streans_fusion(torch.cat(streams_outputs,1))
        streams_outputs.append(stream_fusion)
        return streams_outputs


class aSeg(nn.Module):
    def __init__(self):
        super(aSeg, self).__init__()
        self.c0=nn.Conv2d(1,16,5,1,2)
        self.c1=nn.Conv2d(16,32,5,2,2)
        self.c2=nn.Conv2d(32,64,5,1,2)
        self.c3=nn.Conv2d(64,128,5,2,2)
        self.c4=nn.Conv2d(128,128,3,2,1)
        self.c5=nn.Conv2d(128,128,3,2,1)
        self.c6=nn.Conv2d(128,128,3,2,1)
        self.c7=nn.Conv2d(128,128,5,1,0)
        self.c8=nn.Conv2d(128,1,4,1,0)

    def forward(self, x):
        #256
        n=self.c0(x)
        n=F.relu(n)
        n=self.c1(n)
        n=F.relu(n)
        #128
        n=self.c2(n)
        n=F.relu(n)
        n=self.c3(n)
        n=F.relu(n)
        #64
        n=self.c4(n)
        n=F.relu(n)
        #32
        n=self.c5(n)
        n=F.relu(n)
        #16
        n=self.c6(n)
        n=F.relu(n)
        #8
        n=self.c7(n)
        n=F.relu(n)
        #4
        n=self.c8(n)
        n=torch.sigmoid(n)
        n=n.view(-1)
        return n
class aParseSeg(nn.Module):
    def __init__(self):
        super(aParseSeg, self).__init__()
        self.c0=nn.Conv2d(60,16,5,1,2)
        self.c1=nn.Conv2d(16,32,5,2,2)
        self.c2=nn.Conv2d(32,64,5,1,2)
        self.c3=nn.Conv2d(64,128,5,2,2)
        self.c4=nn.Conv2d(128,128,3,2,1)
        self.c5=nn.Conv2d(128,128,3,2,1)
        self.c6=nn.Conv2d(128,128,3,2,1)
        self.c7=nn.Conv2d(128,128,5,1,0)
        self.c8=nn.Conv2d(128,1,4,1,0)

    def forward(self, x):
        #256
        n=self.c0(x)
        n=F.relu(n)
        n=self.c1(n)
        n=F.relu(n)
        #128
        n=self.c2(n)
        n=F.relu(n)
        n=self.c3(n)
        n=F.relu(n)
        #64
        n=self.c4(n)
        n=F.relu(n)
        #32
        n=self.c5(n)
        n=F.relu(n)
        #16
        n=self.c6(n)
        n=F.relu(n)
        #8
        n=self.c7(n)
        n=F.relu(n)
        #4
        n=self.c8(n)
        n=torch.sigmoid(n)
        n=n.view(-1)
        return n

class aInSeg(nn.Module):
    def __init__(self):
        super(aInSeg, self).__init__()

    def forward(self, *input):
        pass


if __name__=='__main__':

    a1=torch.rand([1,3,512,512])
    a2=torch.rand([1,1,512,512])
    # b=nn.Conv2d(1,1,1,1,(1,1,0,0),bias=False)
    b=gParseSeg(2,1)
    c0=b(a1,a2)

    # c=b(a)
    print(c0.shape)