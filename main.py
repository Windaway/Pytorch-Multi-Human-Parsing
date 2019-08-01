import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from yadataset import *
from utils import *
import torch.utils
import torch.optim as optim
from torch.optim import lr_scheduler
from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_integer("prt_num_epochs", 5,
                     "Number of times to run pretrain.")
flags.DEFINE_integer("step1_num_epochs", 20,
                     "Number of times to run step1 train.")
flags.DEFINE_integer("step2_num_epochs", 20,
                     "Number of times to run step2 train.")
flags.DEFINE_integer("stepe2e_num_epochs", 20,
                     "Number of times to run stepe2e train.")
flags.DEFINE_string("save_path", './model/',
                     "Path to save.")
flags.DEFINE_integer("batchsize",4,
                     "bs")

def main(argv):
    device = torch.device("cuda:0")
    trainset = MHPdataset(root='C:/LV-MHP-v2/train/', mode='train', scale=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS.batchsize, num_workers=12, shuffle=True)
    valset = MHPdataset(root='C:/LV-MHP-v2/val/', mode='val', scale=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, num_workers=6, shuffle=False)
    bgfg_net = gSeg(1)
    seg_net = gParseSeg(59)
    aseg_net=aSeg()
    aparseseg_net=aParseSeg()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        bgfg_net = nn.DataParallel(bgfg_net)
        seg_net = nn.DataParallel(seg_net)
        aseg_net = nn.DataParallel(aseg_net)
        aparseseg_net = nn.DataParallel(aparseseg_net)
    bgfg_net.to(device)
    seg_net.to(device)
    aseg_net.to(device)
    aparseseg_net.to(device)
    optim_bgfg=optim.SGD(bgfg_net.parameters(), lr=0.001, momentum=0.9)
    optim_seg=optim.SGD(seg_net.parameters(), lr=0.001, momentum=0.9)
    optim_base=optim.SGD(BaseMainBody.parameters(), lr=0.001, momentum=0.9)
    optim_aseg=optim.Adam(aseg_net.parameters(), lr=0.001)
    optim_aparseseg=optim.Adam(aparseseg_net.parameters(), lr=0.001)
    dice_loss=DiceLoss()
    class_loss=ClassLoss()
    aseg_loss=nn.BCELoss()
    aparse_loss=nn.BCELoss()

    for epoch in range(FLAGS.prt_num_epochs):
        print('Epoch {}/{}'.format(epoch+1, FLAGS.prt_num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            running_loss = 0.0
            if phase == 'train':
                for i,data in enumerate(trainloader):
                    image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                    image = image.to(device)
                    segs = seg.to(device)
                    optim_bgfg.zero_grad()
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = bgfg_net(image)
                        _, preds = torch.max(outputs, 1)
                        loss = dice_loss(outputs, segs)
                        running_loss += loss.item()
                        if phase == 'train':
                            loss.backward()
                            optim_bgfg.step()
                epoch_loss = running_loss / len(trainset)
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            else:
                with torch.set_grad_enabled(phase=='train'):
                    for i, data in enumerate(valloader):
                        image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                        image = image.to(device)
                        segs = seg.to(device)
                        outputs = bgfg_net(image)
                        loss = dice_loss(outputs, segs)
                        running_loss += loss.item()
                    epoch_loss = running_loss / len(valset)
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    torch.save(bgfg_net, FLAGS.save_path+'pretrain')
    optim_bgfg=optim.Adam(bgfg_net.parameters(), lr=0.001)


    for epoch in range(FLAGS.step1_num_epochs):
        print('Epoch {}/{}'.format(epoch+1, FLAGS.step1_num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            running_loss = 0.0
            if phase == 'train':
                for i,data in enumerate(trainloader):
                    optim_bgfg.zero_grad()
                    optim_aseg.zero_grad()
                    image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                    image = image.to(device)
                    segs = seg.to(device)
                    a_seg_label=torch.cat([torch.ones(FLAGS.batchsize),torch.zeros(FLAGS.batchsize)],-1)
                    a_seg_label=a_seg_label.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = bgfg_net(image)
                        seg_loss = dice_loss(outputs, segs)
                        a_seg_input=torch.cat([F.interpolate(segs,scale_factor=0.5),outputs],0)
                        a_seg_output=aseg_net(a_seg_input)
                        a_seg_loss=aseg_loss(a_seg_output,a_seg_label)
                        loss=a_seg_loss+seg_loss
                        loss.backward()
                        optim_aseg.step()
                        optim_bgfg.step()
            else:
                with torch.set_grad_enabled(phase == 'train'):
                    for i, data in enumerate(valloader):
                        image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                        image = image.to(device)
                        segs = seg.to(device)
                        outputs = bgfg_net(image)
                        loss = dice_loss(outputs, segs)
                        running_loss += loss.item()
                    epoch_loss = running_loss / len(valset)
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    torch.save(bgfg_net, FLAGS.save_path+'pretrain')
    optim_bgfg=optim.Adam(bgfg_net.parameters(), lr=0.001)

    for epoch in range(FLAGS.prt_num_epochs):
        print('Epoch {}/{}'.format(epoch+1, FLAGS.prt_num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            running_loss = 0.0
            if phase == 'train':
                for i,data in enumerate(trainloader):
                    optim_bgfg.zero_grad()
                    optim_aseg.zero_grad()
                    image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                    image = image.to(device)
                    parse= parse.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        bgfgouts = bgfg_net(image)
                        bgfgouts=torch.cat([F.interpolate(bgfgouts,scale_factor=2.),image],1)
                        outputs = seg_net(bgfgouts)
                        loss = class_loss(outputs, parse)
                        loss.backward()
                        optim_bgfg.zero_grad()
                        optim_seg.step()
            else:
                with torch.set_grad_enabled(phase=='train'):
                    for i, data in enumerate(valloader):
                        image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                        image = image.to(device)
                        parse = parse.to(device)
                        bgfgout = bgfg_net(image)
                        bgfgouts=torch.cat([ F.interpolate(bgfgout,scale_factor=2.),image],1)
                        outputs = seg_net(bgfgouts)
                        loss = class_loss(outputs, parse)
                        running_loss += loss.item()
                    epoch_loss = running_loss / len(valset)
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))


    optim_seg=optim.Adam(bgfg_net.parameters(), lr=0.001)
    for epoch in range(FLAGS.step2_num_epochs):
        print('Epoch {}/{}'.format(epoch+1, FLAGS.step2_num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            running_loss = 0.0
            if phase == 'train':
                for i,data in enumerate(trainloader):
                    optim_bgfg.zero_grad()
                    optim_aseg.zero_grad()
                    image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                    image = image.to(device)
                    segs=seg.to(device)
                    parse=parse.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        bgfgout = bgfg_net(image)
                        bgfgouts=torch.cat([F.interpolate(bgfgout,scale_factor=2),image],1)
                        outputs = seg_net(bgfgouts)
                        a_parse_label = torch.cat([torch.ones(FLAGS.batchsize), torch.zeros(FLAGS.batchsize)], -1)
                        a_parse_label = a_parse_label.to(device)
                        parse_onehot=torch.zeros(FLAGS.batchsize, 59,512,512).to(device).scatter(1, torch.Tensor.long(parse)   ,1)
                        a_parse_inputt=torch.cat([segs,parse_onehot],1)
                        a_parse_inputf=torch.cat([bgfgout,outputs],1)
                        a_parse_input=torch.cat([F.interpolate(a_parse_inputt,scale_factor=0.5),a_parse_inputf],0)
                        a_parse_output=aparseseg_net(a_parse_input)
                        closs = class_loss(outputs, parse)
                        a_parse_loss=aparse_loss(a_parse_output,a_parse_label)
                        loss=closs+a_parse_loss
                        loss.backward()
                        optim_base.zero_grad()
                        optim_aparseseg.step()
                        optim_seg.step()
            else:
                with torch.set_grad_enabled(phase=='train'):
                    for i, data in enumerate(valloader):
                        image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                        image = image.to(device)
                        parse = parse.to(device)
                        bgfgout = bgfg_net(image)
                        bgfgouts=torch.cat([ F.interpolate(bgfgout,scale_factor=2.),image],1)
                        outputs = seg_net(bgfgouts)
                        loss = class_loss(outputs, parse)
                        running_loss += loss.item()
                        print('pass1 train')
                        break
                    epoch_loss = running_loss / len(valset)
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    #fin
    optim_bgfg=optim.Adam(bgfg_net.parameters(), lr=0.001)
    optim_seg=optim.Adam(seg_net.parameters(), lr=0.001)
    optim_base=optim.Adam(BaseMainBody.parameters(), lr=0.001)
    optim_aseg=optim.Adam(aseg_net.parameters(), lr=0.001)
    optim_aparseseg=optim.Adam(aparseseg_net.parameters(), lr=0.001)

    for epoch in range(FLAGS.stepe2e_num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, FLAGS.stepe2e_num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            running_loss = 0.0
            if phase == 'train':
                for i, data in enumerate(trainloader):
                    optim_bgfg.zero_grad()
                    optim_aseg.zero_grad()
                    image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                    image = image.to(device)
                    segs = seg.to(device)
                    parse = parse.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        bgfgout = bgfg_net(image)
                        bgfgouts = torch.cat([ F.interpolate(bgfgout,scale_factor=2.) , image],1)
                        outputs = seg_net(bgfgouts)
                        a_seg_label = torch.cat([torch.ones(FLAGS.batchsize), torch.zeros(FLAGS.batchsize)], -1)
                        a_seg_label = a_seg_label.to(device)
                        a_parse_label = torch.cat([torch.ones(FLAGS.batchsize), torch.zeros(FLAGS.batchsize)], -1)
                        a_parse_label = a_parse_label.to(device)
                        a_seg_input=torch.cat([F.interpolate(segs,scale_factor=0.5) ,bgfgout],0)
                        a_seg_output=aseg_net(a_seg_input)
                        parse_onehot=torch.zeros(FLAGS.batchsize, 59,512,512).to(device).scatter(1,  torch.Tensor.long(parse),1)
                        a_parse_inputt = torch.cat([segs, parse_onehot], 1)
                        a_parse_inputf = torch.cat([bgfgout, outputs], 1)
                        a_parse_input = torch.cat([ F.interpolate(a_parse_inputt,scale_factor=0.5) , a_parse_inputf], 0)
                        a_parse_output = aparseseg_net(a_parse_input)
                        bfgfloss = dice_loss(bgfgout, segs)
                        closs = class_loss(outputs, parse)
                        a_parse_loss = aparse_loss(a_parse_output, a_parse_label)
                        a_seg_loss = aseg_loss(a_seg_output, a_seg_label)
                        loss = closs + a_parse_loss+bfgfloss+a_seg_loss
                        loss.backward()
                        optim_aparseseg.step()
                        optim_aseg.step()
                        optim_bgfg.step()
                        optim_base.zero_grad()
                        optim_seg.step()
            else:
                with torch.set_grad_enabled(phase == 'train'):
                    for i, data in enumerate(valloader):
                        image, seg, parse, instance, count1, c2, name, (img_w, img_h), out_size = data
                        image = image.to(device)
                        segs = seg.to(device)
                        parse = parse.to(device)
                        bgfgout = bgfg_net(image)
                        bgfgouts = torch.cat([F.interpolate(bgfgout,scale_factor=2.), image],1)
                        outputs = seg_net(bgfgouts)
                        bfgfloss = dice_loss(bgfgout, segs)
                        closs = class_loss(outputs, parse)
                        loss = closs+bfgfloss
                        running_loss += loss.item()
                        print('pass1 train')
                        break
                    epoch_loss = running_loss / len(valset)
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    torch.save(bgfg_net, FLAGS.save_path+'bg')
    torch.save(seg_net, FLAGS.save_path+'sg')






if __name__ == '__main__':
    app.run(main)
