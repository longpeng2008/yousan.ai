#coding:utf8
import time
import cv2
import torch 
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='./', help='preTrained model')
parser.add_argument('--imagedir', default='images', help='test image dir')
parser.add_argument('--resultdir', default='results', help='result image dir')
parser.add_argument('--size', type=int, default=256, help='input size')
parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

args = parser.parse_args()
torch.set_grad_enabled(False)

#################################
if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("----------------------------------------------------------")
        print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
        print("----------------------------------------------------------")
        device = torch.device('cuda:0,1')

#################################
def load_model(args):
    print('Loading model from {}...'.format(args.model))
    if args.without_gpu:
        myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(args.model)

    myModel.eval()
    myModel.to(device)
    
    return myModel

## 处理主代码
def seg_process(args, image, net):
    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (args.size,args.size), interpolation=cv2.INTER_CUBIC)
    image_clone = image_resize.copy().astype(np.float)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0

    tensor_4D = torch.FloatTensor(1, 3, args.size, args.size)
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)

    t0 = time.time()

    ## 模型预测
    trimap, alpha = net(inputs)

    print(str(time.time() - t0)+' s')

    if args.without_gpu:
        alpha_np = alpha[0,0,:,:].data.numpy()
        trimap_np = trimap[0,:,:,:].data.numpy()
    else:
        alpha_np = alpha[0,0,:,:].cpu().data.numpy()
        trimap_np = trimap[0,:,:,:].cpu().data.numpy()

    print("min max alpha="+str(np.min(alpha_np))+' '+str(np.max(alpha_np)))

    alpha_np[alpha_np>1] = 1
    alpha_np[alpha_np<0] = 0

    fg = np.multiply(alpha_np[..., np.newaxis], image_clone)
    fg = fg.astype(np.uint8)

    return alpha_np,fg,trimap_np

def main(args):
    torch.no_grad()

    myModel  = torch.load(args.model, map_location=lambda storage, loc: storage)
    myModel.eval()
    #myModel = load_model(args)
    #dummy_input = torch.randn((1,3,256,256))
    #torch.onnx.export(myModel, dummy_input, "model.onnx", verbose=True)

    cv2.namedWindow("result",0)
    images = os.listdir(args.imagedir)
    for imagepath in images:
        imageid = imagepath.split('.')[0]
        image = cv2.imread(os.path.join(args.imagedir,imagepath))
        imagealpha,imagematting,imagetrimap = seg_process(args, image, myModel)
        imagealpha = (imagealpha*255).astype(np.uint8)
        imagemask = (imagealpha > 127)
        trimaplabel = np.argmax(imagetrimap, axis=0)
        print('trimaplabel'+str(np.unique(trimaplabel)))

        image = cv2.resize(image, (args.size,args.size), interpolation=cv2.INTER_CUBIC)
        imageseg = np.multiply(imagemask[..., np.newaxis], image)
        imagemask = imagemask * 255

        cv2.imwrite(os.path.join(args.resultdir,imageid+'_matting.png'),imagematting)
        cv2.imwrite(os.path.join(args.resultdir,imageid+'_alpha.png'),imagealpha)
        cv2.imwrite(os.path.join(args.resultdir,imageid+'_mask.png'),imagemask)
        cv2.imwrite(os.path.join(args.resultdir,imageid+'_seg.png'),imageseg)
        cv2.imwrite(os.path.join(args.resultdir,imageid+'_trimap.png'),trimaplabel*127)

        ##cv2.imshow("result",np.concatenate((image,imagematting),axis=1))
        ##k = cv2.waitKey(10)
        ##if k == ord('q'):
            ##break
     
if __name__ == "__main__":
    main(args)

     

