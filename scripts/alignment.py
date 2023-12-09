import math 
import numpy as np 
from numpy.linalg import norm, inv, pinv 
import cv2 
from PIL import Image

import torch 
from torch.autograd import Variable 

def main(depth_path, image_path): 
    # new_size = (160, 120)
    new_size = (640, 480)
    depth = np.array(Image.open(depth_path).resize(new_size)) / 255 * 30 
    img = np.array(Image.open(image_path).resize(new_size))

    # intrinsics
    w = int(img.shape[1])
    h = int(img.shape[0])
    u_0 = w / 2
    v_0 = h / 2 
    vfovd = 55 
    f = v_0 / math.tan((vfovd / 2) * math.pi / 180.0) 
    cx = u_0 
    cy = v_0 
    K = np.array([[f,0,cx],[0,f,cy],[0,0,1]])
    iK = inv(K) 

    # meshgrid of source image 
    u = np.linspace(0, w-1, w)
    v = np.linspace(0, h-1, h) 
    uv, vv = np.meshgrid(u, v) 
    # shape (3, h, w)
    img_coords = np.transpose(
            np.dstack((uv, vv, np.ones((h, w)))), (2, 0, 1)
    )
    # shape (3, h*w)
    img_coords_flat = img_coords.reshape((3, -1))

    # back projection 
    x_hat = np.dot(iK, img_coords_flat) 
    norms = norm(x_hat, axis=0) 
    Xt = np.vstack((x_hat * 1/norms * depth.reshape(1, h*w), 
            np.ones((1, h*w))
        ))
    
    x_ = Xt[:3, :] 
    p_ = np.zeros((6, 1)) 
    p_[5] -= 3.0
    p = Variable(torch.Tensor(p_), requires_grad=True) 
    x = Variable(torch.Tensor(x_)) 
    K = Variable(torch.Tensor(K))
    
    # angle axis to rotation matrix 
    theta = torch.norm(p[:3])
    r = p[:3] / (theta.expand_as(p[:3]) + 1e-5) 
    cos_theta = torch.cos(theta).expand_as(torch.eye(3)) 
    one_minus_cos_theta = (1 - torch.cos(theta)).expand_as(torch.eye(3)) 
    sin_theta = torch.sin(theta).expand_as(torch.eye(3)) 

    rx = r[0].expand_as(torch.eye(3))
    ry = r[1].expand_as(torch.eye(3))
    rz = r[2].expand_as(torch.eye(3))

    Rx = Variable(torch.Tensor([[0,0,0],[0,0,-1],[0,1,0]]))*rx
    Ry = Variable(torch.Tensor([[0,0,1],[0,0,0],[-1,0,0]]))*ry
    Rz = Variable(torch.Tensor([[0,-1,0],[1,0,0],[0,0,0]]))*rz

    R1 = cos_theta * Variable(torch.eye(3))
    R2 = one_minus_cos_theta * torch.ger(r.squeeze(), r.squeeze())
    R3 = (Rx+Ry+Rz) * sin_theta

    R = R1+R2+R3

    x2 = torch.mm(R, x) + p[3:].expand_as(x) 

    x3 = torch.mm(K, x2) 
    piv = x3[1] / x3[2] 
    piu = x3[0] / x3[2] 
    P = torch.vstack((piu, piv)) 

    accum = np.zeros((h, w))

    P_reshape = P.reshape(2, h, w)
    Py = P_reshape[1].detach().numpy().astype(np.float32)
    Px = P_reshape[0].detach().numpy().astype(np.float32)
    accum = cv2.remap(img, Px, Py, cv2.INTER_LINEAR)

    pgrad = []
    for pi in [piu, piv]: 
        m = x_.shape[1]
        Jacobian = torch.Tensor(m, 6).zero_()
        for i in range(m): 
            grad_mask = torch.zeros(m)
            grad_mask[i] = 1 
            pi.backward(grad_mask, retain_graph=True)
            Jacobian[i, :] = p.grad.data.squeeze() 
            p.grad.data.zero_()
        pgrad.append(Jacobian)

    # calculate image gradient 
    graduimg = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=int(31))
    gradvimg = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=int(31))    
    wGradu_flat = graduimg.reshape((1,-1)).squeeze()
    wGradv_flat = gradvimg.reshape((1,-1)).squeeze()

    pgrad_u = pgrad[0]
    pgrad_v = pgrad[1]
    Wp = np.vstack([pgrad_u.numpy(), pgrad_v.numpy()])
    delI = np.hstack([np.diag(wGradu_flat), np.diag(wGradv_flat)])

    Jac = delI.dot(Wp)

    # solve for parameter update
    # residual error
    r = (accum - img).reshape(-1,1)
    delP = pinv(Jac).dot(r)
    p_ = p_ - delP


if __name__ == "__main__":
    # depth is from image2 
    depth_path = "./data/depth.png"
    image_path = "./data/image1.png"
    # image_path = "./data/image2.png"
    main(depth_path, image_path)