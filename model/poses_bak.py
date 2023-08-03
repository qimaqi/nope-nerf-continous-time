import torch
import torch.nn as nn
from model.common import make_c2w, convert3x4_4x4
from model.posenet import TransNet, RotsNet, RotsNet_

class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, cfg, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        # if (cam_id-4)%8 == 0:
        cam_id = int(cam_id)
        r = self.r[cam_id]  # (3, 3) rot angle
        t = self.t[cam_id]  # (3, )
        # print('r', r.size())
        # print('t', t.size())
        c2w = make_c2w(r, t)  # (4, 4)
        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]
        return c2w
    def get_t(self):
       return self.t
   


class LearnPoseNet_couple(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, cfg, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPoseNet_couple, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.cfg = cfg
        self.transnet = TransNet(cfg)
        self.rotsnet = RotsNet_(cfg)
        self.t = torch.zeros(size=(num_cams, 3))
        self.r = torch.zeros(size=(num_cams, 3))

        # self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        # self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def memorize(self, optimizer_pose):
        optimizer_pose[0].zero_grad()
        optimizer_pose[1].zero_grad()

        for init_i in range(200):
            idx_total = torch.arange(start=0, end=self.num_cams).to(self.cfg['pose']['device']) 

            estimated_new_cam_trans = self.transnet.forward(idx_total)
            # print("estimated_new_cam_trans",estimated_new_cam_trans.size())
            estimated_new_cam_quad = self.rotsnet.forward(idx_total)
            # print("estimated_new_cam_quad",estimated_new_cam_quad.size())
            if init_i % 100 == 0:
                print("init_i",init_i,"trans", estimated_new_cam_trans[0])
                print("init_i",init_i,"rots", estimated_new_cam_quad[0])

            loss_trans = (torch.abs(estimated_new_cam_trans - self.t_m.to(self.cfg['pose']['device']))*self.pose_weight[0]).mean()
            loss_trans.backward()

            loss_quad = (torch.abs(estimated_new_cam_quad - self.r_m.to(self.cfg['pose']['device']) )*self.pose_weight[0]).mean()
            loss_quad.backward()
            # print("loss_quad", loss_quad)

            optimizer_pose[0].step()
            optimizer_pose[1].step()
            optimizer_pose[0].zero_grad()
            optimizer_pose[1].zero_grad() 

    def clean_memory(self):
        self.t_m = torch.zeros(size=(self.num_cams, 3))
        self.r_m = torch.zeros(size=(self.num_cams, 3))        
        self.record = torch.zeros(size=(self.num_cams, 1)) 


    def init_posenet_train(self, optimizer_pose):
        
        # not div
        optimizer_pose.zero_grad()

        for init_i in range(100):
            idx_total = torch.arange(start=0, end=self.num_cams).to(self.cfg['pose']['device']) 

            estimated_new_cam_trans = self.transnet.forward(idx_total)
            # print("estimated_new_cam_trans",estimated_new_cam_trans.size())
            estimated_new_cam_quad = self.rotsnet.forward(idx_total)
            if init_i % 250 == 0:
                print("init_i",init_i,"trans", estimated_new_cam_trans[-1])
                print("init_i",init_i,"rots", estimated_new_cam_quad[-1])

            loss_trans = torch.abs(estimated_new_cam_trans - torch.tensor([0,0,0]).to(self.cfg['pose']['device']) ).mean()
            loss_trans.backward()

            loss_quad = torch.abs(estimated_new_cam_quad - torch.tensor([0,0,0] ).to(self.cfg['pose']['device']) ).mean()
            loss_quad.backward()
            # print("loss_quad", loss_quad)

            optimizer_pose.step()
            optimizer_pose.zero_grad()

    def forward(self, cam_id):
        # couple
        cam_id = int(cam_id)
        r = self.rotsnet(cam_id).reshape(-1)  # (3, 3) rot angle
        # print(r)
        t = self.transnet(cam_id).reshape(-1)  # (3, )
        if self.record[cam_id] == 0 and cam_id!=0:
            self.record[cam_id]+=1
            self.t_m[cam_id] = t.clone().detach()
            self.r_m[cam_id] = r.clone().detach()

        self.r[cam_id] = r.clone().detach()
        self.t[cam_id] = t.clone().detach()
        c2w = make_c2w(r, t)  # (4, 4)
        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]
        return c2w

    def get_t(self):
       return self.t


class LearnPoseNet_decouple(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, cfg, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPoseNet_decouple, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.cfg = cfg
        self.transnet = TransNet(cfg)
        self.rotsnet = RotsNet(cfg)
        self.t = torch.zeros(size=(num_cams, 3))
        self.r = torch.zeros(size=(num_cams, 3))
        self.t_m = torch.zeros(size=(self.num_cams, 3))
        self.r_m = torch.zeros(size=(self.num_cams, 3, 3))  
        self.pose_weight = torch.ones(size=(self.num_cams, 1)).to(self.cfg['pose']['device']) 
        self.pose_weight[0] = 5 # first pose take more weight
        # self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        # self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def memorize(self, optimizer_pose):
        optimizer_pose[0].zero_grad()
        optimizer_pose[1].zero_grad()

        for init_i in range(200):
            idx_total = torch.arange(start=0, end=self.num_cams).to(self.cfg['pose']['device']) 

            estimated_new_cam_trans = self.transnet.forward(idx_total)
            # print("estimated_new_cam_trans",estimated_new_cam_trans.size())
            estimated_new_cam_quad = self.rotsnet.forward(idx_total)
            # print("estimated_new_cam_quad",estimated_new_cam_quad.size())
            if init_i % 100 == 0:
                print("init_i",init_i,"trans", estimated_new_cam_trans[0])
                print("init_i",init_i,"rots", estimated_new_cam_quad[0])

            loss_trans = (torch.abs(estimated_new_cam_trans - self.t_m.to(self.cfg['pose']['device']))*self.pose_weight[0]).mean()
            loss_trans.backward()

            loss_quad = (torch.abs(estimated_new_cam_quad - self.r_m.to(self.cfg['pose']['device']) )*self.pose_weight[0]).mean()
            loss_quad.backward()
            # print("loss_quad", loss_quad)

            optimizer_pose[0].step()
            optimizer_pose[1].step()
            optimizer_pose[0].zero_grad()
            optimizer_pose[1].zero_grad() 

    def clean_memory(self):
        self.t_m = torch.zeros(size=(self.num_cams, 3))
        self.r_m = torch.zeros(size=(self.num_cams, 3, 3))        
        self.r_m[0] = torch.eye(3)
        self.record = torch.zeros(size=(self.num_cams, 1))     

    def init_posenet_train(self, optimizer_pose):

        optimizer_pose[0].zero_grad()
        optimizer_pose[1].zero_grad()

        for init_i in range(500):
            idx_total = torch.arange(start=0, end=self.num_cams).to(self.cfg['pose']['device']) 

            estimated_new_cam_trans = self.transnet.forward(idx_total)
            # print("estimated_new_cam_trans",estimated_new_cam_trans.size())
            estimated_new_cam_quad = self.rotsnet.forward(idx_total)
            if init_i % 250 == 0:
                print("init_i",init_i,"trans", estimated_new_cam_trans[-1])
                print("init_i",init_i,"rots", estimated_new_cam_quad[-1])

            loss_trans = torch.abs(estimated_new_cam_trans - torch.tensor([0,0,0]).to(self.cfg['pose']['device']) ).mean()
            loss_trans.backward()

            loss_quad = torch.abs(estimated_new_cam_quad - torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).to(self.cfg['pose']['device']) ).mean()
            loss_quad.backward()
            # print("loss_quad", loss_quad)

            optimizer_pose[0].step()
            optimizer_pose[1].step()
            optimizer_pose[0].zero_grad()
            optimizer_pose[1].zero_grad() 

    def forward(self, cam_id):
        # decouple
        # if cam_id == 0:
        #     c2w = make_c2w(torch.zeros(3), torch.zeros(3))  # (4, 4)
        #     return c2w.to(self.cfg['pose']['device'])

        cam_id = int(cam_id)
        r = self.rotsnet(cam_id)
        t = self.transnet(cam_id)
        self.t[cam_id] = t.clone().detach()
        if self.record[cam_id] == 0 and cam_id!=0:
            self.record[cam_id]+=1
            self.t_m[cam_id] = t.clone().detach()
            self.r_m[cam_id] = r.reshape(3,3).clone().detach()
        # r = self.r[cam_id]  # (3, ) axis-angle
        # t = self.t[cam_id]  # (3, )
        c2w = torch.cat([r.reshape(3,3), t.reshape(3,1)], dim=-1)
        c2w = torch.cat([c2w, torch.tensor([0,0,0,1]).reshape(1,4).to(c2w.device) ], dim=0)
        c2w = torch.inverse(c2w)
        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        # if cam_id == 0 or cam_id>=104:
        #     print("cam_id",cam_id)
        #     print(c2w)
        
        return c2w
    def get_t(self):
       return self.t
    
    

