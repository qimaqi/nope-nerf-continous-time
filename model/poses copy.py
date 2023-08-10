import torch
import torch.nn as nn
from model.common import make_c2w, convert3x4_4x4, make_c2w_quad
from model.posenet import TransNet, RotsNet_quad3, RotsNet_quad4, RotsNet_so3
from torch.nn import functional as F


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
        # print("r", r)
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


class LearnPoseNet_decouple_quad4(nn.Module): # _quad 4 
    def __init__(self, num_cams, learn_R, learn_t, cfg, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPoseNet_decouple_quad4, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.cfg = cfg
        self.transnet = TransNet(cfg)
        self.rotsnet = RotsNet_quad4(cfg)
        self.t = torch.zeros(size=(num_cams, 3))
        self.r = torch.zeros(size=(num_cams, 4))
        self.t_m = torch.zeros(size=(self.num_cams, 3))
        self.r_m = torch.zeros(size=(self.num_cams, 4))  
        self.pose_weight = torch.ones(size=(self.num_cams, 1)).to(self.cfg['pose']['device']) 
        self.pose_weight[0] = 1
        
    def q_to_R(self,q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa,qb,qc,qd = q.unbind(dim=-1)
        R = torch.stack([torch.stack([1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],dim=-1),
                         torch.stack([2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],dim=-1),
                         torch.stack([2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)],dim=-1)],dim=-2)
        return R
    
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

            loss_trans = (self.pose_weight*torch.abs(estimated_new_cam_trans - self.t_m.to(self.cfg['pose']['device']))).mean()
            loss_trans.backward()

            loss_quad = (self.pose_weight*torch.abs(estimated_new_cam_quad - self.r_m.to(self.cfg['pose']['device']))).mean()
            loss_quad.backward()
            # print("loss_quad", loss_quad)

            optimizer_pose[0].step()
            optimizer_pose[1].step()
            optimizer_pose[0].zero_grad()
            optimizer_pose[1].zero_grad() 

    def clean_memory(self):
        self.t_m = torch.zeros(size=(self.num_cams, 3))
        self.r_m = torch.zeros(size=(self.num_cams, 4))        
        self.record = torch.zeros(size=(self.num_cams, 1))     

    def init_memory(self):
        self.t_m = self.t.clone().detach().to(self.cfg['pose']['device'])
        self.r_m = self.r.clone().detach().to(self.cfg['pose']['device'])
        # self.t_m[-1] = torch.tensor([0,0,0]).to(self.cfg['pose']['device'])
        self.r_m[-1] = torch.tensor([1,0,0,0]).to(self.cfg['pose']['device'])
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

            loss_quad = torch.abs(estimated_new_cam_quad -torch.tensor([1,0,0,0]).to(self.cfg['pose']['device']) ).mean()
            loss_quad.backward()
            # print("loss_quad", loss_quad)

            optimizer_pose[0].step()
            optimizer_pose[1].step()
            optimizer_pose[0].zero_grad()
            optimizer_pose[1].zero_grad() 

    
    def forward(self, cam_id):

        cam_id = int(cam_id-1)
        r = self.rotsnet(cam_id)
        t = self.transnet(cam_id)
        self.t[cam_id] = t.clone().detach()
        self.r[cam_id] = r.clone().detach()

        if self.cfg['training']['memorize']:
            if self.record[cam_id] == 0 and cam_id!=self.num_cams-1: #leave id -1 always be 0
                self.record[cam_id]+=1
                self.t_m[cam_id] = t.clone().detach()
                self.r_m[cam_id] = r.clone().detach()

        rots_mat = self.q_to_R(r)
        c2w = torch.cat([rots_mat.reshape(3,3), t.reshape(3,1)], dim=-1)
        c2w = torch.cat([c2w, torch.tensor([0,0,0,1]).reshape(1,4).to(c2w.device) ], dim=0)
        
        ## relative rots
        # r0 = self.rotsnet(self.num_cams-1)
        # t0 = self.transnet(self.num_cams-1)
        # rots_mat0 = self.q_to_R(r0)
        # c2w0 = torch.cat([rots_mat0.reshape(3,3), t0.reshape(3,1)], dim=-1)
        # c2w0 = torch.cat([c2w0, torch.tensor([0,0,0,1]).reshape(1,4).to(c2w0.device) ], dim=0)
        # c2w0 = c2w0.clone().detach()
        # c2w = torch.inverse(c2w0) @ c2w

        # if self.cfg['pose']['cam_coord']:
        #     c2w = torch.inverse(c2w)
        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]
        
        return c2w
    def get_t(self):
       return self.t
    

    def cal_anchor_loss(self):
        # force the timestamp0 always be identity
        cam_id = int(self.num_cams)
        t_reg = self.transnet(cam_id).reshape(-1)
        r_reg = self.rotsnet(cam_id).reshape(-1)
        reg_loss = torch.mean(torch.abs(t_reg - torch.tensor([0,0,0]).to(self.cfg['pose']['device']))) + torch.mean(torch.abs(r_reg - torch.tensor([1,0,0,0]).to(self.cfg['pose']['device'])))

        return reg_loss  

    def cal_memorize_loss(self, update_index):
        # force the timestamp0 always be identity
        idx_total = torch.arange(start=0, end=self.num_cams).to(self.cfg['pose']['device']) 
        estimated_cam_trans = self.transnet.forward(idx_total)
        estimated_cam_quad = self.rotsnet.forward(idx_total)
        #estimated_cam_trans = estimated_cam_trans.squeeze(0)
        #estimated_cam_quad = estimated_cam_quad.squeeze(0)
        # print("estimated_cam_trans", estimated_cam_trans.size())
        estimated_cam_trans_rest = torch.cat((estimated_cam_trans[:update_index, :], estimated_cam_trans[update_index+1:, :]), dim=0)
        estimated_cam_quad_rest = torch.cat((estimated_cam_quad[:update_index, :], estimated_cam_quad[update_index+1:, :]), dim=0)

        t_m_rest = torch.cat((self.t_m[:update_index, :], self.t_m[update_index+1:, :]), dim=0)
        r_m_rest = torch.cat((self.r_m[:update_index, :], self.r_m[update_index+1:, :]), dim=0)

        reg_loss = torch.mean(torch.abs(estimated_cam_trans_rest - t_m_rest.to(self.cfg['pose']['device']))) + torch.mean(torch.abs(estimated_cam_quad_rest - r_m_rest.to(self.cfg['pose']['device'])))

        return reg_loss  

class LearnPoseNet_decouple_quad3(nn.Module): # _quad 4 
    def __init__(self, num_cams, learn_R, learn_t, cfg, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPoseNet_decouple_quad3, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.cfg = cfg
        self.transnet = TransNet(cfg)
        self.rotsnet = RotsNet_quad3(cfg)
        self.t = torch.zeros(size=(num_cams, 3))
        self.r = torch.zeros(size=(num_cams, 3))
        self.t_m = torch.zeros(size=(self.num_cams, 3))
        self.r_m = torch.zeros(size=(self.num_cams, 3))  

    
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

            loss_trans = (torch.abs(estimated_new_cam_trans - self.t_m.to(self.cfg['pose']['device']))).mean()
            loss_trans.backward()

            loss_quad = (torch.abs(estimated_new_cam_quad - self.r_m.to(self.cfg['pose']['device']))).mean()
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

            loss_quad = torch.abs(estimated_new_cam_quad -torch.tensor([0,0,0]).to(self.cfg['pose']['device']) ).mean()
            loss_quad.backward()
            # print("loss_quad", loss_quad)

            optimizer_pose[0].step()
            optimizer_pose[1].step()
            optimizer_pose[0].zero_grad()
            optimizer_pose[1].zero_grad() 

    def forward(self, cam_id):
        # decouple
        #     c2w = make_c2w(torch.zeros(3), torch.zeros(3))  # (4, 4)
        #     return c2w.to(self.cfg['pose']['device'])

        cam_id = int(cam_id)
        # if cam_id ==0 :
        #     r = torch.tensor([0.,0.,0.]).to(self.cfg['pose']['device'])
        #     t = torch.tensor([0.,0.,0.]).to(self.cfg['pose']['device'])
        #     self.r[cam_id] = r.clone().detach()
        #     self.t[cam_id] = t.clone().detach()
        #     c2w = make_c2w(r.reshape(-1), t.reshape(-1))  # (4, 4)
        #     return c2w
        r = self.rotsnet(cam_id)
        t = self.transnet(cam_id)
        # self.t[cam_id] = t.clone().detach()
        if self.cfg['pose']['memorize']:
            if self.record[cam_id] == 0 and cam_id!=self.num_cams-1: #leave id -1 always be 0
                self.record[cam_id]+=1
                self.t_m[cam_id] = t.clone().detach()
                self.r_m[cam_id] = r.clone().detach()

        c2w = make_c2w_quad(r, t.reshape(-1))  # (4, 4)

        # relative rots
        r_end = self.rotsnet(self.num_cams-1)
        t_end = self.transnet(self.num_cams-1)
        c2w_end = make_c2w_quad(r_end, t_end.reshape(-1))  # (4, 4)
        c2w_end = c2w_end.clone().detach()

        

        self.r[cam_id] = r.clone().detach()
        self.t[cam_id] = t.clone().detach()

        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]
        

        return c2w
    def get_t(self):
       return self.t
    
    def cal_anchor_loss(self):
        cam_id = self.num_cams - 1
        t_reg = self.transnet(cam_id).reshape(-1)
        r_reg = self.rotsnet(cam_id).reshape(-1)

        reg_loss = torch.mean(torch.abs(t_reg - torch.tensor([0,0,0]).to(self.cfg['pose']['device']))) + torch.mean(torch.abs(r_reg - torch.tensor([0,0,0]).to(self.cfg['pose']['device'])))

        return reg_loss     
     


class LearnPoseNet_decouple_so3(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, cfg, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPoseNet_decouple_so3, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        self.cfg = cfg
        self.transnet = TransNet(cfg)
        self.rotsnet = RotsNet_so3(cfg)
        self.t = torch.zeros(size=(num_cams, 3))
        self.r = torch.zeros(size=(num_cams, 3))
        self.t_m = torch.zeros(size=(self.num_cams, 3))
        self.r_m = torch.zeros(size=(self.num_cams, 3))  


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
                # print("init_i",init_i,"trans", estimated_new_cam_trans[0])
                print("init_i",init_i,"rots", estimated_new_cam_quad[0])

            loss_trans = (torch.abs(estimated_new_cam_trans - self.t_m.to(self.cfg['pose']['device']))).mean()
            loss_trans.backward()

            loss_quad = (torch.abs(estimated_new_cam_quad - self.r_m.to(self.cfg['pose']['device']))).mean()
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

            loss_quad = torch.abs(estimated_new_cam_quad - torch.tensor([0,0,0]).to(self.cfg['pose']['device']) ).mean()
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
        # if cam_id == self.num_cams-1 :
        #     r = torch.tensor([0.,0.,0.]).to(self.cfg['pose']['device'])
        #     t = torch.tensor([0.,0.,0.]).to(self.cfg['pose']['device'])
        #     self.r[cam_id] = r.clone().detach()
        #     self.t[cam_id] = t.clone().detach()
        #     c2w = make_c2w(r.reshape(-1), t.reshape(-1))  # (4, 4)
        #     return c2w

        r = self.rotsnet(cam_id)
        t = self.transnet(cam_id)
        if self.cfg['pose']['memorize']:
            if self.record[cam_id] == 0 and cam_id!=self.num_cams-1:
                self.record[cam_id]+=1
                self.t_m[cam_id] = t.clone().detach()
                self.r_m[cam_id] = r.clone().detach()

        self.r[cam_id] = r.clone().detach()
        self.t[cam_id] = t.clone().detach()
        c2w = make_c2w(r.reshape(-1), t.reshape(-1))  # (4, 4)

        # if cam_id == self.num_cams-1 or cam_id == self.num_cams-2 or cam_id == self.num_cams-3 :
        #     print("before norm",cam_id)
        #     print(c2w)

        ## relative rots
        # r0 = self.rotsnet(self.num_cams-1)
        # t0 = self.transnet(self.num_cams-1)
        # c2w0 = make_c2w(r0.reshape(-1), t0.reshape(-1))  # (4, 4)
        # c2w0 = c2w0.clone().detach()

        c2w = torch.inverse(c2w0) @ c2w

        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w
    def get_t(self):
       return self.t
    

    def cal_anchor_loss(self):
        # force the timestamp0 always be identity
        cam_id = int(self.num_cams-1)
        t_reg = self.transnet(cam_id).reshape(-1)
        r_reg = self.rotsnet(cam_id).reshape(-1)
        #learned_poses = torch.stack([self.pose_param_net(i) for i in range(self.pose_param_net.num_cams)])
        #last_epoch_poses = torch.stack([self.pose_param_net.get_last_epoch_poses(i) for i in range(self.pose_param_net.num_cams)])
        #t_reg = torch.stack([self.transnet(i).reshape(-1) for i in range(self.num_cams)])
        #r_reg = torch.stack([self.rotsnet(i).reshape(-1) for i in range(self.num_cams)])
        # F.mse_loss(t_reg, torch.zeros_like(t_reg)) +
        reg_loss = torch.mean(torch.abs(t_reg - torch.tensor([0,0,0]).to(self.cfg['pose']['device']))) + torch.mean(torch.abs(r_reg - torch.tensor([0,0,0]).to(self.cfg['pose']['device'])))
        # F.mse_loss(t_reg, torch.zeros_like(t_reg)) + F.mse_loss(r_reg, torch.zeros_like(r_reg))  
        # 

        return reg_loss     
    
    

