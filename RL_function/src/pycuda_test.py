import ctypes
import numpy as np



def init_sparsemat(p2p_file="/lf_tool/matrix_cuda/coo",value_file="/lf_tool/matrix_cuda/coo2",rownum=13):
    so=ctypes.cdll.LoadLibrary('/root/git/RL_function/RL_function/Debug/libRL_function.so')

    # 返回稀疏矩阵nzz
    nnz=np.zeros(1,dtype=int)
    nnz_array=(ctypes.c_int*len(nnz))(*nnz)
    so.mat_nnz_gpu(b"/lf_tool/matrix_cuda/coo",13,nnz_array)
    nnz=np.array(nnz_array)[0]


    csr_col=np.zeros(nnz,dtype=int)
    csr_col_array=(ctypes.c_int*len(csr_col))(*csr_col)

    csr_row=np.zeros(rownum+1,dtype=int)
    csr_row_array=(ctypes.c_int*len(csr_row))(*csr_row)

    csr_p2p=np.zeros(nnz,dtype=float)
    csr_p2p_array=(ctypes.c_float*len(csr_p2p))(*csr_p2p)

    csr_value=np.zeros(nnz,dtype=float)
    csr_value_array=(ctypes.c_float*len(csr_value))(*csr_value)

    xs_point=np.zeros(rownum,dtype=int)
    xs_point_array=(ctypes.c_int*len(xs_point))(*xs_point)
    so.create_csr_mat(csr_col_array,csr_row_array,csr_p2p_array,csr_value_array,b"/lf_tool/matrix_cuda/coo",b"/lf_tool/matrix_cuda/coo2",rownum,rownum,xs_point_array)

    col=np.array(csr_col_array)
    row=np.array(csr_row_array)
    p2p=np.array(csr_p2p_array)
    value=np.array(csr_value_array)
    xs_point=np.array(xs_point_array)
    return  nnz,col,row,p2p,value,xs_point

class mat_sparse:
    def __init__(self,row):
        [nnz,col_l,row_l,p2p_l,value_l,xs_p]=init_sparsemat(p2p_file="/lf_tool/matrix_cuda/coo",value_file="/lf_tool/matrix_cuda/coo2",rownum=13)
        self.nzz=nnz
        self.csr_row=row_l
        self.csr_col=col_l
        self.p2p=p2p_l
        self.value=value_l
        self.xs_ps=xs_p
        self.rownum=row

    #出稀疏矩阵p2p的元素
    def v_p2p(self,row,col):
        row_s=self.csr_row[row]
        row_e=self.csr_row[row+1]
        col_list=list(self.csr_col[row_s:row_e])

        if col_list.__contains__(col):
            index=col_list.index(col)
            return self.p2p[self.csr_row[row]+index]
        else:
           return 0

    #出稀疏矩阵value的元素
    def v_v(self,row,col):
        row_s=self.csr_row[row]
        row_e=self.csr_row[row+1]
        col_list=list(self.csr_col[row_s:row_e])
        if col_list.__contains__(col):
            index=col_list.index(col)
            return self.value[self.csr_row[row]+index]
        else:
            return 0

mat=mat_sparse(13)
# print(mat.value)

def TD_beta(mat=mat):
    init_V=np.zeros(mat.rownum,dtype=float)
    init_e=np.zeros(mat.rownum,dtype=float)
    R=1
    beta=0.85
    a=0.99
    t=1
    def at(a,t):
        if a-0.001*t>0.001:
           return a-0.001*t
        else:
           return 0.001
    at_1=at(a,t)
    state=0
    #依据当前状态，更新适度轨迹0增长轨迹，1替代轨迹
    def up_e(e=init_e,stat_now=0,R=1,beta=beta,mode=0):
        rezult=[]
        e=list(e)
        i=0
        for a in e:
          if i!=stat_now:
             rezult.append(R*beta*a)
          else:
             if mode==0:
                rezult.append(R*beta*a+1.0)
             else:
                rezult.append(1.0)
          i=i+1
        return np.array(rezult,dtype=float)


    #判断当前状态是否吸收状态
    def is_xs(state):
        rezult=True
        if mat.xs_ps[state]==1:
           rezult=True
        else:
           rezult=False
        return rezult

    #计算时序差分
    def difference(stats_0=0):
        #计算时序差
        dt=0
        next_stat=0 #在稀疏矩阵nnz中的位置
        next_stat_col=stats_0 #在稀疏矩阵行中的位置
        if mat.xs_ps[stats_0]!=1:#不是吸收点
           show_p=np.random.rand()
           up=mat.csr_row[stats_0]
           down=mat.csr_row[stats_0+1]

           if up<down-1:#多余一个跳转点
              p_list=mat.p2p[up:down]
              next_stat=up
              # print("不是吸收点",stats_0,p_list,show_p,up,down)
              for e in p_list:
                  if show_p<e:
                     break
                  next_stat=next_stat+1
              next_stat_col=mat.csr_col[next_stat]
              # print("不是吸收点",next_stat)
              dt=mat.value[next_stat]+R*init_V[next_stat_col]-init_V[stats_0]
           else:#只有一个跳转点
              next_stat=up
              next_stat_col=mat.csr_col[next_stat]
              dt=mat.value[next_stat]+R*init_V[next_stat_col]-init_V[stats_0]
              # print("只有一个跳转点：",stats_0,next_stat,mat.value[next_stat],init_V[next_stat_col],init_V[stats_0])
        else:#当前点等于吸收点
           # print("吸收点：",stats_0)
           dt=0+R*init_V[stats_0]-init_V[stats_0]
           next_stat=0#进入吸收点后返回初始点
        return dt,next_stat_col

    def up_all_difference():
        for i in range(mat.rownum):
            if init_e[i]!=0:
               [dt,next_state]=difference(stats_0=i)
               # print("stats_0:",i,"next_:",next_state)
               # print("-----------------")
               init_V[i]=init_V[i]+at_1*init_e[i]*dt

    # print("init_e0:",init_e)
    while(True):
        #计算差分
        [dt,next_state]=difference(state)
        #更新所有适度轨迹
        init_e=up_e(e=init_e,stat_now=state,R=1,beta=beta,mode=1)
        # print("init_e1:",init_e)
        #更新所有状态值函数
        up_all_difference()
        t=t+1
        at_1=at(a,t)
        state=next_state
        if is_xs(state):#是否吸收状态
           state=0
        if t>5000:
           print(init_V)
           break
    return init_V

def td_fun(mat=mat):
    N=10
    def norm(state=0,x=1,std=2):
        return np.exp(-1.0*np.power(state-x,2)/(2.0*std))

    def create_f_base(state=0,f_base_num=5):
        rezult=[]
        for i in range(f_base_num):
            rezult.append(norm(state=state,x=i,std=1+i))
        # print("rezult:=",rezult)
        return np.array(rezult)

    def transter(state_0=0,w=0,at=0.9):#科技函数和大小
        base_f_now=create_f_base(state=state_0,f_base_num=N)
        state_next=0
        #计算时序差
        wt=0
        next_stat=0 #在稀疏矩阵nnz中的位置
        next_stat_col=0 #在稀疏矩阵行中的位置
        if mat.xs_ps[state_0]!=1:#不是吸收点
            show_p=np.random.rand()
            up=mat.csr_row[state_0]
            down=mat.csr_row[state_0+1]
            if up<down-1:#多余一个跳转点
                p_list=mat.p2p[up:down]
                next_stat=up
                # print("不是吸收点",state_0,p_list,show_p,up,down)
                for e in p_list:
                    if show_p<e:
                        break
                    next_stat=next_stat+1
                next_stat_col=mat.csr_col[next_stat]
                # print("不是吸收点：",state_0,next_stat_col,mat.value[next_stat])
                wt=w+at*(mat.value[next_stat]+R*np.dot(w,create_f_base(state=next_stat_col,f_base_num=N))-np.dot(w,base_f_now))*base_f_now
            else:#只有一个跳转点
                next_stat=up
                next_stat_col=mat.csr_col[next_stat]
                wt=w+at*(mat.value[next_stat]+R*np.dot(w,create_f_base(state=next_stat_col,f_base_num=N))-np.dot(w,base_f_now))*base_f_now
                # print("只有一个跳转点：",mat.value[next_stat])
                # print("一个点：",state_0,next_stat_col,mat.value[next_stat])
        else:#当前点等于吸收点
            wt=w+at*(0.0+R*np.dot(w,base_f_now)-np.dot(w,base_f_now))*base_f_now
            # print("吸收点======：",state_0,0,at*(0.0+R*np.dot(w,base_f_now)-np.dot(w,base_f_now)))
            next_stat_col=0#进入吸收点后返回初始点
        return wt,next_stat_col

    f_base_num=N
    w=np.random.rand(f_base_num)
    print(w)
    t=500
    at=100/(t+100)
    R=1.0
    next=0
    while True:
       if t>500000:
          break
       else:
          [w,next]=transter(state_0=next,w=w,at=at)
          t=t+1
          if t%10==0:
             at=100/(t+100)
    for i in range(13):
        print(12-i,np.dot(w,create_f_base(state=i,f_base_num=N)))

TD_beta()
td_fun()




