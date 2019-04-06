import ctypes
import numpy as np
np.set_printoptions(suppress=True)


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

#一、表格TD（n）学习法
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


#二、线性值函数逼近求解法
def td_fun(mat=mat):
    N=100
    def norm(state=0,x=1,std=2):
        return np.exp(-1.0*np.power(state-x*0.5,2)/(2.0*std)) #//高斯基
        # return 1.0/(1.0+np.exp(np.power(state-x,2)/(std)))#//反转基

    def create_f_base(state=0,f_base_num=5):
        rezult=[]
        for i in range(f_base_num):
            rezult.append(norm(state=state,x=i,std=2))
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
    t=1000
    at=100/(t+100)
    R=1.0
    next=0
    while True:
       if t>20000:
          break
       else:
          [w,next]=transter(state_0=next,w=w,at=at)
          t=t+1
          if t%10==0:
             at=100/(t+100)
    for i in range(13):
        print(12-i,np.dot(w,create_f_base(state=i,f_base_num=N)))

#。最小二成法更新
def Least_Square():
    N=50000
    fun_num=10
    so=ctypes.cdll.LoadLibrary('/root/git/RL_function/RL_function/Debug/libRL_function.so')

    def norm(state=0,x=1,std=2):
        return np.exp(-1.0*np.power(state-x,2)/(2.0*std)) #//高斯基
        # return 1.0/(1.0+np.exp(np.power(state-x,2)/(std)))#//反转基

    def create_f_base(state=0,f_base_num=fun_num):
        rezult=[]
        for i in range(f_base_num):
            rezult.append(norm(state=state,x=i,std=2))
        # print("rezult:=",np.array(rezult))
        return np.array(rezult)

    def ls_fuc(Aarray,Carray,m,n,nrch):
        so.least_square_cublas(Aarray,Carray,m,n,nrch)
        w=np.array(Carray)[0:n]
        return w

    def sample(mat=mat,num=N):#(xt,rt+1,xt+1)

       def transter(state_0=0):#科技函数和大小
            state_next=0
            #计算时序差
            next_stat=0 #在稀疏矩阵nnz中的位置
            next_stat_col=0 #在稀疏矩阵行中的位置
            reuzlt=[]
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
                    rezult=[state_0,mat.value[next_stat],next_stat_col]
                    # print("不是吸收点：",rezult)
                else:#只有一个跳转点
                    next_stat=up
                    next_stat_col=mat.csr_col[next_stat]
                    rezult=[state_0,mat.value[next_stat],next_stat_col]
                    # print("只有一个跳转点：",rezult)
                    # print("一个点：",state_0,next_stat_col,mat.value[next_stat])
            else:#当前点等于吸收点
                rezult=[state_0,0,state_0]
                # print("吸收点======：",rezult)
                # print("----------------------------------------")
            return rezult

       i=0
       state_0=0
       sample_list=[]
       while i<num:
           if mat.xs_ps[state_0]!=1:
              [now,r,next]=transter(state_0)
              sample_list.append([now,r,next])
              i=i+1
              state_0=next
           else:
              [now,r,next]=transter(state_0)
              sample_list.append([now,r,next])
              i=i+1
              state_0=0#返回原点
       return sample_list

    sam=sample(mat=mat,num=N)
    Aarray=[]
    Carray=[]
    R=1
    w=np.zeros(fun_num)
    t=1
    # while True:
    #     for  e  in  sam:
    #          value_now=create_f_base(state=e[0],f_base_num=fun_num)
    #          value_next=np.dot(create_f_base(state=e[2],f_base_num=fun_num),w)
    #          rward=e[1]+value_next*R
    #          Aarray.append(value_now)
    #          Carray.append(rward)
    #     Aarray=np.array(Aarray).T
    #     Aarray=np.reshape(Aarray,N*fun_num)
    #     Carray=np.array(Carray)
    #     Aarray_array=(ctypes.c_float*len(Aarray))(*Aarray)
    #     Carray_array=(ctypes.c_float*len(Carray))(*Carray)
    #     w=ls_fuc(Aarray_array,Carray_array,N,fun_num,1)
    #     Aarray=[]
    #     Carray=[]
    #一次迭代法
    for e  in sam:
         value_now=create_f_base(state=e[0],f_base_num=fun_num)
         value_next=create_f_base(state=e[2],f_base_num=fun_num)
         rward=e[1]
         Aarray.append(np.array(value_now)-R*np.array(value_next))
         Carray.append(rward)
    Aarray=np.array(Aarray).T
    Aarray=np.reshape(Aarray,N*fun_num)
    Carray=np.array(Carray)
    Aarray_array=(ctypes.c_float*len(Aarray))(*Aarray)
    Carray_array=(ctypes.c_float*len(Carray))(*Carray)
    w=ls_fuc(Aarray_array,Carray_array,N,fun_num,1)
    print(w)
    for i in range(13):
        print(i,np.dot(create_f_base(state=i,f_base_num=fun_num),w))

#[-33.607815  41.806618 -49.396786  29.50096  -16.299845]
Least_Square()
# TD_beta()
# td_fun()

# x = np.array([[1, 50, 5, 200], [1, 50, 5, 400], [1, 50, 5, 600], [1, 50, 5, 800], [1, 50, 5, 1000], [1, 50, 10, 200], [1, 50, 10, 400], [1, 50, 10, 600], [1, 50, 10, 800], [1, 50, 10, 1000], [1, 60, 5, 200], [1, 60, 5, 400], [1, 60, 5, 600], [1, 60, 5, 800], [1, 60, 5, 1000], [1, 60, 10, 200], [1, 60, 10, 400], [1, 60, 10, 600], [1, 60, 10, 800], [1, 60, 10, 1000], [1, 70, 5, 200], [1, 70, 5, 400], [1, 70, 5, 600], [1, 70, 5, 800], [1, 70, 5, 1000], [1, 70, 10, 200], [1, 70, 10, 400]])
# x=x.T
# x=np.reshape(x,108)
# y = np.array([7.434, 3.011, 1.437, 0.6728, 0.00036, 5.518, 2.556, 1.341, 0.6824, 0.0001, 18.22, 7.344, 4.066, 1.799, 1.218, 16.11, 9.448, 4.752, 2.245, 1.539, 18.14, 12.88, 7.29, 3.449, 2.533, 15.76, 16.24])
#



