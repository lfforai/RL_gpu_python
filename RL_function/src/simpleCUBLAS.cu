#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
#include <math.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <typeinfo>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif
#define MIN(a,b) ((a)<(b)? (a):(b))


#include "cusparse.h"
#include <iostream>
#include <unistd.h>
#include<sys/types.h>
#include<fcntl.h>
#include<string.h>
#include<stdio.h>
#include<unistd.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <sys/mman.h>
using namespace std;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define AP_LOWER(i,j,ld) (i-j+((2*ld-j+1)*j)/2)//三角矩阵压缩模式lower
#define AP_UPPER(i,j) (i+(j*(j+1))/2)//三角矩阵压缩模式upper

//----------------------------Frist-file_input------------------------
//一.catch file length
int getFileSize(const string &filename)
{
    int size = 0;
    FILE *fp = NULL;

    fp=fopen(filename.c_str(),"r");
    if( NULL == fp)
    {
        return size;
    }

    fseek(fp,0L,SEEK_END);
    size = ftell(fp);
    fclose(fp);
    return size;
}

//把矩阵数据文件读取到内存中去，等到转化到ｃｕｂｌａｓ用的标准矩阵数据格式
void mmapSaveDataIntoFiles(const string &filename,char *rezult)
{
    int fileLength = 0;
    int dataLength = 0;
    int offset = 0;
    /* 获取文件大小和数据长度 */
    fileLength = getFileSize(filename);
    int fd = open(filename.c_str(),O_CREAT |O_RDWR|O_APPEND,00777);
    if(fd < 0)
    {
        cout<<"OPEN FILE ERROR!"<<endl;
    }
    char *buffer = (char*)mmap(NULL,fileLength,PROT_READ,MAP_SHARED,fd,0);
    close(fd);
    memcpy(rezult,buffer,fileLength);
    rezult[fileLength]='\0';
    munmap(buffer,fileLength);
}

//----------------------------Seconde-cublas------------------------
enum d_or_v{ondevice,onhost};//0返回值在device上，１／返回值在host上

enum matrix_type
{g_packed,//普通矩阵
 g_banded,//gb,普通带状矩阵，需要ｋｌ，ｍｌ表示上下斜线的条数
 symmetric_banded, //sb,对称带状矩阵，ｋ用来表示ｓｕｐｅｒ或ｓｕｂ斜线的条数，cublasFillMode_t参数储存上下三角
 symmetric_packed,//sp,对称原始矩阵：cublasFillMode_t
 triangular_packed,//tp,三角原始矩阵：cublasDiagType_t是否有对角线，cublasFillMode_t上下三角
 triangular_banded,//tb,三角带状矩阵：cublasDiagType_t是否有对角线，cublasFillMode_t上下三角，ｋ用来表示ｓｕｐｅｒ或ｓｕｂ斜线的条数
 Hermitian_packed,//ｈp,自共轭普通矩阵，Ａ＝ＡＨ,cublasFillMode_t
 Hermitian_banded//hb,自共轭带状矩阵,对称带状矩阵，ｋ用来表示ｓｕｐｅｒ或ｓｕｂ斜线的条数，cublasFillMode_t参数储存上下三角
};

template<class T>
struct matrix_info{
	T** matrix_data_TT;//*[]
	T*  matrix_data_T;//*
	long  row;
	long  colomn;
    long  idx;
    int   batchCount=1;
    matrix_type type_mat=g_packed;
};

//gpu point be converted to cpu point for printf() using
//return menory in cpu
template<class T> //convert dd2hh
T** dp2printf(T** devcie_m,int bithsize,int row_num,int col_num){
//	cout<<"多bithsize矩阵，输出在device上的矩阵"<<endl;
//	printf("bithsize:%d,row_num:%d,col_num:%d\n",bithsize,row_num,col_num);
	int array_length=row_num*col_num;
	T** host_m=(T**)malloc(bithsize*sizeof(*host_m));
	T** host_print=(T**)malloc(bithsize*sizeof(*host_m));
	for(int i=0;i<bithsize;i++){
	    host_print[i]=(T* )malloc(array_length*sizeof(host_print[0][0]));
	}
	checkCudaErrors(cudaMemcpy(host_m,devcie_m,bithsize*sizeof(host_m[0]),cudaMemcpyDeviceToHost));
	for(int i=0;i<bithsize;i++){
		checkCudaErrors(cudaMemcpy(host_print[i],host_m[i],array_length*sizeof(host_m[0][0]),cudaMemcpyDeviceToHost));
		cout<<"the ith_matrix:="<<i<<"******************************"<<endl;
	    string output="[";
	    for(int row_N=0;row_N<row_num;row_N++){
	    	for (int col_N=0;col_N<col_num;col_N++) {
	//    		cout<<"row:="<<i<<"|col:="<<j<<"|value:"<<rezult[IDX2C(i,j,m)]<<endl;
	    		stringstream ss;
	    		ss<<host_print[i][IDX2C(row_N,col_N,row_num)];
	    		string temp;
	    		ss>>temp;
	            output+=temp;
	            if(col_N!=(col_num-1))
	               output+=",";
	    		ss.clear();
			}
	    	if (row_N!=row_num-1)
	    	   output+="\n";
	    }
	    output+=("]\n");
		cout<<output<<endl;
	}
	return host_print;
}

template<class T> //**T 把ｄｅｖｉｃｅ上的内容打印出来
T* dp1_matirx_printf(T* devcie_m,int m,int n,bool ondevice=true){
//	cout<<"输出在device上的矩阵(vector)"<<endl;
	T* rezult;
	if (ondevice==true)
    {rezult=(T*)malloc(n*m*sizeof(rezult[0]));
     checkCudaErrors(cudaMemcpy(rezult,devcie_m,m*n*sizeof(rezult[0]),cudaMemcpyDeviceToHost));
    }else{
     cout<<"host"<<endl;
     memcpy(rezult,devcie_m,m*n*sizeof(rezult[0]));
    }
    string output="[";
    for(int i=0;i<m;i++){
    	for (int j=0;j<n;j++) {
//    		cout<<"row:="<<i<<"|col:="<<j<<"|value:"<<rezult[IDX2C(i,j,m)]<<endl;
    		stringstream ss;
    		ss<<rezult[IDX2C(i,j,m)];
    		string temp;
    		ss>>temp;
            output+=temp;
            if(j!=(n-1))
               output+=",";
    		ss.clear();
		}
    	if (i!=m-1)
    	   output+="\n";
    }
    output+=("]\n");
    cout<<output<<endl;
	return rezult;
}

//转换为列存储的矩阵g_packed,存储到device上去
template<class T>
matrix_info<T>* m2g_packed(char* data_txt,int batchCount,long row_num,long col_num){
//	cout<<"导入dense矩阵开始"<<endl;
	matrix_info<T>* rezult=(matrix_info<T>*)malloc(sizeof(matrix_info<T>));
	if (batchCount==1)
	   {rezult->row=row_num;
		rezult->colomn=col_num;
		rezult->idx=row_num;
		rezult->batchCount=1;
		T* temp_data=(T*)malloc(row_num*col_num*sizeof(T));//所有分配在ｈｏｓｔ上完成，按需求拷贝到ｄｅｖｉｃｅ
		checkCudaErrors(cudaMalloc((void **)&(rezult->matrix_data_T),row_num*col_num*sizeof((rezult->matrix_data_T[0]))));
		long ld=row_num;
		stringstream ss(data_txt);
		string line;
		int row=0;
		int col=0;
		while (getline(ss, line, '\n')) {
			//开始一行的数据导入
			col=0;
			stringstream ss_in(line);
			string value;
			while (getline(ss_in, value, ',')){
				stringstream ss_inn(value);
				ss_inn>>temp_data[IDX2C(row,col,ld)];
				cout<<value<<"|at_matrix[:="<<temp_data[IDX2C(row,col,ld)]<<"]"<<endl;
				col++;
			}
	//		cout<< line << endl;
			++row;
		}
	   int len=row_num*col_num;
//	   if (len>10) len=16;//only out put 10 recored
//	   for(long i=0;i<row_num;i++){
//		   for(long j=0;j<col_num;j++)
//			  {cout<<i<<":="<<temp_data[IDX2C(i,j,ld)]<<endl;}
//		}
	   cout<<"matrix:="<<endl;
	   cout<<data_txt<<endl;
	   checkCudaErrors(cudaMemcpy(rezult->matrix_data_T,temp_data,len*sizeof(T),cudaMemcpyHostToDevice));
	   }
	   else
	   {    //when  batchCount used
	    	rezult->row=row_num;
			rezult->colomn=col_num;
			rezult->idx=row_num;
			rezult->batchCount=batchCount;

			checkCudaErrors(cudaMalloc((void **)&(rezult->matrix_data_TT),batchCount*sizeof(rezult->matrix_data_TT[0])));
			long len=row_num*col_num;
			T** temp_data=(T**)malloc(batchCount*sizeof(*temp_data));//所有分配在ｈｏｓｔ上完成，按需求拷贝到ｄｅｖｉｃｅ
			for(int i=0;i<batchCount;i++){
			   *(temp_data+i)=(T*)malloc(len*sizeof(temp_data[0][0]));
			}

			long ld=row_num;
			stringstream ss(data_txt);
			string line;
			int batch_each=-1;
            int row=0;
			int col=0;
			while (getline(ss, line, '\n')) {
					if(line=="m")
					{//开始一行的数据导入
					 ++batch_each;
					 row=0;
					}else if(line!="#"){
//						cout<<line<<endl;
						col=0;
						stringstream ss_in(line);
						string value;
						while (getline(ss_in, value, ',')){
							stringstream ss_inn(value);
							ss_inn>>temp_data[batch_each][IDX2C(row,col,ld)];
							col++;
						}
						row++;
					}else{
//						printf("g_packed matrix in menory::\n");
//						for(int batch_each=0;batch_each<batchCount;batch_each++)
//						{   cout<<"batch_each:="<<batch_each<<"******************************"<<endl;
//							for(int row=0;row<row_num;row++)
//							{  for(int col=0;col<col_num;col++)
//								  cusparseMatDescr_t debsrC;
					            //{cout<<"batch_each:"<<batch_each<<"|row:"<<row<<"|col:"<<col<<"|value:="<<temp_data[batch_each][IDX2C(row,col,ld)]<<endl;}
//							}
//						}
						break;
					}
			}

			//copy to device
	        T** temp_2=(T**)malloc(batchCount*sizeof(*temp_2));
	        for(int i=0;i<batchCount;i++){
	            checkCudaErrors(cudaMalloc((void **)&(temp_2[i]),len*sizeof(temp_2[0][0])));
	            checkCudaErrors(cudaMemcpy(temp_2[i],temp_data[i],len*sizeof(temp_data[0][0]),cudaMemcpyHostToDevice));
	        }
			cudaMemcpy(rezult->matrix_data_TT,temp_2, batchCount*sizeof(*temp_2),cudaMemcpyHostToDevice);
	     }
//	cout<<"导入dense矩阵结束"<<endl;
	return rezult;
}

//symmetric packed matrix,对称方阵
template<class T>
matrix_info<T>* m2g_symmetric_packed(char* data_txt,cublasFillMode_t uplo=CUBLAS_FILL_MODE_LOWER,int batchCount=1,long row_num=4){
//	cout<<"导入dense矩阵开始"<<endl;
	matrix_info<T>* rezult=(matrix_info<T>*)malloc(sizeof(matrix_info<T>));
	if (batchCount==1)
	   {rezult->row=row_num;
		rezult->colomn=row_num;
		rezult->idx=row_num;
		rezult->batchCount=1;
		int pitch=(int)(row_num*(row_num+1)/2);
		T* temp_data=(T* )malloc(pitch*sizeof(temp_data[0]));//所有分配在ｈｏｓｔ上完成，按需求拷贝到ｄｅｖｉｃｅ
		checkCudaErrors(cudaMalloc((void **)&(rezult->matrix_data_T),pitch*sizeof((rezult->matrix_data_T[0]))));
		long ld=row_num;
		stringstream ss(data_txt);
		string line;
		int row=0;
		int col=0;
		cout<<AP_LOWER(0,1,ld)<<endl;
		while (getline(ss, line, '\n')) {
			//开始一行的数据导入
			col=0;
			stringstream ss_in(line);
			string value;
			while (getline(ss_in, value, ',')){
				if(row-col>=0 and uplo==CUBLAS_FILL_MODE_LOWER)
				   {stringstream ss_inn(value);
				    ss_inn>>temp_data[AP_LOWER(row,col,ld)];
				    cout<<"row:="<<AP_LOWER(row,col,ld)<<"|value:="<<temp_data[AP_LOWER(row,col,ld)]<<endl;
				   }
				if(row-col<=0 and uplo==CUBLAS_FILL_MODE_UPPER)
				   {stringstream ss_inn(value);
				    ss_inn>>temp_data[AP_UPPER(row,col)];
				    cout<<"row:="<<AP_UPPER(row,col)<<"|value:="<<temp_data[AP_UPPER(row,col)]<<endl;
				   }
				col++;
			}
	//		cout<< line << endl;
			++row;
		}
		for (int i=0;i<pitch;++i) {
		  cout<<temp_data[i]<<endl;
		}
	   checkCudaErrors(cudaMemcpy(rezult->matrix_data_T,temp_data,pitch*sizeof(rezult->matrix_data_T[0]),cudaMemcpyHostToDevice));
	   dp1_matirx_printf(rezult->matrix_data_T,1,pitch,true);
	   }
	   else
	   {    //when  batchCount used
	    	rezult->row=row_num;
			rezult->colomn=row_num;
			rezult->idx=row_num;
			rezult->batchCount=batchCount;

			checkCudaErrors(cudaMalloc((void **)&(rezult->matrix_data_TT),batchCount*sizeof(rezult->matrix_data_TT[0])));
			long len=(int)(row_num*(row_num+1)/2);
			T** temp_data=(T**)malloc(batchCount*sizeof(*temp_data));//所有分配在ｈｏｓｔ上完成，按需求拷贝到ｄｅｖｉｃｅ
			for(int i=0;i<batchCount;i++){
			   *(temp_data+i)=(T*)malloc(len*sizeof(temp_data[0][0]));
			}

			long ld=row_num;
			stringstream ss(data_txt);
			string line;
			int batch_each=-1;
            int row=0;
			int col=0;
			while (getline(ss, line, '\n')) {
					if(line=="m")
					{//开始一行的数据导入
					 ++batch_each;
					 row=0;
					}else if(line!="#"){
//						cout<<line<<endl;
						col=0;
						stringstream ss_in(line);
						string value;
						while (getline(ss_in, value, ',')){
							stringstream ss_inn(value);
							ss_inn>>temp_data[batch_each][IDX2C(row,col,ld)];
							col++;
						}
						row++;
					}else{
//						printf("g_packed matrix in menory::\n");
//						for(int batch_each=0;batch_each<batchCount;batch_each++)
//						{   cout<<"batch_each:="<<batch_each<<"******************************"<<endl;
//							for(int row=0;row<row_num;row++)
//							{  for(int col=0;col<col_num;col++)
//								  cusparseMatDescr_t debsrC;
					            //{cout<<"batch_each:"<<batch_each<<"|row:"<<row<<"|col:"<<col<<"|value:="<<temp_data[batch_each][IDX2C(row,col,ld)]<<endl;}
//							}
//						}
						break;
					}
			}

			//copy to device
	        T** temp_2=(T**)malloc(batchCount*sizeof(*temp_2));
	        for(int i=0;i<batchCount;i++){
	            checkCudaErrors(cudaMalloc((void **)&(temp_2[i]),len*sizeof(temp_2[0][0])));
	            checkCudaErrors(cudaMemcpy(temp_2[i],temp_data[i],len*sizeof(temp_data[0][0]),cudaMemcpyHostToDevice));
	        }
			cudaMemcpy(rezult->matrix_data_TT,temp_2, batchCount*sizeof(*temp_2),cudaMemcpyHostToDevice);
	     }
//	cout<<"导入dense矩阵结束"<<endl;
	return rezult;
}

//这里的带状矩阵都是方阵,row=colomn
//g_banded带状矩阵,symmetric_banded对称带状矩阵(ku设置为０，为sub存储：kｌ设置为０，为super存储)
template<class T>
matrix_info<T>* m2g_banded(char * data_txt,int batchCount=1,long row_num=5,int kl=2,int ku=2){
	matrix_info<T>* rezult=(matrix_info<T>*)malloc(sizeof(matrix_info<T>));
	rezult->row=row_num;
	rezult->colomn=row_num;
	rezult->idx=row_num;
	rezult->batchCount=batchCount;

	if(batchCount==1)
	{T* temp_data=(T*)malloc((kl+ku+1)*row_num*sizeof(T));//所有分配在ｈｏｓｔ上完成，按需求拷贝到ｄｅｖｉｃｅ
	checkCudaErrors(cudaMalloc((void **)&(rezult->matrix_data_T),(kl+ku+1)*row_num*sizeof(rezult->matrix_data_T[0])));
	memset(temp_data,0,(kl+ku+1)*row_num*sizeof(T));
	long ld=kl+ku+1;
	stringstream ss(data_txt);
	string line;
	int row=0;
	int col=0;
	T temp_value;
	while (getline(ss, line, '\n')) {
		//开始一行的数据导入
		col=0;
		stringstream ss_in(line);
		string value;
		while (getline(ss_in, value, ',')){
			if(col-row>0 and col-row<=ku)//superline
			  {stringstream ss_inn(value);
			   ss_inn>>temp_value;
			   temp_data[IDX2C(ku-(col-row),col,ld)]=temp_value;
			  }

			if(row-col>0 and row-col<=kl)//sublin
			  {stringstream ss_inn(value);
						ss_inn>>temp_data[IDX2C(row-col+ku,col,ld)];}
			if(row==col)//diagonl
			  {stringstream ss_inn(value);
			   ss_inn>>temp_data[IDX2C(ku,col,ld)];}
//			cout<<value<<"|at_matrix[:="<<IDX2C(row,col,ld)<<"]"<<endl;
			col++;
		}
//		cout<< line << endl;
        ++row;
	}

	//结果输出
	checkCudaErrors(cudaMemcpy(rezult->matrix_data_T, temp_data,(kl+ku+1)*row_num*sizeof(T), cudaMemcpyHostToDevice));

	//打印结果
	 string output="[";
		for(int i=0;i<ld;i++){//row
			for (int j=0;j<row_num;j++) {//col
	   		//cout<<"row:="<<i<<"|col:="<<j<<"|value:"<<rezult[IDX2C(i,j,m)]<<endl;
				stringstream ss;
				ss<<temp_data[IDX2C(i,j,ld)];
				string temp;
				ss>>temp;
				output+=temp;
				if(j!=(row_num-1))
				   output+=",";
				ss.clear();
			}
			if (i!=ld-1)
			   output+="\n";
		}
		output+=("]\n");
		cout<<output<<endl;
	}else{
		cout<<"cublas is not to suport  batched matrix function!"<<endl;
	}
	return rezult;
}

template<class T>//return2维的在ｄｅｖｉｃｅ上的矩阵
T** cudamenory_create(int batchCount,int array_len){
	T** rezult=(T**)malloc(batchCount*sizeof(*rezult));
	for(int i=0;i<batchCount;i++){
	checkCudaErrors(cudaMalloc((void **)&(rezult[i]),array_len*sizeof(rezult[0][0])));
	}
	T** rezult_d;
	checkCudaErrors(cudaMalloc((void **)&(rezult_d),batchCount*sizeof(*rezult_d)));
	checkCudaErrors(cudaMemcpy(rezult_d,rezult,batchCount*sizeof(*rezult),cudaMemcpyHostToDevice));
	return rezult_d;
}


//功能函数封装
//一、cublasgetrfBatched_mat invoke  for matix_LU--start
template<class T>
struct cublasgetrfBatched_mat{
	T **A_value;//output
	T **Carray;
	int *PivotArray;//output
	int *infoArray;//output
	int batchSize;
	int n;

	void init_from_txt(string file_path,long batchsize,long n){
		int char_len=getFileSize(file_path);
		char* matrix_data=(char*)malloc((char_len+20)*sizeof(char));
		mmapSaveDataIntoFiles(file_path,matrix_data);
		matrix_info<T>* A=m2g_packed<T>(matrix_data,batchsize,n,n);
		this->batchSize=A->batchCount;
		this->n=A->row;//A->colomn=A->row;
		this->A_value=A->matrix_data_TT;

		//Carray-----------------todd_start----------------------------
	    //allocate T** hostpoint_hh_N on host an assign value
				int size_N=this->batchSize;
				int pitch_N=(this->n)*(this->n);
				T **hostPointer_hh_N=(T **)malloc(size_N*sizeof(hostPointer_hh_N[0]));
				for (int i= 0;i<size_N;i++) {
					hostPointer_hh_N[i]=(T* )malloc(pitch_N*sizeof(hostPointer_hh_N[0][0]));
						  // assign value to hostPointer[i][j]
					for (int j=0;j<pitch_N;j++) {
						 //assign some value
						hostPointer_hh_N[i][j]=0;
					}
				}
				//allocate T* hd on host  and T** hd on device
				//todd shoud be used frist from the template todd
				//hostPointer_N be created by to hh and  make some value to hh
				T **hostPointer_hd=(T **)malloc(size_N*sizeof(hostPointer_hd[0]));
				for (int i= 0;i<size_N;i++) {
					checkCudaErrors(cudaMalloc((void **)(&hostPointer_hd[i]),pitch_N*sizeof(hostPointer_hd[0][0])));
					checkCudaErrors(cudaMemcpy(hostPointer_hd[i], hostPointer_hh_N[i],pitch_N*sizeof(hostPointer_hh_N[0][0]), cudaMemcpyHostToDevice));
				}
				checkCudaErrors(cudaMalloc((void **)(&this->Carray),size_N*sizeof(this->Carray[0])));
				checkCudaErrors(cudaMemcpy(this->Carray, hostPointer_hd,size_N*sizeof(this->Carray[0]), cudaMemcpyHostToDevice));
		//-----------------todd_end----------------------------------

		//--------*PivotArray
	    checkCudaErrors(cudaMalloc(&this->PivotArray, batchsize*n*sizeof(this->PivotArray[0])));
		//-----------------todd_end----------------------------------

		//------------------*infoArray
		checkCudaErrors(cudaMalloc(&this->infoArray, batchsize*sizeof(this->infoArray[0])));
        cout<<"inited all right!"<<endl;
	}
};

template<class T>
cublasgetrfBatched_mat<T>* cublasgetrfBatched(string path="/tool-lf/matix_data/QR.txt",long batchsize=3,long n=4,bool print_yorn=true)
{cout<<"cublasgetrfBatched  computing......"<<endl;
 cublasHandle_t handle_cublas;
 cublasCreate(&handle_cublas);
 cublasgetrfBatched_mat<T>* getrfBatched_mat=(cublasgetrfBatched_mat<T>* )malloc(sizeof(cublasgetrfBatched_mat<T>));
 getrfBatched_mat->init_from_txt(path,batchsize,n);
 cout<<"compute result is all right? "<<cublasSgetrfBatched(handle_cublas,
		                           getrfBatched_mat->n,
                                   getrfBatched_mat->A_value,
                                   getrfBatched_mat->n,
                                   getrfBatched_mat->PivotArray,
                                   getrfBatched_mat->infoArray,
                                   getrfBatched_mat->batchSize)<<endl;

  //dp1_matirx_printf(A->matrix_data_T,m,n);
  if (print_yorn==true)
     { cout<<"LU_mat:"<<endl;
	   dp2printf(getrfBatched_mat->A_value,3,4,4);
	   cout<<"PivotArray:"<<endl;
       dp1_matirx_printf(getrfBatched_mat->PivotArray,4,3 );
       cout<<"infoArray:"<<endl;
       dp1_matirx_printf(getrfBatched_mat->infoArray,1,3);
     }
  return getrfBatched_mat;
}

//二、cublasgetrifBatched_mat invoke  for matix_inverse--start
template<class T>
cublasgetrfBatched_mat<T>* cublasgetrifBatched(string path="/tool-lf/matix_data/QR.txt",long batchsize=3,long n=4,bool print_yorn=true)
{cout<<"cublasgetrfBatched  computing......"<<endl;
 cublasHandle_t handle_cublas;
 cublasCreate(&handle_cublas);
 cublasgetrfBatched_mat<T>* getrfBatched_mat=(cublasgetrfBatched_mat<T>* )malloc(sizeof(cublasgetrfBatched_mat<T>));
 getrfBatched_mat->init_from_txt(path,batchsize,n);
 cout<<"compute result is all right? "<<cublasSgetrfBatched(handle_cublas,
		                           getrfBatched_mat->n,
                                   getrfBatched_mat->A_value,
                                   getrfBatched_mat->n,
                                   getrfBatched_mat->PivotArray,
                                   getrfBatched_mat->infoArray,
                                   getrfBatched_mat->batchSize)<<endl;

 cout<<"compute result is all right? "<<cublasSgetriBatched(handle_cublas,
		                                getrfBatched_mat->n,
		                                getrfBatched_mat->A_value,
		                                getrfBatched_mat->n,
		                                getrfBatched_mat->PivotArray,
		                                getrfBatched_mat->Carray,
		                                getrfBatched_mat->n,
		                                getrfBatched_mat->infoArray,
		                                getrfBatched_mat->batchSize)<<endl;

  //dp1_matirx_printf(A->matrix_data_T,m,n);
  if (print_yorn==true)
     { cout<<"inverse_mat:"<<endl;
	   dp2printf(getrfBatched_mat->Carray,3,4,4);
	   cout<<"PivotArray:"<<endl;
       dp1_matirx_printf(getrfBatched_mat->PivotArray,4,3 );
       cout<<"infoArray:"<<endl;
       dp1_matirx_printf(getrfBatched_mat->infoArray,1,3);
     }
  return getrfBatched_mat;
}

//三、cublasgetrsfBatched_mat invoke  for matix_LU--start，invoking  is false!
template<class T>
struct cublasgetrsfBatched_mat:cublasgetrfBatched_mat<T>{
	T** Barray;
	int nrhs;
	void init_from_txt2(string file_path,long batchsize,long n,long nrhs){
		int char_len=getFileSize(file_path);
		char* matrix_data=(char*)malloc((char_len+20)*sizeof(char));
		mmapSaveDataIntoFiles(file_path,matrix_data);
		matrix_info<T>* A=m2g_packed<T>(matrix_data,batchsize,n,n);
		this->batchSize=A->batchCount;
		this->n=A->row;//A->colomn=A->row;
		this->A_value=A->matrix_data_TT;
		this->nrhs=nrhs;

		//Barray-----------------todd_start----------------------------
		//allocate T** hostpoint_hh_N on host an assign value
		int size_N=this->batchSize;
		int pitch_N=(this->n)*(this->nrhs);
		T **hostPointer_hh_N=(T **)malloc(size_N*sizeof(hostPointer_hh_N[0]));
		for (int i= 0;i<size_N;i++) {
			hostPointer_hh_N[i]=(T* )malloc(pitch_N*sizeof(hostPointer_hh_N[0][0]));
				  // assign value to hostPointer[i][j]
			for (int j=0;j<pitch_N;j++) {
				 //assign some value
				hostPointer_hh_N[i][j]=0;
			}
		}
		//allocate T* hd on host  and T** hd on device
		//todd shoud be used frist from the template todd
		//hostPointer_N be created by to hh and  make some value to hh
		T **hostPointer_hd=(T **)malloc(size_N*sizeof(hostPointer_hd[0]));
		for (int i= 0;i<size_N;i++) {
			checkCudaErrors(cudaMalloc((void **)(&hostPointer_hd[i]),pitch_N*sizeof(hostPointer_hd[0][0])));
			checkCudaErrors(cudaMemcpy(hostPointer_hd[i], hostPointer_hh_N[i],pitch_N*sizeof(hostPointer_hh_N[0][0]), cudaMemcpyHostToDevice));
		}
		checkCudaErrors(cudaMalloc((void **)(&this->Barray),size_N*sizeof(this->Barray[0])));
		checkCudaErrors(cudaMemcpy(this->Barray, hostPointer_hd,size_N*sizeof(this->Barray[0]), cudaMemcpyHostToDevice));
		//-----------------todd_end----------------------------------

		//--------*PivotArray
	    checkCudaErrors(cudaMalloc(&this->PivotArray, batchsize*n*sizeof(this->PivotArray[0])));
		//-----------------todd_end----------------------------------

		//------------------*infoArray
	    this->infoArray=(int *)malloc(batchsize*sizeof(this->infoArray[0]));
		cout<<"inited all right!"<<endl;
	}
};

template<class T>
cublasgetrsfBatched_mat<T>* cublasgetrsfBatched(string path="/tool-lf/matix_data/QR.txt",long batchsize=3,long n=4,long nrhs=4,bool print_yorn=true)
{cout<<"cublasgetrsfBatched  computing..cublasmatinvBatched...."<<endl;
 cublasHandle_t handle_cublas;
 cublasCreate(&handle_cublas);
 cublasgetrsfBatched_mat<T>* getrsBatched_mat=(cublasgetrsfBatched_mat<T>*)malloc(sizeof(cublasgetrsfBatched_mat<T>));
 getrsBatched_mat->init_from_txt2(path,batchsize,n,nrhs);
 cout<<"compute result is all right? "<<cublasSgetrsBatched(handle_cublas,
		                           CUBLAS_OP_N,
		                           getrsBatched_mat->n,
		                           getrsBatched_mat->nrhs,
		                           getrsBatched_mat->A_value,
		                           getrsBatched_mat->n,
		                           getrsBatched_mat->PivotArray,
		                           getrsBatched_mat->Barray,
		                           getrsBatched_mat->n,
		                           getrsBatched_mat->infoArray,
		                           getrsBatched_mat->batchSize)<<endl;

  //dp1_matirx_printf(A->matrix_data_T,m,n);
  if (print_yorn==true)
     { cout<<"LU_mat:"<<endl;
	   dp2printf(getrsBatched_mat->Barray,3,4,4);
	   cout<<"PivotArray:"<<endl;
       dp1_matirx_printf(getrsBatched_mat->PivotArray,4,3);
       cout<<"infoArray:"<<endl;
       dp1_matirx_printf(getrsBatched_mat->infoArray,1,3,false);
     }
  return getrsBatched_mat;
}

//四、less then 32  matrix_inverse
template<class T>
struct cublasmatinvBatched_mat:cublasgetrfBatched_mat<T>{
	T** Ainv;
	void init_from_txt3(string file_path,long batchsize,long n){
		int char_len=getFileSize(file_path);
		char* matrix_data=(char*)malloc((char_len+20)*sizeof(char));
		mmapSaveDataIntoFiles(file_path,matrix_data);
		matrix_info<T>* A=m2g_packed<T>(matrix_data,batchsize,n,n);
		this->batchSize=A->batchCount;
		this->n=A->row;//A->colomn=A->row;
		this->A_value=A->matrix_data_TT;

		//Barray-----------------todd_start----------------------------
		//allocate T** hostpoint_hh_N on host an assign value
		int size_N=this->batchSize;
		int pitch_N=(this->n)*(this->n);
		T **hostPointer_hh_N=(T **)malloc(size_N*sizeof(hostPointer_hh_N[0]));
		for (int i= 0;i<size_N;i++) {
			hostPointer_hh_N[i]=(T* )malloc(pitch_N*sizeof(hostPointer_hh_N[0][0]));
				  // assign value to hostPointer[i][j]
			for (int j=0;j<pitch_N;j++) {
				 //assign some value
				hostPointer_hh_N[i][j]=0;
			}
		}
		//allocate T* hd on host  and T** hd on device
		//todd shoud be used frist from the template todd
		//hostPointer_N be created by to hh and  make some value to hh
		T **hostPointer_hd=(T **)malloc(size_N*sizeof(hostPointer_hd[0]));
		for (int i= 0;i<size_N;i++) {
			checkCudaErrors(cudaMalloc((void **)(&hostPointer_hd[i]),pitch_N*sizeof(hostPointer_hd[0][0])));
			checkCudaErrors(cudaMemcpy(hostPointer_hd[i], hostPointer_hh_N[i],pitch_N*sizeof(hostPointer_hh_N[0][0]), cudaMemcpyHostToDevice));
		}
		checkCudaErrors(cudaMalloc((void **)(&this->Ainv),size_N*sizeof(this->Ainv[0])));
		checkCudaErrors(cudaMemcpy(this->Ainv, hostPointer_hd,size_N*sizeof(this->Ainv[0]), cudaMemcpyHostToDevice));
		//-----------------todd_end----------------------------------

		//------------------*infoArray
		checkCudaErrors(cudaMalloc(&this->infoArray, batchsize*sizeof(this->infoArray[0])));
		cout<<"inited all right!"<<endl;
	}
};

template<class T>
cublasmatinvBatched_mat<T>* cublasmatinvBatched(string path="/tool-lf/matix_data/QR.txt",long batchsize=3,long n=4,bool print_yorn=true)
{cout<<"cublasgetrsfBatched  computing......"<<endl;
 cublasHandle_t handle_cublas;
 cublasCreate(&handle_cublas);
 cublasmatinvBatched_mat<T>* getrsBatched_mat=(cublasmatinvBatched_mat<T>*)malloc(sizeof(cublasmatinvBatched_mat<T>));
 getrsBatched_mat->init_from_txt3(path,batchsize,n);
 cout<<"compute result is all right? "<<cublasSmatinvBatched(handle_cublas,
			                       getrsBatched_mat->n,
		                           getrsBatched_mat->A_value,
		                           getrsBatched_mat->n,
		                           getrsBatched_mat->Ainv,
		                           getrsBatched_mat->n,
		                           getrsBatched_mat->infoArray,
		                           getrsBatched_mat->batchSize)<<endl;

  //dp1_matirx_printf(A->matrix_data_T,m,n);
  if (print_yorn==true)
     { cout<<"LU_mat:"<<endl;
	   dp2printf(getrsBatched_mat->A_value,3,4,4);
	   cout<<"Ainv:"<<endl;
	   dp2printf(getrsBatched_mat->Ainv,3,4,4);
       cout<<"infoArray:"<<endl;
       dp1_matirx_printf(getrsBatched_mat->infoArray,1,3);
     }
  return getrsBatched_mat;
}

//五、QR分解
template<class T>
struct cublasSgeqrfBatched_mat:cublasgetrfBatched_mat<T>{
	T** TauArray;
	int m;
	void init_from_txt2(string file_path,long batchsize,long n,long m){
		int char_len=getFileSize(file_path);
		char* matrix_data=(char*)malloc((char_len+20)*sizeof(char));
		mmapSaveDataIntoFiles(file_path,matrix_data);
		matrix_info<T>* A=m2g_packed<T>(matrix_data,batchsize,n,m);
		this->batchSize=A->batchCount;
		this->n=A->row;//A->colomn=A->row;
		this->A_value=A->matrix_data_TT;
		this->m=A->colomn;

		//TauArray-----------------todd_start----------------------------
		//allocate T** hostpoint_hh_N on host an assign value
		int size_N=this->batchSize;
		int pitch_N=MIN((this->n),(this->m));
		T **hostPointer_hh_N=(T **)malloc(size_N*sizeof(hostPointer_hh_N[0]));
		for (int i= 0;i<size_N;i++) {
			hostPointer_hh_N[i]=(T* )malloc(pitch_N*sizeof(hostPointer_hh_N[0][0]));
				  // assign value to hostPointer[i][j]
			for (int j=0;j<pitch_N;j++) {
				 //assign some value
				hostPointer_hh_N[i][j]=0;
			}
		}
		//allocate T* hd on host  and T** hd on device
		//todd shoud be used frist from the template todd
		//hostPointer_N be created by to hh and  make some value to hh
		T **hostPointer_hd=(T **)malloc(size_N*sizeof(hostPointer_hd[0]));
		for (int i= 0;i<size_N;i++) {
			checkCudaErrors(cudaMalloc((void **)(&hostPointer_hd[i]),pitch_N*sizeof(hostPointer_hd[0][0])));
			checkCudaErrors(cudaMemcpy(hostPointer_hd[i], hostPointer_hh_N[i],pitch_N*sizeof(hostPointer_hh_N[0][0]), cudaMemcpyHostToDevice));
		}
		checkCudaErrors(cudaMalloc((void **)(&this->TauArray),size_N*sizeof(this->TauArray[0])));
		checkCudaErrors(cudaMemcpy(this->TauArray, hostPointer_hd,size_N*sizeof(this->TauArray[0]), cudaMemcpyHostToDevice));
		//-----------------todd_end----------------------------------

		//------------------*infoArray
		this->infoArray=(int *)malloc(batchsize*sizeof(this->infoArray[0]));
		//checkCudaErrors(cudaMalloc(&this->infoArray, batchsize*sizeof(this->infoArray[0])));
		cout<<"inited all right!"<<endl;
	}
};

template<class T>
cublasSgeqrfBatched_mat<T>* cublasSgeqrfBatched(string path="/tool-lf/matix_data/QR1.txt",long batchsize=3,long n=3,long m=3,bool print_yorn=true)
{cout<<"cublasgetrsfBatched  computing......"<<endl;
 cublasHandle_t handle_cublas;
 cublasCreate(&handle_cublas);
 cublasSgeqrfBatched_mat<T>* getrsBatched_mat=(cublasSgeqrfBatched_mat<T>*)malloc(sizeof(cublasSgeqrfBatched_mat<T>));
 getrsBatched_mat->init_from_txt2(path,batchsize,n,m);
 dp2printf(getrsBatched_mat->A_value,batchsize,getrsBatched_mat->n,getrsBatched_mat->m);

 if(typeid(T) == typeid(float))
 {cout<<"compute result is all right? "<<cublasSgeqrfBatched(handle_cublas,
		                           getrsBatched_mat->n,//row
			                       getrsBatched_mat->m,//col
		                           getrsBatched_mat->A_value,
		                           getrsBatched_mat->n,
		                           getrsBatched_mat->TauArray,
		                  		   getrsBatched_mat->infoArray,
		                           getrsBatched_mat->batchSize)<<endl;}

// if(typeid(T) == typeid(double))
// {cout<<"compute result is all right? "<<cublasDgeqrfBatched(handle_cublas,
//			                       getrsBatched_mat->n,//row
//			                       getrsBatched_mat->m,//col
//		                           getrsBatched_mat->A_value,
//		                           getrsBatched_mat->n,
//		                           getrsBatched_mat->TauArray,
//		                  		   getrsBatched_mat->infoArray,
//		                           getrsBatched_mat->batchSize)<<endl;}


  if (print_yorn==true)
     { cout<<"R of QR_mat:"<<endl;
	   dp2printf(getrsBatched_mat->A_value,batchsize,getrsBatched_mat->n,getrsBatched_mat->m);
	   cout<<"TauArray:"<<endl;
	   dp2printf(getrsBatched_mat->TauArray,batchsize,1,MIN(getrsBatched_mat->m,getrsBatched_mat->n));
       cout<<"infoArray:"<<endl;
       dp1_matirx_printf(getrsBatched_mat->infoArray,1,batchsize,false);
     }
  return getrsBatched_mat;
}

//六,最小二乘法cublasSgelsBatched
template<class T>
struct cublasSgelsBatched_mat:cublasgetrfBatched_mat<T>{
	int m;//row>col,m>n
	int nrhs;
    int *info;
	void init_from_txt2(string file_path_x,string file_path_y,long batchsize,long m,long n,long nrhs){
		int char_len=getFileSize(file_path_x);
		char* matrix_data=(char*)malloc((char_len+20)*sizeof(char));

		//input x
		mmapSaveDataIntoFiles(file_path_x,matrix_data);
		matrix_info<T>* A=m2g_packed<T>(matrix_data,batchsize,m,n);//m=row,n=col
		this->batchSize=A->batchCount;
		this->m=A->row;//A->colomn=A->row;
		this->A_value=A->matrix_data_TT;
		this->n=A->colomn;

		//input carray=y
		int char_len_y=getFileSize(file_path_y);
		char* matrix_data_y=(char*)malloc((char_len_y+20)*sizeof(char));
		mmapSaveDataIntoFiles(file_path_y,matrix_data_y);
	    matrix_info<T>* carray=m2g_packed<T>(matrix_data_y,batchsize,m,nrhs);//m=row,n=col,m>n,output n*nrhs
		this->nrhs=nrhs;
		this->Carray=carray->matrix_data_TT;

		//------------------*info
        this->info=(int *)malloc(batchsize*sizeof(this->info[0]));
//	    checkCudaErrors(cudaMalloc(&this->info, batchsize*sizeof(this->info[0])));

	    //------------------*infoArray
//		this->infoArray=(int *)malloc(batchsize*sizeof(this->infoArray[0]));
		checkCudaErrors(cudaMalloc(&this->infoArray, batchsize*sizeof(this->infoArray[0])));
		cout<<"inited all right!"<<endl;
	}
};

template<class T>
cublasSgelsBatched_mat<T>* cublasSgelsBatched(string path_x="/tool-lf/matix_data/lq_x.txt",string path_y="/tool-lf/matix_data/lq_y.txt",long batchsize=2,long m=27,long n=4,long nrhs=1,bool print_yorn=true)
{cout<<"cublasgetrsfBatched  computing......"<<endl;
 cublasHandle_t handle_cublas;
 cublasCreate(&handle_cublas);
 cublasSgelsBatched_mat<T>* getrsBatched_mat=(cublasSgelsBatched_mat<T>*)malloc(sizeof(cublasSgelsBatched_mat<T>));
 getrsBatched_mat->init_from_txt2(path_x,path_y,batchsize,m,n,nrhs);
// dp2printf(getrsBatched_mat->A_value,batchsize,getrsBatched_mat->m,getrsBatched_mat->n,nrhs);
 if(typeid(T) == typeid(float))
 {cout<<"if 0 all right ,eles wrong!"<<cublasSgelsBatched(handle_cublas,
		 CUBLAS_OP_N,
		 getrsBatched_mat->m,
		 getrsBatched_mat->n,
		 getrsBatched_mat->nrhs,
		 getrsBatched_mat->A_value,
		 getrsBatched_mat->m,
         getrsBatched_mat->Carray,
         getrsBatched_mat->m,
         getrsBatched_mat->info,
         getrsBatched_mat->infoArray,
         getrsBatched_mat->batchSize)<<endl;
  }

  if (print_yorn==true)
     { cout<<"A_value of QR_mat:"<<endl;
	   dp2printf(getrsBatched_mat->A_value,batchsize,getrsBatched_mat->m,getrsBatched_mat->n);
	   cout<<"Carray of QR_mat:"<<endl;
	   dp2printf(getrsBatched_mat->Carray,batchsize,getrsBatched_mat->n,getrsBatched_mat->nrhs);
	   cout<<"info:"<<endl;
	   dp1_matirx_printf(getrsBatched_mat->info,1,batchsize,false);
       cout<<"infoArray:"<<endl;
       dp1_matirx_printf(getrsBatched_mat->infoArray,1,batchsize);
     }
  return getrsBatched_mat;
}

//----------------------------Third-cuspares------------------------
template<class T>
struct coo_mat_h{
	T* cooValA_h;
	int* cooRowIndA;
	int* cooColIndA;
	long nnz;
	int mb;
	int nb;
	cusparseMatDescr_t matdes;
};

template<class T>
struct csr_mat_h{
   long nnz;
   T*  csrValA;
   int* csrRowPtrA;
   int* csrColIndA;
   int mb;
   int nb;
   cudaDataType csrValAtype;
   cusparseMatDescr_t matdes;
};

template<class T>
struct csc_mat_h{
   long nnz;
   T*  cscValA;
   int* cscRowPtrA;
   int* cscColIndA;
   int mb;
   int nb;
   cusparseMatDescr_t matdes;
};

template<class T>
struct ELL_mat_h{
   cusparseMatDescr_t matdes;
};

template<class T>
struct HYB_mat_h{
   cusparseMatDescr_t matdes;
};

template<class T>
struct bsr_mat_h{
   int blockDim=2;
   int mb;
   int nb;
   int nnzb;
   T*  bsrValA;
   int* bsrRowPtrA;
   int* bsrColIndA;
   cusparseMatDescr_t matdes;
   cusparseDirection_t dirA;
};

template<class T>
struct bsrx_mat_h{
   int blockDim=2;
   long mb;
   long nb;
   long nnzb;
   T*  bsrValA;
   int* bsrRowPtrA;
   int* bsrEndPtrA;
   int* bsrColIndA;
   cusparseMatDescr_t matdes;
   cusparseDirection_t dirA;
};

//host读入txt把一个dense结构的matrix,转化为coo格式的sparse_matrix在cpu上
template<class T>
coo_mat_h<T>* mat_s2coo(const char* file_path,int rownum,int colomnnum){
	coo_mat_h<T>* coo_matrix=(coo_mat_h<T>*)malloc(sizeof(coo_mat_h<T>));
	coo_matrix->cooRowIndA=(int *)malloc(rownum*colomnnum*sizeof(int));
	coo_matrix->cooColIndA=(int *)malloc(rownum*colomnnum*sizeof(int));
	coo_matrix->cooValA_h=(T* )malloc(sizeof(T)*(rownum*colomnnum));
	char *data_txt=(char *)malloc(sizeof(char)*(rownum*colomnnum*10));//整体长度设计
	mmapSaveDataIntoFiles(file_path,data_txt);
	//逐行扫描获取矩阵内容
	//checkCudaErrors(cudaMalloc((void **)&(coo_matrix->cooValA_h),row_num*col_num*sizeof(*(rezult->matrix_data_T))));
	stringstream ss(data_txt);
	string line;
	int row=0;
	int col=0;
	int index=0;
	T value_T;
	string value;
	while (getline(ss, line, '\n')) {
		//开始一行的数据导入
		col=0;
		stringstream ss_in(line);
		while(getline(ss_in,value,',')){
			stringstream ss_inn(value);
			ss_inn>>value_T;
			if(value_T!=(T)0.0f)
			 {coo_matrix->cooValA_h[index]=value_T;
			  coo_matrix->cooRowIndA[index]=row;
			  coo_matrix->cooColIndA[index]=col;
			  index++;
			 }
			col++;
		}
		row++;
	}
	coo_matrix->nnz=index;
//	cout<<"nnz:="<<coo_matrix->nnz<<endl;
	//打印结果
//	for(int i=0;i<coo_matrix->nnz;i++)
//	{cout<<"coo_value:="<<coo_matrix->cooValA_h[i]<<",row:="<<coo_matrix->cooRowIndA[i]<<"col:="<<coo_matrix->cooColIndA[i]<<endl;
//	}
	delete data_txt;

	//output rezult
	coo_mat_h<T>* coo_matrix_d=(coo_mat_h<T>*)malloc(sizeof(coo_mat_h<T>));
	coo_matrix_d->mb=rownum;
	coo_matrix_d->nb=colomnnum;
	coo_matrix_d->nnz=coo_matrix->nnz;
	checkCudaErrors(cudaMalloc(&coo_matrix_d->cooColIndA,coo_matrix->nnz*sizeof(int)));
	checkCudaErrors(cudaMemcpy(coo_matrix_d->cooColIndA,coo_matrix->cooColIndA,coo_matrix->nnz*sizeof(coo_matrix_d->cooColIndA[0]), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&coo_matrix_d->cooRowIndA,coo_matrix->nnz*sizeof(int)));
	checkCudaErrors(cudaMemcpy(coo_matrix_d->cooRowIndA, coo_matrix->cooRowIndA,coo_matrix->nnz*sizeof(coo_matrix_d->cooRowIndA[0]), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&coo_matrix_d->cooValA_h,coo_matrix->nnz*sizeof(coo_matrix_d->cooValA_h[0])));
	checkCudaErrors(cudaMemcpy(coo_matrix_d->cooValA_h, coo_matrix->cooValA_h,coo_matrix->nnz*sizeof(coo_matrix->cooValA_h[0]), cudaMemcpyHostToDevice));
	return coo_matrix_d;
}

//dense vector struct
template<class T>
struct vector_dense_d{
	 int nnz=0;
	 int incx=1;
     T* value_d;
     void init(T* vlaue_h,int incx,long nnz=1){
        checkCudaErrors(cudaMalloc((void **)&this->value_d,nnz*sizeof(this->value_d[0])));
		checkCudaErrors(cudaMemcpy(this->value_d, vlaue_h,nnz*sizeof(this->value_d[0]), cudaMemcpyHostToDevice));
		this->incx=incx;//deflaut=1dp1_matirx_printf
		this->nnz=nnz;//100
     }

     void init_txt(char* txtfile,int incx,long nnz=1){
		T *temp_vector=(T *)malloc(nnz*sizeof(T));
		stringstream ss(txtfile);
		string line;
		int i=0;
		while (getline(ss, line, ',')) {
			stringstream ss_inn(line);
			ss_inn>>temp_vector[i];
			i++;
		}

        checkCudaErrors(cudaMalloc((void **)&this->value_d,nnz*sizeof(this->value_d[0])));
		checkCudaErrors(cudaMemcpy(this->value_d,temp_vector,nnz*sizeof(this->value_d[0]), cudaMemcpyHostToDevice));
		this->incx=incx;//deflaut=1dp1_matirx_printf
		this->nnz=nnz;//100
     }
};

//sparse vector struct
template<class T>
struct vector_sparse_d{
	 long    nnz;//total length of the vector
	 int*  xInd;//index of the vector
     T* value_d;

     void init(T* vlaue_h,int* xInd_h,long nnz){
        checkCudaErrors(cudaMalloc((void **)&this->value_d,nnz*sizeof(this->value[0])));
		checkCudaErrors(cudaMemcpy(this->value_d, vlaue_h,nnz*sizeof(this->value[0]), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void**)&this->xInd,nnz*sizeof(xInd_h[0])));
		checkCudaErrors(cudaMemcpy(this->xInd, xInd_h,nnz*sizeof(xInd_h[0]), cudaMemcpyHostToDevice));
		this->nnz=nnz;
     }

     void init_txt(char* txtfile){
        int j=0;
        stringstream ss(txtfile);
        string line;
        T value;
 		while (getline(ss, line, ',')) {
 				stringstream ss_inn(line);
 				ss_inn>>value;
 			    if (value!=0)
 				   {j++;}
 		}

 		this->nnz=j;//100
        int len=j;
		T* temp_vector=(T*)malloc(j*sizeof(T));
		int* temp_xInd=(int*)malloc(j*sizeof(int));

		ss.str("");
		int i=0;
	    j=0;
	    stringstream ss1(txtfile);
		while (getline(ss1, line, ',')) {
			stringstream ss_inn(line);
			ss_inn>>value;
		    if (value!=0)
			   {temp_vector[i]=value;
			    temp_xInd[i]=j;
//			    cout<<value<<endl;
			    i++;
			   }
		    j++;
		}

        checkCudaErrors(cudaMalloc((void **)&this->value_d,this->nnz*sizeof(this->value_d[0])));
		checkCudaErrors(cudaMemcpy(this->value_d,temp_vector,this->nnz*sizeof(this->value_d[0]), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc((void**)&this->xInd,this->nnz*sizeof(temp_xInd[0])));
		checkCudaErrors(cudaMemcpy(this->xInd,temp_xInd,this->nnz*sizeof(temp_xInd[0]), cudaMemcpyHostToDevice));

		for (int i=0;i<len;++i) {
				cout<<"i:="<<temp_xInd[i]<<"|value:="<<temp_vector[i]<<endl;
		}
     }
};

template<class T> //coo be converted to csr
csr_mat_h<T>* create_csr(string filepath="/lf_tool/matrix_cuda/coo",int row=5,int col=5){
	    cusparseHandle_t handle=0;
		cusparseCreate(&handle);

		csr_mat_h<T>* csr_mat_d=(csr_mat_h<T>*)malloc(sizeof(csr_mat_h<T>));
		coo_mat_h<T>* coo_mat_d=mat_s2coo<T>(filepath.c_str(),row,col);
		checkCudaErrors(cudaMalloc(&csr_mat_d->csrRowPtrA,(row+1)*sizeof(int)));
		csr_mat_d->mb=row;
		csr_mat_d->nb=col;
		csr_mat_d->nnz=coo_mat_d->nnz;

		cusparseStatus_t status=cusparseXcoo2csr(handle,
				coo_mat_d->cooRowIndA,
				coo_mat_d->nnz,
				5,
		        csr_mat_d->csrRowPtrA,
		        CUSPARSE_INDEX_BASE_ZERO);

        if(status!=0)
		  {cout<<"cusparseXcoo2csr erres"<<endl;
		   exit(status);
		  }

		int *hostPointer=(int *)malloc((row+1)*sizeof(int));
		checkCudaErrors(cudaMemcpy(hostPointer,csr_mat_d->csrRowPtrA,(row+1)*sizeof(int),cudaMemcpyDeviceToHost));
//		for (int i=0;i<(row+1);++i){
//			cout<<"csrRow:="<<hostPointer[i]<<endl;
//		}

		csr_mat_d->csrColIndA=coo_mat_d->cooColIndA;
		csr_mat_d->csrValA=coo_mat_d->cooValA_h;
		cusparseCreateMatDescr(&csr_mat_d->matdes);

		int *hostPointer_2=(int *)malloc((coo_mat_d->nnz)*sizeof(int));
		checkCudaErrors(cudaMemcpy(hostPointer_2,csr_mat_d->csrColIndA,(coo_mat_d->nnz)*sizeof(int),cudaMemcpyDeviceToHost));
//		for (int i=0;i<coo_mat_d->nnz;++i) {
//				cout<<"csrcol:="<<hostPointer_2[i]<<endl;
//			}
//		cusparseSetMatDiagType(csr_mat_d->matdes,CUSPARSE_DIAG_TYPE_UNIT);
		cout<<"create_csr finished!"<<endl;
		return csr_mat_d;
};

template<class T>
csc_mat_h<T>* csr2csc(csr_mat_h<T>* csr_mat){
cusparseHandle_t handle=0;
cusparseCreate(&handle);
csc_mat_h<T>* csc_mat=(csc_mat_h<T>*)malloc(sizeof(csc_mat_h<T>));
csc_mat->mb=csr_mat->mb;
csc_mat->nb=csr_mat->nb;
csc_mat->nnz=csr_mat->nnz;
checkCudaErrors(cudaMalloc((void **)&(csc_mat->cscValA),(csr_mat->nnz)*sizeof(csc_mat->cscValA[0])));
checkCudaErrors(cudaMalloc((void **)&csc_mat->cscRowPtrA,(csr_mat->nnz)*sizeof(csc_mat->cscRowPtrA[0])));
checkCudaErrors(cudaMalloc((void **)&csc_mat->cscColIndA,(csc_mat->nb+1)*sizeof(csc_mat->cscColIndA[0])));
cusparseStatus_t status=cusparseScsr2csc(handle,
		                                 csr_mat->mb,
		                                 csr_mat->nb,
		                                 csr_mat->nnz,
		                                 csr_mat->csrValA,
		                                 csr_mat->csrRowPtrA,
		                                 csr_mat->csrColIndA,
		                                 csc_mat->cscValA,
		                                 csc_mat->cscRowPtrA,
		                                 csc_mat->cscColIndA,
		                                 CUSPARSE_ACTION_SYMBOLIC,
		                                 CUSPARSE_INDEX_BASE_ZERO);
cout<<cusparseScsr2csc<<status<<endl;
//dp1_matirx_printf(csc_mat->cscRowPtrA,1,csc_mat->nnz);
//dp1_matirx_printf(csc_mat->cscColIndA,1,csc_mat->nb+1);
}

template<class T>
bsr_mat_h<T>* csr2bsr(csr_mat_h<T>* csr_mat,int blockdim=2){
  cusparseHandle_t handle=0;
  cusparseCreate(&handle);
  //block_use
  int m_b=(csr_mat->mb + blockdim-1)/blockdim;
  int n_b=(csr_mat->nb + blockdim-1)/blockdim;

  bsr_mat_h<T>* bsr_mat_rezult=(bsr_mat_h<T>*)malloc(sizeof(bsr_mat_h<T>));
  bsr_mat_rezult->blockDim=blockdim;
  bsr_mat_rezult->mb=csr_mat->mb;
  bsr_mat_rezult->nb=csr_mat->nb;

  cusparseCreateMatDescr(&bsr_mat_rezult->matdes);
  cudaMalloc((void**)&bsr_mat_rezult->bsrRowPtrA, sizeof(int)*(m_b+1));
  bsr_mat_rezult->dirA= CUSPARSE_DIRECTION_COLUMN;
  cusparseStatus_t status=cusparseXcsr2bsrNnz(handle,bsr_mat_rezult->dirA,csr_mat->mb,csr_mat->nb,
		  csr_mat->matdes,csr_mat->csrRowPtrA,csr_mat->csrColIndA,blockdim,
          bsr_mat_rezult->matdes,bsr_mat_rezult->bsrRowPtrA,&(bsr_mat_rezult->nnzb));

  if(status!=0){
	  cout<<"cusparseXcsr2bsrNnz errose"<<endl;
  }

  cudaMalloc((void**)&bsr_mat_rezult->bsrColIndA, sizeof(int)*bsr_mat_rezult->nnzb);
  cudaMalloc((void**)&bsr_mat_rezult->bsrValA, sizeof(float)*(blockdim*blockdim)*bsr_mat_rezult->nnzb);

  status=cusparseScsr2bsr(handle,bsr_mat_rezult->dirA,csr_mat->mb,csr_mat->nb,
		  csr_mat->matdes,csr_mat->csrValA,csr_mat->csrRowPtrA,csr_mat->csrColIndA,blockdim,
          bsr_mat_rezult->matdes
          ,bsr_mat_rezult->bsrValA,bsr_mat_rezult->bsrRowPtrA, bsr_mat_rezult->bsrColIndA);

  if(status!=0){
	  cout<<"cusparseScsr2bsr errose"<<endl;
  }

  //output bsrrow.
  int b=(bsr_mat_rezult->mb+bsr_mat_rezult->blockDim-1)/bsr_mat_rezult->blockDim+1;
  int *hostPointer=(int *)malloc(b*sizeof(hostPointer[0]));
  checkCudaErrors(cudaMemcpy(hostPointer,bsr_mat_rezult->bsrRowPtrA,b*sizeof(hostPointer[0]),cudaMemcpyDeviceToHost));

  for (int i=0;i<b;++i) {
	cout<<hostPointer[i]<<endl;
  }
  cout<<"csr2bsr finished!"<<endl;
  return bsr_mat_rezult;
}

//bsr2bsrx
template<class T>
bsrx_mat_h<T>* bsr2bsrx(bsr_mat_h<T>* bsr_mat){
    bsrx_mat_h<T>* rezult=(bsrx_mat_h<T>*)malloc(sizeof(bsrx_mat_h<T>));
    rezult->blockDim=bsr_mat->blockDim;
    rezult->bsrColIndA=bsr_mat->bsrColIndA;
    rezult->bsrValA=bsr_mat->bsrValA;
    rezult->bsrColIndA=bsr_mat->bsrColIndA;
	rezult->dirA=bsr_mat->dirA;
    rezult->matdes=bsr_mat->matdes;
    rezult->mb=bsr_mat->mb;
    rezult->nb=bsr_mat->nb;
    rezult->nnzb=bsr_mat->nnzb;

    //row and rowend
    int b=(rezult->mb+rezult->blockDim-1)/rezult->blockDim+1;
    int *bsrRow_h=(int *)malloc(b*sizeof(bsrRow_h[0]));
    checkCudaErrors(cudaMemcpy(bsrRow_h,bsr_mat->bsrRowPtrA,b*sizeof(bsrRow_h[0]),cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMalloc(&rezult->bsrRowPtrA,(b-1)*sizeof(rezult->bsrRowPtrA[0])));
    checkCudaErrors(cudaMalloc(&rezult->bsrEndPtrA,(b-1)*sizeof(rezult->bsrEndPtrA[0])));

    int* bsrxrow=(int *)malloc((b-1)*sizeof(int));
    int* bsrxend=(int *)malloc((b-1)*sizeof(int));

    for (int i=0;i<b-1;++i) {
    	bsrxrow[i]=bsrRow_h[i];
    	bsrxend[i]=bsrRow_h[i+1];
	}

    checkCudaErrors(cudaMemcpy(rezult->bsrRowPtrA,bsrxrow,(b-1)*sizeof(bsrxrow[0]),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rezult->bsrEndPtrA,bsrxend,(b-1)*sizeof(bsrxend[0]),cudaMemcpyHostToDevice));
    cout<<"bsrRow:"<<endl;
    dp1_matirx_printf(bsr_mat->bsrRowPtrA,1,b);
    cout<<"bsrxRow:"<<endl;
    dp1_matirx_printf(rezult->bsrRowPtrA,1,b-1);
    cout<<"bsrxEnd:"<<endl;
    dp1_matirx_printf(rezult->bsrEndPtrA,1,b-1);
    return rezult;
}

template<class T> //csr be converted to sr
bsr_mat_h<T>* create_bsr(string filepath="/tool-lf/matix_data/spares_mat",int row=5,int col=5,int blockdim=2){
	csr_mat_h<T>* csr_mat_d=create_csr<T>(filepath,row,col);
	bsr_mat_h<T>* rezult=csr2bsr<T>(csr_mat_d,blockdim);
	return rezult;
}

//create bsrxmv
template<class T>
vector_dense_d<T>* cusparse_bsrmv(string filepath="/lf_tool/matrix_cuda/coo",char * vector="1,1,1,1,1",int m=5,int n=5,int blockdim=2){
    vector_dense_d<T>* rezult=(vector_dense_d<T> *)malloc(sizeof(vector_dense_d<T>));
    cusparseHandle_t handle=0;
    cusparseCreate(&handle);

    //create bsr
   	bsr_mat_h<float>* bsr_mat=create_bsr<float>(filepath,m,n,blockdim);
   	cusparseSetMatDiagType(bsr_mat->matdes,CUSPARSE_DIAG_TYPE_UNIT);

//   	cusparseDiagType_t one=cusparseGetMatDiagType(bsr_mat->matdes);
//   	cout<<one<<endl;
   	//create vector from txt
	vector_dense_d<float>  vector_d;

	int n_b=(bsr_mat->nb+bsr_mat->blockDim-1)/bsr_mat->blockDim;
   	T *temp_vector=(T *)malloc(n_b*bsr_mat->blockDim*sizeof(T));

	stringstream ss(vector);
	string line;
	int i=0;
    while (getline(ss, line, ',')) {
		stringstream ss_inn(line);
	    ss_inn>>temp_vector[i];
	    i++;
    }

    if(i<n_b*bsr_mat->blockDim){
    	for (int index=0;index<n_b*bsr_mat->blockDim-i;++index) {
    		temp_vector[i+index]=0;
		}
    }

   	T beta=0.0;
   	T alpha=1.0;

   	vector_d.init(temp_vector,1,n_b*bsr_mat->blockDim);
   	cusparseStatus_t status=cusparseSbsrmv(handle,bsr_mat->dirA,CUSPARSE_OPERATION_NON_TRANSPOSE,bsr_mat->mb,bsr_mat->nb, bsr_mat->nnzb, &alpha,
       bsr_mat->matdes, bsr_mat->bsrValA, bsr_mat->bsrRowPtrA,bsr_mat->bsrColIndA,bsr_mat->blockDim,vector_d.value_d, &beta,vector_d.value_d);
    cout<<"cusparseSbsrmv status:="<<status<<endl;
    dp1_matirx_printf(vector_d.value_d,1,n_b*bsr_mat->blockDim);
    return rezult;
}


//create bsrxmv
template<class T>
vector_dense_d<T>* cusparse_bsrxmv(string filepath,char* vector,int m,int n,int blockdim,int sizeOfMask_h,int* bsrMaskPtr_h,int* bsrRowPtrA_hh,int* bsrEndPtrA_hh)
{
    vector_dense_d<T>* rezult=(vector_dense_d<T> *)malloc(sizeof(vector_dense_d<T>));
    cusparseHandle_t handle=0;
    cusparseCreate(&handle);

    //create bsr
   	bsr_mat_h<T>* bsr_mat=create_bsr<T>(filepath,m,n,blockdim);
   	bsrx_mat_h<T>* bsrx_mat=bsr2bsrx<T>(bsr_mat);

   	//create vector from txt
	vector_dense_d<T>  vector_d;

	int n_b=(bsrx_mat->nb+bsrx_mat->blockDim-1)/bsrx_mat->blockDim;
   	T *temp_vector=(T *)malloc(n_b*bsrx_mat->blockDim*sizeof(T));

	stringstream ss(vector);
	string line;
	int i=0;
    while (getline(ss, line, ',')) {
		stringstream ss_inn(line);
	    ss_inn>>temp_vector[i];
	    i++;
    }

    if(i<n_b*bsrx_mat->blockDim){
    	for (int index=0;index<n_b*bsr_mat->blockDim-i;++index) {
    		temp_vector[i+index]=0;
		}
    }

   	T beta=0.0;
   	T alpha=1.0;
   	int sizeOfMask=sizeOfMask_h;
   	int* bsrMaskPtr=bsrMaskPtr_h;

   	int *bsrMaskPtr_d;
	checkCudaErrors(cudaMalloc(&bsrMaskPtr_d, sizeOfMask*sizeof(bsrMaskPtr[0])));
	checkCudaErrors(cudaMemcpy(bsrMaskPtr_d,bsrMaskPtr,sizeOfMask*sizeof(bsrMaskPtr[0]), cudaMemcpyHostToDevice));

	//redefine matrix shape
	int* bsrRowPtrA_h=bsrRowPtrA_hh;
	int* bsrEndPtrA_h=bsrEndPtrA_hh;
	checkCudaErrors(cudaMemcpy(bsrx_mat->bsrRowPtrA,bsrRowPtrA_h,n_b*sizeof(bsrRowPtrA_h[0]),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(bsrx_mat->bsrEndPtrA,bsrEndPtrA_h,n_b*sizeof(bsrEndPtrA_h[0]),cudaMemcpyHostToDevice));

   	vector_d.init(temp_vector,1,n_b*bsrx_mat->blockDim);
   	cusparseStatus_t status=cusparseSbsrxmv(handle,bsrx_mat->dirA,CUSPARSE_OPERATION_NON_TRANSPOSE,sizeOfMask,bsrx_mat->mb,bsrx_mat->nb, bsrx_mat->nnzb, &alpha,
       bsrx_mat->matdes, bsrx_mat->bsrValA,bsrMaskPtr_d,bsrx_mat->bsrRowPtrA,bsrx_mat->bsrEndPtrA,bsrx_mat->bsrColIndA,bsrx_mat->blockDim,vector_d.value_d, &beta,vector_d.value_d);

   	cout<<" cusparseSbsrxmv status:="<<status<<endl;
    dp1_matirx_printf(vector_d.value_d,1,n_b*bsrx_mat->blockDim);
    return rezult;
}


template<class T>
void test(string filepath="/lf_tool/matrix_cuda/coo",int row=5,int col=5){
	// Suppose that L is m x m sparse matrix represented by CSR format,
	// L is lower triangular with unit diagonal.
	// Assumption:
	// - dimension of matrix L is m,
	// - matrix L has nnz number zero elements,
	// - handle is already created by cusparseCreate(),
	// - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of L on device memory,
	// - d_x is right hand side vector on device memory,
	// - d_y is solution vector on device memory.
    cusparseHandle_t handle=0;
    cusparseCreate(&handle);
	csr_mat_h<T>* csr_mat_d=create_csr<T>(filepath,row,col);

	cusparseMatDescr_t descr =csr_mat_d->matdes;
	csrsv2Info_t info = 0;
	int pBufferSize;
	void *pBuffer = 0;
	int structural_zero;
	int numerical_zero;
	const double alpha = 1.;
	const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

	// step 1: create a descriptor which contains
	// - matrix L is base-1
	// - matrix L is lower triangular
	// - matrix L has unit diagonal, specified by parameter CUSPARSE_DIAG_TYPE_UNIT
	//   (L may not have all diagonal elements.)
	cusparseCreateMatDescr(&descr);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
	cudaDeviceSynchronize();

	// step 2: create a empty info structure
	cusparseCreateCsrsv2Info(&info);


	// step 3: query how much memory used in csrsv2, and allocate the buffer
	cusparseDcsrsv2_bufferSize(handle, trans,csr_mat_d->mb,csr_mat_d->nnz, descr,
			csr_mat_d->csrValA,csr_mat_d->csrRowPtrA,csr_mat_d->csrColIndA,info,&pBufferSize);
	// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.

	cudaMalloc((void**)&pBuffer, pBufferSize);

    //cout<<"cusparseDcsrsv2_bufferSize finished"<<endl;
	// step 4: perform analysis
	cusparseDcsrsv2_analysis(handle, trans,csr_mat_d->mb,csr_mat_d->nnz, descr,
			csr_mat_d->csrValA,csr_mat_d->csrRowPtrA,csr_mat_d->csrColIndA,
	        info, policy, pBuffer);

	// L has unit diagonal, so no structural zero is reported.
	cusparseStatus_t status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){
	   printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
	}

	// step 5: solve L*y = x
	vector_dense_d<double> d_x;
	d_x.init_txt("1,1,1,1,1",1,5);
	vector_dense_d<double> d_y;
	d_y.init_txt("1,1,1,1,1",1,5);

	cusparseDcsrsv2_solve(handle,trans,csr_mat_d->mb,csr_mat_d->nnz,&alpha,descr,
			csr_mat_d->csrValA,csr_mat_d->csrRowPtrA,csr_mat_d->csrColIndA, info,
	   d_x.value_d, d_y.value_d, policy, pBuffer);

    dp1_matirx_printf<double>(d_y.value_d,1,5);
//	cout<<"cusparseDcsrsv2_solve finished"<<endl;
	// L has unit diagonal, so no numerical zero is reported.
	status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
	if (CUSPARSE_STATUS_ZERO_PIVOT == status){
	   printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
	}

//	cout<<"cusparseDcsrsv2_bufferSize finished"<<endl;
	// step 6: free resources
	cudaFree(pBuffer);
	cusparseDestroyCsrsv2Info(info);
	cusparseDestroyMatDescr(descr);
	cusparseDestroy(handle);
	cout<<"test finished"<<endl;
}

//template<class T>
//void cusparseXgemvi(string filepath="/lf_tool/matrix_cuda/coo",int row=5,int col=5){
//	int len=getFileSize(filepath);
//	char* data=(char*)malloc((len+10)*sizeof(char));
//	mmapSaveDataIntoFiles(filepath,data);
//	matrix_info<T>* matrix_packed=m2g_packed<float>(data,1,5,5);
////  matrix_packed->matrix_data_T;
//	vector_sparse_d<float> x_
//
//	The general procedure is as follows:
//
//	int baseC, nnzC;d;
//	x_d.init_txt("1,0,1,0,1");
//	vector_dense_d<float> y_d;
//	y_d.init_txt("0,0,0,0,0",1,5);
//
//    cusparseHandle_t handle=0;
//    cusparseCreate(&handle);
//    T alpha=1.0;
//    T beta=0.0;
//    int pBufferSize;
//
//    cusparseStatus_t status=cusparseSgemvi_bufferSize(handle,
//    		                  CUSPARSE_OPERATION_NON_TRANSPOSE,
//    		                  matrix_packed->row,
//    		                  matrix_packed->row,
//    		                  x_d.nnz,
//                              &pBufferSize);
//
//    cout<<"pBufferSize:="<<pBufferSize<<endl;
//
//    void* pBuffer=0;
//    cudaMalloc((void**)&pBuffer, pBufferSize);
//
//
//    The general procedure is as follows:
//
//    int baseC, nnzC;
//    cout<<"status"<<status<<endl;
//    status=cusparseSgemvi(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
//    		       matrix_packed->row,matrix_packed->row,&alpha,
//    		       matrix_packed->matrix_data_T,
//    		       matrix_packed->row,x_d.nnz,
//                   x_d.value_d,
//                   x_d.xInd,
//                   &beta,
//                   y_d.value_d,
//                   CUSPARSE_INDEX_BASE_ZERO,
//                   pBuffer);
//
//    cout<<"status"<<status<<endl;
//    dp1_matirx_printf(y_d.value_d,1,5);
//}


//a=r*r^T,wan
template<class T>
void csric0(string filepath="/lf_tool/matrix_cuda/coo",int row=5,int col=5){

	csr_mat_h<T>* csr_mat=create_csr<T>(filepath,row,col);

    cusparseHandle_t handle=0;
    cusparseCreate(&handle);

    dp1_matirx_printf(csr_mat->csrColIndA,1,csr_mat->nnz);
    cusparseSetMatType(csr_mat->matdes,CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatFillMode(csr_mat->matdes,CUSPARSE_FILL_MODE_LOWER);

    cusparseSolveAnalysisInfo_t info;
    cusparseCreateSolveAnalysisInfo(&info);
	cusparseStatus_t status=cusparseScsrsv_analysis(handle,
			                CUSPARSE_OPERATION_NON_TRANSPOSE,
	                        csr_mat->mb,
	                        csr_mat->nnz,
	                        csr_mat->matdes,
	                        csr_mat->csrValA,
	                        csr_mat->csrRowPtrA,
	                        csr_mat->csrColIndA,
	                        info);
//	cudaDeviceSynchronize();
    cout<<"cusparseScsrsv_analysis:"<<status<<endl;

	status=cusparseScsric0(handle,
			        CUSPARSE_OPERATION_NON_TRANSPOSE,
			        csr_mat->mb,
			        csr_mat->matdes,
                    csr_mat->csrValA,
                    csr_mat->csrRowPtrA,
                    csr_mat->csrColIndA,
                    info);
//	cudaDeviceSynchronize();
    cout<<"cusparseScsric0:"<<status<<endl;
    dp1_matirx_printf(csr_mat->csrValA,1,csr_mat->nnz);
    dp1_matirx_printf(csr_mat->csrColIndA,1,csr_mat->nnz);
}

//a=r*r^T,wan
template<class T>
void csric02(string filepath="/lf_tool/matrix_cuda/coo_2",int row=5,int col=5){

	csr_mat_h<T>* csr_mat=create_csr<T>(filepath,row,col);
    cusparseHandle_t handle=0;
    cusparseCreate(&handle);

    int structural_zero;
    int numerical_zero;
    const cusparseSolvePolicy_t policy_M  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;

    cusparseSetMatType(csr_mat->matdes,CUSPARSE_MATRIX_TYPE_GENERAL);
//    cusparseSetMatFillMode(csr_mat->matdes,CUSPARSE_FILL_MODE_LOWER);
    dp1_matirx_printf(csr_mat->csrValA,1,csr_mat->nnz);
    dp1_matirx_printf(csr_mat->csrColIndA,1,csr_mat->nnz);
    dp1_matirx_printf(csr_mat->csrRowPtrA,1,csr_mat->mb+1);

    csric02Info_t info;
    cusparseCreateCsric02Info(&info);

    int pBufferSize=0;
    void *pBuffer = 0;
    cusparseStatus_t status=cusparseScsric02_bufferSize(handle,
                        csr_mat->mb,
                        csr_mat->nnz,
                        csr_mat->matdes,
                        csr_mat->csrValA,
                        csr_mat->csrRowPtrA,
                        csr_mat->csrColIndA,
                        info,
                        &pBufferSize);

    checkCudaErrors(cudaMalloc((void**)&pBuffer, pBufferSize));
    cusparseScsric02_analysis(handle,csr_mat->mb, csr_mat->nnz, csr_mat->matdes,
    		csr_mat->csrValA,csr_mat->csrRowPtrA,csr_mat->csrColIndA,info,
        policy_M, pBuffer);

    status = cusparseXcsric02_zeroPivot(handle, info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
       printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // step 5: M = L * L'
    cusparseScsric02(handle,csr_mat->mb, csr_mat->nnz, csr_mat->matdes,
    		csr_mat->csrValA,csr_mat->csrRowPtrA,csr_mat->csrColIndA,info,
        policy_M, pBuffer);

    status = cusparseXcsric02_zeroPivot(handle, info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
       printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    //cudaDeviceSynchronize();
    cout<<"cusparseScsric0:"<<status<<endl;
    dp1_matirx_printf(csr_mat->csrValA,1,csr_mat->nnz);
    dp1_matirx_printf(csr_mat->csrColIndA,1,csr_mat->nnz);
    dp1_matirx_printf(csr_mat->csrRowPtrA,1,csr_mat->mb+1);

}




/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include csrmvmp_example.cpp
 *   g++ -fopenmp -o csrmvmp_example csrmvmp_example.o -L/usr/local/cuda/lib64 -lcublas -lcusparse -lcudart
 *
 */

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

void eigenvalue()
{
      cublasHandle_t cublasH = NULL;
      cusparseHandle_t cusparseH = NULL;
      cudaStream_t stream = NULL;
      cusparseMatDescr_t descrA = NULL;

      cublasStatus_t cublasStat = CUBLAS_STATUS_SUCCESS;
      cusparseStatus_t cusparseStat = CUSPARSE_STATUS_SUCCESS;
      cudaError_t cudaStat1 = cudaSuccess;
      cudaError_t cudaStat2 = cudaSuccess;
      cudaError_t cudaStat3 = cudaSuccess;
      cudaError_t cudaStat4 = cudaSuccess;
      cudaError_t cudaStat5 = cudaSuccess;
      const int n = 4;
      const int nnzA = 9;
/*
 *      |    1     0     2     3   |
 *      |    0     4     0     0   |
 *  A = |    5     0     6     7   |
 *      |    0     8     0     9   |
 *
 * eigevales are { -0.5311, 7.5311, 9.0000, 4.0000 }
 *
 * The largest eigenvaluse is 9 and corresponding eigenvector is
 *
 *      | 0.3029  |
 * v =  |     0   |
 *      | 0.9350  |
 *      | 0.1844  |
 */
        const int csrRowPtrA[n+1] = { 0, 3, 4, 7, 9 };
        const int csrColIndA[nnzA] = {0, 2, 3, 1, 0, 2, 3, 1, 3 };
        const double csrValA[nnzA] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
        const double lambda_exact[n] = { 9.0000, 7.5311, 4.0000, -0.5311 };
        const double x0[n] = {1.0, 2.0, 3.0, 4.0 }; /* initial guess */
        double x[n]; /* numerical eigenvector */

        int *d_csrRowPtrA = NULL;
        int *d_csrColIndA = NULL;
        double *d_csrValA = NULL;

        double *d_x = NULL; /* eigenvector */
        double *d_y = NULL; /* workspace */

        const double tol = 1.e-6;
        const int max_ites = 30;

        const double h_one  = 1.0;
        const double h_zero = 0.0;

        printf("example of csrmv_mp \n");
        printf("tol = %E \n", tol);
        printf("max. iterations = %d \n", max_ites);

        printf("1st eigenvaluse is %f\n", lambda_exact[0] );
        printf("2nd eigenvaluse is %f\n", lambda_exact[1] );

        double alpha = lambda_exact[1]/lambda_exact[0] ;
        printf("convergence rate is %f\n", alpha );

        double est_iterations = log(tol)/log(alpha);
        printf("# of iterations required is %d\n", (int)ceil(est_iterations)  );

    /* step 1: create cublas/cusparse handle, bind a stream */
        cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        assert(cudaSuccess == cudaStat1);

        cublasStat = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublasStat);

        cublasStat = cublasSetStream(cublasH, stream);
        assert(CUBLAS_STATUS_SUCCESS == cublasStat);

        cusparseStat = cusparseCreate(&cusparseH);
        assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

        cusparseStat = cusparseSetStream(cusparseH, stream);
        assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    /* step 2: configuration of matrix A */
        cusparseStat = cusparseCreateMatDescr(&descrA);
        assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

        cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

        /* step 3: copy A and x0 to device */
            cudaStat1 = cudaMalloc ((void**)&d_csrRowPtrA, sizeof(int) * (n+1) );
            cudaStat2 = cudaMalloc ((void**)&d_csrColIndA, sizeof(int) * nnzA );
            cudaStat3 = cudaMalloc ((void**)&d_csrValA   , sizeof(double) * nnzA );
            cudaStat4 = cudaMalloc ((void**)&d_x         , sizeof(double) * n );
            cudaStat5 = cudaMalloc ((void**)&d_y         , sizeof(double) * n );
            assert(cudaSuccess == cudaStat1);
            assert(cudaSuccess == cudaStat2);
            assert(cudaSuccess == cudaStat3);
            assert(cudaSuccess == cudaStat4);
            assert(cudaSuccess == cudaStat5);

            cudaStat1 = cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (n+1)   , cudaMemcpyHostToDevice);
            cudaStat2 = cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * nnzA    , cudaMemcpyHostToDevice);
            cudaStat3 = cudaMemcpy(d_csrValA   , csrValA   , sizeof(double) * nnzA , cudaMemcpyHostToDevice);
            assert(cudaSuccess == cudaStat1);
            assert(cudaSuccess == cudaStat2);
            assert(cudaSuccess == cudaStat3);


        /*
         * step 4: power method
         */
            double lambda = 0.0;
            double lambda_next = 0.0;

        /*
         *  4.1: initial guess x0
         */
            cudaStat1 = cudaMemcpy(d_x, x0, sizeof(double) * n, cudaMemcpyHostToDevice);
            assert(cudaSuccess == cudaStat1);

            for(int ite = 0 ; ite < max_ites ; ite++ ){
        /*
         *  4.2: normalize vector x
         *      x = x / |x|
         */
                double nrm2_x;
                cublasStat = cublasDnrm2_v2(cublasH,
                                            n,
                                            d_x,
                                            1, // incx,
                                            &nrm2_x  /* host pointer */
                                           );

                assert(CUBLAS_STATUS_SUCCESS == cublasStat);

                double one_over_nrm2_x = 1.0 / nrm2_x;
                cublasStat = cublasDscal_v2( cublasH,
                                             n,
                                             &one_over_nrm2_x,  /* host pointer */
                                             d_x,
                                             1 // incx
                                            );
                assert(CUBLAS_STATUS_SUCCESS == cublasStat);
                /*
                 *  4.3: y = A*x
                 */
                        cusparseStat = cusparseDcsrmv_mp(cusparseH,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n,
                                                         n,
                                                         nnzA,
                                                         &h_one,
                                                         descrA,
                                                         d_csrValA,
                                                         d_csrRowPtrA,
                                                         d_csrColIndA,
                                                         d_x,
                                                         &h_zero,
                                                         d_y);
                        assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

                /*
                 *  4.4: lambda = y**T*x
                 */
                        cublasStat = cublasDdot_v2 ( cublasH,
                                                     n,
                                                     d_x,
                                                     1, // incx,
                                                     d_y,
                                                     1, // incy,
                                                     &lambda_next  /* host pointer */
                                                   );
                        dp1_matirx_printf(d_x,1,4);
                        dp1_matirx_printf(d_y,1,4);
                        cout<<lambda_next<<endl;
                        assert(CUBLAS_STATUS_SUCCESS == cublasStat);

                        double lambda_err = fabs( lambda_next - lambda_exact[0] );
                        printf("ite %d: lambda = %f, error = %E\n", ite, lambda_next, lambda_err );
                /*
                 *  4.5: check if converges
                 */
                        if ( (ite > 0) &&
                             fabs( lambda - lambda_next ) < tol
                        ){
                            break; // converges
                        }

                /*
                 *  4.6: x := y
                 *       lambda = lambda_next
                 *
                 *  so new approximation is (lambda, x), x is not normalized.
                 */
                        cudaStat1 = cudaMemcpy(d_x, d_y, sizeof(double) * n , cudaMemcpyDeviceToDevice);
                        assert(cudaSuccess == cudaStat1);

                        lambda = lambda_next;
                    }
            /*
             * step 5: report eigen-pair
             */
                cudaStat1 = cudaMemcpy(x, d_x, sizeof(double) * n, cudaMemcpyDeviceToHost);
                assert(cudaSuccess == cudaStat1);

                printf("largest eigenvalue is %E\n", lambda );

                printf("eigenvector = (matlab base-1)\n");
                printMatrix(n, 1, x, n, "V0");
                printf("=====\n");


            /* free resources */
                if (d_csrRowPtrA  ) cudaFree(d_csrRowPtrA);
                if (d_csrColIndA  ) cudaFree(d_csrColIndA);
                if (d_csrValA     ) cudaFree(d_csrValA);
                if (d_x           ) cudaFree(d_x);
                if (d_y           ) cudaFree(d_y);

                if (cublasH       ) cublasDestroy(cublasH);
                if (cusparseH     ) cusparseDestroy(cusparseH);
                if (stream        ) cudaStreamDestroy(stream);
                if (descrA        ) cusparseDestroyMatDescr(descrA);

                cudaDeviceReset();
}


template<class T>
void eigenvalue2(string file_path="/lf_tool/matrix_cuda/eigenvalue",int row=5)
{
      cublasHandle_t cublasH = NULL;
      cusparseHandle_t cusparseH = NULL;
//      cudaStream_t stream = NULL;
//      cusparseMatDescr_t descrA = NULL;

/*
 *      |    1     0     2     3   |
 *      |    0     4     0     0   |
 *  A = |    5     0     6     7   |
 *      |    0     8     0     9   |
 *
 * eigevales are { -0.5311, 7.5311, 9.0000, 4.0000 }
 *
 * The largest eigenvaluse is 9 and corresponding eigenvector is
 *
 *      | 0.3029  |
 * v =  |     0   |
 *      | 0.9350  |
 *      | 0.1844  |
 */
      csr_mat_h<T>* csr_mat=create_csr<T>(file_path,row,row);

      T* v=(T*)malloc(row*sizeof(T));
      for(int i=0;i<row;++i) {
		v[i]=i;
	  }

      T* v_d;
	  checkCudaErrors(cudaMalloc((void**)&v_d,row*sizeof(v_d[0])));
	  checkCudaErrors(cudaMemcpy(v_d, v,row*sizeof(T), cudaMemcpyHostToDevice));

	  size_t bufferSizeInBytes=0;
	  T alpha=1.0;
      T beta=0.0;

      cusparseStatus_t status;
	  status=cusparseCsrmvEx_bufferSize(cusparseH,
			                           CUSPARSE_ALG_MERGE_PATH,
			                           CUSPARSE_OPERATION_NON_TRANSPOSE,
			                           csr_mat->mb,
			                           csr_mat->nb,
			                           csr_mat->nnz,
	                                   &alpha,CUDA_R_32F,
									   csr_mat->matdes,
									   csr_mat->csrValA,CUDA_R_32F,
									   csr_mat->csrRowPtrA,
									   csr_mat->csrColIndA,
									   v_d,CUDA_R_32F,
									   &beta,CUDA_R_32F,
									   v_d,CUDA_R_32F,
									   CUDA_R_32F,
									   &bufferSizeInBytes);

     cout<<"cusparseCsrmvEx_bufferSize status:="<<status<<endl;

     void *buffer;
     checkCudaErrors(cudaMalloc((void **)&buffer,bufferSizeInBytes));

	 status=cusparseCsrmvEx(cusparseH,
			               CUSPARSE_ALG_MERGE_PATH,
						   CUSPARSE_OPERATION_NON_TRANSPOSE,
						   csr_mat->mb,
						   csr_mat->nb,
						   csr_mat->nnz,
						   &alpha,CUDA_R_32F,
						   csr_mat->matdes,
						   csr_mat->csrValA,CUDA_R_32F,
						   csr_mat->csrRowPtrA,
						   csr_mat->csrColIndA,
						   v_d,CUDA_R_32F,
						   &beta,CUDA_R_32F,
						   v_d,CUDA_R_32F,
						   CUDA_R_32F,
                           buffer);

	  cout<<"cusparseCsrmvEx status:="<<status<<endl;
      dp1_matirx_printf(v_d,1,row);
      cudaDeviceReset();
}

////////////////////////////////////////////////bsrRowPtrC////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    cout<<"---------------------"<<endl;
//    csric0<float>("/lf_tool/matrix_cuda/coo5",4,4);
//    eigenvalue();
    eigenvalue2<float>();

	return 0;
}
