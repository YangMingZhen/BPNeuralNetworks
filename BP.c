#include "./BP.h"

#define DATASET "MAGIC"

#define TRAINX_FILE "./magic04x.csv"
#define TRAINY_FILE "./magic04y.csv"
#define M 19020

#define TESTX_FILE "testx.csv"
#define TESTX_M 6000

#define ALPHA 0.8
#define EPCHOS 100
#define INPUT 10
#define HIDDEN 100
#define OUTPUT 2
#define BATCH_EPCHOS 1

int stepEpchos = BATCH_EPCHOS;
int semop = 0;

//训练
void BPtrain(BP*net)
{
	int i,j,k,step;
	int checkstep = 1;
	int enStepDisplay = 0;
	int isDisplay = 0;
	int epchos = stepEpchos;
	double alpha = ALPHA;
	
	double temp;
		
	pthread_t tid =  pthread_self();
				
	for ( step = 0; step<epchos;step++)
	{		
		if(enStepDisplay||isDisplay&&(step % checkstep == 0))
			printf("============  epchos[ %d ] ============\n",step+1);
		
		//输入层到隐层传播
		if(isDisplay&(step % checkstep == 0))
			printf("输入层到隐层传播\n");
		for(i = 0;i<(*net).m;i++)
		{
			for(j = 0;j<(*net).hidden_num;j++)
			{
				temp = 0;
				for(k = 0;k<(*net).input_num;k++)
				{
					temp = temp+(*net).input[i][k]*(*net).v[k][j];
				}
				(*net).hidden[i][j] = sigm(temp - (*net).vt[j]);
			}
		}
				
		//隐层到输出层传播
		if(isDisplay&(step % checkstep == 0))
			printf("隐层到输出层传播\n");
		for(i = 0;i<(*net).m;i++)
		{
			for(j = 0;j<(*net).output_num;j++)
			{
				temp = 0;
				for(k = 0;k<(*net).hidden_num;k++)
				{
					temp = temp+(*net).hidden[i][k]*(*net).w[k][j];
				}
				(*net).output[i][j] = sigm(temp - (*net).wt[j]);
			}
		}
		
		//计算误差
		if(isDisplay&(step % checkstep == 0))
			printf("计算误差\n");
		temp = 0;
		for(i = 0;i<(*net).m;i++)
		{
			for(j = 0;j<(*net).output_num;j++)
			{
				temp = temp + fabs((*net).y[i][j]-(*net).output[i][j]);
			}
		}
		///(*net).er[step] = temp/(*net).output_num/(*net).m;
		
		//计算传播误差
		if(isDisplay&(step % checkstep == 0))
			printf("计算传播误差\n");
		
		for(i = 0;i<(*net).m;i++)
		{
			for(j = 0;j<(*net).output_num;j++)
			{
				(*net).g[i][j] = (*net).output[i][j] * (1-(*net).output[i][j]) * ((*net).y[i][j]-(*net).output[i][j]);
			}
		}
		
		
		for(i = 0;i<(*net).m;i++)
		{
			for(j = 0;j<(*net).hidden_num;j++)
			{
				temp = 0;
				for(k = 0;k<(*net).output_num;k++)
					temp = temp+(*net).g[i][k]*(*net).w[j][k];
				(*net).e[i][j] = (*net).hidden[i][j] * (1-(*net).hidden[i][j]) * temp;
			}
		}
		
		
		//修正wt
		if(isDisplay&(step % checkstep == 0))
			printf("修正wt ...\n");
		for(i = 0 ; i<(*net).output_num;i++)
		{
			temp = 0;
			for(j = 0;j<(*net).m;j++)
			{
				temp = temp + (*net).g[j][i];
			}
			(*net).wt[i] = (*net).wt[i] - alpha*temp/(*net).m;
		}
		
		//修正vt
		if(isDisplay&(step % checkstep == 0))
			printf("修正vt ...\n");
		for(i = 0 ; i<(*net).hidden_num;i++)
		{
			temp = 0;
			for(j = 0;j<(*net).m;j++)
			{
				temp = temp + (*net).e[j][i];
			}
			(*net).vt[i] = (*net).vt[i] - alpha*temp/(*net).m;
		}
		
		//修正w
		if(isDisplay&(step % checkstep == 0))
			printf("修正w ...\n");
		for(i = 0 ; i<(*net).output_num;i++)
		{
			for(j = 0;j<(*net).hidden_num;j++)
			{
				temp = 0;
				for(k = 0;k<(*net).m;k++)
					temp = temp + (*net).g[k][i]*(*net).hidden[k][j];
				(*net).w[j][i] = (*net).w[j][i] + alpha*temp/(*net).m;
			}
		}
		
		//修正v
		if(isDisplay&(step % checkstep == 0))
			printf("修正v ...\n");
		for(i = 0 ; i<(*net).hidden_num;i++)
		{
			for(j = 0;j<(*net).input_num;j++)
			{
				temp = 0;
				for(k = 0;k<(*net).m;k++)
					temp = temp + (*net).e[k][i]*(*net).input[k][j];
				(*net).v[j][i] = (*net).v[j][i] + alpha*temp/(*net).m;
			}
		}
		//printWeight(net);
	}//end for step
	
	semop -- ;
}

//测试
void BPtest(BP*net,const char*testx,const char *outfile,int m)
{
	int i,j,k;
	double temp;
	FILE *fp = fopen(testx,"r");
	FILE *fpout = fopen(outfile,"w");
	
	//释放内存
	for(i = 0;i <(*net).m;i++)
	{
		free((*net).input[i]);
		free((*net).hidden[i]);
		free((*net).output[i]);
		free((*net).y[i]);
	}
	free((*net).input);
	free((*net).hidden);
	free((*net).output);
	free((*net).y);
	
	//重新申请空间
	(*net).m = m;
	(*net).input = (double**)malloc(sizeof(double*)*(*net).m);
	(*net).hidden = (double**)malloc(sizeof(double*)*(*net).m);
	(*net).output = (double**)malloc(sizeof(double*)*(*net).m);
	(*net).y = (double**)malloc(sizeof(double*)*(*net).m);
	for(i = 0; i<m;i++)
	{
		(*net).input[i] = (double*)malloc(sizeof(double)*(*net).input_num);
		(*net).hidden[i] = (double*)malloc(sizeof(double)*(*net).hidden_num);
		(*net).output[i] = (double*)malloc(sizeof(double)*(*net).output_num);
		(*net).y[i] = (double*)malloc(sizeof(double)*(*net).output_num);
	}
	
	//读取测试数据
	for(i = 0; i<m;i++)
	{
		for(j = 0;j<(*net).input_num;j++)
		{
			fscanf(fp,"%lf,",&((*net).input[i][j]));
		}
	}
	fclose(fp);
	
	//输入层到隐层传播
	for(i = 0;i<(*net).m;i++)
	{
		for(j = 0;j<(*net).hidden_num;j++)
		{
			temp = 0;
			for(k = 0;k<(*net).input_num;k++)
			{
				temp = temp+(*net).input[i][k]*(*net).v[k][j];
			}
			(*net).hidden[i][j] = sigm(temp - (*net).vt[j]);
		}
	}
	
	//隐层到输出层传播
	for(i = 0;i<(*net).m;i++)
	{
		for(j = 0;j<(*net).output_num;j++)
		{
			temp = 0;
			for(k = 0;k<(*net).hidden_num;k++)
			{
				temp = temp+(*net).hidden[i][k]*(*net).w[k][j];
			}
			(*net).output[i][j] = sigm(temp - (*net).wt[j]);
			if(j != (*net).output_num-1)
				fprintf(fpout,"%lf,",(*net).output[i][j]);
			else
				fprintf(fpout,"%lf",(*net).output[i][j]);
		}
		fprintf(fpout,"\n");
	}
	fclose(fpout);
}

//读取训练数据
void trainDataRead(double **trainx,double **trainy,char *trainxFile,char * trainyFile,int m,int n,int ny)
{
	int i,j;
	int *ind;
	
	FILE *fp = fopen(trainxFile,"r");
	FILE *fp2 = fopen(trainyFile,"r");
	
	ind = randperm(m);
	
	for(i = 0; i<m;i++)
	{
		for(j = 0;j<n;j++)
		{
			fscanf(fp,"%lf,",&trainx[ind[i]][j]);
		}
		for(j = 0;j<ny;j++)
			fscanf(fp2,"%lf,",&trainy[ind[i]][j]);
	}
	fclose(fp);
	fclose(fp2);
}


//将数据载入神经网络
void dataCopy(BP*net,double **trainx,double **trainy)
{
	int i,j;
	for(i = 0;i<(*net).m;i++)
	{
		for(j = 0;j<(*net).input_num;j++)
		{
			(*net).input[i][j] = trainx[i][j];
		}
		for(j = 0;j<(*net).output_num;j++)
			(*net).y[i][j] = trainy[i][j];
	}
}

void updateTime(struct timeval *t)
{
	gettimeofday( &t, NULL );
	printf("%ld.%ld\n",(*t).tv_sec,(*t).tv_usec);
}


//
void parllel(int nthreads,const char *predfile)
{
	//------------------------------------
	int i,j;
	int sumEpchos = 0;
	int md = 1;
	int md0;
	int *mx;
	int ret = 0;
	char trainxfile[] = TRAINX_FILE;
	char trainyfile[] = TRAINY_FILE;
	//------------------------------------
	double **trainx;
	double **trainy;
	//------------------------------------
	long int tp;
	struct timeval start,end;
	//------------------------------------
	pthread_t *tid;
	//------------------------------------
	BP **net;
	BP *netm;
	BP *nets;
	//------------------------------------
	//训练数据文件读取
	trainx = (double**)malloc(sizeof(double*)*M);
	trainy = (double**)malloc(sizeof(double*)*M);
	for(i = 0;i<M;i++)
	{
		trainx[i] = (double*)malloc(sizeof(double)*INPUT);
		trainy[i] = (double*)malloc(sizeof(double)*OUTPUT);
	}
	trainDataRead(trainx,trainy,trainxfile,trainyfile,M,INPUT,OUTPUT);
	
	//------------------------------------
	tid = (pthread_t*)malloc(sizeof(pthread_t)*nthreads);
	netm = (BP*)malloc(sizeof(BP));
	net = (BP**)malloc(sizeof(BP*)*nthreads);
	for(i = 0;i<nthreads;i++)
		net[i] = (BP*)malloc(sizeof(BP));
	
	//------------------------------------
	BPset(netm,INPUT,HIDDEN,OUTPUT,md);
	BPinit(netm);
	
	mx = split(M,nthreads);
	md0 = 0;
	for(i = 0;i<nthreads;i++)
	{
		md = mx[i]-md0+1;
		BPset(net[i],INPUT,HIDDEN,OUTPUT,md);
		BPinit(net[i]);
		weightCopy(net[i],netm);
		dataCopy(net[i],&trainx[md0],&trainy[md0]);
		md0 = mx[i]+1;
	}
	
	//---------------------------------------------
	//开启多线程训练
	gettimeofday( &start, NULL );
	printf("%ld.%ld,",start.tv_sec,start.tv_usec);
	
	sumEpchos = 0;
	for (i = 0;sumEpchos<EPCHOS;i++ )
	{
		if(sumEpchos+BATCH_EPCHOS > EPCHOS) stepEpchos = EPCHOS - BATCH_EPCHOS;
		else stepEpchos = BATCH_EPCHOS;
		semop = nthreads;
		
		// open multi threads
		for(j = 0;j< nthreads;j++)
			ret = pthread_create(&tid[j],NULL,(void *) BPtrain,net[j]);
		
		// wait until the process end 
		for(j = 0;j<nthreads;j++){
			ret = pthread_join(tid[j],NULL);
		}
		
		// update weight
		
		//while(semop);
		weightCopy(netm,net[0]);
		for(j = 1;j<nthreads;j++)
			BPmerge(netm,net[j]);
		for(j = 0;j<nthreads;j++)
			weightCopy(net[j],netm);
		//
		sumEpchos = sumEpchos + stepEpchos;
	}
	
	gettimeofday( &end, NULL );
	printf("%ld.%ld,",end.tv_sec,end.tv_usec);
	tp = end.tv_sec-start.tv_sec;
	
	
	//BPtest(netm,TESTX_FILE,predfile,TESTX_M);
	//----------------------------------------------
	printf("%ld\n",tp);
}

int main()
{
	printf("============[ MAGIC BATCH_EPCHOS = %d ]===========\n",BATCH_EPCHOS);
	parllel(1,"pred1.csv");
	parllel(2,"pred2.csv");
	parllel(4,"pred4.csv");
	parllel(8,"pred8.csv");
	parllel(16,"pred16.csv");
	parllel(28,"pred28.csv");
	parllel(32,"pred32.csv");
	parllel(56,"pred56.csv");
	parllel(64,"pred64.csv");
	printf("\n");
	return 0;
}





