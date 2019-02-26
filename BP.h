#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

typedef struct BP{
	int m;
	int input_num;
	int hidden_num;
	int output_num;
	double **v;
	double *vt;
	double **w;
	double *wt;
	double **input;
	double **hidden;
	double **output;
	double **y;
	double *er;
} BP;


double sigm(double x)
{
	double y;
	y = 1/(1+exp(-x));
	return y;
}

//
int *randperm(int m)
{
	int i;
	int *x;
	int n;
	int temp;
	
	x = (int*)malloc(sizeof(int)*m);
	srand(time(NULL));
	for(i = 0;i<m;i++)
		x[i] = i;
	
	for(i = 0;i<m;i++)
	{
		n = rand()%(m-i);
		temp = x[i];
		x[i] = x[i+n];
		x[i+n] = temp;
	}
	return x;
}

int * split(int m,int n)
{
	int i;
	int *x;

	double sum ;
	double d = m-1;
	
	d = d/n;
	x = (int*)malloc(sizeof(int)*n);

	sum = d;
	for(i = 0;i<n;i++)
	{
		x[i] = (int)sum;		
		sum = sum + d;
	}
	return x;
}


/*
* 初始化BP神经网络，随机初始化权值
*/
void BPinit(BP *net)
{
	int i,j;
	double *sumv;
	double sumvt;
	double *sumw;
	double sumwt;

	sumv = (double*)malloc(sizeof(double)*(*net).hidden_num);
	sumw = (double*)malloc(sizeof(double)*(*net).output_num);
	for(i = 0;i<(*net).hidden_num;i++)
		sumv[i] = 0;
	for(i = 0;i<(*net).output_num;i++)
		sumw[i] = 0;

	srand(time(NULL));
	
	//初始化输入到隐层权值
	(*net).v = (double**)malloc(sizeof(double*)*(*net).input_num);
	for(i = 0;i<(*net).input_num;i++)
	{
		(*net).v[i] = (double*)malloc(sizeof(double)*(*net).hidden_num);
		for(j = 0;j<(*net).hidden_num;j++)
		{
			(*net).v[i][j] = (double)rand()/RAND_MAX;
			sumv[j] += (*net).v[i][j];
		}
	}

	for(i = 0;i<(*net).input_num;i++)
		for(j = 0;j<(*net).hidden_num;j++)
			(*net).v[i][j] = (*net).v[i][j]/sumv[j];
	
	//初始化输入到隐层阈值
	(*net).vt = (double*)malloc(sizeof(double)*(*net).hidden_num);
	sumvt = 0;
	for(i = 0;i<(*net).hidden_num;i++)
	{
		(*net).vt[i] = (double)rand()/RAND_MAX;
		sumvt += (*net).vt[i];
	}
	for(i = 0;i<(*net).hidden_num;i++)
		(*net).vt[i] = (*net).vt[i]/sumvt;
	
	//初始化隐层到输出层权值
	(*net).w = (double**)malloc(sizeof(double*)*(*net).hidden_num);
	for(i = 0;i<(*net).hidden_num;i++)
	{
		(*net).w[i] = (double*)malloc(sizeof(double)*(*net).output_num);
		for(j = 0;j<(*net).output_num;j++)
		{
			(*net).w[i][j] = (double)rand()/RAND_MAX;
			sumw[j] += (*net).w[i][j];
		}
	}

	for(i = 0;i<(*net).hidden_num;i++)
		for(j = 0;j<(*net).output_num;j++)
			(*net).w[i][j] = (*net).w[i][j]/sumw[j];
	
	//初始化隐层到输出层阈值
	(*net).wt = (double*)malloc(sizeof(double)*(*net).output_num);
	for(i = 0;i<(*net).output_num;i++)
	{
		(*net).wt[i] = (double)rand()/RAND_MAX;
		sumwt += (*net).wt[i];
	}
	for(i = 0;i<(*net).output_num;i++)
		(*net).wt[i] = (*net).wt[i] / sumwt;
}

//构建BP模型
void BPset(BP *net,int input_num,int hidden_num,int output_num,int m)
{
	int i;
	(*net).input_num = input_num;
	(*net).hidden_num = hidden_num;
	(*net).output_num = output_num;
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
}


void BPcopy(BP*net1,BP*net2)
{
	int i,j;
	BPset(net1,(*net2).input_num,(*net2).hidden_num,(*net2).output_num,(*net2).m);
	BPinit(net1);
	//复制输入到隐层权值
	for(i = 0;i<(*net1).input_num;i++)
	{
		
		for(j = 0;j<(*net1).hidden_num;j++)
		{
			(*net1).v[i][j] = (*net2).v[i][j];
		}
	}
	
	//复制输入到隐层阈值
	for(i = 0;i<(*net1).hidden_num;i++)
	{
		(*net1).vt[i] = (*net2).vt[i];
	}
	
	//复制隐层到输出层权值
	for(i = 0;i<(*net1).hidden_num;i++)
	{
		
		for(j = 0;j<(*net1).output_num;j++)
		{
			(*net1).w[i][j] = (*net2).w[i][j];
		}
	}
	
	//复制隐层到输出层阈值
	for(i = 0;i<(*net1).output_num;i++)
	{
		(*net1).wt[i] = (*net2).wt[i];
	}
}

void weightCopy(BP*net1,BP*net2)
{
	int i,j;
	//复制输入到隐层权值
	for(i = 0;i<(*net1).input_num;i++)
	{
		
		for(j = 0;j<(*net1).hidden_num;j++)
		{
			(*net1).v[i][j] = (*net2).v[i][j];
		}
	}
	
	//复制输入到隐层阈值
	for(i = 0;i<(*net1).hidden_num;i++)
	{
		(*net1).vt[i] = (*net2).vt[i];
	}
	
	//复制隐层到输出层权值
	for(i = 0;i<(*net1).hidden_num;i++)
	{
		
		for(j = 0;j<(*net1).output_num;j++)
		{
			(*net1).w[i][j] = (*net2).w[i][j];
		}
	}
	
	//复制隐层到输出层阈值
	for(i = 0;i<(*net1).output_num;i++)
	{
		(*net1).wt[i] = (*net2).wt[i];
	}
}




//
void BPmerge(BP*net1,BP*net2)
{
	int i,j;
	
	//复制输入到隐层权值
	for(i = 0;i<(*net1).input_num;i++)
	{
		
		for(j = 0;j<(*net1).hidden_num;j++)
		{
			(*net1).v[i][j] = ((*net1).v[i][j]+(*net2).v[i][j])/2;
		}
	}
	
	//复制输入到隐层阈值
	for(i = 0;i<(*net1).hidden_num;i++)
	{
		(*net1).vt[i] = ((*net2).vt[i]+(*net1).vt[i])/2;
	}
	
	//复制隐层到输出层权值
	for(i = 0;i<(*net1).hidden_num;i++)
	{
		
		for(j = 0;j<(*net1).output_num;j++)
		{
			(*net1).w[i][j] = ((*net2).w[i][j]+(*net1).w[i][j])/2;
		}
	}
	
	//复制隐层到输出层阈值
	for(i = 0;i<(*net1).output_num;i++)
	{
		(*net1).wt[i] = ((*net1).wt[i]+(*net2).wt[i])/2;
	}
}




//
void printData(BP*net,int type)
{
	int i,j;
	int m = (*net).m;
	FILE *fpw;
	
	if(type == 1 || type == 5)
	{
		fpw = fopen("input.csv","w");
		for(i = 0;i<m;i++)
		{
			for(j = 0;j<(*net).input_num;j++)
				fprintf(fpw,"%lf ",(*net).input[i][j]);
			fprintf(fpw,"\n");
		}
		fclose(fpw);
	}
	
	if(type == 2 || type == 5)
	{
		fpw = fopen("hidden.csv","w");
		for(i = 0;i<m;i++)
		{
			for(j = 0;j<(*net).hidden_num;j++)
				fprintf(fpw,"%lf ",(*net).hidden[i][j]);
			fprintf(fpw,"\n");
		}
		fclose(fpw);
	}
	
	if(type == 3 || type == 5)
	{
		fpw = fopen("output.csv","w");
		for(i = 0;i<m;i++)
		{
			for(j = 0;j<(*net).output_num;j++)
				fprintf(fpw,"%lf ",(*net).output[i][j]);
			fprintf(fpw,"\n");
		}
		fclose(fpw);
	}
	
	if(type == 4 || type == 5)
	{
		fpw = fopen("y.csv","w");
		for(i = 0;i<m;i++)
		{
			for(j = 0;j<(*net).output_num;j++)
				fprintf(fpw,"%lf ",(*net).y[i][j]);
			fprintf(fpw,"\n");
		}
		fclose(fpw);
	}
}

//
void printWeight(BP*net)
{
	int i,j;
	FILE *fpw = fopen("weight.csv","w");
	//v
	//printf("v = \n");
	for(i = 0;i<(*net).input_num;i++)
	{
		for(j = 0;j<(*net).hidden_num;j++)
		{
			fprintf(fpw,"%lf ",(*net).v[i][j]);
		}
		fprintf(fpw,"\n");
	}
	
	//vt
	//printf("vt = \n");
	for(i = 0;i<(*net).hidden_num;i++)
	{
		fprintf(fpw,"%lf ",(*net).vt[i]);
	}
	fprintf(fpw,"\n");
	
	
	//w
	//printf("w = \n");
	for(i = 0;i<(*net).hidden_num;i++)
	{
		for(j = 0;j<(*net).output_num;j++)
		{
			fprintf(fpw,"%lf ",(*net).w[i][j]);
		}
		fprintf(fpw,"\n");
	}
	
	//wt
	//printf("wt = \n");
	for(i = 0;i<(*net).output_num;i++)
	{
		fprintf(fpw,"%lf ",(*net).wt[i]);
	}
	fprintf(fpw,"\n");
	fclose(fpw);
}