#include "MetricLearning.hpp"
#include "MetricLearningDiag.hpp"

const double ALPHA = 1e-6; //更新幅初期値
const double GAMMA = 0.05; //smooth度
const unsigned LOOP_MAX = INT_MAX - 1;
const double R = 0.9;
const double QUIT = 0.01;
const bool FULL_METRIC = true;// 対角の場合false


const string result="result/";
const string dataset="../dataset/";


double metricPath(MetricLearningInterface &metric,const MetricLearningInterface::Bound &bound,int k,double lambda_max,double lambda_min,int freq,double eps, double train, ostream &os=cout);
MetricLearningInterface::Bound select(const string& bound);

int main(int argc, char** argv) {
    
    if(argc==9 || argc==10){
        string filename(argv[1]);
        int k=atoi(argv[2]);
        double lambda_max=atof(argv[3]);
        double lambda_min=atof(argv[4]);
        int freq=atoi(argv[5]);
        string bound(argv[6]);
        double eps=atof(argv[7]);
        double train = atof(argv[8]);
        if (train >= 1)
            exit(1);
        if (argc==9){
            MetricLearningInterface *metric;
            if (FULL_METRIC)
                metric=new MetricLearning(dataset+filename);
            else
                metric=new MetricLearningDiag(dataset+filename);
            cout << filename << endl;
            metricPath(*metric, select(bound), k, lambda_max, lambda_min, freq, eps, train);
            delete metric;
        }else{
            ofstream output(result + filename +"_"+ bound);
            if (output.fail()) {
                cout << "file open failed" <<endl;
                exit(1);
            }
            MetricLearningInterface *metric;
            if (FULL_METRIC)
                metric=new MetricLearning(dataset+filename);
            else
                metric=new MetricLearningDiag(dataset+filename);
            output << filename << endl;
            size_t count = atoi(argv[9]);
            vector<string> bounds;
            if (bound=="ALL"){
                bounds.push_back("RRPB_PGB");
                bounds.push_back("RRPB");
                bounds.push_back("PGB");
                bounds.push_back("DGB");
                bounds.push_back("GB");
                bounds.push_back("NO");
            }else{
                bounds.push_back(bound);
            }
            vector<double> means(bounds.size(),0);
            for (size_t i=0;i < count;++i){
                for (size_t j=0;j<bounds.size();++j){
                    srand(i+1);
                    ofstream outputfile(result + filename +"_"+ bounds[j] +"_" + to_string(i));
                    if (outputfile.fail()) {
                        cout << "file open failed" <<endl;
                        exit(1);
                    }
                    means[j] += metricPath(*metric, select(bounds[j]), k, lambda_max, lambda_min, freq, eps, train, outputfile);
                    outputfile.close();
                }
            }
            for (size_t j=0;j<bounds.size();++j){
                means[j] /= count;
                output <<"bound = "<<bounds[j]<<", mean time = "<<means[j]<<endl;
            }
            output.close();
            delete metric;
        }
    }else{
        cout<< "usage : ./run filename k lambda_max lambda_min freq bound eps train (count)"<<endl;
        exit(1);
    }
    return 0;
}

MetricLearningInterface::Bound select(const string& bound){
    if (bound=="NO"){
        return MetricLearningInterface::NO;
    }else if (bound=="GB"){
        return MetricLearningInterface::GB;
    }else if(bound=="PGB"){
        return MetricLearningInterface::PGB;
    }else if(bound=="DGB"){
        return MetricLearningInterface::DGB;
    }else if(bound=="RPB"){
        return MetricLearningInterface::RPB;
    }else if(bound=="RRPB"){
        return MetricLearningInterface::RRPB;
    }else if(bound=="GB_LINEAR"){
        return MetricLearningInterface::GB_LINEAR;
    }else if(bound=="GB_SEMIDEF"){
        return MetricLearningInterface::GB_SEMIDEF;
    }else if(bound=="PGB_SEMIDEF"){
        return MetricLearningInterface::PGB_SEMIDEF;
    }else if(bound=="RRPB_PGB"){
        return MetricLearningInterface::RRPB_PGB;
    }
    exit(1);
}


double metricPath(MetricLearningInterface &metric, const MetricLearningInterface::Bound &bound, int k, double lambda_max,double lambda_min,int freq,double eps, double train, ostream &os) {

    Index sampleSize = metric.getSampleSize();
    vector<size_t> random = randomIndex(sampleSize);
    int exceptSize = (int) (sampleSize*(1-train));
    set<size_t> except;
    for (int i=0;i<exceptSize;++i)
        except.insert(random[i]);
    metric.initialize(k,except);
    
    int Kmax = (int)sqrt(sampleSize-exceptSize);
    os<<"Kmax = "<<Kmax<<endl;
    
    set<size_t> valid;
    set<size_t> test;
    
    for (int i=0;i<exceptSize/2;++i)
        valid.insert(random[i]);
    for (int i=exceptSize/2;i<exceptSize;++i)
        test.insert(random[i]);
    
    vector<double> validError;
    vector<double> testError;
    
    double lambda = lambda_max, loss = -1;
    double lambda_prev = lambda;
    clock_t start = clock();
    while (lambda > lambda_min) {
        os << "*******************" << endl;
        loss = metric.run(bound, lambda_prev, lambda, GAMMA, ALPHA, eps, freq, LOOP_MAX,os);
        os << "loss = "<< loss << endl;
        
        clock_t s=clock();
        vector<double> eV=metric.error(valid,Kmax), eT=metric.error(test,Kmax);
        for (int i=0;i<Kmax;++i){
            os<<i+1<<"-nn: ValidError = "<<eV[i]<<", TestError = "<<eT[i]<<endl;
            validError.push_back(eV[i]);
            testError.push_back(eT[i]);
        }
        clock_t e=clock();
        start += (e-s);
        lambda_prev = lambda;
        lambda *= R;
    }
    clock_t end=clock(); 
    double time = (double) (end-start) / CLOCKS_PER_SEC;
    os << "Total time = " <<time<< endl;

    double min=DBL_MAX, argMin=-1;
    os <<"validError = [";
    for (size_t i=0;i<validError.size();++i){
        if (min>validError[i])
            min=validError[i], argMin=i;
        os << validError[i] <<", " ;
    }
    os <<"]"<<endl;
    os <<"argMin = "<<argMin<<", validMin = "<<min<<endl;
    os << "testError = "<<testError[argMin]<<endl;
    return time;
}

