#include "MetricLearning.hpp"
#include "MetricLearningNO.hpp"
#include "MetricLearningPGB.hpp"

const double ALPHA = 1; //çXêVïùèâä˙íl
const unsigned LOOP_MAX = INT_MAX - 1;
const double R = 0.9;
const double QUIT = 0.01;

const string result="result/";
const string dataset="../dataset/";

double metricPath(MetricLearning &metric,int k,double lambda_max,int freq,double eps, double train, ostream &os=cout);


int main(int argc, char** argv) {
    
    if(argc==7 || argc==9){
        string filename(argv[1]);
        int k=atoi(argv[2]);
        double lambda_max=atof(argv[3]);
        int freq=atoi(argv[4]);
        string bound(argv[5]);
        double eps=atof(argv[6]);
        if (argc==7){
            MetricLearning *metric;
            if (bound=="PGB")
                metric=new MetricLearningPGB(dataset+filename);
            else
                metric=new MetricLearningNO(dataset+filename);
            cout << filename << endl;
            metricPath(*metric, k, lambda_max, freq, eps, 1);
            delete metric;
        }else{
            ofstream output(result + filename +"_"+ bound);
            if (output.fail()) {
                cout << "file open failed" <<endl;
                exit(1);
            }
            MetricLearning *metric;
            if (bound=="PGB")
                metric=new MetricLearningPGB(dataset+filename);
            else
                metric=new MetricLearningNO(dataset+filename);
            output << filename << endl;
            double train = atof(argv[7]);
            size_t count = atoi(argv[8]);
            if (train > 1)
                exit(1);
            string bound = "PGB";
            double mean = 0;
            for (size_t i=0;i < count;++i){
                srand(i+1);
                ofstream outputfile(result + filename +"_"+ bound +"_" + to_string(i));
                if (outputfile.fail()) {
                    cout << "file open failed" <<endl;
                    exit(1);
                }
                mean += metricPath(*metric, k, lambda_max, freq, eps, train, outputfile);
                outputfile.close();
            }
            mean /= count;
            output <<"bound = "<<bound<<", mean time = "<<mean<<endl;
            output.close();
            delete metric;
        }
    }else{
        cout<< "usage : ./run filename k lambda_max freq bound eps (train count)"<<endl;
        exit(1);
    }
    return 0;
}

double metricPath(MetricLearning &metric, int k, double lambda_max,int freq,double eps, double train, ostream &os) {

    Index sampleSize = metric.getSampleSize();
    vector<size_t> random = randomIndex(sampleSize);
    int exceptSize = (int) (sampleSize*(1-train));
    set<size_t> except;
    for (int i=0;i<exceptSize;++i)
        except.insert(random[i]);
    metric.initialize(k,except);

    double lambda = lambda_max, loss = -1;
    double lambda_prev = lambda, loss_prev = loss;
    clock_t start = clock();
    while (lambda > 0) {
        os << "*******************" << endl;
        loss_prev = loss;
        loss = metric.run(lambda_prev, lambda, ALPHA, eps, freq, LOOP_MAX,os);
        os << "loss = "<< loss << endl;
        double quit = ((loss_prev-loss)/loss_prev) * (lambda_prev/(lambda_prev-lambda));
        os << "quit = " << quit << endl;
        if (lambda!=lambda_prev && quit < QUIT)
            break;
        lambda_prev = lambda;
        lambda *= R;
    }
    clock_t end=clock(); 
    double time = (double) (end-start) / CLOCKS_PER_SEC;
    os << "Total time = " <<time<< endl;
    return time;
}
