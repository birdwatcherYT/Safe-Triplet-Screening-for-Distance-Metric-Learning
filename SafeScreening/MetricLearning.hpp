#ifndef METRICLEARNING_HPP
#define METRICLEARNING_HPP

#include "MetricLearningInterface.hpp"

using namespace std;
using namespace Eigen;

class MetricLearning : public MetricLearningInterface {
protected:
    int K;
    set<size_t> EXCEPT;
    
    VectorXi y;
    MatrixXd X;
    MatrixXd M;
    MatrixXd Euclid;
    unsigned dim;
    Index sampleSize;
    vector< tuple<Index, Index, Index> > tripletIndex;
    vector< double > tripletNorm;

    unsigned count_L_Active;
    MatrixXd H_L_Active;
    vector<bool> L_Active;

    MatrixXd Morg;

    static bool exist(double C, const MatrixXd &P, const MatrixXd &Q, double r2, const MatrixXd &Q_project, double eps = 1e-6);
    static bool existLoose(double C, const MatrixXd &P, const MatrixXd &Q_project, double rp2, double eps = 1e-6);
public:
    Index getSampleSize() const;
    MetricLearning(const string& filename);
    void initialize(int k, set<size_t> except);
    double run(const Bound &bound, double lambda_prev, double lambda, double gamma, double alpha, double eps, int freq, unsigned loopMax, ostream &os=cout) ;
    double error(const set<size_t> &indexSet) const;
};

#endif /* METRICLEARNING_HPP */
