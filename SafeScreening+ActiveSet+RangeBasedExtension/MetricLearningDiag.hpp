#ifndef METRICLEARNINGDIAG_HPP
#define METRICLEARNINGDIAG_HPP

#include "MetricLearningInterface.hpp"

using namespace std;
using namespace Eigen;

class MetricLearningDiag : public MetricLearningInterface{
protected:
    int K;
    set<size_t> EXCEPT;
    
    VectorXi y;
    MatrixXd X;
    VectorXd M;
    MatrixXd Euclid;
    unsigned dim;
    Index sampleSize;
    vector< tuple<Index, Index, Index> > tripletIndex;
    vector< double > tripletNorm;

    vector< pair<double, bool> > tripletLR;//true:L, false:R
    double accuracy;// ||M-M^*|| <= accuracy

    unsigned count_L_Active;
    VectorXd H_L_Active;
    vector<bool> L_Active;

    static bool exist(double C, const VectorXd &p, const VectorXd &q, double r2, const VectorXd &q_project, double eps = 1e-6);
    static double minimize(const VectorXd &p, const VectorXd &q, double r2);
public:
    Index getSampleSize() const;
    MetricLearningDiag(const string& filename);
    void initialize(int k, set<size_t> except);
    double run(const Bound &bound, double lambda_prev, double lambda, double gamma, double alpha, double eps, int freq, unsigned loopMax, ostream &os=cout) ;
    double error(const set<size_t> &indexSet) const;
};

#endif /* METRICLEARNINGDIAG_HPP */
