#ifndef METRICLEARNING_HPP
#define METRICLEARNING_HPP

#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <map>
#include <algorithm>
#include <tuple>
#include <set>
#include "Tools.hpp"

using namespace std;
using namespace Eigen;

class MetricLearning {
protected:
    int K;
    set<size_t> EXCEPT;
    
    VectorXi y;
    MatrixXd X;
    MatrixXd M;
    VectorXd z;
    MatrixXd Euclid;
    unsigned dim;
    Index sampleSize;
    vector< tuple<Index, Index, Index> > triplet;
    vector< double > tripletNorm;
    static bool exist(const MatrixXd &P, const MatrixXd &Q, double r2, const MatrixXd &Q_project, double eps = 1e-6);
    static bool existLoose(const MatrixXd &P, const MatrixXd &Q_project, double rp2, double eps = 1e-6);
public:
    Index getSampleSize() const;
    MetricLearning(const string& filename);
    size_t initialize(int k, const set<size_t> &except);
    virtual double run(double lambda_prev, double lambda, double alpha, double eps, int freq, unsigned loopMax, ostream &os=cout) = 0;
    double error(const set<size_t> &indexSet) const;
    virtual ~MetricLearning(){};
};

#endif /* METRICLEARNING_HPP */
