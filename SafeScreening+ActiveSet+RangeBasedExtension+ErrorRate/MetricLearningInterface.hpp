#ifndef METRICLEARNINGINTERFACE_HPP
#define METRICLEARNINGINTERFACE_HPP

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

class MetricLearningInterface {
public:
    enum Bound{NO, GB, PGB, DGB, RPB, RRPB, GB_LINEAR, GB_SEMIDEF, PGB_SEMIDEF, RRPB_PGB};
    virtual Index getSampleSize() const=0;
    virtual void initialize(int k, set<size_t> except)=0;
    virtual double run(const Bound &bound, double lambda_prev, double lambda, double gamma, double alpha, double eps, int freq, unsigned loopMax, ostream &os=cout) = 0;
    virtual vector<double> error(const set<size_t> &indexSet, int Ksample) const=0;
    virtual ~MetricLearningInterface(){};
};

#endif /* METRICLEARNINGINTERFACE_HPP */
