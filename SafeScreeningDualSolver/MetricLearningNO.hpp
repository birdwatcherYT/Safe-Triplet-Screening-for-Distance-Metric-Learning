#ifndef METRICLEARNING_NO_HPP
#define METRICLEARNING_NO_HPP

#include "MetricLearning.hpp"

using namespace std;
using namespace Eigen;

class MetricLearningNO : public MetricLearning {
public:
    MetricLearningNO(const string& filename) : MetricLearning(filename){};

    double run(double lambda_prev, double lambda, double alpha, double eps, int freq, unsigned loopMax,ostream &os) {
        os << "k, lambda, alpha, eps, loopMax, dim, freq" << endl;
        os << K << ", "<< lambda << ", " << alpha << ", " << eps << ", " << loopMax << ", " << dim << ", " << freq << endl;
        const size_t length = triplet.size();
        os << "#triplet = " << length << endl;

        vector<size_t> index(length);
        for (size_t i = 0; i < length; ++i)
            index[i] = i;
        VectorXd z_prev = z;
        double dual_prev = -DBL_MAX;

        M = MatrixXd::Zero(dim, dim);
        double M2;

        VectorXd dDdz = VectorXd::Zero(length);
        VectorXd dDdz_prev = dDdz;
        double loss = -1;

        clock_t start = clock(), screeningTime = 0;
        for (unsigned loop = 0; loop < loopMax; ++loop) {
            os << "-----" << "loop = " << loop << "-----" << endl;

            M = MatrixXd::Zero(dim,dim);
            double z_sum = 0;
            for (size_t i : index) {
                if (z.coeff(i)==0)
                    continue;
                const VectorXd& a = X.row(get<0>(triplet[i])) - X.row(get<2>(triplet[i]));
                const VectorXd& b = X.row(get<0>(triplet[i])) - X.row(get<1>(triplet[i]));
                const MatrixXd& Hijl =  a*a.transpose()-b*b.transpose();
                if (z.coeff(i)==1)
                    M += Hijl, z_sum += 1;
                else
                    M += z.coeff(i) * Hijl, z_sum += z.coeff(i);
            }
            M = projectSemidefinite(M) / lambda;
            M2 = M.squaredNorm();

            double dual = -0.5 * lambda * M2 + z_sum;
            if (dual < dual_prev){
                alpha *= 0.1;
                os << "alpha = " << alpha << endl;
                z = z_prev;
                --loop;
            } else {
                double primal = 0;
                for (size_t i : index) {
                    const VectorXd& a = X.row(get<0>(triplet[i])) - X.row(get<2>(triplet[i]));
                    const VectorXd& b = X.row(get<0>(triplet[i])) - X.row(get<1>(triplet[i]));
                    //const MatrixXd& Hijl = a * a.transpose() - b * b.transpose();
                    const double value = a.dot(M*a)-b.dot(M*b);
                    dDdz.coeffRef(i) = 1 - value;
                    if (dDdz.coeff(i) > 0)
                        primal += dDdz.coeff(i);
                }

                primal += 0.5 * lambda * M2;
                double gap = primal - dual;
                os << "dual = " << dual  << ", primal = " << primal <<  ", gap = " << gap << endl;
                if (gap <= eps * primal){
                    loss = primal - 0.5 * lambda * M2;
                    break;
                }

                if (loop != 0){
                    double dot=0, zSqr=0, dDdzSqr=0;
                    for (size_t i : index) {
                        dot += (z.coeff(i)-z_prev.coeff(i))*(dDdz.coeff(i)-dDdz_prev.coeff(i));
                        zSqr += (z.coeff(i)-z_prev.coeff(i))*(z.coeff(i)-z_prev.coeff(i));
                        dDdzSqr += (dDdz.coeff(i)-dDdz_prev.coeff(i))*(dDdz.coeff(i)-dDdz_prev.coeff(i));
                    }
                    alpha = 0.5*fabs(dot/dDdzSqr + zSqr/dot);
                    if (std::isnan(alpha)){
                        os<<"alpha is nan."<<endl;
                        break;
                    }
                }

                z_prev = z, dual_prev = dual, dDdz_prev = dDdz;
            }

            for (size_t i : index) {
                z.coeffRef(i) += alpha * dDdz.coeff(i);
                if (z.coeff(i) < 0)
                    z.coeffRef(i) = 0;
                else if (z.coeff(i) > 1)
                    z.coeffRef(i) = 1;
            }
        }
        clock_t end = clock();
        os << "time = " << (double) (end - start) / CLOCKS_PER_SEC << endl;
        os << "totalScreeningTime = " << (double) screeningTime / CLOCKS_PER_SEC << endl;
        //os << "M = " << endl << M << endl;
        os << "||M||^2 = " << M2 << endl;
        unsigned count_L_fact = 0, count_C_fact = 0, count_R_fact = 0;
        for (Index i = 0;i <(int) length;++i){
            if (z.coeff(i)==0)
                ++count_R_fact;
            else if (z.coeff(i)==1)
                ++count_L_fact;
            else
                ++count_C_fact;
        }
        os << "#L* = " << count_L_fact << endl;
        os << "#C* = " << count_C_fact << endl;
        os << "#R* = " << count_R_fact << endl;
        return loss;
    }
};

#endif /* METRICLEARNING_NO_HPP */
