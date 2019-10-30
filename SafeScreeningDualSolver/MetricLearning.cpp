#include "MetricLearning.hpp"

MetricLearning::MetricLearning(const string& filename) {
    // data load
    pair<MatrixXd, VectorXi> dataset = dataLoad(filename);
    X = dataset.first;
    y = dataset.second;
    sampleSize = X.rows();
    dim = X.cols();
    Euclid = MatrixXd::Zero(sampleSize, sampleSize);
    for (Index i = 0; i < sampleSize; ++i) 
        for (Index j = 0; j < sampleSize; ++j) 
            if (Euclid.coeffRef(i,j) == 0)
                Euclid.coeffRef(i,j) = Euclid.coeffRef(j,i) = (i==j ? 0 : (X.row(i) - X.row(j)).squaredNorm());
}

size_t MetricLearning::initialize(int k, const set<size_t> &except){
    K = k, EXCEPT = except;
    triplet.clear();
    tripletNorm.clear();
    // triplet
    if (K > 0){
        
        for (Index i = 0; i < sampleSize; ++i) {
            if (EXCEPT.count(i))
                continue;
            vector< pair<double, Index> > distanceJ;
            for (Index j = 0; j < sampleSize; ++j)
                if (!EXCEPT.count(j) && i != j && y.coeff(i) == y.coeff(j))
                    distanceJ.push_back(pair<double, Index>(Euclid.coeff(i,j), j));
            sort(distanceJ.begin(), distanceJ.end());

            vector< pair<double, Index> > distanceL;
            for (Index l = 0; l < sampleSize; ++l)
                if (!EXCEPT.count(l) && y.coeff(i) != y.coeff(l))
                    distanceL.push_back(pair<double, Index>(Euclid.coeff(i,l), l));
            sort(distanceL.begin(), distanceL.end());

            for (size_t k1 = 0, size = distanceJ.size(); k1 < (size_t) K && k1 < size; ++k1) {
                Index j = distanceJ[k1].second;
                const VectorXd& b = X.row(i) - X.row(j);
                double b2=b.squaredNorm();
                for (size_t k2 = 0, size = distanceL.size(); k2 < (size_t) K && k2 < size; ++k2) {
                    Index l = distanceL[k2].second;
                    const VectorXd& a = X.row(i) - X.row(l);
                    double a2=a.squaredNorm(), ab=a.dot(b);
                    triplet.push_back(tuple<Index,Index,Index>(i,j,l));
                    tripletNorm.push_back(a2*a2-2*ab*ab+b2*b2);
                }
            }
        }
        
        /*
        for (Index i = 0; i < sampleSize; ++i) {
            if (EXCEPT.count(i))
                continue;
            vector< pair<double, Index> > distance;
            for (Index j = 0; j < sampleSize; ++j)
                if (!EXCEPT.count(j) && i != j && y.coeff(i) == y.coeff(j))
                    distance.push_back(pair<double, Index>(Euclid.coeff(i,j), j));
            sort(distance.begin(), distance.end());
            for (size_t k = 0, size = distance.size(); k < (size_t) K && k < size; ++k) {
                Index j = distance[k].second;
                for (Index l = 0; l < sampleSize; ++l)
                    if (!EXCEPT.count(l) && y.coeff(i) != y.coeff(l) && Euclid.coeff(i,l) <= distance[k].first + W){
                        const VectorXd& a = X.row(i) - X.row(l);
                        const VectorXd& b = X.row(i) - X.row(j);
                        double a2=a.squaredNorm(), b2=b.squaredNorm(), ab=a.dot(b);
                        triplet.push_back(tuple<Index,Index,Index>(i,j,l));
                        tripletNorm.push_back(a2*a2-2*ab*ab+b2*b2);
                    }
            }
        }
        */
    } else {
        for (Index i = 0; i < sampleSize; ++i){
            if (EXCEPT.count(i))
                continue;
            for (Index j = 0; j < sampleSize; ++j){
                if (!EXCEPT.count(j) && i != j && y.coeff(i) == y.coeff(j)){
                    for (Index l = 0; l < sampleSize; ++l)
                        if (!EXCEPT.count(l) && y.coeff(i) != y.coeff(l)){
                            const VectorXd& a = X.row(i) - X.row(l);
                            const VectorXd& b = X.row(i) - X.row(j);
                            double a2=a.squaredNorm(), b2=b.squaredNorm(), ab=a.dot(b);
                            triplet.push_back(tuple<Index,Index,Index>(i,j,l));
                            tripletNorm.push_back(a2*a2-2*ab*ab+b2*b2);
                        }
                }
            }
        }
    }
    z.setOnes(triplet.size());
    return triplet.size();
}

bool MetricLearning::exist(const MatrixXd &P, const MatrixXd &Q, double r2, const MatrixXd &Q_project, double eps) {
    // âÇ™ë∂ç›Ç∑ÇÈ:true
    double y = 0;
    MatrixXd X = Q_project;
    double dDdy = 1 - P.cwiseProduct(X).sum();
    const double Q2 = Q.squaredNorm();
    double D = -X.squaredNorm() + 2 * y + Q2;
    double dDdy_prev, y_prev, D_prev;
    double alpha = 1.0 / P.squaredNorm();
    //cout << "---------------------------------" << endl;
    while (true) {
        dDdy_prev = dDdy, y_prev = y, D_prev = D;
        if (D > r2)
            return false;
        if (fabs(dDdy) <= eps)
            break;
        y += alpha * dDdy;
        X = projectSemidefinite(Q + P * y);
        dDdy = 1 - P.cwiseProduct(X).sum();
        D = -X.squaredNorm() + 2 * y + Q2;
        //cout << dDdy << " " << y << " " << alpha << " " << D << endl;
        if (fabs(dDdy_prev) < fabs(dDdy) || D_prev > D) {
            alpha *= 0.5;
            if (alpha <= 1e-20)
                return true;
            dDdy = dDdy_prev, y = y_prev, D = D_prev;
            continue;
        }
        if (fabs(D - D_prev) <= 1e-10 * fabs(D))
            break;
        if (dDdy != dDdy_prev)
            alpha = -(y - y_prev) / (dDdy - dDdy_prev);
    }
    return true;
}

bool MetricLearning::existLoose(const MatrixXd &P, const MatrixXd &Q_project, double rp2, double eps) {
    // âÇ™ë∂ç›Ç∑ÇÈ:true
    double y = 0;
    MatrixXd X = Q_project;
    double dDdy = 1 - P.cwiseProduct(X).sum();
    const double Q_project2 = Q_project.squaredNorm();
    double D = 0;
    double dDdy_prev, y_prev, D_prev;
    double alpha = 1.0 / P.squaredNorm();
    //cout << "-------------------------------------" << endl;
    VectorXd v = VectorXd::Random(P.rows());
    while (true) {
        dDdy_prev = dDdy, y_prev = y, D_prev = D;
        if (D > rp2)
            return false;
        if (fabs(dDdy) <= eps)
            break;
        y += alpha * dDdy;
        // X = projectOneNegative(Q_project + P * y);
        X = Q_project + P * y;
        const pair<double, VectorXd>& lam_v = smallestEigh(X, v);
        v = lam_v.second;
        if (lam_v.first < 0)
            X -= lam_v.first * v * v.transpose();
        //-----------------------------------
        dDdy = 1 - P.cwiseProduct(X).sum();
        D = -X.squaredNorm() + 2 * y + Q_project2;
        //cout << dDdy << " " << y << " " << alpha << " " << D << endl;
        if (fabs(dDdy_prev) < fabs(dDdy) || D_prev > D) {
            alpha *= 0.5;
            if (alpha <= 1e-20)
                return true;
            dDdy = dDdy_prev, y = y_prev, D = D_prev;
            continue;
        }
        if (fabs(D - D_prev) <= 1e-10 * fabs(D))
            break;
        if (dDdy != dDdy_prev)
            alpha = -(y - y_prev) / (dDdy - dDdy_prev);
    }
    return true;
}

double MetricLearning::error(const set<size_t> &indexSet) const{
    if(indexSet.empty())
        return 0;
    double err=0;
    for (size_t index : indexSet){
        vector< pair<double, Index> > distance;
        for (Index i = 0; i < sampleSize; ++i){
            if (!EXCEPT.count(i)){
                VectorXd dx = X.row(i) - X.row(index);
                distance.push_back(pair<double, Index>(dx.dot(M*dx), i));
            }
        }
        sort(distance.begin(), distance.end());
        map<int,int> classMap;
        for (size_t k = 0, size = distance.size(); k < (size_t) K && k < size; ++k)
            classMap[y[distance[k].second]]++;
        int max=-1, argmax=0;
        for (const pair<int,int> &p : classMap){
            if (p.second > max)
                max = p.second, argmax = p.first;
        }
        err += (y[index] != argmax);
    }
    return err / indexSet.size();
}


Index MetricLearning::getSampleSize() const{
    return sampleSize;
}

