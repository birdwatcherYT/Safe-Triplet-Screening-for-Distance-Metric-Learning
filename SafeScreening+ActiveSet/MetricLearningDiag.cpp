#include "MetricLearningDiag.hpp"

MetricLearningDiag::MetricLearningDiag(const string& filename) {
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

void MetricLearningDiag::initialize(int k, set<size_t> except){
    K = k, EXCEPT = except;
    tripletIndex.clear();
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
                const VectorXd& bb = b.cwiseProduct(b);
                for (size_t k2 = 0, size = distanceL.size(); k2 < (size_t) K && k2 < size; ++k2) {
                    Index l = distanceL[k2].second;
                    const VectorXd& a = X.row(i) - X.row(l);
                    const VectorXd& aa = a.cwiseProduct(a);
                    const VectorXd& Hijl = aa - bb;
                    tripletIndex.push_back(tuple<Index,Index,Index>(i,j,l));
                    tripletNorm.push_back(Hijl.squaredNorm());
                }
            }
        }
    }
    M = VectorXd::Zero(dim);
    H_L_Active = VectorXd::Zero(dim);
    L_Active.clear();
    L_Active.resize(tripletIndex.size(), false);
    count_L_Active=0;
}

bool MetricLearningDiag::exist(double C, const VectorXd &p, const VectorXd &q, double r2, const VectorXd &q_project, double eps) {
    // âÇ™ë∂ç›Ç∑ÇÈ:true
    double y = 0;
    VectorXd x = q_project;
    double dDdy = C - p.dot(x);
    const double q2 = q.squaredNorm();
    double D = -x.squaredNorm() + 2 * C * y + q2;
    double dDdy_prev, y_prev, D_prev;
    double alpha = 1.0 / p.squaredNorm();
    //cout << "---------------------------------" << endl;
    while (true) {
        dDdy_prev = dDdy, y_prev = y, D_prev = D;
        if (D > r2)
            return false;
        if (fabs(dDdy) <= eps)
            break;
        y += alpha * dDdy;
        x = projectPositive(q + p * y);
        dDdy = C - p.dot(x);
        D = -x.squaredNorm() + 2 * y * C + q2;
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

double MetricLearningDiag::minimize(const VectorXd& p, const VectorXd& q, double r2) {
    set<double> alphaset;
    alphaset.insert(0);
    Index size = p.size();
    for (Index i = 0; i < size; ++i) {
        if (q.coeff(i) != 0 && p.coeff(i) / q.coeff(i) >= 0)
            alphaset.insert(0.5 * p.coeff(i) / q.coeff(i));
    }
    const unsigned length = alphaset.size();
    const vector<double> alpha(alphaset.begin(), alphaset.end());
    for (unsigned i = 0; i < length; ++i) {
        const double alphaMid = alpha[i] + (i + 1 >= length ? 2 * alpha[i] : alpha[i + 1]);
        double R2 = r2, pq = 0, pp = 0;
        for (Index j = 0; j < size; ++j) {
            if (p.coeff(j) > alphaMid * q.coeff(j)) {
                R2 -= q.coeff(j) * q.coeff(j);
            } else {
                pq += p.coeff(j) * q.coeff(j);
                pp += p.coeff(j) * p.coeff(j);
            }
        }
        if (R2 < 0)
            continue;
        if (pp == 0) {
            if (R2 == 0 || alpha[i] == 0)
                return pq;
        } else if (R2 != 0) {
            double a = 0.5 * sqrt(pp) / sqrt(R2);
            if (a >= alpha[i] && (i + 1 >= length || a < alpha[i + 1]))
                return pq - 0.5 * pp / a;
        }
    }
    return DBL_MAX;
}

double MetricLearningDiag::error(const set<size_t> &indexSet) const{
    if(indexSet.empty())
        return 0;
    double err=0;
    for (size_t index : indexSet){
        vector< pair<double, Index> > distance;
        for (Index i = 0; i < sampleSize; ++i){
            if (!EXCEPT.count(i)){
                VectorXd dx = X.row(i) - X.row(index);
                distance.push_back(pair<double, Index>(dx.dot(M.cwiseProduct(dx)), i));
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


Index MetricLearningDiag::getSampleSize() const{
    return sampleSize;
}

double MetricLearningDiag::run(const Bound &bound, double lambda_prev, double lambda, double gamma, double alpha, double eps, int freq, unsigned loopMax, ostream &os) {
    os<<"lambda = "<<lambda<<endl;
    os << "k, gamma, alpha, eps, loopMax, dim, freq" << endl;
    os << K << ", " << gamma << ", " << alpha << ", " << eps << ", " << loopMax << ", " << dim << ", " << freq << endl;
    const size_t length = tripletIndex.size();
    os << "#triplet = " << length << endl;

    double primal_prev = DBL_MAX;
    unsigned count_L = 0, count_R = 0;

    double M2;
    VectorXd M_prev = M;

    VectorXd Grad = VectorXd::Zero(dim);
    VectorXd Grad_prev = Grad;
    
    vector<double> dot(length);
    double loss=0;

    vector<bool> activeSet(length, true);
    bool reset = true;
    vector<bool> screening(length, false);

    clock_t start = clock(), screeningTime = 0;
    for (unsigned loop = 0; loop < loopMax; ++loop) {
        os << "-----" << "loop = " << loop << "-----" << endl;

        loss=0;
        reset = (reset || loop%freq==0);
        VectorXd H = VectorXd::Zero(dim);
        double z_sum = 0, z2 = 0;
        for (size_t i = 0; i < length; ++i) {
            if (!activeSet[i] || screening[i])
                continue;
            const VectorXd& a = X.row(get<0>(tripletIndex[i])) - X.row(get<2>(tripletIndex[i]));
            const VectorXd& b = X.row(get<0>(tripletIndex[i])) - X.row(get<1>(tripletIndex[i]));
            double value = a.dot(M.cwiseProduct(a))-b.dot(M.cwiseProduct(b));
            dot[i] = value;
            if (value < 1-gamma){
                if (!L_Active[i])
                    H_L_Active += a.cwiseProduct(a)-b.cwiseProduct(b), ++count_L_Active, L_Active[i]=true;
                //loss += 1-value-0.5*gamma;
            }else if (L_Active[i] || value < 1){
                const VectorXd& Hijl =  a.cwiseProduct(a)-b.cwiseProduct(b);
                if (L_Active[i])
                    H_L_Active -= Hijl, --count_L_Active, L_Active[i]=false;
                if(value < 1){
                    loss += 0.5*(1-value)*(1-value)/gamma, H += ((1-value)/gamma) * Hijl, z_sum += ((1-value)/gamma), z2 += ((1-value)/gamma)*((1-value)/gamma);
                }
            }
        }
        M2 = M.squaredNorm();
        double primal = loss + ((1-0.5*gamma)*count_L_Active - M.dot(H_L_Active)) + 0.5 * lambda * M2;
        if (primal > primal_prev) {
            os << "primal = " << primal  << endl;
            alpha *= 0.1;
            os << "alpha = " << alpha << endl;
            --loop;
            M = projectPositive(M_prev - alpha * Grad);
            continue;
        }
        if (reset){
            for (size_t i = 0; i < length; ++i) {
                if (activeSet[i] || screening[i])
                    continue;
                const VectorXd& a = X.row(get<0>(tripletIndex[i])) - X.row(get<2>(tripletIndex[i]));
                const VectorXd& b = X.row(get<0>(tripletIndex[i])) - X.row(get<1>(tripletIndex[i]));
                double value = a.dot(M.cwiseProduct(a))-b.dot(M.cwiseProduct(b));
                dot[i] = value;
                if (value < 1-gamma){
                    if (!L_Active[i])
                        H_L_Active += a.cwiseProduct(a)-b.cwiseProduct(b), ++count_L_Active, L_Active[i]=true;
                    //loss += 1-value-0.5*gamma;
                }else if (L_Active[i] || value < 1){
                    const VectorXd& Hijl =  a.cwiseProduct(a)-b.cwiseProduct(b);
                    if (L_Active[i])
                        H_L_Active -= Hijl, --count_L_Active, L_Active[i]=false;
                    if(value < 1){
                        loss += 0.5*(1-value)*(1-value)/gamma, H += ((1-value)/gamma) * Hijl, z_sum += ((1-value)/gamma), z2 += ((1-value)/gamma)*((1-value)/gamma);
                    }
                }
            }
            loss += (1-0.5*gamma)*count_L_Active - M.dot(H_L_Active);
            primal = loss + 0.5 * lambda * M2;
        }
        H += H_L_Active, z_sum += count_L_Active, z2 += count_L_Active;
        double alphaH2 = projectPositive(H).squaredNorm();
        double dual = -0.5 * alphaH2/lambda + z_sum - 0.5 * gamma * z2;
        double gap = primal - dual;
        os << "primal = " << primal  << ", dual = " << dual <<  ", gap = " << gap << endl;
        if (primal - dual <= eps * fabs(primal)){
            if (reset)
                break;
            reset=true, --loop;
            os<<"convergence"<<endl;
            continue;
        }

        Grad = -H + lambda * M;

        if (loop % freq == 0) {
            // ----------Screening Start----------
            clock_t screeningStart = clock(); 
            switch(bound){
                case NO:{
                }break;
                case GB:{
                    const VectorXd &gradientDiv2Lam = Grad / (2*lambda);
                    const VectorXd &Q = M - gradientDiv2Lam;
                    const double r2 = gradientDiv2Lam.squaredNorm();
                    for (size_t i = 0; i < length; ++i) {
                        if (screening[i])
                            continue;
                        if (dot[i] <= 1-gamma || 1 <= dot[i]){
                            const VectorXd& a = X.row(get<0>(tripletIndex[i])) - X.row(get<2>(tripletIndex[i]));
                            const VectorXd& b = X.row(get<0>(tripletIndex[i])) - X.row(get<1>(tripletIndex[i]));
                            const double Hijl2 = tripletNorm[i];
                            const double HijlQ = a.dot(Q.cwiseProduct(a)) - b.dot(Q.cwiseProduct(b));
                            const double sqrtHijl2r2 = sqrt(Hijl2 * r2);
                            if (HijlQ + sqrtHijl2r2 <= 1-gamma) {
                                ++count_L, screening[i]=true;
                                continue;
                            } else if (HijlQ - sqrtHijl2r2 >= 1) {
                                ++count_R, screening[i]=true;
                                continue;
                            }
                        }
                    }
                }break;
                case PGB:{
                    const VectorXd &gradientDiv2Lam = Grad / (2*lambda);
                    const VectorXd &Q = M - gradientDiv2Lam;
                    const double r2 = gradientDiv2Lam.squaredNorm();
                    const VectorXd &Qp = projectPositive(Q);
                    const double rp2 = r2 - (Q - Qp).squaredNorm();
                    for (size_t i = 0; i < length; ++i) {
                        if (screening[i])
                            continue;
                        if (dot[i] <= 1-gamma || 1 <= dot[i]){
                            const VectorXd& a = X.row(get<0>(tripletIndex[i])) - X.row(get<2>(tripletIndex[i]));
                            const VectorXd& b = X.row(get<0>(tripletIndex[i])) - X.row(get<1>(tripletIndex[i]));
                            const double Hijl2 = tripletNorm[i];
                            const double HijlQp = a.dot(Qp.cwiseProduct(a)) - b.dot(Qp.cwiseProduct(b));
                            const double sqrtHijl2rp2 = sqrt(Hijl2 * rp2);
                            if (HijlQp + sqrtHijl2rp2 <= 1-gamma) {
                                ++count_L, screening[i]=true;
                                continue;
                            } else if (HijlQp - sqrtHijl2rp2 >= 1) {
                                ++count_R, screening[i]=true;
                                continue;
                            }
                        }
                    }
                }break;
                case DGB:{
                    const double r2 = 2 * gap / lambda;
                    for (size_t i = 0; i < length; ++i) {
                        if (screening[i])
                            continue;
                        const double Hijl2 = tripletNorm[i];
                        const double HijlQ = dot[i];
                        const double sqrtHijl2r2 = sqrt(Hijl2 * r2);
                        if (HijlQ + sqrtHijl2r2 <= 1-gamma) {
                            ++count_L, screening[i]=true;
                            continue;
                        } else if (HijlQ - sqrtHijl2r2 >= 1) {
                            ++count_R, screening[i]=true;
                            continue;
                        }
                    }
                }break;
                case RPB:{
                    if (loop == 0 && lambda != lambda_prev) {
                        const double r2 = M2*((lambda_prev-lambda)/(2*lambda))*((lambda_prev-lambda)/(2*lambda));
                        for (size_t i = 0; i < length; ++i) {
                            if (screening[i])
                                continue;
                            const double Hijl2 = tripletNorm[i];
                            const double HijlQ = dot[i] * ((lambda_prev+lambda)/(2*lambda));
                            const double sqrtHijl2r2 = sqrt(Hijl2 * r2);
                            if (HijlQ + sqrtHijl2r2 <= 1-gamma) {
                                ++count_L, screening[i]=true;
                                continue;
                            } else if (HijlQ - sqrtHijl2r2 >= 1) {
                                ++count_R, screening[i]=true;
                                continue;
                            }
                        }
                    }
                }break;
                case RRPB:{
                    const double r = (loop == 0 && lambda < lambda_prev) ? 
                    (((lambda_prev-lambda)/(2*lambda)) * sqrt(M2) 
                        + (lambda_prev/lambda) * sqrt(2*(loss+0.5*lambda_prev*M2+0.5*alphaH2/lambda_prev-z_sum + 0.5 * gamma * z2)/lambda_prev))
                    : sqrt(2 * gap / lambda);
                    const double coeff = (loop == 0 && lambda < lambda_prev) ? ((lambda_prev+lambda)/(2*lambda)) : 1;
                    for (size_t i = 0; i < length; ++i) {
                        if (screening[i])
                            continue;
                        const double Hijl2 = tripletNorm[i];
                        const double HijlQ = dot[i] * coeff;
                        const double sqrtHijl2r2 = r * sqrt(Hijl2);
                        if (HijlQ + sqrtHijl2r2 <= 1-gamma) {
                            ++count_L, screening[i]=true;
                            continue;
                        } else if (HijlQ - sqrtHijl2r2 >= 1) {
                            ++count_R, screening[i]=true;
                            continue;
                        }
                    }
                }break;
                case GB_LINEAR:{
                    const VectorXd& gradientDiv2Lam = Grad / (2*lambda);
                    const VectorXd& Q = M - gradientDiv2Lam;
                    const double r2 = gradientDiv2Lam.squaredNorm();
                    const VectorXd& Qp = projectPositive(Q);
                    const VectorXd& Qm = Q - Qp;
                    const double Qm2 = Qm.squaredNorm();
                    for (size_t i = 0; i < length; ++i) {
                        if (screening[i])
                            continue;
                        if (dot[i] <= 1-gamma || 1 <= dot[i]){
                            const VectorXd& a = X.row(get<0>(tripletIndex[i])) - X.row(get<2>(tripletIndex[i]));
                            const VectorXd& b = X.row(get<0>(tripletIndex[i])) - X.row(get<1>(tripletIndex[i]));
                            const double Hijl2 = tripletNorm[i];
                            double QHijl = a.dot(Q.cwiseProduct(a)) - b.dot(Q.cwiseProduct(b)), sqrtHijl2r2=sqrt(Hijl2*r2);
                            if (dot[i] <= 1 - gamma) {
                                if (QHijl + sqrtHijl2r2 <= 1-gamma) {
                                    ++count_L, screening[i]=true;
                                    continue;
                                }
                                double QmHijl = a.dot(Qm.cwiseProduct(a)) - b.dot(Qm.cwiseProduct(b));
                                if (Qm2+r2*QmHijl/sqrtHijl2r2 <= 0)
                                    continue;
                                double alpha = -sqrt( (Hijl2*Qm2-QmHijl*QmHijl)/(Qm2*(r2-Qm2)) );
                                double beta = alpha - QmHijl/Qm2;
                                if (QHijl-(beta*QmHijl+Hijl2)/alpha <= 1-gamma){
                                    ++count_L, screening[i]=true;
                                    continue;
                                }
                            } else {
                                if (QHijl - sqrtHijl2r2 >= 1) {
                                    ++count_R, screening[i]=true;
                                    continue;
                                }
                                double QmHijl = a.dot(Qm.cwiseProduct(a)) - b.dot(Qm.cwiseProduct(b));
                                if (Qm2-r2*QmHijl/sqrtHijl2r2 <= 0)
                                    continue;
                                double alpha= sqrt( (Hijl2*Qm2-QmHijl*QmHijl)/(Qm2*(r2-Qm2)) );
                                double beta = alpha - QmHijl/Qm2;
                                if (QHijl-(beta*QmHijl+Hijl2)/alpha >= 1){
                                    ++count_R, screening[i]=true;
                                    continue;
                                }
                            }
                        }
                    }
                }break;
                case GB_SEMIDEF:{
                    const VectorXd& gradientDiv2Lam = Grad / (2*lambda);
                    const VectorXd& Q = M - gradientDiv2Lam;
                    const double r2 = gradientDiv2Lam.squaredNorm();
                    const VectorXd& Qp = projectPositive(Q);
                    for (size_t i = 0; i < length; ++i) {
                        if (screening[i])
                            continue;
                        if (dot[i] <= 1-gamma || 1 <= dot[i]){
                            const VectorXd& a = X.row(get<0>(tripletIndex[i])) - X.row(get<2>(tripletIndex[i]));
                            const VectorXd& b = X.row(get<0>(tripletIndex[i])) - X.row(get<1>(tripletIndex[i]));
                            const VectorXd& Hijl = a.cwiseProduct(a) - b.cwiseProduct(b);
                            const double Hijl2 = tripletNorm[i];
                            if (dot[i] <= 1 - gamma) {
                                if (Hijl.dot(Q) + sqrt(Hijl2 * r2) <= 1 - gamma || !exist(1-gamma, Hijl, Q, r2, Qp)
                                    //|| -minimize(-Hijl, Q, r2) <= 1 - gamma
                                ){
                                    ++count_L, screening[i]=true;
                                    continue;
                                }
                            } else {
                                if (Hijl.dot(Q) - sqrt(Hijl2 * r2) >= 1 || !exist(1, Hijl, Q, r2, Qp)
                                    //|| minimize(Hijl, Q, r2) >= 1
                                ){
                                    ++count_R, screening[i]=true;
                                    continue;
                                }
                            }
                        }
                    }
                }break;
                case PGB_SEMIDEF:{
                    const VectorXd &gradientDiv2Lam = Grad / (2*lambda);
                    const VectorXd &Q = M - gradientDiv2Lam;
                    const double r2 = gradientDiv2Lam.squaredNorm();
                    const VectorXd &Qp = projectPositive(Q);
                    const double rp2 = r2 - (Q - Qp).squaredNorm();
                    for (size_t i = 0; i < length; ++i) {
                        if (screening[i])
                            continue;
                        if (dot[i] <= 1-gamma || 1 <= dot[i]){
                            const VectorXd& a = X.row(get<0>(tripletIndex[i])) - X.row(get<2>(tripletIndex[i]));
                            const VectorXd& b = X.row(get<0>(tripletIndex[i])) - X.row(get<1>(tripletIndex[i]));
                            const VectorXd& Hijl = a.cwiseProduct(a) - b.cwiseProduct(b);
                            const double Hijl2 = tripletNorm[i];
                            if (dot[i] <= 1 - gamma) {
                                if (Hijl.dot(Qp) + sqrt(Hijl2 * rp2) <= 1 - gamma || !exist(1 - gamma, Hijl, Qp, rp2, Qp)
                                    //|| -minimize(-Hijl, Qp, rp2) <= 1 - gamma
                                ){
                                    ++count_L, screening[i]=true;
                                    continue;
                                }
                            } else {
                                if (Hijl.dot(Qp) - sqrt(Hijl2 * rp2) >= 1 || !exist(1, Hijl, Qp, rp2, Qp)
                                    //|| minimize(Hijl, Qp, rp2) >= 1
                                ){
                                    ++count_R, screening[i]=true;
                                    continue;
                                }
                            }
                        }
                    }
                }break;
                case RRPB_PGB:{
                    const VectorXd &gradientDiv2Lam = Grad / (2*lambda);
                    const VectorXd &Q = M - gradientDiv2Lam;
                    const double r2 = gradientDiv2Lam.squaredNorm();
                    const VectorXd &Qp = projectPositive(Q);
                    const double rp2 = r2 - (Q - Qp).squaredNorm();
                    const double rRRPB = (loop == 0 && lambda < lambda_prev) ? 
                    (((lambda_prev-lambda)/(2*lambda)) * sqrt(M2) 
                        + (lambda_prev/lambda) * sqrt(2*(loss+0.5*lambda_prev*M2+0.5*alphaH2/lambda_prev-z_sum + 0.5 * gamma * z2)/lambda_prev))
                    : sqrt(2 * gap / lambda);
                    const double coeff = (loop == 0 && lambda < lambda_prev) ? ((lambda_prev+lambda)/(2*lambda)) : 1;
                    for (size_t i = 0; i < length; ++i) {
                        if (screening[i])
                            continue;
                        if (dot[i] <= 1-gamma || 1 <= dot[i]){
                            const double Hijl2 = tripletNorm[i];
                            const double HijlQRRPB = dot[i] * coeff;
                            const double sqrtHijl2r2RRPB = rRRPB * sqrt(Hijl2);
                            if (HijlQRRPB + sqrtHijl2r2RRPB <= 1-gamma) {
                                ++count_L, screening[i]=true;
                                continue;
                            } else if (HijlQRRPB - sqrtHijl2r2RRPB >= 1) {
                                ++count_R, screening[i]=true;
                                continue;
                            }
                            const VectorXd& a = X.row(get<0>(tripletIndex[i])) - X.row(get<2>(tripletIndex[i]));
                            const VectorXd& b = X.row(get<0>(tripletIndex[i])) - X.row(get<1>(tripletIndex[i]));
                            const double HijlQp = a.dot(Qp.cwiseProduct(a)) - b.dot(Qp.cwiseProduct(b));
                            const double sqrtHijl2rp2 = sqrt(Hijl2 * rp2);
                            if (HijlQp + sqrtHijl2rp2 <= 1-gamma) {
                                ++count_L, screening[i]=true;
                                continue;
                            } else if (HijlQp - sqrtHijl2rp2 >= 1) {
                                ++count_R, screening[i]=true;
                                continue;
                            }
                        }
                    }
                }break;
            }
            clock_t screeningEnd = clock(); 
            screeningTime += screeningEnd-screeningStart;
            os << "#L = " <<count_L<< endl;
            os << "#R = " <<count_R<< endl;
            os <<"screeningTime = "<< (double)(screeningEnd-screeningStart) / CLOCKS_PER_SEC << endl;
            // ----------Screening End----------
        }

        if (loop != 0){
            const VectorXd &M_diff = M-M_prev;
            const VectorXd &G_diff = Grad-Grad_prev;
            const double dot = M_diff.dot(G_diff);
            const double G_sqr = G_diff.dot(G_diff);
            const double M_sqr = M_diff.dot(M_diff);
            alpha = 0.5 * fabs(dot/G_sqr + M_sqr/dot);
            if (std::isnan(alpha)){
                os<<"alpha is nan."<<endl;
                exit(1);
            }
        }
        if (reset){
            unsigned active=0;
            for (size_t i = 0; i < length; ++i)
                if (!screening[i])
                    activeSet[i] = (dot[i]<1), active += activeSet[i];
            os<<"#active = "<<active<<endl;
            reset=false;
        }
        primal_prev = primal, M_prev = M, Grad_prev = Grad;
        M = projectPositive(M - alpha * Grad);
    }
    clock_t end = clock();
    os << "time = " << (double) (end - start) / CLOCKS_PER_SEC << endl;
    os << "totalScreeningTime = " << (double) screeningTime / CLOCKS_PER_SEC << endl;
    //os << "M = " << endl << M << endl;
    os << "||M||^2 = " << M2 << endl;
    unsigned count_L_fact = 0, count_C_fact = 0, count_R_fact = 0;
    for (size_t i = 0; i < length; ++i) {
        if ( dot[i] < 1 - gamma )
            ++count_L_fact;
        else if ( dot[i] > 1 )
            ++count_R_fact;
        else
            ++count_C_fact;
    }
    os << "#L* = " << count_L_fact << endl;
    os << "#C* = " << count_C_fact << endl;
    os << "#R* = " << count_R_fact << endl;
    return loss;
}
