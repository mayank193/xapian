/* svmranker.cc: SVM ranker to rank the training data.
 *
 * Copyright (C) 2012 Parth Gupta
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
 * USA
 */


#include <xapian.h>
#include <xapian/intrusive_ptr.h>
#include <xapian/types.h>
#include <xapian/visibility.h>

#include "ranker.h"
#include "ranklist.h"
#include "svmranker.h"
//#include "evalmetric.h"

#include "str.h"
#include "stringutils.h"
#include <string.h>

#include <list>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <math.h>

#include <libsvm/svm.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;
using namespace Xapian;

struct svm_parameter param;
struct svm_problem prob;
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

static string get_cwd() {
    char temp[200];
    return (getcwd(temp, 200) ? std::string(temp) : std::string());
}

SVMRanker::SVMRanker() {
    this->model_file_name = get_cwd().append("/model.txt");
    for(int i = 0; i<num_of_features;++i){
        weight[i] = 0.0;
    }
}

/* Override all the four methods below in the ranker sub-classes files
 * wiz svmranker.cc , listnet.cc, listmle.cc and so on
 */

void SVMRanker::load_model(const std::string & model_file){
    (void)model_file;
}

std::vector<double> SVMRanker::rank(const Xapian::RankList & rl) {
    // FIXME
    (void)rl;
    return std::vector<double>();
}

void SVMRanker::learn_model(){

    // default values
    param.svm_type = 4;             // set to 4 (corresponds to nu-SVR)
    param.kernel_type = 0;          // set to 0 (corresponds to linear Kernel)
    param.degree = 3;               // set degree in kernel function
    param.gamma = 0;                // set gamma in kernel function (default 1/num_features)
    param.coef0 = 0;                // set coef0 in kernel function
    param.nu = 0.5;                 // set the parameter nu of nu-SVC, one-class SVM, and nu-SVR
    param.cache_size = 100;         // set cache memory size in MB
    param.C = 1;                    // set the parameter C of C-SVC, epsilon-SVR, and nu-SVR
    param.eps = 1e-3;               // set tolerance of termination criterion
    param.p = 0.1;                  // for EPSILON_SVR
    param.shrinking = 1;            // whether to use the shrinking heuristics, 0 or 1
    param.probability = 0;          // whether to train an SVC or SVR model for probability estimates, 0 or 1
    param.nr_weight = 0;            // set the parameter C of class i to weight*C in C-SVC
    param.weight_label = NULL;      // for C_SVC
    param.weight = NULL;            // for C_SVC


    vector<Xapian::RankList> rl = this->traindata;
    
    /* prob.l is the size of training dataset */
    prob.l = 0;
    for(unsigned int i = 0; i < rl.size(); ++i){
        prob.l += rl[i].get_num_of_feature_vectors();
    }
    cout<<"Number of features are "<<prob.l<<endl;

    /* Allocating memory to  the labels(y) and feature vectors(x)*/
    prob.y = new double [prob.l];
    prob.x = new svm_node* [prob.l];
    cout<<__FILE__<<":"<<__LINE__<<endl;
    
    /* Get the number of non-zero features in the feature vector
     * and assign memory to number of non-zero features plus one
     * sentential feature.
     */
    int feature_index = 0;
    for(unsigned int i = 0; i < rl.size(); ++i){
        vector<Xapian::FeatureVector> fv = rl[i].get_data();
        for(unsigned int j = 0; j < fv.size(); ++j){
            int non_zero_elements = fv[j].get_non_zero_features();
            cout<<feature_index<<" Non zero features "<< non_zero_elements<< endl;
            prob.x[feature_index] = new svm_node [non_zero_elements+1];
            feature_index++;
        }
    }

    cout<<__FILE__<<":"<<__LINE__<<endl;

    /* Generating the feature vector prob.x with all the non-zero
     * features (sparse matrix representation) and the sentential
     * feature with index set to -1  and value to -1.
     */
    feature_index = 0;
    cout<<"Size of ranklist"<<rl.size()<<endl;
    for(unsigned int i = 0; i < rl.size(); ++i){
        cout<<"Value of i"<<i<<endl;
        vector<Xapian::FeatureVector> fv = rl[i].get_data();
        for(unsigned int j = 0; j < fv.size(); ++j){
            prob.y[feature_index] = fv[j].get_label();
            cout<<feature_index<<" Label: "<<prob.y[feature_index]<<endl;
            map <int,double> fvals = fv[j].get_fvals();
            int last_nonzero_value = -1;
            for(unsigned int z = 1; z <= fvals.size(); ++z){
                
                if(fvals[z] != 0){
                    prob.x[feature_index][z-1].index = z;
                    prob.x[feature_index][z-1].value = fvals[z];
                    cout<<"("<<prob.x[feature_index][z-1].index<<","<<prob.x[feature_index][z-1].value<<") ";
                    last_nonzero_value++;
                }

            } // endfor
            cout<<endl;
            prob.x[feature_index][last_nonzero_value+1].index = -1;
            prob.x[feature_index][last_nonzero_value+1].value = -1;
            feature_index++;
        } // endfor

    } // endfor

    for(int i = 0; i < prob.l; i++){
        for(unsigned int j = 0 ; j <= 19; ++j){
            cout<<"("<<prob.x[i][j].index<<","<<prob.x[i][j].value<<") ";
        }
        cout<<endl;
    }
    // if (param.kernel_type == PRECOMPUTED)
    // for (int i = 0; i < prob.l; ++i) {
    //     if (prob.x[i][0].index != 0) {
    //     fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
    //     exit(1);
    //     }
    //     if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index) {
    //     fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
    //     exit(1);
    //     }
    // }
    cout<<__FILE__<<":"<<__LINE__<<endl;

    const char *error_msg;

    error_msg = svm_check_parameter(&prob, &param);
    if (error_msg) {
    fprintf(stderr, "svm_check_parameter failed: %s\n", error_msg);
    exit(1);
    }
    cout<<__FILE__<<":"<<__LINE__<<endl;

    model = svm_train(&prob, &param);
    cout<<__FILE__<<":"<<__LINE__<<endl;
    if (svm_save_model(model_file_name.c_str(), model)) {
    fprintf(stderr, "can't save model to file %s\n", model_file_name.c_str());
    exit(1);
    }
}


void SVMRanker::save_model() {
    // FIXME
}
 
double SVMRanker::score(const Xapian::FeatureVector & fv) {
    // FIXME
    (void)fv;
    return 3.14;
}

//double SVMRanker::score(const Xapian::FeatureVector & fv);
