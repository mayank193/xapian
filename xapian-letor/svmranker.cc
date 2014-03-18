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
#include "letor_internal.h"
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

struct svm_node *x;

static void
read_problem(const char *filename) {
    int elements, max_index, inst_max_index, i, j;
    FILE *fp = fopen(filename, "r");
    char *endptr;
    char *idx, *val, *label;

    if (fp == NULL) {
    fprintf(stderr, "can't open input file %s\n", filename);
    exit(1);
    }

    prob.l = 0;
    elements = 0;
    max_line_len = 1024;
    line = Malloc(char, max_line_len);

    while (readline(fp) != NULL) {
    char *p = strtok(line, " \t"); // label

    // features
    while (1) {
        p = strtok(NULL, " \t");
        if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
        break;
        ++elements;
    }
    ++elements;
    ++prob.l;
    }
    rewind(fp);

    prob.y = Malloc(double, prob.l);
    prob.x = Malloc(struct svm_node *, prob.l);
    x_space = Malloc(struct svm_node, elements);

    max_index = 0;
    j = 0;

    for (i = 0; i < prob.l; ++i) {
    inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
    readline(fp);
    prob.x[i] = &x_space[j];
    label = strtok(line, " \t\n");
    if (label == NULL) // empty line
        exit_input_error(i + 1);
    prob.y[i] = strtod(label, &endptr);
    if (endptr == label || *endptr != '\0')
        exit_input_error(i + 1);

    while (1) {
        idx = strtok(NULL, ":");
        val = strtok(NULL, " \t");

        if (val == NULL)
        break;

        errno = 0;
        x_space[j].index = (int)strtol(idx, &endptr, 10);

        if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
        exit_input_error(i + 1);
        else
        inst_max_index = x_space[j].index;

        errno = 0;
        x_space[j].value = strtod(val, &endptr);

        if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
        exit_input_error(i + 1);

        ++j;
    }

    if (inst_max_index > max_index)
        max_index = inst_max_index;
    x_space[j++].index = -1;
    }

    if (param.gamma == 0 && max_index > 0)
    param.gamma = 1.0 / max_index;

    if (param.kernel_type == PRECOMPUTED)
    for (i = 0; i < prob.l; ++i) {
        if (prob.x[i][0].index != 0) {
        fprintf(stderr, "Wrong input format: first column must be 0:sample_serial_number\n");
        exit(1);
        }
        if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index) {
        fprintf(stderr, "Wrong input format: sample_serial_number out of range\n");
        exit(1);
        }
    }
    fclose(fp);
}
SVMRanker() {
    this->model = get_cwd().append("/model.txt");
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
    //cross_validation = 0;

    printf("Learning the model..");
    string input_file_name;
    string model_file_name;
    //const char *error_msg;

    input_file_name = get_cwd().append("/train.txt");
    model_file_name = get_cwd().append("/model.txt");
    
    read_problem(input_file_name.c_str());
    error_msg = svm_check_parameter(&prob, &param);
    if (error_msg) {
    fprintf(stderr, "svm_check_parameter failed: %s\n", error_msg);
    exit(1);
    }

    model = svm_train(&prob, &param);
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
