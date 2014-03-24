/* featurevector.cc: The file responsible for transforming the document into the feature space.
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



#include <config.h>

#include <xapian.h>
#include <xapian/intrusive_ptr.h>
#include <xapian/types.h>
#include <xapian/visibility.h>

#include "featurevector.h"
#include "featuremanager.h"

#include <list>
#include <map>

#include "str.h"
#include "stringutils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "safeerrno.h"
#include "safeunistd.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


using namespace std;

using namespace Xapian;

FeatureVector::FeatureVector() {
    this->label = 0.0;
    this->score = 0.0;
    this->fcount = 0;
    this->did = 0;
}

FeatureVector::FeatureVector(const FeatureVector & o) {
    this->label = o.label;
    this->score = o.score;
    this->fcount = o.fcount;
    this->fvals = o.fvals;
    this->did = o.did;
}

// set methods
void
FeatureVector::set_label(double label1) {
    this->label=label1;
}

void
FeatureVector::set_score(double score1) {
    this->score=score1;
}

void
FeatureVector::set_fvals(map<int,double> fvals1) {
    this->fvals=fvals1;
}

void
FeatureVector::set_fcount(int fcount1) {
    this->fcount = fcount1;
}

void
FeatureVector::set_did(const Xapian::docid & did1) {
    this->did=did1;
}

void
FeatureVector::set_feature_value(int index, double value) {
    this->fvals[index] = value;
}

//get methods
double
FeatureVector::get_label() {
    return this->label;
}

double
FeatureVector::get_score() {
    return this->score;
}

std::map<int,double>
FeatureVector::get_fvals() {
    return this->fvals;
}

int
FeatureVector::get_fcount(){
    return this->fcount;
}

Xapian::docid
FeatureVector::get_did() {
    return this->did;
}

double
FeatureVector::get_feature_value(int index) {
    map<int,double>::const_iterator iter;
    iter = this->fvals.find(index);
    if(iter == this->fvals.end())
	return 0;
    else
	return (*iter).second;
}

int
FeatureVector::get_non_zero_features(){
    int non_zero_fv = 0;
    for(unsigned int i = 1; i <= this->fvals.size(); ++i){     // FIXME: hardcoded to 20 
        if(fvals[i] != 0)   
            non_zero_fv++;
    }
    return non_zero_fv;
}

void
FeatureVector::printFeatureVector() const{
    std::map<int,double> fv =  this->fvals;
    for(int j=1; j<=19; ++j) {
        cout<<j<<":"<<fv[j]<<" ";
    }
        cout<<endl;
}
