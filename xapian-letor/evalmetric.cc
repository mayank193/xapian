/* evalmetric.h: The abstract evaluation score file.
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

#include <ranklist.h>
#include <evalmetric.h>

#include <list>
#include <map>
#include <cmath>
#include <iostream>

using namespace std;


using namespace Xapian;


EvalMetric::EvalMetric() {
}

double
EvalMetric::precision(Xapian::RankList & rl, int n){
	vector <Xapian::FeatureVector> fv = rl.get_data();
	int postive_labels = 0;
	for(int i = 0; i < n; ++i){
		if(fv[i].get_label() != 0){
			postive_labels++;
		}
	}
	double precision_value = postive_labels/double(n);
	return precision_value;
}

double
EvalMetric::average_precision(Xapian::RankList & rl){
	double total_precision = 0.0;
	vector <Xapian::FeatureVector> fv = rl.get_data();
	for(unsigned int i =0; i < fv.size(); ++i){
		if(fv[i].get_label()){
			total_precision += precision(rl,i+1);
		}
	}
	double avg_precision = 0.0;
	if(fv.size() != 0)
	avg_precision = total_precision/precision(rl,fv.size());
	return avg_precision;
}

double
EvalMetric::map_score(vector<Xapian::RankList> & rl){
	double mean_avg_precision = 0.0;
	for(unsigned int i = 0; i < rl.size(); ++i){
		mean_avg_precision += average_precision(rl[i]);
		cout << average_precision(rl[i])<<endl;
	}
	mean_avg_precision /= rl.size();
	return mean_avg_precision;
}

double
EvalMetric::discount_cumulative_gain(vector <Xapian::FeatureVector> fv){
	double dcg = 0.0;		// dcg stands for discout cumulative gain
	for(unsigned int i = 0; i < fv.size(); ++i){
		//dcg += (math.pow(2,fv[i].get_score()) / math.log2(2+i));
		// Here it is log(2+j) instead of log(1+j) because i starts with 0.
	}
	return dcg;
}

double
EvalMetric::ndcg_score(vector<Xapian::RankList> & rl){
	double mean_ndcg = 0.0;
	for(unsigned int i = 0; i < rl.size(); ++i){
		vector <Xapian::FeatureVector> fv = rl[i].get_data();
		vector <Xapian::FeatureVector> fv_sorted = rl[i].sort_by_score();
		double normalized_dcg = discount_cumulative_gain(fv)/discount_cumulative_gain(fv_sorted);
		mean_ndcg += normalized_dcg;
	}
	mean_ndcg /= rl.size();
	return mean_ndcg;
}

    /* override this in the sub-class like MAP, NDCG, MRR, etc*/
double
EvalMetric::score(const Xapian::RankList & /*rl*/) {
return 1.0;
}

