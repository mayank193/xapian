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

using namespace std;


using namespace Xapian;


EvalMetric::EvalMetric() {
}

double
precision(const Xapian::RankList & rl, int n){
	vector <Xapian::FeatureVector> fv = rl.get_data();
	int postive_labels = 0;
	for(int i = 0; i < n; ++i){
		if(fv[i].get_label != 0){
			postive_labels++;
		}
	}
	double precision = postive_labels/n;
	return precision;
}

double
average_precision(const Xapian::RankList & rl){
	double total_precision = 0.0;
	vector <Xapian::FeatureVector> fv = rl.get_data();
	for(int i =0; i < fv.size(); ++i){
		if(fv.get_label()){
			total_precision += precision(rl,i);
		}
	}

	double avg_precision = total_precision/precision(rl,fv.size());
	return avg_precision;
}

double
map_score(const vector<Xapian::RankList> rl){
	double mean_avg_precision = 0.0;
	for(int i = 0; i < rl.size(); ++i){
		mean_avg_precision += average_precision(rl[i]);
	}
	mean_avg_precision /= rl.size();
	return mean_avg_precision;
}

    /* override this in the sub-class like MAP, NDCG, MRR, etc*/
double
EvalMetric::score(const Xapian::RankList & /*rl*/) {
return 1.0;
}

