/** @file api_stem.cc
 * @brief Test the stemming API
 */
/* Copyright (C) 2010 Olly Betts
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
 */

#include <config.h>

#include "api_stem.h"

#include <xapian.h>

#include "apitest.h"
#include "testsuite.h"
#include "testutils.h"

using namespace std;

class MyStemImpl : public Xapian::StemImplementation {
    string operator()(const string & word) {
	return word.substr(0, 3);
    }

    string get_description() const {
	return "MyStem()";
    }
};

/// Test user stemming algorithms.
DEFINE_TESTCASE(stem1, !backend) {
    Xapian::Stem st(new MyStemImpl);

    TEST_EQUAL(st("a"), "a");
    TEST_EQUAL(st("foo"), "foo");
    TEST_EQUAL(st("food"), "foo");

    return true;
}